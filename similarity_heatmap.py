"""Generate a similarity heatmap between DeepLabV3+ and SegFormer intermediate features.

The similarity is measured using centered kernel alignment (CKA). The script collects
intermediate features from both networks, computes pair-wise linear CKA scores and
visualises the result as a heatmap.
"""
from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models.d3p import DEFAULT_STAGE_INDICES, create_model
from models.segformer_wrapper import SegFormerWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeplab-encoder", type=str, default="mobilenet_v2",
                        help="Encoder backbone for DeepLabV3+ (see segmentation_models_pytorch encoders).")
    parser.add_argument("--deeplab-weights",
                        type=lambda v: None if v.lower() in {"none", "null"} else v,
                        default="imagenet",
                        help="Pre-trained weights name for the DeepLab encoder. Use 'none' for random init.")
    parser.add_argument("--deeplab-stages", type=str,
                        default=",".join(str(i) for i in DEFAULT_STAGE_INDICES),
                        help="Comma separated indices of DeepLab encoder stages to compare.")
    parser.add_argument("--segformer-name", type=str, default="segformerb0",
                        help="SegFormer variant declared in models/segformer_wrapper.py")
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=16,
                        help="Adaptive AvgPool spatial size before flatten. 0 disables pooling.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use torch.cuda.amp.autocast for faster inference.")
    return parser.parse_args()


def parse_stage_indices(stage_str: str) -> Sequence[int]:
    if not stage_str:
        return DEFAULT_STAGE_INDICES
    indices = []
    for part in stage_str.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return tuple(indices)


def build_models(
    deeplab_encoder: str,
    deeplab_weights: str | None,
    deeplab_stages: Sequence[int],
    segformer_name: str,
    num_classes: int,
    device: torch.device,
):
    deeplab = create_model(
        encoder_name=deeplab_encoder,
        encoder_weights=deeplab_weights,
        in_channels=3,
        classes=num_classes,
        stage_indices=deeplab_stages,
    )
    segformer = SegFormerWrapper(name=segformer_name, num_classes=num_classes)

    deeplab.to(device).eval()
    segformer.to(device).eval()
    return deeplab, segformer


def _flatten_feature(feat: torch.Tensor, pool_size: int | None) -> torch.Tensor:
    """Flatten to shape (N, D) on the SAME device."""
    if feat.dim() == 4:
        if pool_size and pool_size > 0:
            feat = F.adaptive_avg_pool2d(feat, (pool_size, pool_size))
        return feat.flatten(1)
    if feat.dim() == 3:
        # (B, seq, C) -> (B, seq*C)
        return feat.reshape(feat.shape[0], -1)
    return feat.reshape(feat.shape[0], -1)


@torch.inference_mode()
def collect_features(
    model: torch.nn.Module,
    inputs: Iterable[torch.Tensor],
    pool_size: int | None,
    use_amp: bool,
) -> List[torch.Tensor]:
    """Run a model and accumulate intermediate features on DEVICE (no .cpu())."""
    collected: List[List[torch.Tensor]] = []
    use_autocast = torch.cuda.is_available() and use_amp

    for batch in inputs:
        if use_autocast:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, features = model(batch, return_feats=True)
        else:
            logits, features = model(batch, return_feats=True)

        if not collected:
            collected = [[] for _ in range(len(features))]
        for idx, feat in enumerate(features):
            flat = _flatten_feature(feat, pool_size)
            collected[idx].append(flat)  # keep on device

    if not collected:
        raise RuntimeError("No features were collected. Ensure at least one batch was provided.")
    return [torch.cat(feats, dim=0) for feats in collected]  # (N, D) each, ON DEVICE


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    H = I - (1.0 / n)
    # H @ K @ H  ; implement as HK = K - row/col means + grand mean
    K_centered = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
    return K_centered


def linear_cka_samplewise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fast linear CKA using sample-wise Gram matrices (works well when D >> N).
    x, y: (N, D) on SAME device.
    Returns a scalar tensor.
    """
    # Center features (per-dim)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    Kx = x @ x.T          # (N, N)
    Ky = y @ y.T          # (N, N)

    Kx = _center_gram(Kx)
    Ky = _center_gram(Ky)

    num = (Kx * Ky).sum()
    denom = Kx.norm(p="fro") * Ky.norm(p="fro")
    return (num / (denom + 1e-12))


def compute_similarity_matrix(
    deeplab_feats: Sequence[torch.Tensor],
    segformer_feats: Sequence[torch.Tensor],
) -> torch.Tensor:
    # all tensors are on device
    m, n = len(deeplab_feats), len(segformer_feats)
    out = torch.zeros(m, n, device=deeplab_feats[0].device)
    for i, df in enumerate(deeplab_feats):
        for j, sf in enumerate(segformer_feats):
            # align N if needed
            N = min(df.shape[0], sf.shape[0])
            out[i, j] = linear_cka_samplewise(df[:N], sf[:N])
    return out  # ON DEVICE


def prepare_inputs(
    device: torch.device,
    batch_size: int,
    num_batches: int,
    height: int,
    width: int,
    seed: int,
) -> Iterable[torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    for _ in range(num_batches):
        yield torch.rand(batch_size, 3, height, width, device=device, generator=g)


def plot_heatmap(
    matrix: torch.Tensor,  # can be on device
    deeplab_labels: Sequence[str],
    segformer_labels: Sequence[str],
) -> None:
    mat = matrix.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(1.5 * len(segformer_labels), 1.2 * len(deeplab_labels)))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xticks(range(len(segformer_labels)))
    ax.set_yticks(range(len(deeplab_labels)))
    ax.set_xticklabels(segformer_labels, rotation=45, ha="right")
    ax.set_yticklabels(deeplab_labels)
    ax.set_xlabel("SegFormer stages")
    ax.set_ylabel("DeepLabV3+ stages")
    ax.set_title("Similarity heatmap (linear CKA)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # speed knobs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    deeplab_stages = parse_stage_indices(args.deeplab_stages)

    deeplab, segformer = build_models(
        deeplab_encoder=args.deeplab_encoder,
        deeplab_weights=args.deeplab_weights,
        deeplab_stages=deeplab_stages,
        segformer_name=args.segformer_name,
        num_classes=args.num_classes,
        device=device,
    )

    inputs = list(
        prepare_inputs(
            device=device,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            height=args.input_height,
            width=args.input_width,
            seed=args.seed,
        )
    )

    # collect features ON GPU
    deeplab_feats = collect_features(deeplab, inputs, pool_size=args.pool_size, use_amp=args.use_amp)
    segformer_feats = collect_features(segformer, inputs, pool_size=args.pool_size, use_amp=args.use_amp)

    # compute CKA ON GPU (Gram on samples => fast)
    cka_matrix = compute_similarity_matrix(deeplab_feats, segformer_feats)

    deeplab_labels = [f"Stage {idx}" for idx in deeplab_stages]
    segformer_labels = [f"Stage {i+1}" for i in range(len(segformer_feats))]

    plot_heatmap(cka_matrix, deeplab_labels, segformer_labels)


if __name__ == "__main__":
    main()
