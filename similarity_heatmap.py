"""Generate a similarity heatmap between DeepLabV3+ and SegFormer intermediate features.

The similarity is measured using centered kernel alignment (CKA). The script collects
intermediate features from both networks on real CamVid samples provided by
``data_loader.py``, computes pair-wise linear CKA scores and visualises the result as a
heatmap.
"""
from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# --- 변경점 1: DataLoader 및 data_loader import 추가 ---
from torch.utils.data import DataLoader
import data_loader as project_data_loader
# ---------------------------------------------------

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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=8,
                        help="Number of batches to sample for the CKA computation.")
    parser.add_argument("--pool-size", type=int, default=16,
                        help="Adaptive AvgPool spatial size before flatten. 0 disables pooling.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use torch.cuda.amp.autocast for faster inference.")

    # --- 변경점 2: 데이터 로더 관련 인자 다시 추가 ---
    parser.add_argument(
        "--data-split",
        type=str,
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split declared in data_loader.py used to source real images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the DataLoader that reads real images.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin host memory for faster host to GPU transfers.",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        help="Shuffle the dataset before sampling batches (enabled by default for train split).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable shuffling even when the training split is used.",
    )
    parser.set_defaults(shuffle=None)
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the last incomplete batch from the data loader when sampling inputs.",
    )
    # -----------------------------------------------
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
                _, features = model(batch, return_feats=True)
        else:
            _, features = model(batch, return_feats=True)

        if not collected:
            collected = [[] for _ in range(len(features))]
        for idx, feat in enumerate(features):
            flat = _flatten_feature(feat, pool_size)
            collected[idx].append(flat)  # keep on device

    if not collected:
        raise RuntimeError("No features were collected. Ensure at least one batch was provided.")
    return [torch.cat(feats, dim=0) for feats in collected]  # (N, D) each, ON DEVICE


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    K_centered = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
    return K_centered


def linear_cka_samplewise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fast linear CKA using sample-wise Gram matrices (works well when D >> N)."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    Kx = x @ x.T
    Ky = y @ y.T

    Kx = _center_gram(Kx)
    Ky = _center_gram(Ky)

    num = (Kx * Ky).sum()
    denom = Kx.norm(p="fro") * Ky.norm(p="fro")
    return num / (denom + 1e-12)


def compute_similarity_matrix(
    deeplab_feats: Sequence[torch.Tensor],
    segformer_feats: Sequence[torch.Tensor],
) -> torch.Tensor:
    m, n = len(deeplab_feats), len(segformer_feats)
    out = torch.zeros(m, n, device=deeplab_feats[0].device)
    for i, df in enumerate(deeplab_feats):
        for j, sf in enumerate(segformer_feats):
            N = min(df.shape[0], sf.shape[0])
            out[i, j] = linear_cka_samplewise(df[:N], sf[:N])
    return out


# --- 변경점 3: 더미 데이터 생성 함수를 실제 데이터 로더용 함수로 교체 ---
def prepare_inputs(
    loader: DataLoader,
    device: torch.device,
    num_batches: int,
) -> List[torch.Tensor]:
    """DataLoader에서 num_batches 만큼 데이터를 추출하여 device로 보냅니다."""
    batches: List[torch.Tensor] = []
    for batch_idx, (images, _labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batches.append(images.to(device, non_blocking=True))
    if not batches:
        raise RuntimeError(
            "No batches were drawn from the dataset. Reduce --num-batches or check the data paths."
        )
    return batches
# -------------------------------------------------------------


def plot_heatmap(
    matrix: torch.Tensor,
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

    # --- 변경점 4: main 함수에 실제 데이터 로딩 로직 추가 ---
    dataset_by_split = {
        "train": project_data_loader.train_dataset,
        "val": project_data_loader.val_dataset,
        "test": project_data_loader.test_dataset,
    }
    try:
        dataset = dataset_by_split[args.data_split]
    except KeyError as exc:
        raise ValueError(f"Unsupported split '{args.data_split}'.") from exc
    if dataset is None:
        raise RuntimeError(
            f"Dataset for split '{args.data_split}' is not available. Check data_loader.py initialisation."
        )

    shuffle = args.shuffle if args.shuffle is not None else args.data_split == "train"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    input_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        generator=loader_generator,
    )

    inputs = prepare_inputs(
        loader=input_loader,
        device=device,
        num_batches=args.num_batches,
    )
    # ----------------------------------------------------

    deeplab_feats = collect_features(deeplab, inputs, pool_size=args.pool_size, use_amp=args.use_amp)
    segformer_feats = collect_features(segformer, inputs, pool_size=args.pool_size, use_amp=args.use_amp)

    cka_matrix = compute_similarity_matrix(deeplab_feats, segformer_feats)

    deeplab_labels = [f"Stage {idx}" for idx in deeplab_stages]
    segformer_labels = [f"Stage {i+1}" for i in range(len(segformer_feats))]

    plot_heatmap(cka_matrix, deeplab_labels, segformer_labels)


if __name__ == "__main__":
    main()