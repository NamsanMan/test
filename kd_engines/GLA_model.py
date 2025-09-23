"""Utilities for Global-Local Attention based feature alignment."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["get_GLA_model", "GLA_model", "GlobalLocalAttentionAlign"]


def _pair(value: int) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class PatchEmbed(nn.Module):
    """A lightweight patch embedding layer used by the original GLA module."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: Optional[int] = None) -> None:
        super().__init__()
        patch_size_2d = _pair(patch_size)
        stride = stride if stride is not None else patch_size
        stride_2d = _pair(stride)

        self.patch_size = patch_size_2d
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size_2d, stride=stride_2d, padding=0)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Pad so that spatial dims are divisible by patch size
        H, W = x.shape[-2:]
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x = self.proj(x)
        _, _, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, (Hp, Wp)


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(patches)
        k = self.key(patches)
        v = self.value(patches)

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / (q.size(-1) ** 0.5)
        attn = self.softmax(attn)
        out = torch.bmm(attn, v)
        return out, attn


class GlobalLocalAttentionAlign(nn.Module):
    """Global-Local Attention distillation loss.

    The original repository assumed fixed feature shapes.  This implementation
    keeps the spirit of the method while adapting all projections lazily to the
    incoming feature tensors from the current code base.
    """

    def __init__(self, embed_dim: int = 64, patch_size: int = 8, stride: Optional[int] = None) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.stride = stride if stride is not None else patch_size

        # Modules are built lazily once feature shapes are known
        self.student_proj: Optional[nn.Module] = None
        self.teacher_proj: Optional[nn.Module] = None
        self.student_patch: Optional[nn.Module] = None
        self.teacher_patch: Optional[nn.Module] = None
        self.student_attn: Optional[nn.Module] = None
        self.teacher_attn: Optional[nn.Module] = None

    def _build_if_needed(self, student_c: int, teacher_c: int, device: torch.device) -> None:
        if self.student_proj is not None:
            return

        self.student_proj = nn.Conv2d(student_c, self.embed_dim, kernel_size=1, bias=False).to(device)
        self.teacher_proj = nn.Conv2d(teacher_c, self.embed_dim, kernel_size=1, bias=False).to(device)

        self.student_patch = PatchEmbed(self.embed_dim, self.embed_dim, self.patch_size, self.stride).to(device)
        self.teacher_patch = PatchEmbed(self.embed_dim, self.embed_dim, self.patch_size, self.stride).to(device)

        self.student_attn = SelfAttention(self.embed_dim).to(device)
        self.teacher_attn = SelfAttention(self.embed_dim).to(device)

    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        device = student_feat.device
        self._build_if_needed(student_feat.shape[1], teacher_feat.shape[1], device)
        assert self.student_proj is not None and self.teacher_proj is not None
        assert self.student_patch is not None and self.teacher_patch is not None
        assert self.student_attn is not None and self.teacher_attn is not None

        target_size = student_feat.shape[-2:]
        teacher_feat = F.interpolate(teacher_feat, size=target_size, mode="bilinear", align_corners=False)

        s_proj = self.student_proj(student_feat)
        t_proj = self.teacher_proj(teacher_feat)

        s_tokens, _ = self.student_patch(s_proj)
        t_tokens, _ = self.teacher_patch(t_proj)

        # If padding leads to slightly different token counts, align them
        if s_tokens.shape[1] != t_tokens.shape[1]:
            min_tokens = min(s_tokens.shape[1], t_tokens.shape[1])
            s_tokens = s_tokens[:, :min_tokens, :]
            t_tokens = t_tokens[:, :min_tokens, :]

        _, s_attn = self.student_attn(s_tokens)
        _, t_attn = self.teacher_attn(t_tokens)

        return F.mse_loss(s_attn, t_attn)


class GLA_model(GlobalLocalAttentionAlign):
    """Backward compatible wrapper name kept from the original source."""

    def __init__(self, batchsize: Optional[int] = None, **kwargs) -> None:  # noqa: D401 - signature kept for compatibility
        # ``batchsize`` is ignored but kept for API compatibility with the
        # original implementation which relied on a fixed batch size.
        del batchsize
        super().__init__(**kwargs)


def get_GLA_model(batchsize: Optional[int] = None, **kwargs) -> GLA_model:
    model = GLA_model(batchsize=batchsize, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model