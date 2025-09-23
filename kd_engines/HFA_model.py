"""Utilities for Heterogeneous Feature Alignment distillation."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "get_heterogeneous_feature_align_model",
    "HFA_model",
    "HeterogeneousFeatureAlignLoss",
]


class RRB(nn.Module):
    """Residual Refinement Block used to stabilise alignment."""

    def __init__(self, features: int, out_features: int) -> None:
        super().__init__()
        mid = max(1, out_features // 4)
        self.unify = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(out_features, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_features, kernel_size=3, padding=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(out_features)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.unify(feats)
        residual = self.residual(feats)
        feats = self.norm(feats + residual)
        return feats


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        squeeze = x.view(b, c, -1).mean(dim=2)
        excitation = torch.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        return x * excitation


def compute_weighted_attention_map(feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = feature_map.shape
    flat = feature_map.view(b, c, -1)
    attention_scores = torch.bmm(flat.permute(0, 2, 1), flat)
    attention_map = F.softmax(attention_scores, dim=-1)
    weighted = torch.bmm(attention_map, flat.permute(0, 2, 1))
    weighted = weighted.permute(0, 2, 1).view(b, c, h, w)
    return attention_map, weighted


class HeterogeneousFeatureAlignLoss(nn.Module):
    """Alignment loss between heterogeneous teacher/student features."""

    def __init__(self, aligned_channels: int = 160, reduction: int = 16) -> None:
        super().__init__()
        self.aligned_channels = int(aligned_channels)
        self.reduction = int(reduction)

        self.teacher_proj: Optional[nn.Module] = None
        self.student_proj: Optional[nn.Module] = None
        self.teacher_rrb: Optional[RRB] = None
        self.student_rrb: Optional[RRB] = None
        self.teacher_ca: Optional[ChannelAttention] = None
        self.student_ca: Optional[ChannelAttention] = None
        self.offset_gen: Optional[nn.Module] = None

    def _build_if_needed(self, teacher_c: int, student_c: int, device: torch.device) -> None:
        if self.teacher_proj is not None:
            return

        self.teacher_proj = nn.Conv2d(teacher_c, self.aligned_channels, kernel_size=1, bias=False).to(device)
        self.student_proj = nn.Conv2d(student_c, self.aligned_channels, kernel_size=1, bias=False).to(device)

        self.teacher_rrb = RRB(self.aligned_channels, self.aligned_channels).to(device)
        self.student_rrb = RRB(self.aligned_channels, self.aligned_channels).to(device)

        self.teacher_ca = ChannelAttention(self.aligned_channels, self.reduction).to(device)
        self.student_ca = ChannelAttention(self.aligned_channels, self.reduction).to(device)

        self.offset_gen = nn.Sequential(
            nn.Conv2d(self.aligned_channels * 2, self.aligned_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.aligned_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.aligned_channels, 2, kernel_size=3, padding=1, bias=False),
        ).to(device)
        nn.init.zeros_(self.offset_gen[-1].weight)

    def _bilinear_interpolate(self, input: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        n, _, h, w = input.shape
        ys = torch.linspace(-1.0, 1.0, h, device=input.device, dtype=input.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=input.device, dtype=input.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        norm = torch.tensor([[[[w / 2.0, h / 2.0]]]], device=input.device, dtype=input.dtype)
        grid = base_grid + delta.permute(0, 2, 3, 1) / norm
        return F.grid_sample(input, grid, align_corners=True)

    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        device = student_feat.device
        self._build_if_needed(teacher_feat.shape[1], student_feat.shape[1], device)
        assert self.teacher_proj is not None and self.student_proj is not None
        assert self.teacher_rrb is not None and self.student_rrb is not None
        assert self.teacher_ca is not None and self.student_ca is not None
        assert self.offset_gen is not None

        target_size = teacher_feat.shape[-2:]
        student_resized = F.interpolate(student_feat, size=target_size, mode="bilinear", align_corners=False)

        t_proj = self.teacher_proj(teacher_feat)
        s_proj = self.student_proj(student_resized)

        t_proj = self.teacher_rrb(t_proj)
        s_proj = self.student_rrb(s_proj)

        t_proj = self.teacher_ca(t_proj)
        s_proj = self.student_ca(s_proj)

        offsets = self.offset_gen(torch.cat([t_proj, s_proj], dim=1))
        t_aligned = self._bilinear_interpolate(t_proj, offsets)

        return F.mse_loss(t_aligned, s_proj)


class HFA_model(HeterogeneousFeatureAlignLoss):
    """Backward compatible wrapper name kept from the original source."""

    def __init__(self, batchsize: Optional[int] = None, **kwargs) -> None:  # noqa: D401
        del batchsize
        super().__init__(**kwargs)


def get_heterogeneous_feature_align_model(batchsize: Optional[int] = None, **kwargs) -> HFA_model:
    model = HFA_model(batchsize=batchsize, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model