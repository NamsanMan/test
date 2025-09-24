# hsam_hfa.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "get_heterogeneous_feature_align_model",
    "HFA_model",
    "HeterogeneousFeatureAlignLoss",
]


# ----------------------------
# Building Blocks
# ----------------------------
class RRB(nn.Module):
    """
    Residual Refinement Block (논문의 RCB에 해당: 1x1 정합 + 3x3-3x3 잔차)
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = max(1, out_ch // 4)
        self.unify = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(out_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unify(x)
        res = self.residual(x)
        x = self.norm(x + res)  # 합 뒤 BN으로 안정화 (ReLU는 생략: 값 범위 팽창 억제)
        return x


class SpatialSelfAttention(nn.Module):
    """
    학생 특징에 적용하는 간단한 공간 Self-Attention (논문 식 (7)의 Att(·))
    입력/출력 shape 유지 (B, C, H, W)
    """
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch, 1, bias=False)
        self.k = nn.Conv2d(ch, ch, 1, bias=False)
        self.v = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).view(b, c, -1).transpose(1, 2)  # (B, HW, C)
        k = self.k(x).view(b, c, -1)                  # (B, C, HW)
        v = self.v(x).view(b, c, -1).transpose(1, 2)  # (B, HW, C)

        attn = (q @ k) / (c ** 0.5)                   # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                # (B, HW, C)

        return out.transpose(1, 2).view(b, c, h, w)   # (B, C, H, W)


# ----------------------------
# HSAM / HFA Loss
# ----------------------------
class HeterogeneousFeatureAlignLoss(nn.Module):
    """
    HSAM 기반 HFA Loss (논문 식 (6)~(15)을 충실히 구현)
    - aligned_channels: proj 후 채널 수 (논문 예: 160 등)
    - offset_scale: tanh로 제한되는 최대 픽셀 오프셋(±offset_scale)
    - align_corners: interpolate/grid_sample 일관 옵션
    """
    def __init__(
        self,
        aligned_channels: int = 160,
        offset_scale: float = 2.0,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.aligned_channels = int(aligned_channels)
        self.offset_scale = float(offset_scale)
        self.align_corners = bool(align_corners)

        # Lazy build (입력 채널을 보고 생성)
        self.teacher_proj: Optional[nn.Conv2d] = None
        self.student_proj: Optional[nn.Conv2d] = None
        self.teacher_rrb: Optional[RRB] = None
        self.student_rrb: Optional[RRB] = None
        self.student_attn: Optional[SpatialSelfAttention] = None

        # h3 위한 3x3 두 갈래 + DELTA_GEN
        self.branch1: Optional[nn.Conv2d] = None
        self.branch2: Optional[nn.Conv2d] = None
        self.delta_gen: Optional[nn.Sequential] = None

    # ---------- internal utils ----------
    def _build_if_needed(self, teacher_c: int, student_c: int, device: torch.device) -> None:
        if self.teacher_proj is not None:
            return

        ch = self.aligned_channels

        # 1) 1x1 projection (채널 정합)
        self.teacher_proj = nn.Conv2d(teacher_c, ch, kernel_size=1, bias=False).to(device)
        self.student_proj = nn.Conv2d(student_c, ch, kernel_size=1, bias=False).to(device)

        # 2) RRB (= RCB) 정제
        self.teacher_rrb = RRB(ch, ch).to(device)
        self.student_rrb = RRB(ch, ch).to(device)

        # 3) 학생 공간 self-attention
        self.student_attn = SpatialSelfAttention(ch).to(device)

        # 4) h3용 3x3 두 갈래
        self.branch1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False).to(device)
        self.branch2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False).to(device)

        # 5) DELTA_GEN: (2*ch) -> ch -> 2 (dx, dy)
        self.delta_gen = nn.Sequential(
            nn.Conv2d(2 * ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, kernel_size=3, padding=1, bias=False),
        ).to(device)
        # offset 0으로 시작 → 안정적
        nn.init.zeros_(self.delta_gen[-1].weight)

    @staticmethod
    def _make_base_grid(n: int, h: int, w: int, device, dtype) -> torch.Tensor:
        """
        grid_sample용 base grid: (B, H, W, 2) with x in last-dim 0, y in last-dim 1, range [-1,1]
        """
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")          # (H, W)
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)       # (1, H, W, 2)
        return base.repeat(n, 1, 1, 1)                                  # (N, H, W, 2)

    def _bilinear_refactor(self, feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """
        논문 식 (12)~(14): offset=(dx,dy)을 [-1,1] 정규화하여 grid_sample로 bilinear 재샘플.
        - align_corners에 맞춰 정규화 계수를 정확히 사용.
        - padding_mode='border'로 경계 안정화.
        """
        n, _, h, w = feat.shape
        base = self._make_base_grid(n, h, w, feat.device, feat.dtype)

        # offset 채널 명시: (B,2,H,W) → (dx,dy)
        dx = offset[:, 0]  # (B, H, W)
        dy = offset[:, 1]  # (B, H, W)

        # 정규화: align_corners=True → 2/(size-1), False → 2/size
        if self.align_corners:
            nx = 2.0 * dx / max(w - 1, 1)
            ny = 2.0 * dy / max(h - 1, 1)
        else:
            nx = 2.0 * dx / max(w, 1)
            ny = 2.0 * dy / max(h, 1)

        grid = torch.empty_like(base)
        grid[..., 0] = base[..., 0] + nx   # x
        grid[..., 1] = base[..., 1] + ny   # y

        out = F.grid_sample(
            feat, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )
        return out

    # ---------- forward ----------
    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """
        Args
        - teacher_feat: (B, C_t, H_t, W_t)
        - student_feat: (B, C_s, H_s, W_s)
        Returns
        - scalar loss (MSE): MSE( f_t_re, rc_s^3 )
        """
        device = student_feat.device
        self._build_if_needed(teacher_feat.shape[1], student_feat.shape[1], device)
        assert self.teacher_proj and self.student_proj
        assert self.teacher_rrb and self.student_rrb
        assert self.student_attn and self.branch1 and self.branch2 and self.delta_gen

        # (식 6) 교사 → 학생 해상도 정렬
        target_size = student_feat.shape[-2:]
        t_resized = F.interpolate(
            teacher_feat, size=target_size, mode="bilinear", align_corners=self.align_corners
        )

        # 1x1 proj (채널 정합)
        t_proj = self.teacher_proj(t_resized)      # (B, C, H, W)
        s_proj = self.student_proj(student_feat)   # (B, C, H, W)

        # (식 7) 학생 공간 self-attention
        s_proj = self.student_attn(s_proj)

        # (식 8) RRB 정제
        rc_t = self.teacher_rrb(t_proj)
        rc_s = self.student_rrb(s_proj)

        # (식 9) 요소곱 융합 h3
        h3 = rc_t * rc_s

        # (식 10) 3x3 두 갈래 후 concat
        b1 = F.relu(self.branch1(h3), inplace=True)
        b2 = F.relu(self.branch2(h3), inplace=True)
        hc3 = torch.cat([b1, b2], dim=1)  # (B, 2C, H, W)

        # (식 11) DELTA_GEN → offset (dx, dy), 안정화를 위해 tanh * scale
        raw_offset = self.delta_gen(hc3)           # (B, 2, H, W)
        offset = torch.tanh(raw_offset) * self.offset_scale

        # (식 12)~(14) Refactor: 교사 특징 재구성
        t_re = self._bilinear_refactor(rc_t, offset)  # (B, C, H, W)

        # (식 15) MSE
        loss = F.mse_loss(t_re, rc_s)
        return loss


# ----------------------------
# Wrapper / Factory
# ----------------------------
class HFA_model(HeterogeneousFeatureAlignLoss):
    """
    Backward compatible wrapper (batchsize 인자만 호환용으로 유지)
    """
    def __init__(self, batchsize: Optional[int] = None, **kwargs) -> None:
        del batchsize
        super().__init__(**kwargs)


def get_heterogeneous_feature_align_model(batchsize: Optional[int] = None, **kwargs) -> HFA_model:
    model = HFA_model(batchsize=batchsize, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
