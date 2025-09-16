# kd_engines/transtocnn_gl.py
# Group-wise Linear projector KD (GL-KD):
#   - Teacher: SegFormer (MiT-B5)
#   - Student: DeepLabV3+ (MobileNetV2)
# Projector:
#   - A 4x4 spatial residue group shares one FC -> total 16 FCs
#   - Dropout applied for robustness
# Loss:
#   L_total = w_ce * CE(student logits, GT) + w_gl * || h_T - h'_S ||_2^2
#
# Usage:
#   teacher = SegFormerWrapper("segformerb5", num_classes=12).to(device).eval()
#   student = create_deeplab(in_channels=3, classes=12).to(device)
#   engine  = TransToCNN_GLKD(teacher, student, ignore_index=255, num_classes=12)
#   out = engine.compute_losses(imgs, masks, device=device)
#   out["total"].backward(); optimizer.step()
#
# Notes:
#   - Teacher backbone is frozen (no grad).
#   - Projector is lazy-built on first forward once channel dims are known.
#   - Void/ignore_index masked for the GL loss as well as CE.

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- small utilities ----------
def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

def _valid_mask_like(masks: torch.Tensor, ref: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # masks: [B,H,W] -> [B,1,H',W'] (nearest for labels)
    m = F.interpolate(masks.unsqueeze(1).float(), size=ref.shape[-2:], mode="nearest").squeeze(1).long()
    return (m != ignore_index).float().unsqueeze(1)


# ---------- Group-wise Linear projector ----------
class GroupWiseLinearProjector(nn.Module):
    """
    4x4 spatial residue groups -> 16 shared Linear layers.
    For each token at (i,j), select FC by (i%4, j%4) to map C_s -> C_t.
    Dropout is applied on tokens before FC.

    forward(S:[B,Cs,H,W]) -> S':[B,Ct,H,W]
    """
    def __init__(self, in_ch: int, out_ch: int, p_dropout: float = 0.1):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.p = float(p_dropout)

        # 16 FC layers (no bias, you can enable bias if you want)
        self.fcs = nn.ModuleList(
            [nn.Linear(self.in_ch, self.out_ch, bias=False) for _ in range(16)]
        )
        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")

        # token-wise dropout
        self.dropout = nn.Dropout(p=self.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Cs, H, W]
        return: [B, Ct, H, W]
        """
        B, Cs, H, W = x.shape
        out = x.new_zeros(B, self.out_ch, H, W)

        # Iterate over residue groups
        for r in range(4):
            for c in range(4):
                xs = x[:, :, r::4, c::4]                         # [B, Cs, Hr, Wr]
                Hr, Wr = xs.shape[-2:]
                if Hr == 0 or Wr == 0:
                    continue
                tokens = xs.permute(0, 2, 3, 1).reshape(-1, Cs)   # [B*Hr*Wr, Cs]
                tokens = self.dropout(tokens)                     # dropout on tokens
                proj = self.fcs[r * 4 + c](tokens)                # [B*Hr*Wr, Ct]
                proj = proj.view(B, Hr, Wr, self.out_ch).permute(0, 3, 1, 2).contiguous()
                out[:, :, r::4, c::4] = proj
        return out


# ---------- GL-KD engine ----------
class TransToCNN_GLKD(nn.Module):
    """
    Group-wise Linear projector KD:
      - Take encoder last features (teacher/student)
      - Resize teacher -> student spatial size
      - h'_S = GLProjector(s_last)  (C_s -> C_t per 4x4 residue group)
      - L_gl  = mean_s ( || h_T - h'_S ||_2^2 ), masked by ignore_index
      - total = w_ce * CE(s_logits, GT) + w_gl * L_gl

    Args:
        teacher, student: (imgs, return_feats=True) -> (logits, feats[list])
        ignore_index: void label value
        num_classes: segmentation classes
        p_dropout: token dropout prob inside GL projector
        w_ce: CE weight
        w_gl: GL projector loss weight
        match_space: currently teacher->student (fixed)
    """
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        ignore_index: int,
        num_classes: int = 12,
        p_dropout: float = 0.1,
        w_ce: float = 1.0,
        w_gl: float = 1.0,
    ):
        super().__init__()
        # freeze teacher backbone
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = student
        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)

        self.p_dropout = float(p_dropout)
        self.w_ce = float(w_ce)
        self.w_gl = float(w_gl)

        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # lazy projector
        self._gl: Optional[GroupWiseLinearProjector] = None
        self._inited = False

    @torch.no_grad()
    def _t_forward(self, x: torch.Tensor):
        return self.teacher(x, return_feats=True)

    @torch.no_grad()
    def _init_if_needed(self, s_last: torch.Tensor, t_last: torch.Tensor):
        if self._inited:
            return
        Cs = int(s_last.shape[1])
        Ct = int(t_last.shape[1])
        self._gl = GroupWiseLinearProjector(Cs, Ct, p_dropout=self.p_dropout)
        self._inited = True
        self.to(s_last.device)

    def get_extra_parameters(self) -> List[nn.Parameter]:
        if not self._inited or self._gl is None:
            return []
        return list(self._gl.parameters())

    def compute_losses(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        device: Optional[torch.device] = None,
        **_
    ) -> Dict[str, torch.Tensor]:

        # unify device if passed
        if device is not None:
            if next(self.parameters(), torch.empty(0, device=device)).device != device:
                self.to(device)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

        # forward student
        s_logits, s_feats = self.student(imgs, return_feats=True)
        s_last = s_feats[-1]  # [B, Cs, Hs, Ws]

        # forward teacher (frozen)
        with torch.no_grad():
            t_logits, t_feats = self._t_forward(imgs)
            t_last = t_feats[-1]         # [B, Ct, Ht, Wt]
            t_last = _resize_like(t_last, s_last)  # teacher -> student spatial

        # lazy init GL projector (Cs -> Ct)
        self._init_if_needed(s_last, t_last)
        assert self._gl is not None

        # ---- losses ----
        # (1) CE on logits
        ce = self.ce(s_logits, masks)

        # (2) GL projector loss: || h_T - h'_S ||_2^2  (void-masked mean)
        hS = self._gl(s_last)                           # [B, Ct, Hs, Ws]
        # mask
        valid = _valid_mask_like(masks, t_last, self.ignore_index)  # [B,1,Hs,Ws]
        denom = valid.sum().clamp_min(1.0)
        diff2 = (hS - t_last) ** 2
        l_gl = (diff2 * valid).sum() / denom

        total = self.w_ce * ce + self.w_gl * l_gl

        return {
            "total": total,
            "ce_student": ce.detach(),
            "gl_loss": l_gl.detach(),
        }
