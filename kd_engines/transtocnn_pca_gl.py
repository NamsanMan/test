# kd_engines/transtocnn_pca_gl.py
# Wrapper: run PCA-KD and GL-KD together in one pass
#
# total = w_ce * CE(student logits, GT)
#        + w_pca * ( w_attn * L_attn + w_v * L_v )
#        + w_gl  * L_gl
#
# Note:
# - Reuses existing engines' internals to avoid double forward & code dup.
# - Teacher backbone is frozen.
# - get_extra_parameters() returns the union of (student-side) adapters.

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# 이 경로는 네 프로젝트 구조에 맞춰 수정하세요.
from kd_engines.transtocnn_pca import TransToCNN_PCAKD, _resize_like, _valid_mask_like
from kd_engines.transtocnn_gl import TransToCNN_GLKD


class TransToCNN_PCA_GL_KD(nn.Module):
    """
    Combine PCA-KD and GL-KD in a single engine:
      - Single forward for teacher & student
      - Lazy-build projectors in both sub-engines
      - Compute CE + PCA losses + GL loss, then weighted sum

    Args:
        teacher, student: (imgs, return_feats=True) -> (logits, feats[list])
        ignore_index: void label
        num_classes: segmentation classes
        # PCA params
        d_attn: Q/K/V channel
        p_replace: probability for partially-cross replacement in PCA
        # GL params
        p_dropout: token dropout prob in GL projector
        # Weights
        w_ce, w_pca, w_attn, w_v, w_gl
        use_mask_on_pca: apply ignore mask to PCA losses as well
    """
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        ignore_index: int,
        num_classes: int = 12,
        # PCA-KD
        d_attn: int = 256,
        p_replace: float = 0.5,
        use_mask_on_pca: bool = True,
        # GL-KD
        p_dropout: float = 0.1,
        # Weights
        w_ce: float = 1.0,
        w_pca: float = 1.0,
        w_attn: float = 1.0,
        w_v: float = 1.0,
        w_gl: float = 1.0,
    ):
        super().__init__()
        # freeze teacher
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = student

        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)

        # weights
        self.w_ce = float(w_ce)
        self.w_pca = float(w_pca)
        self.w_attn = float(w_attn)
        self.w_v = float(w_v)
        self.w_gl = float(w_gl)

        # sub-engines (teacher/student는 공유)
        self.pca = TransToCNN_PCAKD(
            teacher=self.teacher,
            student=self.student,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
            d_attn=int(d_attn),
            p_replace=float(p_replace),
            w_ce=1.0,     # wrapper에서 CE를 통일 계산하므로 여기선 사용 안 함
            w_pca=1.0,    # 내부 합산을 wrapper에서 수행
            w_attn=1.0,
            w_v=1.0,
            use_mask_on_pca=bool(use_mask_on_pca),
        )
        self.gl  = TransToCNN_GLKD(
            teacher=self.teacher,
            student=self.student,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
            p_dropout=float(p_dropout),
            w_ce=1.0,   # wrapper에서 CE 계산
            w_gl=1.0,   # wrapper에서 가중
        )

        # 하나의 CE로 통일
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # lazy flags는 각 엔진 내부에서 관리
        self._inited = False

    @torch.no_grad()
    def _t_forward(self, x: torch.Tensor):
        return self.teacher(x, return_feats=True)

    def get_extra_parameters(self) -> List[nn.Parameter]:
        """
        Return union of student-side projector params from both PCA/GL engines.
        """
        params: List[nn.Parameter] = []
        # PCA student projector
        if getattr(self.pca, "_inited", False) and getattr(self.pca, "_proj_s", None) is not None:
            params += list(self.pca._proj_s.parameters())
        # GL projector
        if getattr(self.gl, "_inited", False) and getattr(self.gl, "_gl", None) is not None:
            params += list(self.gl._gl.parameters())
        return params

    def compute_losses(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        device: Optional[torch.device] = None,
        **_
    ) -> Dict[str, torch.Tensor]:

        # device 호환
        if device is not None:
            if next(self.parameters(), torch.empty(0, device=device)).device != device:
                self.to(device)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

        # ---------- forward (공유) ----------
        # student
        s_logits, s_feats = self.student(imgs, return_feats=True)
        s_last = s_feats[-1]  # [B, Cs, Hs, Ws]

        # teacher (frozen)
        with torch.no_grad():
            t_logits, t_feats = self._t_forward(imgs)
            t_last = t_feats[-1]  # [B, Ct, Ht, Wt]
            t_last = _resize_like(t_last, s_last)  # teacher -> student spatial

        # ---------- lazy build projectors ----------
        # PCA projectors
        self.pca._init_if_needed(s_last, t_last)   # builds _proj_s/_proj_t
        # GL projector
        self.gl._init_if_needed(s_last, t_last)    # builds _gl

        # ---------- CE (공통) ----------
        ce = self.ce(s_logits, masks)

        # ---------- PCA losses (reuse engine's internal function) ----------
        L_attn, L_v = self.pca._pca_losses(s_last, t_last, masks=masks)
        L_pca = self.w_attn * L_attn + self.w_v * L_v

        # ---------- GL loss (manual, reuse GL projector) ----------
        assert self.gl._gl is not None
        hS = self.gl._gl(s_last)            # [B, Ct, Hs, Ws]
        valid = _valid_mask_like(masks, t_last, self.ignore_index)
        denom = valid.sum().clamp_min(1.0)
        l_gl = (((hS - t_last) ** 2) * valid).sum() / denom

        total = self.w_ce * ce + self.w_pca * L_pca + self.w_gl * l_gl

        return {
            "total": total,
            # components
            "ce_student": ce.detach(),
            "pca_attn": L_attn.detach(),
            "pca_v": L_v.detach(),
            "pca_total": L_pca.detach(),
            "gl_loss": l_gl.detach(),
        }
