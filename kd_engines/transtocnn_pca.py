# kd_engines/transtocnn_pca.py
# Partially Cross Attention KD (PCA-KD):
# SegFormer (teacher) -> DeepLabV3+ MobileNetV2 (student)
#
# Loss = CE(student logits, GT)
#      + w_pca * [ w_attn * || Attn_T - PCAttn_S ||_2^2
#                 + w_v    * || (||V_T||^2 / sqrt(d)) - (||V_S||^2 / sqrt(d)) ||_2^2 ]
#
# Paper ref: "Cross-Architecture Knowledge Distillation" (PCA projector) — equations (1-4) 개념.  ⟶ teacher Q/K/V attention space에 student를 맵핑해 모방.
# (교차 아키텍처 KD에서 attention 공간 정렬로 전역 관계 학습을 유도)  :contentReference[oaicite:1]{index=1}

from typing import Dict, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Small utilities (재사용/동일 시그니처) ----------
def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

def _valid_mask_like(masks: torch.Tensor, ref: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # masks: [B,H,W] -> [B,1,H',W'] (nearest)
    m = F.interpolate(masks.unsqueeze(1).float(), size=ref.shape[-2:], mode="nearest").squeeze(1).long()
    return (m != ignore_index).float().unsqueeze(1)

def _flatten_hw(t: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, N, C]  (tokens-first)
    B, C, H, W = t.shape
    return t.view(B, C, H * W).permute(0, 2, 1).contiguous()

def _attn_single(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, d: int) -> torch.Tensor:
    # Q,K,V: [B, N, d]
    # return: [B, N, d]
    attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d)  # [B, N, N]
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, V)  # [B, N, d]
    return out

def _build_3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    m = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
    nn.init.kaiming_normal_((m.weight), mode="fan_out", nonlinearity="relu")
    return m


class _PCAProjector(nn.Module):
    """
    3x(3x3 conv)로 Q/K/V를 생성하는 projector.
    forward(feat:[B,C,H,W]) -> (Q,K,V) where each is [B, d, H, W]
    """
    def __init__(self, in_ch: int, d: int):
        super().__init__()
        self.q = _build_3x3(in_ch, d)
        self.k = _build_3x3(in_ch, d)
        self.v = _build_3x3(in_ch, d)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(feat), self.k(feat), self.v(feat)


class TransToCNN_PCAKD(nn.Module):
    """
    PCA-KD 엔진:
      - teacher/student의 encoder-last feature로 Q/K/V 만들고 self-attention 계산
      - student는 Q/K/V를 p 확률로 teacher 토큰으로 교체한 partially-cross attention으로 계산
      - 두 attention L2 + V-정규항 L2를 합해 PCA loss 구성
      - segmentation CE 포함

    Args:
        teacher, student: (imgs, return_feats=True)->(logits, feats[list])
        ignore_index: void 라벨
        num_classes: 세그 클래스 수
        d_attn: attention 채널 차원(d)
        p_replace: Q/K/V에서 teacher 토큰으로 교체할 확률 p
        w_pca: PCA loss 전체 가중치
        w_attn: PCA 내 attention 항의 가중치
        w_v: PCA 내 V-정규항 가중치
        use_mask_on_pca: void 제외 마스킹을 PCA에도 적용할지
    """
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        ignore_index: int,
        num_classes: int = 12,
        d_attn: int = 256,
        p_replace: float = 0.5,
        w_ce: float = 1.0,
        w_pca: float = 1.0,
        w_attn: float = 1.0,
        w_v: float = 1.0,
        use_mask_on_pca: bool = True,
    ):
        super().__init__()
        # freeze teacher
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = student

        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)
        self.d = int(d_attn)
        self.p = float(p_replace)
        self.w_ce = float(w_ce)
        self.w_pca = float(w_pca)
        self.w_attn = float(w_attn)
        self.w_v = float(w_v)
        self.use_mask_on_pca = bool(use_mask_on_pca)

        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # projector는 첫 forward에서 in_ch를 보고 lazy-build
        self._proj_s: Optional[_PCAProjector] = None
        self._proj_t: Optional[_PCAProjector] = None
        self._inited = False

    @torch.no_grad()
    def _init_if_needed(self, s_last: torch.Tensor, t_last: torch.Tensor):
        if self._inited:
            return
        s_ch = s_last.shape[1]
        t_ch = t_last.shape[1]

        self._proj_s = _PCAProjector(in_ch=int(s_ch), d=self.d)
        self._proj_t = _PCAProjector(in_ch=int(t_ch), d=self.d)
        # teacher projector는 고정(타겟 공간 고정)
        for p in self._proj_t.parameters():
            p.requires_grad_(False)

        self._inited = True
        self.to(s_last.device)

    @torch.no_grad()
    def _t_forward(self, x: torch.Tensor):
        return self.teacher(x, return_feats=True)

    def _pca_losses(
        self,
        s_last: torch.Tensor,
        t_last: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (L_attn, L_v)  — 각자 평균 스칼라
        """
        # 공간 정합: teacher -> student 크기
        t_last = _resize_like(t_last, s_last)

        # Q/K/V 생성
        assert self._proj_s is not None and self._proj_t is not None
        Qs, Ks, Vs = self._proj_s(s_last)         # [B,d,Hs,Ws]
        with torch.no_grad():
            Qt, Kt, Vt = self._proj_t(t_last)     # [B,d,Hs,Ws] (detach target)

        B, d, H, W = Qs.shape
        N = H * W

        # flatten tokens
        Qs_f = _flatten_hw(Qs)  # [B,N,d]
        Ks_f = _flatten_hw(Ks)
        Vs_f = _flatten_hw(Vs)
        Qt_f = _flatten_hw(Qt)
        Kt_f = _flatten_hw(Kt)
        Vt_f = _flatten_hw(Vt)

        # teacher self-attention
        with torch.no_grad():
            AttnT = _attn_single(Qt_f, Kt_f, Vt_f, self.d)  # [B,N,d]

        # partially-cross replacement: element-wise 확률 p로 teacher로 교체
        if self.p > 0.0:
            mask_q = torch.rand_like(Qs_f) < self.p
            mask_k = torch.rand_like(Ks_f) < self.p
            mask_v = torch.rand_like(Vs_f) < self.p
            Qmix = torch.where(mask_q, Qt_f, Qs_f)
            Kmix = torch.where(mask_k, Kt_f, Ks_f)
            Vmix = torch.where(mask_v, Vt_f, Vs_f)
        else:
            Qmix, Kmix, Vmix = Qs_f, Ks_f, Vs_f

        PCAttnS = _attn_single(Qmix, Kmix, Vmix, self.d)  # [B,N,d]

        # ----- Loss 1: attention L2 -----
        # [B,N,d] -> [B, d, H, W] (masking에 쓰기 위해)
        AttnT_map = AttnT.permute(0, 2, 1).view(B, d, H, W)
        PCAttnS_map = PCAttnS.permute(0, 2, 1).view(B, d, H, W)

        if self.use_mask_on_pca and (masks is not None):
            valid = _valid_mask_like(masks, AttnT_map, self.ignore_index)  # [B,1,H,W]
            denom = valid.sum().clamp_min(1.0)
            diffA = (PCAttnS_map - AttnT_map) ** 2
            L_attn = (diffA * valid).sum() / denom
        else:
            L_attn = F.mse_loss(PCAttnS_map, AttnT_map)

        # ----- Loss 2: V-정규항 L2 -----
        # per-token squared-norm / sqrt(d)  -> [B,N,1]
        def _v_stat(Vf: torch.Tensor) -> torch.Tensor:
            v2 = (Vf ** 2).sum(dim=2, keepdim=True) / math.sqrt(self.d)
            return v2  # [B,N,1]

        vT = _v_stat(Vt_f)
        vS = _v_stat(Vs_f)

        if self.use_mask_on_pca and (masks is not None):
            # [B,1,H,W] -> [B,N,1]
            validN = _valid_mask_like(masks, AttnT_map, self.ignore_index).view(B, 1, N).permute(0, 2, 1).contiguous()
            denomN = validN.sum().clamp_min(1.0)
            L_v = ((vS - vT) ** 2 * validN).sum() / denomN
        else:
            L_v = F.mse_loss(vS, vT)

        return L_attn, L_v

    def get_extra_parameters(self) -> List[nn.Parameter]:
        # student projector 파라미터만 반환 (teacher projector는 no_grad)
        if not self._inited or self._proj_s is None:
            return []
        return list(self._proj_s.parameters())

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

        # student forward
        s_logits, s_feats = self.student(imgs, return_feats=True)
        s_last = s_feats[-1]

        # teacher forward (no grad)
        with torch.no_grad():
            t_logits, t_feats = self._t_forward(imgs)
            t_last = t_feats[-1]

        # lazy build projectors
        self._init_if_needed(s_last, t_last)

        # CE
        ce = self.ce(s_logits, masks)

        # PCA losses
        L_attn, L_v = self._pca_losses(s_last, t_last, masks=masks)
        L_pca = self.w_attn * L_attn + self.w_v * L_v

        total = self.w_ce * ce + self.w_pca * L_pca

        return {
            "total": total,
            "ce_student": ce.detach(),
            "pca_attn": L_attn.detach(),
            "pca_v": L_v.detach(),
            "pca_total": L_pca.detach(),
        }
