# kd_engines/transtocnn.py
# Heterogeneous KD: Transformer teacher (SegFormer) -> CNN student (DeepLabV3+)
# - Distill encoder-final feature (single stage)
# - Distill decoder-final logits
# - Student trains; Teacher is frozen/eval.
#
# Usage (예시):
#   teacher = SegFormerWrapper("segformerb5", num_classes=12).to(device).eval()
#   student = create_deeplab(in_channels=3, classes=12).to(device)
#   engine  = TransToCNN_KD(teacher, student, ignore_index=255)
#   out = engine.compute_losses(imgs, masks)
#   out["total"].backward(); optimizer.step()

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1x1(in_ch: int, out_ch: int) -> nn.Module:
    if in_ch == out_ch:
        return nn.Identity()
    m = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    return m


def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _attn_map(feat: torch.Tensor, p: int = 2) -> torch.Tensor:
    # [B,C,H,W] -> [B,1,H,W], 채널 축 p-제곱합 후 L2 정규화
    a = (feat.abs() ** p).sum(dim=1, keepdim=True)
    a = a / (a.norm(p=2, dim=(2, 3), keepdim=True) + 1e-6)
    return a

def _valid_mask_like(masks: torch.Tensor, ref: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """
    masks: [B,H,W] (long), ref: [B,C,Hr,Wr]
    returns: [B,1,Hr,Wr] float {0.,1.}
    """
    # 최근접 보간으로 HxW -> Hr x Wr
    m = F.interpolate(masks.unsqueeze(1).float(), size=ref.shape[-2:], mode="nearest").squeeze(1).long()
    valid = (m != ignore_index).float().unsqueeze(1)
    return valid



class TransToCNN_KD(nn.Module):
    """
    Final-encoder-feature KD + final-logits KD.

    - Feature KD: 학생 encoder 최종 feature (S_last) vs 교사 encoder 최종 feature (T_last)
        · 공간 정합: 기본 'teacher->student' (teacher feature를 학생 feature 크기로 보간)
        · 채널 정합: 1x1 conv (student channels -> teacher channels)
        · 손실: (정규화 L2) + λ_at * (AT L2), 가중치 w_feat

    - Logit KD: 학생 최종 logits vs 교사 최종 logits
        · 교사 logits을 학생 해상도로 보간
        · KL with temperature T, 무시 인덱스 마스킹 지원, 가중치 w_kd

    - CE: 학생 logits vs GT (ignore_index), 가중치 w_ce

    Args:
        teacher, student: 각각 (imgs, return_feats=True) -> (logits, feats[list]) 지원해야 함
        ignore_index: void 인덱스
        num_classes: 세그 클래스 수
        T: 로짓 KD 온도
        w_ce, w_kd, w_feat: 손실 가중치
        lambda_at: feature AT 손실 비중 (0으로 두면 AT 비활성화)
        match_space: "teacher_to_student" | "student_to_teacher"
        use_feat_norm: feature L2 전에 채널방향 정규화 사용할지
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        ignore_index: int,
        num_classes: int,
        T: float,
        w_ce: float = 1.0,
        w_kd: float = 1.0,
        w_feat: float = 0.5,
        lambda_at: float = 0.5,
        match_space: str = "teacher_to_student",
        use_feat_norm: bool = True,
    ):
        super().__init__()
        # teacher 고정
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = student
        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)

        self.T = float(T)
        self.w_ce = float(w_ce)
        self.w_kd = float(w_kd)
        self.w_feat = float(w_feat)
        self.lambda_at = float(lambda_at)
        assert match_space in ("teacher_to_student", "student_to_teacher")
        self.match_space = match_space
        self.use_feat_norm = bool(use_feat_norm)

        # CE (void 무시)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Lazy 초기화 멤버 (첫 forward에서 모양 보고 생성)
        self._proj_s2t: Optional[nn.Module] = None
        self._logit_adapter: Optional[nn.Module] = None
        self._inited: bool = False

    @torch.no_grad()
    def _init_adapters_if_needed(
        self,
        s_last: torch.Tensor,
        t_last: torch.Tensor,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
    ):
        """
        첫 호출 시 채널 수에 맞춰 1x1 conv 등을 초기화한다.
        - _proj_s2t: student encoder last ch -> teacher encoder last ch
        - _logit_adapter: teacher logits ch -> num_classes (필요 시)
        """
        if self._inited:
            return

        s_ch = int(s_last.shape[1])
        t_ch = int(t_last.shape[1])

        self._proj_s2t = _conv1x1(s_ch, t_ch)

        tC = int(t_logits.shape[1])
        if tC != self.num_classes:
            # teacher logits 채널 수가 다르면 1x1 conv로 적응
            self._logit_adapter = nn.Conv2d(tC, self.num_classes, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self._logit_adapter.weight, mode="fan_out", nonlinearity="relu")
        else:
            self._logit_adapter = nn.Identity()

        self._inited = True
        # 엔진 전체를 student와 동일 디바이스로 옮겨두는 것이 안전
        dev = s_last.device
        self.to(dev)

    @torch.no_grad()
    def _t_forward(self, x: torch.Tensor):
        return self.teacher(x, return_feats=True)

    def _kl_div_masked(self, s_logits: torch.Tensor, t_logits: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
        # teacher logits을 student 크기로
        t = _resize_like(t_logits, s_logits)
        if not isinstance(self._logit_adapter, nn.Identity):
            t = self._logit_adapter(t)

        s_log = F.log_softmax(s_logits / self.T, dim=1)
        t_prb = F.softmax(t / self.T, dim=1)
        kl = F.kl_div(s_log, t_prb, reduction="none")  # [B,C,H,W]

        if masks is not None:
            valid = (masks != self.ignore_index).float().unsqueeze(1)
            kl = kl * valid
            denom = valid.sum().clamp_min(1.0)
            return (kl.sum() * (self.T ** 2)) / denom

        return kl.mean() * (self.T ** 2)

    def _feat_kd_single(self, s_last: torch.Tensor, t_last: torch.Tensor,
                        masks: Optional[torch.Tensor]) -> torch.Tensor:
        # 공간 정합
        if self.match_space == "teacher_to_student":
            t_aligned = _resize_like(t_last, s_last)
            s_aligned = s_last
            ref_for_mask = s_aligned
        else:  # "student_to_teacher"
            s_aligned = _resize_like(s_last, t_last)
            t_aligned = t_last
            ref_for_mask = s_aligned  # 손실을 계산하는 공간에 마스크를 맞춥니다.

        # 채널 정합: student -> teacher ch
        s_proj = self._proj_s2t(s_aligned)

        # 정규화 L2 준비
        if self.use_feat_norm:
            s_norm = F.normalize(s_proj, dim=1)
            t_norm = F.normalize(t_aligned, dim=1)
        else:
            s_norm, t_norm = s_proj, t_aligned

        # ----- void 마스크 적용 -----
        if masks is not None:
            valid = _valid_mask_like(masks, ref_for_mask, self.ignore_index)  # [B,1,H,W]
            denom = valid.sum().clamp_min(1.0)

            # L2 (masked)
            diff = (s_norm - t_norm) ** 2  # [B,C,H,W]
            l2 = (diff * valid).sum() / denom

            # AT (masked, 선택)
            if self.lambda_at > 0.0:
                as_ = _attn_map(s_proj)
                at_ = _attn_map(t_aligned)
                at_diff = (as_ - at_) ** 2  # [B,1,H,W]
                at_loss = (at_diff * valid).sum() / denom
                return l2 + self.lambda_at * at_loss
            else:
                return l2
        # ----- 마스크 없음 (기존 방식) -----
        else:
            l2 = F.mse_loss(s_norm, t_norm)
            if self.lambda_at > 0.0:
                as_ = _attn_map(s_proj)
                at_ = _attn_map(t_aligned)
                at_loss = F.mse_loss(as_, at_)
                return l2 + self.lambda_at * at_loss
            else:
                return l2

    def compute_losses(self, imgs: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict(total, ce_student, kd_logit, kd_feat)
        """
        # (안전) 엔진이 아직 다른 디바이스라면 맞춰준다
        if next(self.parameters(), torch.empty(0, device=imgs.device)).device != imgs.device:
            self.to(imgs.device)

        # student forward
        s_logits, s_feats = self.student(imgs, return_feats=True)
        s_last = s_feats[-1]  # encoder 최종 feature

        # teacher forward (no grad)
        with torch.no_grad():
            t_logits, t_feats = self._t_forward(imgs)
            t_last = t_feats[-1]

        # lazy init adapters
        self._init_adapters_if_needed(s_last, t_last, s_logits, t_logits)

        # losses
        loss_ce = self.ce(s_logits, masks)
        loss_kd = self._kl_div_masked(s_logits, t_logits, masks)
        loss_f = self._feat_kd_single(s_last, t_last, masks)

        total = self.w_ce * loss_ce + self.w_kd * loss_kd + self.w_feat * loss_f

        return {
            "total": total,
            "ce_student": loss_ce.detach(),
            "kd_logit": loss_kd.detach(),
            "kd_feat": loss_f.detach(),
        }
