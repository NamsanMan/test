# kd/basic_kd.py

# KD의 loss계산을 정의한 코드 >> MSE loss, logit을 이용한 KL divergence, CE 계산식 정의
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicKD(nn.Module):
    """
    - teacher, student: SegFormer 래퍼 (forward(..., return_feats=True) 지원)
    - compute_losses(imgs, masks, device) -> dict(loss 항목들)
    """
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 stage_weights, t: float,
                 w_ce_student: float, w_ce_teacher: float,
                 w_logit: float, w_feat: float,
                 ignore_index: int,
                 use_logit_kd: bool = True,
                 feat_l2_normalize: bool = True,
                 freeze_teacher: bool = False):
        super().__init__()
        self.teacher = teacher
        self.student = student

        self.stage_weights = stage_weights
        self.t = t
        self.w_ce_student = w_ce_student
        self.w_ce_teacher = w_ce_teacher
        self.w_logit = w_logit
        self.w_feat = w_feat
        self.ignore_index = ignore_index
        self.use_logit_kd = use_logit_kd
        self.feat_l2_normalize = feat_l2_normalize
        self._freeze_teacher = freeze_teacher

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # 교사 고정
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):
        # 부모 동작 유지(학생은 train으로)
        super().train(mode)
        # 교사는 항상 eval 유지
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    @torch.amp.autocast('cuda', enabled=False)  # 혼합정밀을 쓰더라도 KD의 수치안정성 위해 기본 FP32 추천
    def _logit_kd_loss(self, s_logits, t_logits, masks):
        """
        KLDiv( log_softmax(S/T), softmax(T/T) ) * T^2, void 무시
        s_logits, t_logits: (B, C, H, W)
        masks: (B, H, W) with ignore_index
        """
        if not self.use_logit_kd or self.w_logit <= 0.0:
            return s_logits.new_tensor(0.0)

        T = self.t
        B, C, H, W = s_logits.shape

        s = F.log_softmax(s_logits / T, dim=1)
        t = F.softmax(t_logits.detach() / T, dim=1)  # 교사 로짓은 detach

        s = s.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
        t = t.permute(0, 2, 3, 1).reshape(-1, C)
        valid = (masks.view(-1) != self.ignore_index)

        if valid.any():
            s = s[valid]
            t = t[valid]
            # batchmean은 샘플수로 나눔. (여기선 유효픽셀 수)
            
            # L1 loss으로 바꿔보기
            kd = F.kl_div(s, t, reduction='batchmean') * (T * T)
        else:
            kd = s_logits.new_tensor(0.0)
        return kd

    def _feat_kd_loss(self, s_feats, t_feats, masks):
        """
        스테이지별 MSE, void 무시. 스페이셜 불일치시 bilinear align.
        s_feats/t_feats: tuple of 4 tensors, each (B, C_i, H_i, W_i)
        """
        if self.w_feat <= 0.0:
            return s_feats[0].new_tensor(0.0)

        total = s_feats[0].new_tensor(0.0)
        for i, (ws, sf, tf) in enumerate(zip(self.stage_weights, s_feats, t_feats)):
            if ws <= 0.0:
                continue
            # (B,C,H,W) 정렬
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode='bilinear', align_corners=False)

            # void 마스크 다운샘플 → 유효 위치만 평균
            with torch.no_grad():
                valid = (F.interpolate(
                    (masks.unsqueeze(1).float() != self.ignore_index).float(),
                    size=tf.shape[-2:], mode='nearest'
                ).squeeze(1).bool())

            s_use = sf
            t_use = tf.detach()

            if self.feat_l2_normalize:
                s_use = F.normalize(s_use, p=2, dim=1)  # 채널 방향
                t_use = F.normalize(t_use, p=2, dim=1)

            diff2 = (s_use - t_use) ** 2  # (B,C,H,W)
            # 유효 위치만 평균(C 포함)
            denom = valid.sum() * diff2.shape[1] + 1e-6
            stage_loss = (diff2 * valid.unsqueeze(1)).sum() / denom

            total = total + ws * stage_loss
        return total

    def compute_losses(self, imgs, masks, device):
        # forward with features
        s_logits, s_feats = self.student(imgs, return_feats=True)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.teacher.parameters())):
            t_logits, t_feats = self.teacher(imgs, return_feats=True)

        # CE
        ce_student = self.ce_loss(s_logits, masks)
        ce_teacher = self.ce_loss(t_logits, masks) if self.w_ce_teacher > 0.0 and any(
            p.requires_grad for p in self.teacher.parameters()
        ) else s_logits.new_tensor(0.0)

        # KD
        kd_logit = self._logit_kd_loss(s_logits, t_logits, masks) if self.use_logit_kd else s_logits.new_tensor(0.0)
        kd_feat  = self._feat_kd_loss(s_feats, t_feats, masks)

        total = (self.w_ce_student * ce_student +
                 self.w_ce_teacher * ce_teacher +
                 self.w_logit * kd_logit +
                 self.w_feat  * kd_feat)

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "ce_teacher": ce_teacher.detach(),
            "kd_logit": kd_logit.detach(),
            "kd_feat": kd_feat.detach(),
            "s_logits": s_logits  # 평가용 반환
        }
