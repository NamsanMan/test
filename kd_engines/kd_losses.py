import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine

from kd_losses import SoftTarget
from kd_losses import AT

class KDWithLoss(BaseKDEngine):
    def __init__(self, teacher, student, t, p, w_ce_student: float, w_logit: float, w_feat: float, ignore_index: int, freeze_teacher: bool = False):
        super().__init__(teacher, student)
        self.logit_kd = SoftTarget(T=t)   # 로짓 KD
        self.feat_kd = AT(p=p)            # 특징 KD
        self.w_ce_student = float(w_ce_student)
        self.w_logit = float(w_logit)
        self.w_feat = float(w_feat)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # 교사 고정
        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)  # 학생은 mode에 따름
        if self._freeze_teacher:
            self.teacher.eval()  # 교사는 항상 eval
        return self

    def compute_losses(self, imgs, masks, device):
        s_logits, s_feats = self.student(imgs, return_feats=True)
        with torch.no_grad():
            t_logits, t_feats = self.teacher(imgs, return_feats=True)

        # valid_mask를 맨 위에서 한 번만 정의
        valid_mask = masks != self.ignore_index

        ce = self.ce_loss(s_logits, masks)

        # Logit KD (기존 코드와 동일)
        if valid_mask.any():
            s_valid = s_logits.permute(0, 2, 3, 1)[valid_mask]
            t_valid = t_logits.permute(0, 2, 3, 1)[valid_mask]
            kd_logit = self.logit_kd(s_valid, t_valid)
        else:
            kd_logit = torch.tensor(0.0, device=s_logits.device)

        # Feature KD (수정한 코드와 동일), 마지막 feature map을 distillation에 이용
        s_feat = s_feats[-1]
        t_feat = t_feats[-1].detach()
        feat_loss_map = self.feat_kd.get_loss_map(s_feat, t_feat)

        with torch.no_grad():
            # 위에서 정의한 valid_mask를 float 형태로 변환하여 재사용
            feat_mask = F.interpolate(valid_mask.unsqueeze(1).float(),
                                      size=s_feat.shape[-2:],
                                      mode='nearest')

        kd_feat = (feat_loss_map * feat_mask).sum() / (feat_mask.sum() + 1e-6)

        total = (self.w_ce_student * ce + self.w_logit * kd_logit + self.w_feat * kd_feat)
        return {"total": total, "ce_student": ce, "kd_logit": kd_logit, "kd_feat": kd_feat}
