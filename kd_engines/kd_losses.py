import torch
import torch.nn as nn

from .base_engine import BaseKDEngine

from kd_losses import SoftTarget
from kd_losses import AT

class KDWithLoss(BaseKDEngine):
    def __init__(self, teacher, student, t, p, w_ce_student: float, w_logit: float, w_feat: float):
        super().__init__(teacher, student)
        self.ce_loss = nn.CrossEntropyLoss()
        self.logit_kd = SoftTarget(T=t)   # 로짓 KD
        self.feat_kd = AT(p=p)            # 특징 KD
        self.w_ce_student = float(w_ce_student)
        self.w_logit = float(w_logit)
        self.w_feat = float(w_feat)

    def compute_losses(self, imgs, masks, device):
        s_logits, s_feats = self.student(imgs, return_feats=True)
        with torch.no_grad():
            t_logits, t_feats = self.teacher(imgs, return_feats=True)

        ce = self.ce_loss(s_logits, masks)
        kd_logit = self.logit_kd(s_logits, t_logits)  # SoftTarget 사용
        kd_feat = self.feat_kd(s_feats[-1], t_feats[-1])  # AT 사용 예

        total = (self.w_ce_student * ce + self.w_logit * kd_logit + self.w_feat * kd_feat)
        return {"total": total, "ce_student": ce, "kd_logit": kd_logit, "kd_feat": kd_feat}
