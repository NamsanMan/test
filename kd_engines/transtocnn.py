# kd_engines/segtoseg_kd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_engine import BaseKDEngine


class SegformerToDeepLabKD(BaseKDEngine):
    """
    Knowledge Distillation Engine for SegFormer (Teacher) to DeepLabV3+/MobileNetV2 (Student).

    - Teacher (SegFormer): Outputs a list of 4 feature maps from its encoder.
    - Student (DeepLab): Outputs a single feature map from its encoder.
    - Feature KD: Matches the student's single feature map to a specific stage of the teacher's features.
    - Logit KD: Handles mismatched class numbers by treating logits as features and using MSE loss.
    """

    def __init__(self, teacher, student,
                 teacher_feature_stage_idx: int,  # <-- 변경됨: Teacher의 어떤 stage를 쓸지 지정
                 t: float,
                 w_ce_student: float,
                 w_logit: float, w_feat: float,
                 ignore_index: int,
                 freeze_teacher: bool = True):
        super().__init__(teacher, student)

        assert 0 <= teacher_feature_stage_idx < 4, "Teacher stage index must be in [0, 3]"
        self.teacher_feature_stage_idx = teacher_feature_stage_idx
        self.t = float(t)
        self.w_ce_student = float(w_ce_student)
        self.w_logit = float(w_logit)
        self.w_feat = float(w_feat)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # Adaptation layers for KD (built lazily on the first forward pass)
        self._projs_built = False
        self.feature_adaptation = nn.Identity()
        self.logit_adaptation = nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def _build_projs_if_needed(self, s_feat, t_feat, s_logits, t_logits):
        if self._projs_built:
            return

        device = s_feat.device

        # 1. Feature Adaptation Layer
        s_ch, t_ch = s_feat.shape[1], t_feat.shape[1]
        if s_ch != t_ch:
            self.feature_adaptation = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False).to(device)
            nn.init.kaiming_normal_(self.feature_adaptation.weight, mode="fan_out", nonlinearity="relu")

        # 2. Logit Adaptation Layer
        s_cls, t_cls = s_logits.shape[1], t_logits.shape[1]
        if s_cls != t_cls:
            self.logit_adaptation = nn.Conv2d(s_cls, t_cls, kernel_size=1, bias=False).to(device)
            nn.init.kaiming_normal_(self.logit_adaptation.weight, mode="fan_out", nonlinearity="relu")

        self._projs_built = True

    def _forward_teacher(self, imgs):
        # Assumes Hugging Face SegFormer model
        outputs = self.teacher(pixel_values=imgs, output_hidden_states=True)
        logits = outputs.logits
        # Get all 4 encoder feature maps
        features = outputs.hidden_states[-4:]
        return logits, features

    def _forward_student(self, imgs):
        # Assumes a wrapper that returns (logits, single_encoder_feature)
        logits, feature = self.student(imgs)
        return logits, feature

    def _logit_kd_loss_mse(self, s_logits, t_logits, masks):  # <-- 핵심 로직 변경: MSE 기반
        if self.w_logit <= 0.0:
            return s_logits.new_tensor(0.0)

        # KD in FP32
        s_logits, t_logits = s_logits.float(), t_logits.float()

        # 1. Adapt student logit channels to match teacher's
        s_logits_adapted = self.logit_adaptation(s_logits)

        # 2. Upsample teacher logits to match student's spatial size
        t_logits_upsampled = F.interpolate(
            t_logits,
            size=s_logits.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(s_logits_adapted, t_logits_upsampled.detach())  # (B, C, H, W)

        # Mask out ignored pixels
        valid_mask = (masks.unsqueeze(1) != self.ignore_index)  # (B, 1, H, W)

        # Calculate mean loss only on valid pixels
        if valid_mask.any():
            kd_loss = (loss * valid_mask).sum() / valid_mask.sum()
        else:
            kd_loss = s_logits.new_tensor(0.0)

        return kd_loss

    def _feat_kd_loss(self, s_feat, t_feat, masks):  # <-- 핵심 로직 변경: 1-to-1 매칭
        if self.w_feat <= 0.0:
            return s_feat.new_tensor(0.0)

        s_feat, t_feat = s_feat.float(), t_feat.float()

        # Adapt student feature channels to match teacher's
        s_feat_adapted = self.feature_adaptation(s_feat)

        # Align spatial size if they don't match
        if s_feat_adapted.shape[-2:] != t_feat.shape[-2:]:
            s_feat_adapted = F.interpolate(
                s_feat_adapted,
                size=t_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        loss_fn = nn.MSELoss()
        kd_loss = loss_fn(s_feat_adapted, t_feat.detach())
        return kd_loss

    def compute_losses(self, imgs, masks, device):
        # Forward pass for student and teacher
        s_logits, s_feat = self._forward_student(imgs)

        with torch.no_grad():
            t_logits, t_feats = self._forward_teacher(imgs)

        # Select the target teacher feature map based on the index
        t_feat_target = t_feats[self.teacher_feature_stage_idx]

        # Lazily build adaptation layers on the first pass
        self._build_projs_if_needed(s_feat, t_feat_target, s_logits, t_logits)

        # Standard Cross-Entropy loss for the student
        ce_student = self.ce_loss(s_logits, masks)

        # Knowledge Distillation losses
        kd_logit = self._logit_kd_loss_mse(s_logits, t_logits, masks)
        kd_feat = self._feat_kd_loss(s_feat, t_feat_target, masks)

        # Total weighted loss
        total = (self.w_ce_student * ce_student +
                 self.w_logit * kd_logit +
                 self.w_feat * kd_feat)

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "kd_logit": kd_logit.detach(),
            "kd_feat": kd_feat.detach(),
            "s_logits": s_logits
        }

    def get_extra_parameters(self):
        # Expose adaptation layer parameters to the optimizer
        if not self._projs_built:
            return []
        return list(self.feature_adaptation.parameters()) + list(self.logit_adaptation.parameters())