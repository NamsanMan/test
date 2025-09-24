# kd_engines/cross_arch_seg_kd.py
# -*- coding: utf-8 -*-
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine

# torchvision이 있으면 MVG 변환을 더 다양하게 적용
try:
    import torchvision.transforms.functional as TF
    _HAS_TV = True
except Exception:
    _HAS_TV = False


# ----------------------------
# Projectors (PCA / GL)
# ----------------------------
class PCAttentionProjector(nn.Module):
    """
    Partially Cross Attention (PCA) projector.
    입력 특징(h)을 Q, K, V로 매핑.
    """
    def __init__(self, in_channels: int, qk_channels: int, v_channels: int):
        super().__init__()
        self.d_qk = qk_channels
        self.d_v = v_channels
        self.to_q = nn.Conv2d(in_channels, qk_channels, kernel_size=3, padding=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, qk_channels, kernel_size=3, padding=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, v_channels, kernel_size=3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,C,H,W)
        return self.to_q(x), self.to_k(x), self.to_v(x)


class GroupWiseLinearProjector(nn.Module):
    """
    Group-wise Linear (GL) projector.
    4x4 공간 그리드에서 각 위치가 고유 FC를 공유.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_layers = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(16)])
        self.dropout = nn.Dropout(p=dropout_p)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,C_out,H,W)
        B, C, H, W = x.shape
        out = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        for i in range(4):
            for j in range(4):
                group_pixels = x[:, :, i::4, j::4]                  # (B,C,H/4,W/4)
                group_pixels = group_pixels.permute(0, 2, 3, 1)     # (B,H/4,W/4,C)
                fc_idx = i * 4 + j
                projected = self.fc_layers[fc_idx](group_pixels)    # (B,H/4,W/4,C_out)
                projected = projected.permute(0, 3, 1, 2).contiguous()
                out[:, :, i::4, j::4] = projected

        return self.dropout(out)


# ----------------------------
# Cross-view robust training (MVG + MAD)
# ----------------------------
class MVGDiscriminator(nn.Module):
    """
    Discriminator D(h): GAP -> 3-layer MLP -> sigmoid
    h: (B,C,H,W) -> D(h): (B,1)
    """
    def __init__(self, in_channels: int, hidden: Tuple[int, int] = (256, 64)):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden[0], hidden[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden[1], 1),
            nn.Sigmoid()
        )
        self._init()

    def _init(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.gap(h).flatten(1)
        return self.mlp(z)


def _random_patch_mask(x: torch.Tensor, max_frac: float = 0.25) -> torch.Tensor:
    """
    Cutout-like patch zeroing.
    x: (B,C,H,W)
    """
    B, C, H, W = x.shape
    h = int(H * random.uniform(0.05, max_frac))
    w = int(W * random.uniform(0.05, max_frac))
    cy = random.randint(0, max(0, H - h))
    cx = random.randint(0, max(0, W - w))
    x[:, :, cy:cy + h, cx:cx + w] = 0
    return x


def _mvg_trans_single(img: torch.Tensor) -> torch.Tensor:
    """
    Trans(x): color/contrast/saturation jitter + affine + pad-crop + patch mask.
    img: (3,H,W). 확률 0.5로만 적용 (p>=0.5).
    """
    if random.random() < 0.5:
        return img

    x = img.clone()
    if _HAS_TV:
        x = TF.adjust_brightness(x, 0.6 + 0.8 * random.random())
        x = TF.adjust_contrast(x, 0.6 + 0.8 * random.random())
        x = TF.adjust_saturation(x, 0.6 + 0.8 * random.random())

        angle = random.uniform(-15.0, 15.0)
        translate = (random.uniform(-0.05, 0.05) * x.shape[1],
                     random.uniform(-0.05, 0.05) * x.shape[2])
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-8.0, 8.0)
        x = TF.affine(x, angle=angle, translate=translate, scale=scale, shear=shear)

        pad = max(2, int(min(x.shape[1], x.shape[2]) * 0.02))
        x = TF.pad(x, [pad, pad])
        i = random.randint(0, x.shape[1] - 2 * pad)
        j = random.randint(0, x.shape[2] - 2 * pad)
        x = x[:, i:i + img.shape[1], j:j + img.shape[2]]

    x = _random_patch_mask(x.unsqueeze(0), max_frac=0.25).squeeze(0)
    return x


def mvg_augment(batch: torch.Tensor) -> torch.Tensor:
    """
    batch: (B,3,H,W). 정규화 전/후 모두 허용(상대 변화만 적용).
    """
    out = []
    for n in range(batch.shape[0]):
        out.append(_mvg_trans_single(batch[n]))
    return torch.stack(out, dim=0)


# ----------------------------
# KD Engine
# ----------------------------
class CrossArchSegKD(BaseKDEngine):
    """
    Cross-Architecture KD + Cross-view Robust Training.
    - Teacher: Transformer wrapper (e.g., SegFormerWrapper)
    - Student: CNN wrapper (e.g., DeepLabV3+ wrapper)
    - Losses: CE_student, PCA, GL, MAD (D), MVG (G)
    """
    def __init__(self, teacher, student,
                 w_ce_student: float, w_pca: float, w_gl: float,
                 pca_qk_channels: int, pca_v_channels: int,
                 gl_dropout_p: float,
                 ignore_index: int,
                 freeze_teacher: bool = True,
                 # Cross-view robust training
                 w_mad: float = 0.0,           # Discriminator loss weight
                 w_mvg: float = 0.0,           # Generator(Student) loss weight
                 disc_hidden: Tuple[int, int] = (256, 64)):
        super().__init__(teacher, student)

        # weights & hparams
        self.w_ce_student = float(w_ce_student)
        self.w_pca = float(w_pca)
        self.w_gl = float(w_gl)
        self.pca_qk_channels = int(pca_qk_channels)
        self.pca_v_channels = int(pca_v_channels)
        self.gl_dropout_p = float(gl_dropout_p)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        # cross-view weights
        self.w_mad = float(w_mad)
        self.w_mvg = float(w_mvg)
        self._disc_hidden = disc_hidden

        # losses
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.mse_loss = nn.MSELoss()
        self.bce = nn.BCELoss()

        # freeze teacher if requested
        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # delayed builds
        self._projectors_built = False
        self.pca_proj_s = nn.Identity()
        self.pca_proj_t = nn.Identity()
        self.gl_proj_s = nn.Identity()

        self._disc_built = False
        self.disc: MVGDiscriminator | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def _build_projectors_if_needed(self, s_feat: torch.Tensor, t_feat: torch.Tensor):
        if self._projectors_built:
            # Discriminator lazy build
            if (self.w_mad > 0 or self.w_mvg > 0) and not self._disc_built:
                self.disc = MVGDiscriminator(t_feat.shape[1], self._disc_hidden).to(s_feat.device)
                self._disc_built = True
            return

        device = s_feat.device
        s_channels = s_feat.shape[1]
        t_channels = t_feat.shape[1]

        self.pca_proj_s = PCAttentionProjector(s_channels, self.pca_qk_channels, self.pca_v_channels).to(device)
        self.pca_proj_t = PCAttentionProjector(t_channels, self.pca_qk_channels, self.pca_v_channels).to(device)
        self.gl_proj_s = GroupWiseLinearProjector(s_channels, t_channels, self.gl_dropout_p).to(device)

        self._projectors_built = True
        print("✅ Cross-Architecture projectors have been built.")
        print(f"   - Student feat channels: {s_channels}, Teacher feat channels: {t_channels}")
        print(f"   - PCA QK: {self.pca_qk_channels}, V: {self.pca_v_channels}")
        print(f"   - GL out channels: {t_channels}")

        if self.w_mad > 0 or self.w_mvg > 0:
            self.disc = MVGDiscriminator(t_channels, self._disc_hidden).to(device)
            self._disc_built = True
            print(f"   - Discriminator in_channels: {t_channels}")

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _get_last_encoder_feat(self, model, imgs, is_teacher: bool):
        """
        두 래퍼 모두 forward(..., return_feats=True) 지원 가정.
        반환: (logits, last_feat)
        """
        logits, all_feats = model(imgs, return_feats=True)
        last_feat = all_feats[-1]
        return logits, last_feat

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Spatial self-attention: (B,C,H,W) -> (B,Cv,H,W)
        """
        B, C_qk, H, W = q.shape
        C_v = v.shape[1]
        d_k = C_qk

        q_ = q.view(B, C_qk, H * W).permute(0, 2, 1)      # (B,HW,C)
        k_ = k.view(B, C_qk, H * W).permute(0, 2, 1)
        v_ = v.view(B, C_v,  H * W).permute(0, 2, 1)

        attn_scores = torch.bmm(q_, k_.transpose(1, 2))   # (B,HW,HW)
        attn_weights = F.softmax(attn_scores / math.sqrt(d_k), dim=-1)
        out = torch.bmm(attn_weights, v_)                 # (B,HW,Cv)
        out = out.permute(0, 2, 1).view(B, C_v, H, W)
        return out

    def _pca_loss(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        # 공간 크기 정렬
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)

        q_s, k_s, v_s = self.pca_proj_s(s_feat)
        with torch.no_grad():
            q_t, k_t, v_t = self.pca_proj_t(t_feat)
            attn_t = self._calculate_attention(q_t, k_t, v_t)

        # Partial cross (0.5 확률로 teacher 값 대체)
        pc_q_s = torch.where(torch.rand_like(q_s) < 0.5, q_t, q_s)
        pc_k_s = torch.where(torch.rand_like(k_s) < 0.5, k_t, k_s)
        pc_v_s = torch.where(torch.rand_like(v_s) < 0.5, v_t, v_s)

        pc_attn_s = self._calculate_attention(pc_q_s, pc_k_s, pc_v_s)

        d_v = v_s.shape[1]
        loss_attn = self.mse_loss(pc_attn_s, attn_t.detach())
        loss_v = self.mse_loss((v_s ** 2) / math.sqrt(d_v),
                               (v_t.detach() ** 2) / math.sqrt(d_v))
        return loss_attn + loss_v

    def _gl_loss(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        s_feat_prime = self.gl_proj_s(s_feat)
        if s_feat_prime.shape[-2:] != t_feat.shape[-2:]:
            s_feat_prime = F.interpolate(s_feat_prime, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.mse_loss(s_feat_prime, t_feat.detach())

    def _cvrt_losses(self, imgs: torch.Tensor,
                     s_feat_last: torch.Tensor,
                     t_feat_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-view robust training:
        - l_mad: D 학습(학생/교사 특징은 detach)
        - l_mvg: G(학생) 학습(D 고정), log(1 - D(h'_S))
        """
        if self.disc is None:
            z = imgs.new_tensor(0.0)
            return z, z

        # MVG 변환 이미지 생성
        with torch.no_grad():
            imgs_tr = mvg_augment(imgs.detach().cpu()).to(imgs.device, imgs.dtype)

        # 학생에 변환 입력 통과
        _s_logits_tr, s_feat_tr = self._get_last_encoder_feat(self.student, imgs_tr, is_teacher=False)

        # Discriminator 학습 손실 (real: h_T, fake: h'_S)
        real = t_feat_last.detach()
        fake = s_feat_tr.detach()
        pred_real = self.disc(real)
        pred_fake = self.disc(fake)
        l_mad = self.bce(pred_real, torch.ones_like(pred_real)) + \
                self.bce(pred_fake, torch.zeros_like(pred_fake))

        # Generator(학생) 손실: log(1 - D(h'_S)) 최소화
        self._set_requires_grad(self.disc, False)
        pred_fake_for_g = self.disc(s_feat_tr)  # 학생으로만 gradient
        l_mvg = torch.mean(torch.log(1.0 - pred_fake_for_g + 1e-6)) * (-1.0)
        # 비포화 형태로 바꾸고 싶다면:
        # l_mvg = self.bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
        self._set_requires_grad(self.disc, True)

        return l_mad, l_mvg

    def compute_losses(self, imgs: torch.Tensor, masks: torch.Tensor, device) -> dict:
        # logits & last encoder features
        s_logits, s_feat = self._get_last_encoder_feat(self.student, imgs, is_teacher=False)
        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feat = self._get_last_encoder_feat(self.teacher, imgs, is_teacher=True)
        else:
            t_logits, t_feat = self._get_last_encoder_feat(self.teacher, imgs, is_teacher=True)

        # lazy build
        self._build_projectors_if_needed(s_feat, t_feat)

        # losses
        loss_ce = self.ce_loss(s_logits, masks)

        loss_pca = self._pca_loss(s_feat.float(), t_feat.float()) if self.w_pca > 0 else s_logits.new_tensor(0.0)
        loss_gl = self._gl_loss(s_feat.float(), t_feat.float()) if self.w_gl > 0 else s_logits.new_tensor(0.0)

        if (self.w_mad > 0 or self.w_mvg > 0) and self._disc_built:
            loss_mad, loss_mvg = self._cvrt_losses(imgs, s_feat.float(), t_feat.float())
        else:
            loss_mad = s_logits.new_tensor(0.0)
            loss_mvg = s_logits.new_tensor(0.0)

        total = (self.w_ce_student * loss_ce +
                 self.w_pca * loss_pca +
                 self.w_gl * loss_gl +
                 self.w_mad * loss_mad +
                 self.w_mvg * loss_mvg)

        return {
            "total": total,
            "ce_student": loss_ce.detach(),
            "kd_pca": loss_pca.detach(),
            "kd_gl": loss_gl.detach(),
            "mad": loss_mad.detach(),
            "mvg": loss_mvg.detach(),
            "s_logits": s_logits
        }

    def get_extra_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self._projectors_built:
            params += list(self.pca_proj_s.parameters())
            params += list(self.pca_proj_t.parameters())
            params += list(self.gl_proj_s.parameters())
        if self._disc_built and (self.w_mad > 0 or self.w_mvg > 0):
            params += list(self.disc.parameters())
        return params
