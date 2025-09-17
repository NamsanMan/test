import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_engine import BaseKDEngine

class PCAttentionProjector(nn.Module):
    """
    논문의 Partially Cross Attention (PCA) Projector를 구현합니다.
    입력 특징(h)을 Q, K, V 행렬로 변환합니다.
    """
    def __init__(self, in_channels: int, qk_channels: int, v_channels: int):
        super().__init__()
        self.d_qk = qk_channels
        self.d_v = v_channels

        # 3x3 Conv 레이어를 사용해 Q, K, V로 매핑
        self.to_q = nn.Conv2d(in_channels, qk_channels, kernel_size=3, padding=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, qk_channels, kernel_size=3, padding=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, v_channels, kernel_size=3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ (B, C, H, W) -> (Q, K, V) """
        return self.to_q(x), self.to_k(x), self.to_v(x)


class GroupWiseLinearProjector(nn.Module):
    """
    논문의 Group-wise Linear (GL) Projector를 구현합니다.
    4x4 공간적 이웃이 하나의 FC 레이어를 공유하는 구조입니다.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 4x4=16개의 그룹에 대한 FC 레이어 생성
        self.fc_layers = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(16)
        ])
        self.dropout = nn.Dropout(p=dropout_p)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ (B, C_in, H, W) -> (B, C_out, H, W) """
        B, C, H, W = x.shape
        # 결과를 저장할 빈 텐서 생성
        out = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        # 4x4 그룹별로 순회하며 각기 다른 FC 레이어를 적용
        for i in range(4):
            for j in range(4):
                # i, j 그룹에 해당하는 픽셀들을 슬라이싱 (stride=4)
                # (B, C_in, H/4, W/4)
                group_pixels = x[:, :, i::4, j::4]

                # FC 레이어에 맞게 차원 변경: (B, H/4, W/4, C_in)
                group_pixels = group_pixels.permute(0, 2, 3, 1).contiguous()

                # FC 레이어 적용 (입력: (N, C_in), 출력: (N, C_out))
                fc_idx = i * 4 + j
                projected = self.fc_layers[fc_idx](group_pixels)

                # 원래의 4D 텐서 형태로 복원: (B, C_out, H/4, W/4)
                projected = projected.permute(0, 3, 1, 2).contiguous()

                # 결과 텐서의 올바른 위치에 할당
                out[:, :, i::4, j::4] = projected

        return self.dropout(out)

class CrossArchSegKD(BaseKDEngine):
    """
    "Cross-Architecture Knowledge Distillation" 논문 기반 KD 엔진.
    - Teacher: Transformer 계열 (SegFormer)
    - Student: CNN 계열 (DeepLabV3+)
    - KD Methods:
        1. Partially Cross Attention (PCA) Projector
        2. Group-wise Linear (GL) Projector
    """

    def __init__(self, teacher, student,
                 w_ce_student: float, w_pca: float, w_gl: float,
                 pca_qk_channels: int, pca_v_channels: int,
                 gl_dropout_p: float,
                 ignore_index: int,
                 freeze_teacher: bool = True):
        super().__init__(teacher, student)

        # 손실 가중치 및 하이퍼파라미터
        self.w_ce_student = float(w_ce_student)
        self.w_pca = float(w_pca)
        self.w_gl = float(w_gl)
        self.pca_qk_channels = int(pca_qk_channels)
        self.pca_v_channels = int(pca_v_channels)
        self.gl_dropout_p = float(gl_dropout_p)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.mse_loss = nn.MSELoss()

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # Projector들은 첫 forward 시점에 동적으로 생성됩니다.
        self._projectors_built = False
        self.pca_proj_s = nn.Identity()
        self.pca_proj_t = nn.Identity()
        self.gl_proj_s = nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def _build_projectors_if_needed(self, s_feat, t_feat):
        if self._projectors_built:
            return

        device = s_feat.device
        s_channels = s_feat.shape[1]
        t_channels = t_feat.shape[1]

        # PCA Projector 생성
        self.pca_proj_s = PCAttentionProjector(s_channels, self.pca_qk_channels, self.pca_v_channels).to(device)
        self.pca_proj_t = PCAttentionProjector(t_channels, self.pca_qk_channels, self.pca_v_channels).to(device)

        # GL Projector 생성
        self.gl_proj_s = GroupWiseLinearProjector(s_channels, t_channels, self.gl_dropout_p).to(device)

        self._projectors_built = True
        print("✅ Cross-Architecture projectors have been built.")
        print(f"   - Student feat channels: {s_channels}, Teacher feat channels: {t_channels}")
        print(f"   - PCA QK channels: {self.pca_qk_channels}, V channels: {self.pca_v_channels}")
        print(f"   - GL out channels: {t_channels}")

    def _get_last_encoder_feat(self, model, imgs, is_teacher):
        """
        학생(DeepLabV3+ Wrapper)과 교사(SegFormer Wrapper) 모델 모두에서
        로짓과 인코더의 마지막 특징을 추출합니다.

        두 래퍼 모두 `forward(..., return_feats=True)` 인터페이스를 따르므로
        코드를 하나로 통일할 수 있습니다.
        """
        # is_teacher 플래그와 무관하게 동일한 방식으로 호출 가능
        logits, all_feats = model(imgs, return_feats=True)

        # all_feats는 특징 맵의 리스트 또는 튜플이므로 마지막 요소를 선택
        last_feat = all_feats[-1]

        return logits, last_feat

    def _calculate_attention(self, q, k, v):
        """ Spatial Self-Attention을 올바르게 계산하도록 수정된 메서드 """
        B, C_qk, H, W = q.shape
        B, C_v, _, _ = v.shape  # V의 채널 수를 가져옴
        d_k = C_qk  # Key의 채널 차원

        # 1. Spatial Self-Attention을 위해 차원 재정렬
        # (B, C, H, W) -> (B, H*W, C) 형태로 변경
        q_reshaped = q.view(B, C_qk, H * W).permute(0, 2, 1)
        k_reshaped = k.view(B, C_qk, H * W).permute(0, 2, 1)
        v_reshaped = v.view(B, C_v, H * W).permute(0, 2, 1)

        # 2. Attention Score 계산: Q @ K^T
        # (B, H*W, C_qk) @ (B, C_qk, H*W) -> (B, H*W, H*W)
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2))

        # 3. Softmax를 통해 Attention Weights 계산
        attn_weights = F.softmax(attn_scores / math.sqrt(d_k), dim=-1)

        # 4. Attention Weights를 V에 적용
        # (B, H*W, H*W) @ (B, H*W, C_v) -> (B, H*W, C_v)
        output = torch.bmm(attn_weights, v_reshaped)

        # 5. 원래의 이미지 텐서 형태로 복원
        # (B, H*W, C_v) -> (B, C_v, H, W)
        output = output.permute(0, 2, 1).view(B, C_v, H, W)

        return output

    def _pca_loss(self, s_feat, t_feat):
        """ Equation (3), (4) 기반 PCA Loss 계산 """

        # === FIX: 공간적 차원 정렬 (Spatial Dimension Alignment) ===
        # 학생과 교사의 특징 맵 공간 크기가 다를 수 있으므로,
        # 학생의 특징(s_feat)을 교사의 크기(t_feat)에 맞춥니다.
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.interpolate(
                s_feat,
                size=t_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        # ==========================================================

        q_s, k_s, v_s = self.pca_proj_s(s_feat)

        with torch.no_grad():
            q_t, k_t, v_t = self.pca_proj_t(t_feat)
            attn_t = self._calculate_attention(q_t, k_t, v_t)

        # Partially Cross Attention (Equation 3)
        # 이제 q_s와 q_t의 크기가 같으므로 torch.where가 정상적으로 동작합니다.
        pc_q_s = torch.where(torch.rand_like(q_s) < 0.5, q_t, q_s)
        pc_k_s = torch.where(torch.rand_like(k_s) < 0.5, k_t, k_s)
        pc_v_s = torch.where(torch.rand_like(v_s) < 0.5, v_t, v_s)

        pc_attn_s = self._calculate_attention(pc_q_s, pc_k_s, pc_v_s)

        # Loss 계산 (Equation 4)
        d_v = v_s.shape[1]
        loss_attn = self.mse_loss(pc_attn_s, attn_t.detach())
        loss_v = self.mse_loss(
            (v_s ** 2) / math.sqrt(d_v),
            (v_t.detach() ** 2) / math.sqrt(d_v)
        )
        return loss_attn + loss_v

    def _gl_loss(self, s_feat, t_feat):
        """ Equation (6) 기반 GL Loss 계산 """
        # 학생 특징을 교사 특징 공간으로 프로젝션
        s_feat_prime = self.gl_proj_s(s_feat)

        # 교사 특징과 공간적 크기 맞추기
        if s_feat_prime.shape[-2:] != t_feat.shape[-2:]:
            s_feat_prime = F.interpolate(
                s_feat_prime,
                size=t_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return self.mse_loss(s_feat_prime, t_feat.detach())

    def compute_losses(self, imgs, masks, device):
        # 1. 각 모델로부터 로짓(logit)과 인코더 마지막 특징(feature) 추출
        s_logits, s_feat = self._get_last_encoder_feat(self.student, imgs, is_teacher=False)

        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feat = self._get_last_encoder_feat(self.teacher, imgs, is_teacher=True)
        else:
            t_logits, t_feat = self._get_last_encoder_feat(self.teacher, imgs, is_teacher=True)

        # 2. 첫 호출 시 Projector들 생성
        self._build_projectors_if_needed(s_feat, t_feat)

        # 3. 손실 계산
        # 3-1. 학생 모델의 기본 CE Loss
        loss_ce = self.ce_loss(s_logits, masks)

        # 3-2. PCA Loss
        loss_pca = self._pca_loss(s_feat.float(), t_feat.float()) if self.w_pca > 0 else s_logits.new_tensor(0.0)

        # 3-3. GL Loss
        loss_gl = self._gl_loss(s_feat.float(), t_feat.float()) if self.w_gl > 0 else s_logits.new_tensor(0.0)

        # 4. 최종 손실 조합
        total = (self.w_ce_student * loss_ce +
                 self.w_pca * loss_pca +
                 self.w_gl * loss_gl)

        return {
            "total": total,
            "ce_student": loss_ce.detach(),
            "kd_pca": loss_pca.detach(),
            "kd_gl": loss_gl.detach(),
            "s_logits": s_logits  # 평가용
        }

    def get_extra_parameters(self):
        # 학습되어야 할 Projector들의 파라미터를 옵티마이저에 전달
        if not self._projectors_built:
            return []
        return (list(self.pca_proj_s.parameters()) +
                list(self.pca_proj_t.parameters()) +
                list(self.gl_proj_s.parameters()))