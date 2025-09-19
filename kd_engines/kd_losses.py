import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_engine import BaseKDEngine  # 기존 base_engine을 그대로 사용한다고 가정


class BasicLogitKD(BaseKDEngine):
    """
    가장 기본적인 Logit Knowledge Distillation을 수행하는 간단한 KD 엔진입니다.
    Teacher와 Student 모델의 최종 출력(logit) 간의 분포 유사도를 학습합니다.
    """

    def __init__(self, teacher, student,
                 w_ce_student: float,
                 w_kd_logit: float,
                 temperature: float,
                 ignore_index: int,
                 freeze_teacher: bool = True):
        """
        Args:
            teacher (nn.Module): 교사 모델
            student (nn.Module): 학생 모델
            w_ce_student (float): 학생 모델이 정답 레이블(hard target)을 학습하는 CrossEntropy 손실의 가중치
            w_kd_logit (float): 교사 모델의 예측(soft target)을 학습하는 KD 손실의 가중치
            temperature (float): 로짓을 부드럽게 만들어주는 온도 파라미터. 높을수록 분포가 부드러워집니다.
            ignore_index (int): CrossEntropy 손실 계산 시 무시할 레이블 인덱스
            freeze_teacher (bool): 교사 모델의 가중치를 고정할지 여부
        """
        super().__init__(teacher, student)

        self.w_ce_student = float(w_ce_student)
        self.w_kd_logit = float(w_kd_logit)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        # 학생 모델의 주 손실 함수 (정답 레이블과 비교)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # 지식 증류 손실 함수 (Teacher-Student 로짓 분포 비교)
        # KLDivLoss는 입력으로 log-확률, 타겟으로 일반 확률을 기대합니다.
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):
        """모델을 학습 모드로 설정합니다. 교사 모델은 항상 eval 모드를 유지합니다."""
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def compute_losses(self, imgs, masks, device):
        """
        입력 이미지와 마스크를 사용하여 손실을 계산합니다.
        """
        # 1. 학생 모델로부터 로짓(logit) 추출
        # 모델 래퍼가 (logits, features) 튜플을 반환하므로 첫 번째 요소만 사용합니다.
        s_logits, _ = self.student(imgs, return_feats=False)

        # 2. 교사 모델로부터 로짓 추출 (그래디언트 계산 비활성화)
        with torch.no_grad():
            t_logits, _ = self.teacher(imgs, return_feats=False)

        # 3. 손실 계산
        # 3-1. 학생 모델의 기본 CE Loss (Hard Target Loss)
        loss_ce = self.ce_loss(s_logits, masks)

        # 3-2. Logit Distillation Loss (Soft Target Loss)
        # KL Divergence를 사용한 로짓 분포 유사도 계산
        # Teacher의 부드러운 타겟(softmax)과 Student의 부드러운 예측(log_softmax)을 비교
        loss_kd = self.kd_loss(
            F.log_softmax(s_logits / self.temperature, dim=1),
            F.softmax(t_logits / self.temperature, dim=1)
        )

        # Hinton의 논문에 따라 temperature^2를 곱해주어 스케일을 맞춥니다.
        loss_kd = loss_kd * (self.temperature ** 2)

        # 4. 최종 손실 조합
        total = (self.w_ce_student * loss_ce +
                 self.w_kd_logit * loss_kd)

        return {
            "total": total,
            "ce_student": loss_ce.detach(),
            "kd_logit": loss_kd.detach(),
            "s_logits": s_logits  # mIoU 등 평가 지표 계산용
        }

    def get_extra_parameters(self):
        """
        이 엔진은 별도로 학습할 파라미터(프로젝터 등)가 없으므로 빈 리스트를 반환합니다.
        """
        return []