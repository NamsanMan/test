# unet_wrapper.py

from typing import List, Sequence, Tuple, Union
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import DATA

# --- 기본 설정값 (DeepLabV3+ 코드와 동일한 형식 유지) ---
DEFAULT_ENCODER_NAME = "resnet34"  # U-Net에서 자주 사용되는 인코더로 변경
DEFAULT_ENCODER_WEIGHTS = "imagenet"
DEFAULT_IN_CHANNELS = 3
DEFAULT_NUM_CLASSES = DATA.NUM_CLASSES
# U-Net 인코더에서 추출할 중간 특징의 스테이지 인덱스
# smp U-Net의 경우, 0=입력, 1=stage1, 2=stage2, ..., 5=bottleneck
DEFAULT_STAGE_INDICES: Tuple[int, ...] = (1, 2, 3, 4, 5)


# ==============================================================================
# === U-Net을 위한 Wrapper 클래스 ===
# ==============================================================================

class UnetWrapper(nn.Module):
    """
    smp.Unet 모델을 감싸서 Knowledge Distillation에 필요한
    logits와 중간 특징(intermediate features)을 쉽게 반환하도록 만든 래퍼 클래스입니다.
    """

    def __init__(
            self,
            base_model: smp.Unet,
            stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
    ) -> None:
        super().__init__()
        # U-Net 모델도 DeepLabV3+와 동일한 속성들을 가지고 있습니다.
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        # KD에 사용할 인코더 스테이지 인덱스를 저장합니다.
        self.stage_indices: Tuple[int, ...] = tuple(stage_indices)

    def forward(
            self,
            x: torch.Tensor,
            return_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        U-Net은 내부 패딩 처리가 필요 없어 forward 로직이 매우 간단합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서
            return_feats (bool): True일 경우, logits와 함께 중간 특징을 반환합니다.

        Returns:
            - return_feats=False: logits (torch.Tensor)
            - return_feats=True: (logits, features) (Tuple[torch.Tensor, List[torch.Tensor]])
        """
        # 1. 인코더를 통과시켜 각 스테이지의 특징들을 리스트로 얻습니다.
        features: List[torch.Tensor] = self.encoder(x)

        # 2. 디코더와 최종 헤드를 통과시켜 logits를 계산합니다.
        decoder_output: torch.Tensor = self.decoder(*features)
        logits: torch.Tensor = self.segmentation_head(decoder_output)

        # 3. 중간 특징 반환 여부를 결정합니다.
        if not return_feats:
            return logits

        # 4. 지정된 인덱스에 해당하는 인코더 특징들을 선택하여 반환합니다.
        feats_out: List[torch.Tensor] = [features[i] for i in self.stage_indices]

        return logits, feats_out


# ==============================================================================
# === UnetWrapper 인스턴스를 쉽게 생성하기 위한 팩토리 함수 ===
# ==============================================================================

def create_unet_model(
        encoder_name: str = DEFAULT_ENCODER_NAME,
        encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
        in_channels: int = DEFAULT_IN_CHANNELS,
        classes: int = DEFAULT_NUM_CLASSES,
        stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
        **kwargs,
) -> UnetWrapper:
    """
    smp.Unet을 생성하고 UnetWrapper로 감싸서 반환하는 헬퍼 함수입니다.
    """
    # 1. segmentation_models_pytorch 라이브러리에서 U-Net 베이스 모델을 생성합니다.
    base = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )

    # 2. 생성된 베이스 모델을 UnetWrapper로 감싸줍니다.
    return UnetWrapper(
        base_model=base,
        stage_indices=stage_indices,
    )