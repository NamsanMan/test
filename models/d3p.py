import torch.nn as nn
import segmentation_models_pytorch as smp


# --- 1. DeepLabV3+ 모델을 감싸는 Wrapper 클래스 정의 ---
class DeepLabWrapper(nn.Module):
    """
    smp.DeepLabV3Plus 모델을 감싸서 forward pass 시
    (logits, encoder_feature) 튜플을 반환하도록 만드는 래퍼입니다.
    """

    def __init__(self, model):
        super().__init__()
        # smp 모델의 각 구성 요소를 직접 속성으로 가져옵니다.
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.segmentation_head = model.segmentation_head

    def forward(self, x):
        # 1. Encoder를 통과시켜 여러 스케일의 특징 맵 리스트를 얻습니다.
        features = self.encoder(x)

        # 2. Decoder에 이 특징 맵들을 전달합니다.
        decoder_output = self.decoder(features)

        # 3. Segmentation Head를 통과시켜 최종 logit을 얻습니다.
        logits = self.segmentation_head(decoder_output)

        # 4. Distillation에 사용할 Encoder의 최종 특징 맵을 가져옵니다.
        #    smp의 encoder는 깊은 특징 맵일수록 리스트의 뒤쪽에 위치합니다.
        if return_feats:
            return logits, features[-1]  # distillation 용
        else:
            return logits                # 학습/평가 용


# --- 2. 기존 모델 생성 함수 수정 ---
ENCODER_NAME = "mobilenet_v2"
ENCODER_WEIGHTS = "imagenet"


def create_deeplabv3plus(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, in_channels=3, classes=12):
    """
    smp.DeepLabV3Plus 모델을 생성하고 DeepLabWrapper로 감싸서 반환합니다.
    """
    # 1. 기본 smp 모델을 생성합니다.
    base_model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )

    # 2. 생성된 모델을 Wrapper 클래스로 감쌉니다.
    wrapped_model = DeepLabWrapper(base_model)

    return wrapped_model