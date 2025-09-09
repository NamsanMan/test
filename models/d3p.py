import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

ENCODER_NAME = "mobilenet_v2"
ENCODER_WEIGHTS = "imagenet"


class DeepLabV3PlusWrapper(nn.Module):
    def __init__(self, encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                 in_channels=3, classes=12):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # 💡 --- 해결의 핵심 (1/2) --- 💡
        # 272 채널 입력을 받아 256 채널로 출력하는 새로운 Fusion 블록을 정의합니다.
        # 이 블록이 smp의 decoder.block1, block2를 대체합니다.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(272, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # ----------------------------

    def forward(self, x, return_feats: bool = False):
        feats = self.model.encoder(x)

        # ASPP 모듈 실행
        aspp_features = self.model.decoder.aspp(feats[-1])
        # ASPP 출력을 얕은 특징 맵 크기에 맞게 보간
        aspp_features = F.interpolate(aspp_features, size=feats[1].shape[-2:], mode='bilinear', align_corners=False)

        # 특징 맵 결합 (256 + 16 = 272 채널)
        concat_features = torch.cat([aspp_features, feats[1]], dim=1)

        # 💡 --- 해결의 핵심 (2/2) --- 💡
        # 기존 smp의 블록 대신, 우리가 직접 만든 fusion_conv를 사용합니다.
        fused_features = self.fusion_conv(concat_features)
        # ----------------------------

        logits = self.model.segmentation_head(fused_features)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if return_feats:
            return logits, tuple(feats[-4:])
        return logits


def create_deeplabv3plus(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                         in_channels=3, classes=12):
        model = DeepLabV3PlusWrapper(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        return model