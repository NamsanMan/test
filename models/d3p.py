# d3p.py
# DeepLabV3+ (segmentation_models_pytorch) wrapper:
# - forward(x, return_feats=False)  -> logits
# - forward(x, return_feats=True)   -> (logits, [f_s1, f_s2, f_s3, f_s4])
# - get_backbone_channels(...)      -> [c_s1, c_s2, c_s3, c_s4]
# - create_model(...)               -> DeepLabV3PlusWrapper instance

from typing import List, Sequence, Tuple, Union
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ----- Defaults (필요 시 프로젝트 설정과 맞춰 수정) -----
DEFAULT_ENCODER_NAME     = "mobilenet_v2"
DEFAULT_ENCODER_WEIGHTS  = "imagenet"
DEFAULT_IN_CHANNELS      = 3
DEFAULT_NUM_CLASSES      = 12

# SMP encoder가 보통 [x0, x1, x2, x3, x4] 형태로 feature 리스트를 반환.
# KD/분석에서는 보통 저수준부터 고수준까지 4개(stage 1~4)를 사용하므로 아래처럼 선택.
DEFAULT_STAGE_INDICES: Tuple[int, ...] = (1, 2, 3, 4)


class DeepLabV3PlusWrapper(nn.Module):
    """
    SMP DeepLabV3Plus를 래핑하여:
      - 기본은 logits만 반환 (학습 루프/평가 호환)
      - 필요 시 return_feats=True로 스테이지별 encoder feature 리스트까지 함께 반환
    """

    def __init__(
        self,
        base_model: smp.DeepLabV3Plus,
        stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()

        # smp 모델 구성 요소 바인딩
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        # 메타 정보
        self.stage_indices: Tuple[int, ...] = tuple(stage_indices)
        self.num_classes = int(num_classes)

        # encoder의 채널 정보 (SMP encoder는 보통 .out_channels 제공)
        enc_out_ch = getattr(self.encoder, "out_channels", None)
        if isinstance(enc_out_ch, (list, tuple)):
            self._feat_channels = [
                enc_out_ch[i] for i in self.stage_indices if 0 <= i < len(enc_out_ch)
            ]
        else:
            # 안전장치: 알 수 없으면 None (필요 시 get_backbone_channels 함수 사용)
            self._feat_channels = None

    @property
    def feat_channels(self) -> Union[List[int], None]:
        """
        선택된 stage들의 channel 수 목록. (예: [24, 32, 96, 320])
        일부 encoder에서 감지 실패 시 None.
        """
        return self._feat_channels

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: (B, C, H, W)
            return_feats: True면 (logits, feats[list]) 반환.
                          feats는 stage_indices 순서대로 encoder feature들.

        Returns:
            logits 또는 (logits, feats)
        """
        # 1) encoder: 다중 스케일 feature 리스트
        features: List[torch.Tensor] = self.encoder(x)

        # 2) decoder
        decoder_out: torch.Tensor = self.decoder(features)

        # 3) segmentation head -> logits
        logits: torch.Tensor = self.segmentation_head(decoder_out)

        if not return_feats:
            # 학습 루프/평가에서 바로 loss에 넣을 수 있도록 Tensor만 반환
            return logits

        # KD 등에서 사용할 스테이지별 feature 추출
        feats: List[torch.Tensor] = [
            features[i] for i in self.stage_indices if 0 <= i < len(features)
        ]
        return logits, feats


def get_backbone_channels(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = DEFAULT_IN_CHANNELS,
    stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
) -> List[int]:
    """
    주어진 encoder 설정에서 선택된 stage들의 채널 수를 반환.
    (모델 생성 없이 encoder만 임시로 만들어 확인)

    Returns:
        예: [24, 32, 96, 320]
    """
    enc = smp.encoders.get_encoder(
        encoder_name,
        in_channels=in_channels,
        depth=5,  # DeepLabV3+ 기본 depth=5
        weights=encoder_weights,
    )
    out_ch = getattr(enc, "out_channels", [])
    return [out_ch[i] for i in stage_indices if 0 <= i < len(out_ch)]


def create_model(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = DEFAULT_IN_CHANNELS,
    classes: int = DEFAULT_NUM_CLASSES,
    stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
    **kwargs,
) -> DeepLabV3PlusWrapper:
    """
    외부에서 통일된 API로 호출하도록 만든 팩토리.
    segformer_wrapper와 동일한 사용성을 목표:
        - 기본 호출: model = create_model(...)
        - 학습:      logits = model(imgs)
        - KD:        logits, feats = model(imgs, return_feats=True)
    """
    base = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,  # (예: activation 등 추가 옵션)
    )
    return DeepLabV3PlusWrapper(
        base_model=base,
        stage_indices=stage_indices,
        num_classes=classes,
    )
