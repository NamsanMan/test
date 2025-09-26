# d3p.py (drop-in update)
from typing import List, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from config import DATA

DEFAULT_ENCODER_NAME     = "mobilenet_v2"
DEFAULT_ENCODER_WEIGHTS  = "imagenet"
DEFAULT_IN_CHANNELS      = 3
DEFAULT_NUM_CLASSES      = DATA.NUM_CLASSES
DEFAULT_STAGE_INDICES: Tuple[int, ...] = (1, 2, 3, 4, 5)  # smp에서 0은 input이므로, 5개의 stage가 있는 mobilenetV2를 사용한다면 1~5까지 있어야됨

# === 새로 추가: 내부 패딩 설정 ===
AUTO_PAD_STRIDE = 16          # DeepLab(MNv2)에서 안전한 내부 stride >> smp가서 확인해보면 기본 OS(output sride)가 16으로 되어있음 >> 기본값 이용
PAD_MODE = "replicate"        # 'replicate' or 'reflect' 권장 (zeros보다 artifact 적음)

class DeepLabV3PlusWrapper(nn.Module):
    def __init__(
        self,
        base_model: smp.DeepLabV3Plus,
        stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        self.stage_indices: Tuple[int, ...] = tuple(stage_indices)
        self.num_classes = int(num_classes)

        enc_out_ch = getattr(self.encoder, "out_channels", None)
        if isinstance(enc_out_ch, (list, tuple)):
            self._feat_channels = [
                enc_out_ch[i] for i in self.stage_indices if 0 <= i < len(enc_out_ch)
            ]
        else:
            self._feat_channels = None

    @property
    def feat_channels(self) -> Union[List[int], None]:
        return self._feat_channels

    def _pad_to_stride(self, x: torch.Tensor, stride: int = AUTO_PAD_STRIDE):
        B, C, H, W = x.shape
        Ht = math.ceil(H / stride) * stride
        Wt = math.ceil(W / stride) * stride
        pad_h = Ht - H
        pad_w = Wt - W
        if pad_h == 0 and pad_w == 0:
            return x, (H, W), (0, 0)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=PAD_MODE)
        return x_pad, (H, W), (pad_h, pad_w)

    def _crop_spatial(self, t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # 마지막 2차원만 크롭
        return t[..., :H, :W]

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        # 1) 내부에서 stride 배수로 자동 패딩
        x_pad, (H_orig, W_orig), (pad_h, pad_w) = self._pad_to_stride(x, AUTO_PAD_STRIDE)

        # 2) encoder/decoder/seg head
        features: List[torch.Tensor] = self.encoder(x_pad)
        dec_out: torch.Tensor = self.decoder(features)
        logits_pad: torch.Tensor = self.segmentation_head(dec_out)

        # 3) logits는 원본 H×W로 크롭 (CE/KD 바로 사용 가능)
        logits = self._crop_spatial(logits_pad, H_orig, W_orig)

        if not return_feats:
            return logits

        # 4) 스테이지 feat도 원본 공간에 대응되도록 크롭
        #    각 스테이지 stride를 동적으로 추정해 ceil(H/stride), ceil(W/stride)로 자른다.
        Hp, Wp = x_pad.shape[-2], x_pad.shape[-1]
        feats_out: List[torch.Tensor] = []
        for i in self.stage_indices:
            f = features[i]
            fh, fw = f.shape[-2], f.shape[-1]
            # stage stride 추정 (보통 정수: 4/8/16/32)
            sh = max(1, Hp // fh)
            sw = max(1, Wp // fw)
            Hf = math.ceil(H_orig / sh)
            Wf = math.ceil(W_orig / sw)
            f = self._crop_spatial(f, Hf, Wf)
            feats_out.append(f)

        return logits, feats_out


def get_backbone_channels(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = DEFAULT_IN_CHANNELS,
    stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
) -> List[int]:
    enc = smp.encoders.get_encoder(
        encoder_name,
        in_channels=in_channels,
        depth=5,
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
    base = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
    return DeepLabV3PlusWrapper(
        base_model=base,
        stage_indices=stage_indices,
        num_classes=classes,
    )
