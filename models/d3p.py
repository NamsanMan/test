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

    def forward(self, x, return_feats: bool = False):
        feats = self.model.encoder(x)
        decoder_output = self.model.decoder(*feats)
        logits = self.model.segmentation_head(decoder_output)
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