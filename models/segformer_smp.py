import segmentation_models_pytorch as smp

ENCODER_NAME = "mit_b0"
ENCODER_WEIGHTS = "imagenet"


def create_segformer_smp(encoder_name = ENCODER_NAME, encoder_weights = ENCODER_WEIGHTS, in_channels = 3, classes = 12):
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model