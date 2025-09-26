# 1. 각 모델 파일에서 모델 클래스를 가져옵니다.
from .segformer_wrapper import SegFormerWrapper
from .d3p import create_model as d3p
from .model_ddrnet_23slim import DDRNet as DDRNet23Slim
from .segformer_smp import create_segformer_smp as segformer_smp
from .unet import create_unet_model as unet

# 나중에 다른 모델을 추가하면 아래에 계속 추가합니다.
# from .unet import UNet
# from .deeplabv3 import DeepLabV3

import config  # config.py는 프로젝트 루트에 있으므로 바로 import 가능


def create_model(model_name: str):
    """
    모델 이름을 문자열로 받아, 해당 모델의 인스턴스를 생성하고 반환합니다.
    이것을 "모델 팩토리"라고 부릅니다.

    Args:
        model_name (str): 생성할 모델의 이름 (e.g., 'segformer', 'unet')
    """
    model_name = model_name.lower()
    num_classes = config.DATA.NUM_CLASSES

    if model_name in {"segformerb0", "segformerb1", "segformerb3", "segformerb5"}:
        model = SegFormerWrapper(model_name)
        print(f"▶ Model '{[model_name]}' created.", flush=True)

    elif model_name == 'd3p':
        model = d3p(classes=num_classes)
        print(f"▶ Model 'DeepLabV3 plus' created.")

    elif model_name == 'ddrnet23slim':
        model = DDRNet23Slim(pretrained=True, num_classes=num_classes)
        print(f"▶ Model 'DDRNet23Slim' created.")

    elif model_name == 'segformer_smp':
        model = segformer_smp(classes=num_classes)
        print(f"▶ Model 'segformer_smp' created.")

    elif model_name == 'unet':
        model = unet(classes=num_classes)
        print(f"▶ Model 'Unet' created.")

    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    print(f"  - Number of classes: {num_classes}")
    return model