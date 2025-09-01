# 1. 각 모델 파일에서 모델 클래스를 가져옵니다.
from .segformerb0 import SegFormerWrapper as SegFormerB0
from .segformerb1 import SegFormerWrapper as SegFormerB1
from .segformerb3 import SegFormerWrapper as SegFormerB3
from .segformerb4 import SegFormerWrapper as SegFormerB4
from .segformerb5 import SegFormerWrapper as SegFormerB5
from .d3p import create_deeplabv3plus as d3p

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

    if model_name == 'segformerb0':
        model = SegFormerB0(num_classes=num_classes)
        print(f"▶ Model 'SegFormer MiT-B0' created.")
        print(f"  - Pretrained source: 'nvidia/mit-b0'")

    elif model_name == 'segformerb1':
        model = SegFormerB1(num_classes=num_classes)
        print(f"▶ Model 'SegFormer MiT-B1' created.")
        print(f"  - Pretrained source: 'nvidia/mit-b1'")

    elif model_name == 'segformerb3':
        model = SegFormerB3(num_classes=num_classes)
        print(f"▶ Model 'SegFormer MiT-B3' created.")
        print(f"  - Pretrained source: 'nvidia/mit-b3'")

    elif model_name == 'segformerb4':
        model = SegFormerB4(num_classes=num_classes)
        print(f"▶ Model 'SegFormer MiT-B4' created.")
        print(f"  - Pretrained source: 'nvidia/mit-b4'")

    elif model_name == 'segformerb5':
        model = SegFormerB5(num_classes=num_classes)
        print(f"▶ Model 'SegFormer MiT-B5' created.")
        print(f"  - Pretrained source: 'nvidia/mit-b5'")

    elif model_name == 'd3p':
        model = d3p(classes=num_classes)
        print(f"▶ Model 'DeepLabV3 plus' created.")

    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    print(f"  - Number of classes: {num_classes}")
    return model