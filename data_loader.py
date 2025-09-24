import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as T

import config

# Dateset 구현
class CamVidDataset(Dataset):
    #사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다: __init__, __len__, and __getitem__.

    #__init__ 매서드(class안의 함수를 의미) Dataset 객체가 생성(instantiate)될 때 한 번만 실행됩니다. 여기서는 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와 (다음 장에서 자세히 살펴볼) 두가지 변형(transform)을 초기화합니다.
    def __init__(self, images_dir, masks_dir, file_list=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        if file_list:
            with open(file_list) as f:
                self.files = [line.strip() for line in f]
        else:
            self.files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
        self.transform = transform

    #__len__ 함수는 데이터셋의 샘플 개수를 반환합니다.
    def __len__(self):
        return len(self.files)

    #__getitem__ 함수는 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다.
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])       #이미지 파일들이 저장된 디렉토리 경로
        mask_path = os.path.join(self.masks_dir, self.files[idx])       #레이블(마스크)가 저장된 디렉토리 경로
        image = Image.open(img_path).convert('RGB')                     #원본이 흑백이더라도 3채널로 맞춰 주며 3채널(RGB)형식으로 변환
        mask = Image.open(mask_path)  # 클래스 인덱스 그대로  >>>  주의해야될 점이, 마스크 파일이 png이고 0~11값이 class이어야 하고, void가 11이어야 함

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

# train set에 대한 data augmentation: random crop, random flip, random rotation, color jitter
class TrainAugmentation:
    def __init__(
        self,
        size,
        hflip_prob: float = 0.5,
        crop_prob: float = 0.7,
        crop_range: tuple[float, float] = (80.0, 100.0),
        rotation_prob: float = 0.2,
        rotation_degree: float = 5.0,
        brightness: tuple[float, float] = (0.6, 1.4),
        contrast: tuple[float, float]   = (0.7, 1.2),
        saturation: tuple[float, float] = (0.9, 1.3),
        hue: tuple[float, float]        = (-0.05, 0.05),
    ):
        self.size = size
        self.hflip_prob = hflip_prob

        self.crop_prob = crop_prob
        self.crop_min, self.crop_max = crop_range

        self.rotation_prob = rotation_prob
        self.rotation_degree = rotation_degree

        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, img, mask):
        # 0) 초기 리사이즈
        img  = F.resize(img,  self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # 1) 랜덤 크롭 (70% 확률)
        if random.random() < self.crop_prob:
            # 예: 원본의 80~100% 영역을 무작위 크롭 후 원래 크기로 리사이즈
            target_h, target_w = self.size
            scale_min = self.crop_min / 100.0
            scale_max = self.crop_max / 100.0
            crop_h = int(random.uniform(scale_min, scale_max) * target_h)
            crop_w = int(random.uniform(scale_min, scale_max) * target_w)
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(crop_h, crop_w))
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # 2) 랜덤 수평 뒤집기
        if random.random() < self.hflip_prob:
            img  = F.hflip(img)
            mask = F.hflip(mask)

        # 3) 랜덤 회전 (20% 확률, -5도~+5도)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.rotation_degree, self.rotation_degree)
            img  = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR,expand=False)
            mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=11)    #fill에는 빈공간을 void label로 매꾸게 설정(class 11 = void)

        # 4) 컬러 지터
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        h = random.uniform(*self.hue)
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img,   c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img,        h)

        # 5) 텐서 변환 & 정규화
        img  = F.to_tensor(img)
        img  = F.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


# 이미지와 마스크(레이블)을 동시에 전처리하기 위해 만든다
class SegmentationTransform:
    def __init__(self, size):
        # 크기(사이즈)
        self.size = size
    def __call__(self, img, mask):
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


# A_set 만 B_set으로 바꿔서 2fold 진행
train_dataset = CamVidDataset(
    images_dir = config.DATA.TRAIN_IMG_DIR,
    masks_dir  = config.DATA.TRAIN_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform =SegmentationTransform(config.DATA.INPUT_RESOLUTION)
)

val_dataset = CamVidDataset(
    images_dir = config.DATA.VAL_IMG_DIR,
    masks_dir  = config.DATA.VAL_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform =SegmentationTransform(config.DATA.INPUT_RESOLUTION)
)

test_dataset = CamVidDataset(
    images_dir = config.DATA.TEST_IMG_DIR,
    masks_dir  = config.DATA.TEST_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform =SegmentationTransform(config.DATA.INPUT_RESOLUTION)
)

# 데이터셋을 train.py로 넘겨줌
# train_loader에만 shuffle true
# num_workers는 전부 같은 값으로 통일(0 아니면 1)
# val_loader와 test_lodaer의 batch_size는 1로 하는게 맞고, train_loader의 batchsize는 4를 추천
train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE,  shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=1,  shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1,  shuffle=False, num_workers=0)