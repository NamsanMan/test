# sanity_kd_io_check.py
import torch
import torch.nn.functional as F

# 네 프로젝트의 래퍼들 (파일명/함수명은 너가 만든 그대로 사용)
from segformer_wrapper import SegFormerWrapper as create_segformer
from d3p import create_model as create_deeplab

def describe(name, logits, feats=None):
    print(f"[{name}]")
    print(f"  logits: {tuple(logits.shape)}  (C={logits.shape[1]})")
    if feats is not None:
        for i, f in enumerate(feats):
            print(f"  feat[{i}]: {tuple(f.shape)}")
    print("-" * 60)

def main(use_loader=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 12

    # ---------- 입력 준비 ----------
    # segformer 입력 해상도 테스트
    imgs_sf = torch.randn(1, 3, 360, 480, device=device)
    # deeplab 입력 해상도 테스트
    imgs_dl = torch.randn(1, 3, 368, 480, device=device)

    # ---------- 모델 준비 ----------
    teacher = create_segformer(classes=num_classes)         # SegFormer MiT-B5 래퍼
    student = create_deeplab(in_channels=3, classes=num_classes)  # DeepLabV3+ MBv2 래퍼

    teacher.to(device).eval()
    student.to(device).eval()

    with torch.no_grad():
        # 두 모델 모두 (logits, feats) 형태로 받기 위해 return_feats=True
        t_logits, t_feats = teacher(imgs_sf, return_feats=True)
        s_logits, s_feats = student(imgs_dl, return_feats=True)

    # ---------- 결과 출력 ----------
    print(f"Device: {device}")
    print(f"Input : {tuple(imgs.shape)}")
    print("-" * 60)

    describe("Teacher (SegFormer-B5)", t_logits, t_feats)
    describe("Student (DeepLabV3+ MobileNetV2)", s_logits, s_feats)

    # KD용으로 teacher logits을 student 해상도에 맞춰보는 예시
    t_logits_resized = F.interpolate(
        t_logits, size=s_logits.shape[-2:], mode="bilinear", align_corners=False
    )
    print(f"[KD] resized teacher logits -> {tuple(t_logits_resized.shape)}  "
          f"(target student spatial size: {s_logits.shape[-2:]})")

    # 클래스 수 불일치 경고 (ex. teacher가 1000채널로 나오는 상태 감지)
    if t_logits.shape[1] != num_classes:
        print(f"⚠️  Warning: teacher logits C={t_logits.shape[1]} (expected {num_classes}). "
              f"교사 헤드(클래스 수) 확인 필요!")

if __name__ == "__main__":
    # use_loader=True 로 바꾸면 실제 배치로 테스트
    main(use_loader=False)
