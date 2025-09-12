# print_end_feats_and_logits.py
import torch
import torch.nn.functional as F

# 네 프로젝트 래퍼 (경로/이름은 네 코드에 맞춰둠)
from models.segformer_wrapper import SegFormerWrapper
from models.d3p import create_model as create_deeplab


def print_endpoints(name: str, logits: torch.Tensor, feats):
    """
    - feats: 리스트(스테이지별 feature). encoder 최종 feature는 feats[-1]로 정의.
    - logits: decoder까지 거친 최종 출력(logits).
    """
    last_feat = feats[-1]  # encoder 최종 feature
    print(f"[{name}]")
    print(f"  ├─ Encoder 최종 feature: shape={tuple(last_feat.shape)}  (C={last_feat.shape[1]})")
    print(f"  └─ Decoder 최종 logits : shape={tuple(logits.shape)}      (C={logits.shape[1]})")
    print("-" * 60)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 12

    # 입력 텐서 (네가 관측한 해상도에 맞춤)
    imgs_teacher = torch.randn(1, 3, 360, 480, device=device)  # SegFormer 입력 예시
    imgs_student = torch.randn(1, 3, 360, 480, device=device)  # DeepLab 입력 예시

    # 모델 생성
    teacher = SegFormerWrapper("segformerb5", num_classes=num_classes).to(device).eval()
    student = create_deeplab(in_channels=3, classes=num_classes).to(device).eval()

    with torch.no_grad():
        # 둘 다 (logits, feats) 형태로 받음. feats[-1]을 encoder 최종 feature로 사용
        t_logits, t_feats = teacher(imgs_teacher, return_feats=True)
        s_logits, s_feats = student(imgs_student, return_feats=True)

    # 결과 출력 (encoder 마지막 feature & decoder 최종 logits)
    print(f"Device: {device}")
    print(f"Input (Teacher / SegFormer) : {tuple(imgs_teacher.shape)}")
    print(f"Input (Student / DeepLabV3+): {tuple(imgs_student.shape)}")
    print("-" * 60)

    print_endpoints("Teacher (SegFormer-B5)", t_logits, t_feats)
    print_endpoints("Student (DeepLabV3+ MBv2)", s_logits, s_feats)

    # (옵션) KD를 염두에 둔 보간 확인: teacher logits을 student 해상도에 맞추기
    t_logits_resized = F.interpolate(
        t_logits, size=s_logits.shape[-2:], mode="bilinear", align_corners=False
    )
    print(f"[참고] Teacher logits을 Student 해상도에 맞춘 보간 결과: {tuple(t_logits_resized.shape)}")


if __name__ == "__main__":
    main()
