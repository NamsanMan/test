import torch
from transformers import SegformerForSemanticSegmentation

# --- 1. 모델 준비 ---
# Hugging Face Hub에서 사전 학습된 SegFormer-B5 모델을 불러옵니다.
model_name = "nvidia/mit-b5"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

print(f"✅ Hugging Face에서 '{model_name}' 모델을 불러왔습니다.")
print("-" * 50)


# --- 2. 모델 실행 및 결과 확인 ---
# (N, C, H, W) 형태의 임의의 입력 텐서 생성
# 예: (1, 3, 512, 512) 크기의 이미지
dummy_input = torch.randn(1, 3, 360, 480)

print(f"입력 이미지 크기: {dummy_input.shape}\n")

# output_hidden_states=True 플래그를 사용하면 모든 Encoder 스테이지의 출력을 얻을 수 있습니다.
# 모델을 통과시켜 출력을 받습니다.
with torch.no_grad():
    outputs = model(pixel_values=dummy_input, output_hidden_states=True)

# Hugging Face의 출력은 속성으로 접근할 수 있는 객체입니다.
logits = outputs.logits
encoder_hidden_states = outputs.hidden_states


print("\n--- 결과 ---")
# Encoder 출력 확인
print("Encoder가 생성한 다중 스케일 특징 맵 (Hidden States):")
# hidden_states는 튜플(tuple) 형태입니다.
for i, feature_map in enumerate(encoder_hidden_states):
    print(f"  - Hidden State {i}: {feature_map.shape}")

print("\n")
# Decoder 출력 확인
print(f"Decoder 최종 Logit 크기: {logits.shape}")
print("-" * 50)


# --- 3. 최종 출력 업샘플링 (중요!) ---
# Hugging Face SegFormer의 기본 출력(logits)은 원본의 1/4 크기입니다.
# 최종 분할 마스크를 얻으려면 직접 업샘플링해야 합니다.
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=dummy_input.shape[2:],  # 원본 이미지 크기 (H, W)
    mode='bilinear',
    align_corners=False
)
print("✅ 최종 Logit을 원본 이미지 크기로 업샘플링했습니다.")
print(f"업샘플링 후 Logit 크기: {upsampled_logits.shape}")