import torch
# kd/basic_kd.py 파일에서 BasicKD 클래스를 가져왔다고 가정합니다.
from kd.basic_kd import BasicKD
import config
from models import create_model

# --- 사전 준비 (사용자 환경에 맞게 수정) ---

# 1. 실제 모델 로드 (또는 테스트용 Mock 모델 생성)
# 예시: SegFormer B0, B5. 클래스 개수는 12 (11개 + void)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher = create_model(config.KD.TEACHER_NAME).to(device)
student = create_model(config.KD.STUDENT_NAME).to(device)


# 2. BasicKD 모듈 초기화 (가중치는 일단 1로 설정)
# loss 스케일만 볼 것이므로 초기 가중치는 중요하지 않습니다.
kd_module = BasicKD(
    teacher=teacher, student=student,
    stage_weights=config.KD.STAGE_WEIGHTS,
    t=config.KD.T,
    w_ce_student=config.KD.W_CE_STUDENT,
    w_ce_teacher=config.KD.W_CE_TEACHER,
    w_logit=config.KD.W_LOGIT if config.KD.USE_LOGIT_KD else 0.0,
    w_feat=config.KD.W_FEAT,
    ignore_index=config.DATA.IGNORE_INDEX,
    use_logit_kd=config.KD.USE_LOGIT_KD,
    feat_l2_normalize=config.KD.FEAT_L2_NORMALIZE,
    freeze_teacher=config.KD.FREEZE_TEACHER
).to(device)

# 3. 더미 데이터 생성 (실제 데이터로더의 출력과 동일한 형태)
batch_size = 4
img_size = 256
num_classes = 12 # 0~10번 클래스 + 255(ignore)
dummy_imgs = torch.randn(batch_size, 3, img_size, img_size).to(device)
dummy_masks = torch.randint(0, num_classes, (batch_size, img_size, img_size), dtype=torch.long).to(device)

# --- Loss 스케일 확인 ---
student.train()
teacher.eval()

# 모델 forward 및 loss 계산
# compute_losses가 각 loss를 담은 딕셔너리를 반환합니다.
loss_dict = kd_module.compute_losses(dummy_imgs, dummy_masks, device)

# 각 loss의 unweighted 스케일 출력
print("--- Unweighted Loss Scale Check ---")
print(f"CE Student Loss : {loss_dict['ce_student'].item():.4f}")
print(f"Logit KD Loss   : {loss_dict['kd_logit'].item():.4f}")
print(f"Feature KD Loss : {loss_dict['kd_feat'].item():.4f}")

# --- 결과 해석 예시 ---
# 만약 출력이 다음과 같다면:
# CE Student Loss : 2.5123
# Logit KD Loss   : 8.7543
# Feature KD Loss : 0.0432
#
# Feature KD Loss가 다른 loss에 비해 매우 작으므로,
# Feature KD의 가중치를 20~50 정도로 높여야 다른 loss와 영향력이 비슷해질 것이라고 추론할 수 있습니다.
# 반면 Logit KD Loss는 CE Loss보다 크므로, 가중치를 1보다 작은 0.3 정도로 시작해볼 수 있습니다.