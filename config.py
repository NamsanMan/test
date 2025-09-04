from pathlib import Path
import numpy as np
import os

# 'COLAB_GPU'는 Colab 환경에만 존재하는 환경 변수입니다.
IS_COLAB = 'COLAB_GPU' in os.environ

#  2. 플래그 값에 따라 경로를 다르게 설정
if IS_COLAB:
    # --- Colab 환경일 때의 경로 ---
    print("▶ Running in Google Colab environment.")

    # Colab의 구글 드라이브 경로를 기본 경로로 설정
    BASE_DRIVE_DIR = Path('/content/drive/MyDrive/LAB')

    DATA_DIR = BASE_DRIVE_DIR / "datasets/project_use/CamVid_12_2Fold_LR_x4_Bilinear/A_set"
    BASE_DIR = BASE_DRIVE_DIR / "result_files/test_results"

else:
    # --- 로컬 환경일 때의 경로 ---
    print("▶ Running in local environment.")

    # 기존에 사용하시던 로컬 경로 설정
    DATA_DIR = Path(r"E:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\A_set")
    BASE_DIR = Path(r"E:\LAB\result_files\test_results")

# ──────────────────────────────────────────────────────────────────
# 1. GENERAL: 프로젝트 전반 및 실험 관리 설정
# ──────────────────────────────────────────────────────────────────
class GENERAL:
    # 실험 프로젝트 이름
    PROJECT_NAME = "TEMP"

    # 결과 파일을 저장할 기본 경로
    BASE_DIR = BASE_DIR / PROJECT_NAME
    LOG_DIR = BASE_DIR / "log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    SUMMARY_TXT = LOG_DIR / "training_summary.txt"
    SAVE_PLOT = LOG_DIR / "training_progress.png"

    SEED = 42

# ──────────────────────────────────────────────────────────────────
# 2. DATA: 데이터셋 관련 설정
# ──────────────────────────────────────────────────────────────────
class DATA:
    # 데이터셋 경로
    DATA_DIR   = DATA_DIR  # data_loader에서 사용하던 경로
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"

    TRAIN_IMG_DIR = TRAIN_DIR / "images"
    TRAIN_LABEL_DIR = TRAIN_DIR / "labels"
    VAL_IMG_DIR = VAL_DIR / "images"
    VAL_LABEL_DIR = VAL_DIR / "labels"
    TEST_IMG_DIR = TEST_DIR / "images"
    TEST_LABEL_DIR = TEST_DIR / "labels"

    FILE_LIST = None

    # 입력 이미지 해상도 >> 원본 이미지의 크기가 아닌 모델에 들어가게 되는 input size
    INPUT_RESOLUTION = (360, 480)  # H, W

    # 배치 사이즈 및 데이터 로딩 워커 수
    BATCH_SIZE = 4

    # 클래스 정보
    CLASS_NAMES = [
        "Sky", "Building", "Pole", "Road", "Sidewalk",
        "Tree", "SignSymbol", "Fence", "Car",
        "Pedestrian", "Bicyclist", "Void"
    ]
    NUM_CLASSES = len(CLASS_NAMES) # =12
    IGNORE_INDEX = 11  # 'Void' 클래스의 인덱스

    # grayscale label(ground truth 포함)을 공식 컬러 매핑과 동일하게 시각화를 위해 컬러 매핑
    CLASS_COLORS = np.array([
        [128, 128, 128],  # Sky
        [128, 0, 0],  # Building
        [192, 192, 128],  # Pole
        [128, 64, 128],  # Road
        [0, 0, 192],  # Sidewalk
        [128, 128, 0],  # Tree
        [192, 128, 128],  # SignSymbol
        [64, 64, 128],  # Fence
        [64, 0, 128],  # Car
        [64, 64, 0],  # Pedestrian
        [0, 128, 192],  # Bicyclist
        [0, 0, 0],  # Void
    ], dtype=np.uint8)

# ──────────────────────────────────────────────────────────────────
# 3. MODEL: 모델 설정
# ──────────────────────────────────────────────────────────────────

class MODEL:
    NAME = 'segformerb5'

    """
    available models:
    segformerb0
    segformerb1
    segformerb3
    segformerb4
    segformerb5
    d3p
    """

# ──────────────────────────────────────────────────────────────────
# 4. TRAIN: 훈련 과정 관련 설정
# ──────────────────────────────────────────────────────────────────
class TRAIN:
    EPOCHS = 100
    USE_WARMUP = True
    WARMUP_EPOCHS = 5

    # 딕셔너리 형태로 통일
    OPTIMIZER = {
        "NAME": "AdamW",
        "PARAMS": {
            "lr": 1e-4,
            "weight_decay": 1e-2
        }
    }

    SCHEDULER_RoP = {
        "NAME": "ReduceLROnPlateau",
        "PARAMS": {
            "mode": 'min',
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6
        }
    }

    SCHEDULER_CALR = {
        "NAME": "CosineAnnealingLR",
        "PARAMS": {
            "T_max": EPOCHS - WARMUP_EPOCHS,
            "eta_min": 1e-6
        }
    }

    LOSS_FN = {
        "NAME": "CrossEntropyLoss",
        "PARAMS": {
            "ignore_index": 11
        }
    }

    # Warmup도 동일한 구조로 추가
    WARMUP_SCHEDULER = {
        "NAME": "LinearLR",
        "PARAMS": {
            "start_factor": 0.1,
            "end_factor": 1.0,
        }
    }


# ──────────────────────────────────────────────────────────────────
# 5. KD: Knowledge Distillation 관련 설정
# ──────────────────────────────────────────────────────────────────
class KD:
    ENABLE = True

    ENGINE_NAME = "basic"
    """
    available engines:
    basic
    kd_losses
    """

    # 모델 선택
    TEACHER_NAME = 'segformerb5'
    STUDENT_NAME = 'segformerb0'
    # 이미 학습된 teacher .pth 경로 (없으면 None), KD경로는 일단 colab경로로 해놓음
    TEACHER_CKPT = r'E:\LAB\result_files\test_results\Aset_LR_segb5\best_model.pth'  # ← 당신 경로로 변경
    # 교사 고정 여부
    FREEZE_TEACHER = True

    ALL_ENGINE_PARAMS = {
        "basic": {
            "stage_weights": [0.25, 0.5, 0.75, 1.0],  # SegFormer 인코더 4단계 스테이지 가중치
            "t": 2.0,  # KD temperature
            "w_ce_student": 1.0,  # 학생 CE
            "w_ce_teacher": 0.0,  # 교사 CE (교사도 GT로 같이 fine tunning 하지 않으려면 0.0)
            "w_logit": 0.05,  # 로짓 KD
            "w_feat": 0.25,  # 피처 KD
            "ignore_index": DATA.IGNORE_INDEX,
            "use_logit_kd": True,  # logit KD 사용 여부 // teacher와 student의 label이 같지 않으면 false
            "feat_l2_normalize": True,  # 피처 KD 시 채널 방향 L2-정규화 사용 다음 실험때 이거 없애보기
            "freeze_teacher": FREEZE_TEACHER
        },
        "kd_losses": {
            "t": 2.0,
            "p": 2,
            "w_ce_student": 1.0,
            "w_logit": 0.05,
            "w_feat": 0.25
        }
    }

    ENGINE_PARAMS = ALL_ENGINE_PARAMS[ENGINE_NAME]