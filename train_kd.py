import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import data_loader
import config
from models import create_model
import evaluate

from kd_engines import create_kd_engine

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
LOSS_KEY_DISPLAY_OVERRIDES = {
    "total": "Total Loss",
    "ce_student": "CE Student Loss",
    "ce_teacher": "CE Teacher Loss",
    "kd_logit": "KD Logit Loss",
    "kd_feat": "KD Feature Loss",
    "pca_total": "PCA Total Loss",
    "pca_attn": "PCA Attention Loss",
    "pca_v": "PCA Value Loss",
    "gl_loss": "GL Loss",
}

SCALAR_LOSS_KEYS: List[str] = []
LOSS_KEY_TO_HEADER: Dict[str, str] = {}
LOSS_HEADER_ORDER: List[str] = []


def _is_scalar_loss_value(value) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dim() == 0
    return isinstance(value, (int, float))


def _loss_value_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _display_name_for_loss(key: str) -> str:
    if key in LOSS_KEY_DISPLAY_OVERRIDES:
        return LOSS_KEY_DISPLAY_OVERRIDES[key]
    pretty = key.replace('_', ' ').title()
    if "loss" not in key.lower():
        pretty = f"{pretty} Loss"
    return pretty

# model 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher = create_model(config.KD.TEACHER_NAME).to(device)
student = create_model(config.KD.STUDENT_NAME).to(device)
model = student

if config.KD.FREEZE_TEACHER:
    try:
        ckpt_path = Path(config.TEACHER_CKPT)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Teacher 체크포인트가 'model_state' 키를 가지고 있는지 확인하고 로드
            if "model_state" in ckpt:
                teacher.load_state_dict(ckpt["model_state"])        # "model_state"만 로드 하면서 .pth의 내용중 가중치만 불러옴
                print(f"▶ Successfully loaded pretrained teacher weights from: {ckpt_path}")
            else:
                # 키가 다른 경우를 대비하여 직접 로드 시도
                teacher.load_state_dict(ckpt)
                print(f"▶ Successfully loaded pretrained teacher weights from: {ckpt_path} (direct state dict)")

        else:
             print(f"⚠️ WARNING: Teacher checkpoint not found at {ckpt_path}. Using ImageNet pretrained weights.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load teacher checkpoint. Error: {e}. Using ImageNet pretrained weights.")

# ── KD 엔진 구성 ───────────────────────────────────
kd_engine = create_kd_engine(config.KD, teacher, student).to(device)

# --- build KD projections (dry-run) so their params are included in optimizer ---
imgs0, masks0 = next(iter(data_loader.train_loader))
with torch.no_grad():
    dry_run_out = kd_engine.compute_losses(imgs0.to(device, non_blocking=True),
                                           masks0.to(device, non_blocking=True),
                                           device)

if "total" not in dry_run_out:
    raise KeyError("KD engine must return a 'total' loss entry.")

SCALAR_LOSS_KEYS = [
    key for key, value in dry_run_out.items() if _is_scalar_loss_value(value)
]

if "total" in SCALAR_LOSS_KEYS:
    SCALAR_LOSS_KEYS.remove("total")
    SCALAR_LOSS_KEYS.insert(0, "total")
else:
    SCALAR_LOSS_KEYS.insert(0, "total")

LOSS_KEY_TO_HEADER = {key: _display_name_for_loss(key) for key in SCALAR_LOSS_KEYS}

LOSS_HEADER_ORDER = [LOSS_KEY_TO_HEADER[key] for key in SCALAR_LOSS_KEYS]

print("▶ Tracking losses:", ", ".join(LOSS_KEY_TO_HEADER.values()))

# ── 옵티마이저/스케줄러 ─────────────────────────────
params = []
params += list(student.parameters())
params += list(kd_engine.get_extra_parameters())        # KD에서 projection에 이용되는 1x1 conv의 파라미터 추가
if not config.KD.FREEZE_TEACHER and config.KD.ENGINE_PARAMS.get('w_ce_teacher', 0.0) > 0.0:
    params += list(teacher.parameters())

optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])
optimizer = optimizer_class(params, **config.TRAIN.OPTIMIZER["PARAMS"])

scheduler_class = getattr(optim.lr_scheduler, config.TRAIN.SCHEDULER_CALR["NAME"])
scheduler = scheduler_class(optimizer, **config.TRAIN.SCHEDULER_CALR["PARAMS"])

# warm-up
if config.TRAIN.USE_WARMUP:
    warmup_class = getattr(optim.lr_scheduler, config.TRAIN.WARMUP_SCHEDULER["NAME"])
    # LinearLR의 total_iters 파라미터는 따로 계산하여 추가해줍니다.
    warmup_params = config.TRAIN.WARMUP_SCHEDULER["PARAMS"].copy()
    warmup_params["total_iters"] = config.TRAIN.WARMUP_EPOCHS
    warmup = warmup_class(optimizer, **warmup_params)
else:
    warmup = None

# 1epoch당 학습 방법 설정 후 loss값 반환
def train_one_epoch_kd(kd_engine, loader, optimizer, device):
    kd_engine.train()
    if not SCALAR_LOSS_KEYS:
        raise RuntimeError("No scalar losses registered from KD engine dry-run.")
    # 각 손실을 저장할 딕셔너리 초기화
    epoch_losses = {key: 0.0 for key in SCALAR_LOSS_KEYS}
    pbar = tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training")
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = kd_engine.compute_losses(imgs, masks, device)
        total_loss = out.get("total")
        if total_loss is None:
            raise KeyError("KD engine output does not contain 'total' loss.")
        if not isinstance(total_loss, torch.Tensor):
            raise TypeError("'total' loss must be a torch.Tensor for backpropagation.")
        total_loss.backward()
        optimizer.step()

        # 각 손실 값을 누적
        for key in epoch_losses:
            value = out.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            epoch_losses[key] += _loss_value_to_float(value)

        postfix = {}
        for key in SCALAR_LOSS_KEYS:
            value = out.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            postfix_label = LOSS_KEY_TO_HEADER.get(key, key)
            postfix[postfix_label] = f'{_loss_value_to_float(value):.3f}'
        if postfix:
            pbar.set_postfix(postfix)

    # 평균 손실 값 계산
    num_batches = len(loader)
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}
    return avg_losses

# ── (변경) 검증(학생 기준) ─────────────────────────────────
def validate_student(student_model, loader, criterion):
    student_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = student_model(imgs)  # logits
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)

def plot_progress(epochs, train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(str(config.GENERAL.SAVE_PLOT), bbox_inches="tight")
    plt.close()

def write_summary(init=False, best_epoch=None, best_miou=None):
    # 기존 동일 (단, 모델명은 학생/교사 둘 표시 권장)
    with open(config.GENERAL.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
        og = optimizer.param_groups[0]
        f.write(f"Student Model: {student.__class__.__name__}  (source: {config.KD.STUDENT_NAME})\n")
        f.write(f"Teacher Model: {teacher.__class__.__name__}  (source: {config.KD.TEACHER_NAME})\n\n")
        f.write(f"Teacher Freeze: {config.KD.FREEZE_TEACHER}\n")
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write(f"  lr           : {og['lr']}\n")
        f.write(f"  weight_decay : {og.get('weight_decay')}\n")
        f.write(f"Scheduler     : {scheduler.__class__.__name__}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")
        f.write("=== Knowledge Distillation Configuration ===\n")
        f.write(f"Engine NAME        : {config.KD.ENGINE_NAME}\n")
        f.write(f"Teacher Source CKPT: {config.TEACHER_CKPT}\n\n")
        # 1. 현재 설정된 엔진의 파라미터 딕셔너리를 가져옵니다.
        engine_name = config.KD.ENGINE_NAME
        current_engine_params = config.KD.ALL_ENGINE_PARAMS.get(engine_name, {})
        f.write(f"--- Parameters for '{engine_name}' engine ---\n")
        # 2. 가져온 딕셔너리를 반복하면서 모든 파라미터를 자동으로 기록합니다.
        if not current_engine_params:
            f.write("No parameters found for this engine.\n")
        else:
            for key, value in current_engine_params.items():
                # 보기 좋게 정렬하기 위해 key 문자열의 길이를 25로 맞춥니다.
                f.write(f"{key:<25} : {value}\n")
        f.write("\n")

        if init:
            f.write("=== Best Model (to be updated) ===\n")
            f.write("epoch     : N/A\nbest_val_mIoU : N/A\n\n")
        else:
            f.write("=== Best Model ===\n")
            f.write(f"epoch     : {best_epoch}\n")
            f.write(f"best_val_mIoU : {best_miou:.4f}\n\n")

def write_timing(start_dt, end_dt, path=config.GENERAL.SUMMARY_TXT):
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    with open(path, "a", encoding="utf-8") as f:  # append
        f.write("=== Timing ===\n")
        f.write(f"Start : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)\n\n")

# 학습 진행 및 잘 되고있나 성능평가
def run_training(num_epochs):
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_miou = 0.0
    best_epoch = 0
    best_ckpt = config.GENERAL.BASE_DIR / "best_model.pth"

    # CSV 로그 파일 경로 설정 및 헤더 생성
    log_csv_path = config.GENERAL.LOG_DIR / "training_log.csv"
    loss_headers = LOSS_HEADER_ORDER if LOSS_HEADER_ORDER else ["Total Loss"]
    csv_headers = ["Epoch", *loss_headers, "Val Loss", "Val mIoU", "Pixel Acc", "LR"]
    # 클래스별 IoU 헤더 추가
    for class_name in config.DATA.CLASS_NAMES:
        csv_headers.append(f"IoU_{class_name}")

    # 파일이 없으면 헤더를 포함하여 새로 생성
    if log_csv_path.exists():
        try:
            existing_cols = list(pd.read_csv(log_csv_path, nrows=0).columns)
            if len(existing_cols) > 0:
                csv_headers = existing_cols  # 기존 헤더 신뢰
        except Exception:
            pass
    else:
        # 새 파일 생성 시에만 헤더를 기록
        pd.DataFrame(columns=csv_headers).to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []
    loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
    criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

    for epoch in range(1, num_epochs + 1):
        # train_one_epoch_kd는 이제 손실 딕셔너리를 반환
        tr_losses_dict = train_one_epoch_kd(kd_engine, data_loader.train_loader, optimizer, device)
        tr_loss = tr_losses_dict["total"]  # plot을 위한 total loss

        vl_loss = validate_student(student, data_loader.val_loader, criterion)
        metrics = evaluate.evaluate_all(model, data_loader.val_loader, device)
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        if epoch <= config.TRAIN.WARMUP_EPOCHS:
            warmup.step()
        else:
            # ReduceLROnPlateau 사용시
            #scheduler.step(vl_loss)
            # 미사용시
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[{epoch}/{num_epochs}] "
              f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
              f"val_mIoU={miou:.4f},  PA={pa:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        # CSV 파일에 성능 지표 기록
        log_data = {"Epoch": epoch}
        for key in SCALAR_LOSS_KEYS:
            header_name = LOSS_KEY_TO_HEADER.get(key, key)
            log_data[header_name] = tr_losses_dict.get(key, float("nan"))

        log_data.update({
            "Val Loss": vl_loss,
            "Val mIoU": miou,
            "Pixel Acc": pa,
            "LR": current_lr,
        })
        # 클래스별 IoU를 log_data 딕셔너리에 추가
        per_cls_iou = metrics["per_class_iou"]
        for i, class_name in enumerate(config.DATA.CLASS_NAMES):
            log_data[f"IoU_{class_name}"] = per_cls_iou[i]

        # DataFrame으로 변환 후 CSV 파일에 append
        df_new_row = pd.DataFrame([log_data]).reindex(columns=csv_headers)
        df_new_row.to_csv(log_csv_path, mode='a', header=False, index=False)

        # best model 갱신 시 로그 기록
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch

            # 모델 체크포인트 저장
            torch.save({
                "epoch": epoch,
                "model_state": student.state_dict(),
                "teacher_state": teacher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_mIoU": best_miou
            }, best_ckpt)
            print(f"▶ New best val_mIoU at epoch {epoch}: {miou:.4f} → {best_ckpt}")
            write_summary(init=False, best_epoch=best_epoch, best_miou=best_miou)

        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch + 1)), train_losses, val_losses)

    end_dt = datetime.now()
    write_timing(start_dt, end_dt, config.GENERAL.SUMMARY_TXT)

    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    print(f"\nTraining complete.")
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")
    print(f"Best epoch: {best_epoch}, Best val_mIoU: {best_miou:.4f}")

    return best_ckpt

if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)