import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import data_loader
import config
from models import create_model
import evaluate

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# model 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(config.MODEL.NAME)
model.to(device)

# loss function
loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])
# optimizer
optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])
optimizer = optimizer_class(model.parameters(), **config.TRAIN.OPTIMIZER["PARAMS"])
# scheduler
scheduler_class = getattr(optim.lr_scheduler, config.TRAIN.SCHEDULER_RoP["NAME"])
scheduler = scheduler_class(optimizer, **config.TRAIN.SCHEDULER_RoP["PARAMS"])
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
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        #logit tensor를 리턴
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 검증방법 설정 후 loss값 반환
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            # FCN 스타일의 리턴
            #preds = model(imgs)['out']
            # logit tensor를 리턴
            preds = model(imgs)
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
    """
    init=True: config만 기록
    else: best 모델 info 덮어쓰기
    """
    with open(config.GENERAL.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
        og = optimizer.param_groups[0]
        f.write(f"Model         : {model.__class__.__name__}\n")
        f.write(f"Model source  : {config.MODEL.NAME}\n\n")
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write(f"  lr           : {og['lr']}\n")
        f.write(f"  weight_decay : {og.get('weight_decay')}\n")
        f.write(f"Scheduler     : {scheduler.__class__.__name__}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")

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
    # 초기 summary 파일 생성
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_miou   = 0.0
    best_epoch  = 0
    best_ckpt   = config.GENERAL.BASE_DIR / "best_model.pth"

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs+1):
        tr_loss = train_one_epoch(model, data_loader.train_loader, criterion, optimizer)
        vl_loss = validate(model, data_loader.val_loader, criterion)

        # scheduler.step for ReduceLROnPlateau needs val loss
        if epoch <= config.TRAIN.WARMUP_EPOCHS:
            warmup.step()
        else:
            scheduler.step(vl_loss)

        metrics = evaluate.evaluate_all(model, data_loader.val_loader, device)
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[{epoch}/{num_epochs}] "
              f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
              f"val_mIoU={miou:.4f},  PA={pa:.4f}")

        # best model 갱신
        if miou > best_miou:
            best_miou  = miou
            best_epoch = epoch

            # 1) per-class IoU 계산 및 CSV 저장
            per_cls_iou = metrics["per_class_iou"]
            df_iou = pd.DataFrame({
                "Class": config.DATA.CLASS_NAMES,
                "IoU": per_cls_iou
            })
            iou_path = config.GENERAL.LOG_DIR / f"iou_epoch_{epoch}.csv"
            df_iou.to_csv(iou_path, index=False)

            # 2) confusion matrix 계산 및 시각화 저장
            cm = metrics["confusion_matrix"]
            plt.figure(figsize=(8,6))
            plt.imshow(cm, interpolation='nearest')
            plt.title(f"Confusion Matrix (Epoch {epoch})")
            plt.colorbar()
            tick_marks = np.arange(config.DATA.NUM_CLASSES)
            plt.xticks(tick_marks, config.DATA.CLASS_NAMES, rotation=45, ha='right')
            plt.yticks(tick_marks, config.DATA.CLASS_NAMES)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            cm_path = config.GENERAL.LOG_DIR / f"confusion_matrix_epoch_{epoch}.png"
            plt.savefig(str(cm_path), bbox_inches="tight")
            plt.close()
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_mIoU": best_miou
            }, best_ckpt)
            print(f"▶ New best val_mIoU at epoch {epoch}: {miou:.4f} → {best_ckpt}")
            write_summary(init=False, best_epoch=best_epoch, best_miou=best_miou)

        # 10 epoch마다 plot 저장
        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch+1)), train_losses, val_losses)

    end_dt = datetime.now()
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    # NEW: 콘솔 출력
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")

    # NEW: summary_txt에 append
    write_timing(start_dt, end_dt, config.GENERAL.SUMMARY_TXT)

    print(f"Training complete. Best epoch: {best_epoch}, Best val_mIoU: {best_miou:.4f}")
    return best_ckpt

if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)