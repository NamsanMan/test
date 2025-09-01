import torch
import numpy as np
from sklearn.metrics import confusion_matrix

import config

@torch.inference_mode()
def evaluate_all(model, loader, device):
    model.eval()

    all_preds = []
    all_masks = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        preds = torch.argmax(model(imgs), dim=1)

        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    # 전체 예측과 마스크를 하나의 큰 배열로 결합
    preds_np = np.concatenate([p.flatten() for p in all_preds])
    masks_np = np.concatenate([m.flatten() for m in all_masks])

    # 유효한 픽셀만 필터링
    valid = masks_np != config.DATA.IGNORE_INDEX
    preds_np = preds_np[valid]
    masks_np = masks_np[valid]

    # 이 결과로 모든 지표 계산
    # 1. Pixel Accuracy
    den = len(masks_np)
    pa = (np.sum(preds_np == masks_np) / den) if den > 0 else 0.0

    # 2. Confusion Matrix
    cm = confusion_matrix(masks_np, preds_np, labels=list(range(config.DATA.NUM_CLASSES)))

    # 3. mIoU 및 Per-class IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)

    # ignore_index 제외
    iou = np.zeros(config.DATA.NUM_CLASSES, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=(union > 0))
    valid_classes_iou = [iou[c] for c in range(config.DATA.NUM_CLASSES) if c != config.DATA.IGNORE_INDEX and union[c] > 0]
    miou = np.nanmean(valid_classes_iou)

    per_class_iou = np.full(config.DATA.NUM_CLASSES, np.nan)
    for c in range(config.DATA.NUM_CLASSES):
        if c != config.DATA.IGNORE_INDEX and union[c] > 0:
            per_class_iou[c] = iou[c]

    return {
        "mIoU": miou,
        "PixelAcc": pa,
        "per_class_iou": per_class_iou,
        "confusion_matrix": cm
    }