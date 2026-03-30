# train_detection.py — PRODUCTION GRADE (GPU OPTIMIZED)

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.advanced_models import EfficientDetectionModel
from models.resnet_models import TumorDetectionModel

# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 4  # 🔥 Safe for 4GB VRAM (RTX 3050 Ti)
USE_AMP = torch.cuda.is_available()  # Mixed precision on GPU only

# 🔥 GPU benchmark mode — speeds up training with fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
else:
    print("⚠️  No GPU detected — running on CPU")

# ─────────────────────────────────────────────
# 🔥 FOCAL LOSS (FN-FRIENDLY)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def get_model(name):
    if name == "efficientnet":
        print("🚀 Using EfficientNet")
        return EfficientDetectionModel(pretrained=True)
    else:
        print("🚀 Using ResNet101")
        return TumorDetectionModel(pretrained=True)


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def get_loaders(data_dir):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)

    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets)

    weights = 1.0 / class_counts[targets]
    sampler = WeightedRandomSampler(weights, len(weights))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=pin, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=2, pin_memory=pin, persistent_workers=True
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Class distribution: {class_counts}")
    print(f"Batch size: {BATCH_SIZE} | pin_memory: {pin} | num_workers: 2")

    return train_loader, val_loader


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for x, y in loader:
        # 🔥 non_blocking for async GPU transfer (faster with pin_memory)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        if USE_AMP:
            # 🔥 Automatic Mixed Precision — FP16 forward, FP32 gradients
            with autocast():
                out = model(x).squeeze(1)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x).squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            if USE_AMP:
                with autocast():
                    out = model(x).squeeze(1)
            else:
                out = model(x).squeeze(1)

            # 🔥 Guard against NaN/Inf from AMP overflow in early epochs
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
            probs = torch.sigmoid(out).cpu().numpy()

            y_probs.extend(probs)
            y_true.extend(y.numpy())

    y_true  = np.array(y_true)
    y_probs = np.array(y_probs)

    # 🔥 Remove any remaining NaN/Inf values
    valid_mask = np.isfinite(y_probs)
    if valid_mask.sum() < len(y_probs):
        print(f"  ⚠️  Filtered {(~valid_mask).sum()} NaN/Inf probability values")
        y_true  = y_true[valid_mask]
        y_probs = y_probs[valid_mask]

    if len(y_true) == 0:
        print("  ⚠️  No valid predictions — skipping epoch metrics")
        return 0.0, 0.0, 0.0, 0.0, np.array([[0,0],[0,0]]), np.array([]), np.array([]), np.array([]), 0.5

    # 🔥 PR-based threshold (FN-aware)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # 🔥 Lower bound for FN safety
    threshold = max(threshold, 0.30)

    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm, y_true, y_probs, y_pred, threshold


# ─────────────────────────────────────────────
# SAVE PLOTS
# ─────────────────────────────────────────────
def save_plots(y_true, y_probs, y_pred, name, out_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{out_dir}/{name}_cm.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.savefig(f"{out_dir}/{name}_roc.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("PR Curve")
    plt.savefig(f"{out_dir}/{name}_pr.png")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--model", choices=["efficientnet", "resnet101"], default="efficientnet")
    parser.add_argument("--out", default="models")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader = get_loaders(args.data)
    model = get_model(args.model).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss()
    scaler = GradScaler(enabled=USE_AMP)  # 🔥 AMP scaler

    # ── 🎯 STRICT PRIORITY: FN → Recall → F1 (Accuracy NOT used for selection) ──
    best_fn     = float("inf")  # lower is better
    best_recall = 0.0           # higher is better
    best_f1     = 0.0           # higher is better
    metrics_log = []

    print(f"\n🚀 Training {args.model.upper()} | Device: {DEVICE} | AMP: {USE_AMP}\n")

    for epoch in range(args.epochs):
        # 🔥 Clear GPU cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss = train(model, train_loader, optimizer, criterion, scaler)
        acc, prec, rec, f1, cm, y_true, y_probs, y_pred, thr = evaluate(model, val_loader)

        tn, fp, fn, tp = cm.ravel()
        fn_int = int(fn)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss     : {loss:.4f}")
        print(f"  FN       : {fn_int}  ← PRIMARY selection metric")
        print(f"  Recall   : {rec:.4f} ← SECONDARY selection metric")
        print(f"  F1       : {f1:.4f} ← TERTIARY selection metric")
        print(f"  Precision: {prec:.4f}")
        print(f"  Accuracy : {acc:.4f} ← REPORTED ONLY (not used for selection)")
        print(f"  Threshold: {thr:.4f}")

        metrics = {
            "epoch": epoch + 1,
            "loss": round(loss, 6),
            "fn": fn_int,
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "precision": round(prec, 6),
            "accuracy": round(acc, 6),
            "threshold": round(float(thr), 6)
        }
        metrics_log.append(metrics)

        # ── 🎯 STRICT 3-STAGE MODEL SELECTION (FN → Recall → F1) ──
        save_model = False
        reason = ""

        if fn_int < best_fn:
            save_model = True
            reason = f"FN improved: {best_fn} → {fn_int}"
        elif fn_int == best_fn and rec > best_recall:
            save_model = True
            reason = f"FN tied={fn_int}, Recall improved: {best_recall:.4f} → {rec:.4f}"
        elif fn_int == best_fn and rec == best_recall and f1 > best_f1:
            save_model = True
            reason = f"FN+Recall tied, F1 improved: {best_f1:.4f} → {f1:.4f}"

        if save_model:
            best_fn     = fn_int
            best_recall = rec
            best_f1     = f1
            torch.save(model.state_dict(), f"{args.out}/detection_{args.model}.pth")
            save_plots(y_true, y_probs, y_pred, args.model, args.out)
            np.savetxt(f"{args.out}/confusion_matrix_{args.model}.txt", cm, fmt="%d")
            print(f"  ✅ Model saved — {reason}")
        else:
            print(f"  ⏭  No improvement (best FN={best_fn}, Recall={best_recall:.4f}, F1={best_f1:.4f})")

        with open(f"{args.out}/metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=4)

    print("\n🎯 Detection Training Complete")
    print(f"  Best FN     : {best_fn}")
    print(f"  Best Recall : {best_recall:.4f}")
    print(f"  Best F1     : {best_f1:.4f}")


if __name__ == "__main__":
    main()