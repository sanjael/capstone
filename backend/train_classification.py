# train_classification.py — PRODUCTION GRADE (GPU OPTIMIZED)

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.resnet_models import TumorClassificationModel
from models.advanced_models import EfficientClassificationModel

# ─────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 8       # 🔥 Safe for 4GB VRAM (RTX 3050 Ti)
USE_AMP  = torch.cuda.is_available()

# 🔥 GPU benchmark mode
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
else:
    print("⚠️  No GPU detected — running on CPU")


# ─────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────
def get_model(name, num_classes=3):
    if name == "efficientnet":
        print("🚀 Using EfficientNet for classification")
        return EfficientClassificationModel(pretrained=True, num_classes=num_classes)
    else:
        print("🚀 Using ResNet101 for classification")
        return TumorClassificationModel(num_classes=num_classes, pretrained=True)


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def get_loaders(data_dir):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=pin, persistent_workers=True,
        drop_last=True  # prevent batch size=1 for BatchNorm
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=2, pin_memory=pin, persistent_workers=True
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")
    print(f"Batch size: {BATCH_SIZE} | pin_memory: {pin} | num_workers: 2")

    return train_loader, val_loader, train_ds.classes


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for x, y in loader:
        # 🔥 non_blocking for async GPU transfer
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        if USE_AMP:
            with autocast():
                out  = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(x)
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
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            if USE_AMP:
                with autocast():
                    out = model(x)
            else:
                out = model(x)
            # 🔥 Guard against NaN/Inf from AMP overflow
            out   = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm, y_true, y_pred


# ─────────────────────────────────────────────
# SAVE PLOTS
# ─────────────────────────────────────────────
def save_plots(cm, classes, out_dir, prefix):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — Classification ({prefix})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/classification_{prefix}_cm.png")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out",    default="models")
    parser.add_argument(
        "--model",
        choices=["resnet", "efficientnet"],
        default="resnet",
        help="Model architecture: resnet | efficientnet"
    )

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Output filename per model type
    out_weight_name = {
        "resnet":       "classification_model.pth",
        "efficientnet": "classification_efficientnet.pth",
    }[args.model]

    train_loader, val_loader, classes = get_loaders(args.data)
    model = get_model(args.model, num_classes=len(classes)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=USE_AMP)

    # ── 🎯 SELECTION: Highest F1 (Accuracy logged but NOT used) ──
    best_f1      = 0.0
    patience     = 5
    no_improve   = 0
    metrics_log  = []

    print(f"\n🚀 Training {args.model.upper()} | Device: {DEVICE} | AMP: {USE_AMP}\n")

    for epoch in range(args.epochs):
        # 🔥 Clear GPU cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss = train(model, train_loader, optimizer, criterion, scaler)
        acc, prec, rec, f1, cm, y_true, y_pred = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss     : {loss:.4f}")
        print(f"  F1       : {f1:.4f} ← PRIMARY selection metric")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Accuracy : {acc:.4f} ← REPORTED ONLY (not used for selection)")

        metrics = {
            "epoch":     epoch + 1,
            "loss":      round(loss, 6),
            "f1":        round(f1, 6),
            "precision": round(prec, 6),
            "recall":    round(rec, 6),
            "accuracy":  round(acc, 6),
        }
        metrics_log.append(metrics)

        # ── 🎯 SAVE BEST: Highest F1 ──
        if f1 > best_f1:
            best_f1    = f1
            no_improve = 0
            torch.save(model.state_dict(), f"{args.out}/{out_weight_name}")
            save_plots(cm, classes, args.out, args.model)
            np.savetxt(f"{args.out}/classification_{args.model}_confusion.txt", cm, fmt="%d")
            print(f"  ✅ Model saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  ⏭  No F1 improvement ({no_improve}/{patience}) — best={best_f1:.4f}")

        # ── Early stopping ──
        if no_improve >= patience:
            print(f"⏹  Early stopping at epoch {epoch+1}")
            break

        # ── Save metrics.json ──
        metrics_file = f"{args.out}/classification_metrics_{args.model}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_log, f, indent=4)

    print(f"\n🎯 Classification Training Complete ({args.model.upper()})")
    print(f"  Best F1  : {best_f1:.4f}")
    print(f"  Weights  : {args.out}/{out_weight_name}")


if __name__ == "__main__":
    main()