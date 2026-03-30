# train_classification_efficient.py — EfficientNet-B4 Classification (GPU OPTIMIZED)

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.advanced_models import EfficientClassificationModel

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 8   # Safe for RTX 3050 Ti 4GB VRAM
USE_AMP    = torch.cuda.is_available()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
else:
    print("⚠️  No GPU — running on CPU")


def get_model():
    print("🚀 Using EfficientNet-B4 for classification")
    return EfficientClassificationModel(num_classes=3, pretrained=True)


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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin, persistent_workers=True,
                              drop_last=True)  # prevent batch size=1 for BatchNorm
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=2, pin_memory=pin, persistent_workers=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")
    print(f"Batch size: {BATCH_SIZE} | AMP: {USE_AMP} | pin_memory: {pin}")
    return train_loader, val_loader, train_ds.classes


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for x, y in loader:
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


def save_plots(cm, classes, out_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — EfficientNet Classification")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/classification_efficientnet_cm.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out",    default="models")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, classes = get_loaders(args.data)
    model     = get_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=USE_AMP)

    best_f1    = 0.0
    patience   = 5
    no_improve = 0
    metrics_log = []

    print(f"\n🚀 EfficientNet Classification | Device: {DEVICE} | AMP: {USE_AMP}\n")

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        acc, prec, rec, f1, cm, y_true, y_pred = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss     : {loss:.4f}")
        print(f"  F1       : {f1:.4f} ← PRIMARY selection metric")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Accuracy : {acc:.4f} ← REPORTED ONLY (not used for selection)")

        metrics_log.append({
            "epoch":     epoch + 1,
            "loss":      round(loss, 6),
            "f1":        round(f1, 6),
            "precision": round(prec, 6),
            "recall":    round(rec, 6),
            "accuracy":  round(acc, 6),
        })

        # 🎯 SAVE BEST: Highest F1 only
        if f1 > best_f1:
            best_f1    = f1
            no_improve = 0
            torch.save(model.state_dict(), f"{args.out}/classification_efficientnet.pth")
            save_plots(cm, classes, args.out)
            np.savetxt(f"{args.out}/classification_efficientnet_confusion.txt", cm, fmt="%d")
            print(f"  ✅ Saved best EfficientNet classification model (F1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  ⏭  No improvement ({no_improve}/{patience}) — best F1={best_f1:.4f}")

        if no_improve >= patience:
            print(f"⏹  Early stopping at epoch {epoch+1}")
            break

        with open(f"{args.out}/classification_efficientnet_metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=4)

    print(f"\n🎯 EfficientNet Classification Training Complete")
    print(f"  Best F1  : {best_f1:.4f}")
    print(f"  Weights  : {args.out}/classification_efficientnet.pth")


if __name__ == "__main__":
    main()
