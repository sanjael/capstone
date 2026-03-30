"""
generate_ensemble_cm.py — Test Set Evaluation for Dual-Ensemble Models
Evaluates the combined Ensemble detection and classification models on held-out test sets.
Generates metrics and beautiful Confusion Matrix PNGs.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
DETECTION_THRESHOLD = 0.30

print(f"Device: {DEVICE}")

def get_test_loader(data_dir, task):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(os.path.join(data_dir, task, "test"), tf)
    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=2), ds.classes


def evaluate_ensemble(model, loader, is_binary=False, classes=None, title="Confusion Matrix"):
    model.eval()
    y_true, y_pred = [], []
    
    print(f"Evaluating {title} on test set...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            probs = model.predict_proba(x)
            
            if is_binary:
                preds = (probs >= DETECTION_THRESHOLD).long().cpu().numpy()
            else:
                preds = torch.argmax(probs, dim=1).cpu().numpy()

            y_pred.extend(np.array(preds).flatten().tolist())
            y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    avg = "binary" if is_binary else "macro"
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    fn = int(cm[1, 0]) if is_binary and cm.shape == (2, 2) else None

    # Save logic
    out_img = f"models/{title.replace(' ', '_').lower()}.png"
    plt.figure(figsize=(8, 6))
    
    if not classes:
        classes = ["no_tumor", "tumor"] if is_binary else ["glioma", "meningioma", "pituitary"]
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes, yticklabels=classes)
    plt.title(f"{title}\nAccuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    plt.close()

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    if is_binary and fn is not None:
        print(f"  False Negatives: {fn}")
    print(f"✅ Generated {out_img}")
    
    return {"accuracy": acc, "f1": f1, "recall": rec, "precision": prec}


def main():
    os.makedirs("models", exist_ok=True)
    
    print("\n" + "="*50)
    print("  TEST SET EVALUATION — ENSEMBLE PIPELINE")
    print("="*50)

    # 1. ENSEMBLE DETECTION
    try:
        from models.advanced_models import build_efficient_detection
        from models.resnet_models import build_resnet_detection, EnsembleDetectionModel
        
        print("\nLoading Detection Ensemble...")
        det_eff = build_efficient_detection("models/detection_efficientnet.pth", DEVICE)
        det_res = build_resnet_detection("models/detection_resnet101.pth", DEVICE)
        det_ensemble = EnsembleDetectionModel(det_eff, det_res)
        det_ensemble.to(DEVICE)
        
        loader, classes = get_test_loader("dataset", "detection")
        evaluate_ensemble(det_ensemble, loader, is_binary=True, classes=classes, title="Ensemble Detection CM")
        
    except Exception as e:
        print(f"\n[Ensemble Detection] FAILED: {e}")


    # 2. ENSEMBLE CLASSIFICATION
    try:
        from models.resnet_models import build_resnet_classification, EnsembleClassificationModel
        from models.advanced_models import build_efficient_classification
        
        print("\nLoading Classification Ensemble...")
        cls_res = build_resnet_classification("models/classification_model.pth", DEVICE)
        cls_eff = build_efficient_classification("models/classification_efficientnet.pth", DEVICE)
        cls_ensemble = EnsembleClassificationModel(cls_res, cls_eff)
        cls_ensemble.to(DEVICE)
        
        loader, classes = get_test_loader("dataset", "classification")
        evaluate_ensemble(cls_ensemble, loader, is_binary=False, classes=classes, title="Ensemble Classification CM")
        
    except Exception as e:
        print(f"\n[Ensemble Classification] FAILED: {e}")

    print("\n" + "="*50)
    print("FINISHED.")

if __name__ == "__main__":
    main()
