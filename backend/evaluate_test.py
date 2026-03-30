"""
evaluate_test.py — Test Set Evaluation for All 4 Trained Models
Evaluates detection and classification models on held-out test sets.
Saves results to models/test_evaluation_results.json
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

print(f"Device: {DEVICE}")


def get_test_loader(data_dir, task):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(os.path.join(data_dir, task, "test"), tf)
    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=2), ds.classes


def evaluate_model(model, loader, is_binary=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
            if is_binary:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(x)
                    preds = (probs >= 0.4).long().cpu().numpy()
                else:
                    preds = (torch.sigmoid(out).squeeze() >= 0.4).long().cpu().numpy()
            else:
                preds = torch.argmax(out, dim=1).cpu().numpy()

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

    # FN from confusion matrix (binary)
    fn = int(cm[1, 0]) if is_binary and cm.shape == (2, 2) else None

    return {
        "accuracy":          round(float(acc), 4),
        "precision":         round(float(prec), 4),
        "recall":            round(float(rec), 4),
        "f1":                round(float(f1), 4),
        "fn":                fn,
        "confusion_matrix":  cm.tolist(),
    }


def print_result(name, r, is_binary=False):
    print(f"\n[{name}]")
    if is_binary and r.get("fn") is not None:
        print(f"  FN       : {r['fn']}")
    print(f"  Accuracy : {r['accuracy']}")
    print(f"  Recall   : {r['recall']}")
    print(f"  Precision: {r['precision']}")
    print(f"  F1 Score : {r['f1']}")
    print(f"  Confusion Matrix: {r['confusion_matrix']}")


def main():
    results = {}
    print("\n" + "="*50)
    print("  TEST SET EVALUATION — ALL MODELS")
    print("="*50)

    # ── Detection — EfficientNet ──────────────────────────────
    try:
        from models.advanced_models import build_efficient_detection
        loader, classes = get_test_loader("dataset", "detection")
        m = build_efficient_detection("models/detection_efficientnet.pth", DEVICE)
        r = evaluate_model(m, loader, is_binary=True)
        results["detection_efficientnet"] = r
        print_result("Detection — EfficientNet", r, is_binary=True)
    except Exception as e:
        print(f"\n[Detection — EfficientNet] FAILED: {e}")
        results["detection_efficientnet"] = {"error": str(e)}

    # ── Detection — ResNet101 ─────────────────────────────────
    try:
        from models.resnet_models import build_resnet_detection
        loader, classes = get_test_loader("dataset", "detection")
        m = build_resnet_detection("models/detection_resnet101.pth", DEVICE)
        r = evaluate_model(m, loader, is_binary=True)
        results["detection_resnet101"] = r
        print_result("Detection — ResNet101", r, is_binary=True)
    except Exception as e:
        print(f"\n[Detection — ResNet101] FAILED: {e}")
        results["detection_resnet101"] = {"error": str(e)}

    # ── Classification — ResNet101 ────────────────────────────
    try:
        from models.resnet_models import build_resnet_classification
        loader, classes = get_test_loader("dataset", "classification")
        m = build_resnet_classification("models/classification_model.pth", DEVICE)
        r = evaluate_model(m, loader, is_binary=False)
        results["classification_resnet101"] = r
        print_result("Classification — ResNet101", r)
    except Exception as e:
        print(f"\n[Classification — ResNet101] FAILED: {e}")
        results["classification_resnet101"] = {"error": str(e)}

    # ── Classification — EfficientNet ─────────────────────────
    try:
        from models.advanced_models import build_efficient_classification
        loader, classes = get_test_loader("dataset", "classification")
        m = build_efficient_classification("models/classification_efficientnet.pth", DEVICE)
        r = evaluate_model(m, loader, is_binary=False)
        results["classification_efficientnet"] = r
        print_result("Classification — EfficientNet", r)
    except Exception as e:
        print(f"\n[Classification — EfficientNet] FAILED: {e}")
        results["classification_efficientnet"] = {"error": str(e)}

    # ── Save results ──────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    out_path = "models/test_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Test evaluation results saved to {out_path}")
    print("="*50)


if __name__ == "__main__":
    main()
