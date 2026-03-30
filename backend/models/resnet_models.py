# models/resnet_models.py

import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights

TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

# ─────────────────────────────────────────────
# DETECTION MODEL
# ─────────────────────────────────────────────
class ResNetDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet101(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)

# ─────────────────────────────────────────────
# CLASSIFICATION MODEL
# ─────────────────────────────────────────────
class ResNetClassificationModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()

        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet101(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def predict_proba(self, x, temperature=1.5):
        logits = self.forward(x)
        return torch.softmax(logits / temperature, dim=1)

# ─────────────────────────────────────────────
# ENSEMBLE MODEL (VERY IMPORTANT)
# ─────────────────────────────────────────────
class EnsembleDetectionModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x):
        p1 = self.model_a.predict_proba(x)
        p2 = self.model_b.predict_proba(x)
        return (p1 + p2) / 2.0


class EnsembleClassificationModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x):
        p1 = self.model_a.predict_proba(x)
        p2 = self.model_b.predict_proba(x)
        return (p1 + p2) / 2.0


# ─────────────────────────────────────────────
# BUILD FUNCTIONS
# ─────────────────────────────────────────────
def build_resnet_detection(weights_path=None, device="cpu"):
    model = ResNetDetectionModel(pretrained=(weights_path is None))

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Detection weights not found: {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        print("[INFO] ResNet Detection weights loaded from " + weights_path)

    model.to(device)
    model.eval()
    return model


def build_resnet_classification(weights_path=None, device="cpu"):
    model = ResNetClassificationModel(pretrained=(weights_path is None))

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Classification weights not found: {weights_path}")
        
        state = torch.load(weights_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        
        if missing:
            print(f"[WARNING] Missing keys in classification model: {len(missing)}")
            print(f"  First 5: {missing[:5]}")
        if unexpected:
            print(f"[WARNING] Unexpected keys in classification model: {len(unexpected)}")
            print(f"  First 5: {unexpected[:5]}")
        if not missing and not unexpected:
            print("[INFO] Classification weights loaded successfully (all keys matched)")
        else:
            print("[INFO] Classification weights loaded with warnings")

    model.to(device)
    model.eval()
    return model

# =========================================================
# 🔥 BACKWARD COMPATIBILITY (FIX IMPORT ERRORS)
# =========================================================

TumorDetectionModel = ResNetDetectionModel
TumorClassificationModel = ResNetClassificationModel

def build_detection_model(weights_path=None, device="cpu"):
    return build_resnet_detection(weights_path, device)

def build_classification_model(weights_path=None, device="cpu"):
    return build_resnet_classification(weights_path, device)