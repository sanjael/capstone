# models/advanced_models.py

import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

# ─────────────────────────────────────────────
# SIMPLE + STRONG HEAD (NO OVERFIT)
# ─────────────────────────────────────────────
class EfficientHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),  # 0
            nn.Linear(in_features, 512), # 1
            nn.ReLU(inplace=True),        # 2
            nn.Dropout(dropout),          # 3
            nn.Linear(512, num_classes)   # 4
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# DETECTION MODEL
# ─────────────────────────────────────────────
class EfficientDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = EfficientHead(in_features, 1, dropout=0.4)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)

# ─────────────────────────────────────────────
# CLASSIFICATION MODEL
# ─────────────────────────────────────────────
class EfficientClassificationModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = EfficientHead(in_features, num_classes, dropout=0.35)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x, temperature=1.5):
        logits = self.forward(x)
        return torch.softmax(logits / temperature, dim=1)

# ─────────────────────────────────────────────
# BUILD FUNCTIONS
# ─────────────────────────────────────────────
def build_efficient_detection(weights_path=None, device="cpu"):
    model = EfficientDetectionModel(pretrained=(weights_path is None))

    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        print("[INFO] Detection weights loaded from " + weights_path)

    model.to(device)
    model.eval()
    return model


def build_efficient_classification(weights_path=None, device="cpu"):
    model = EfficientClassificationModel(pretrained=(weights_path is None))

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
# 🔥 ENSEMBLE MODELS (PRODUCTION CRITICAL)
# =========================================================

class EnsembleDetectionModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x):
        p1 = self.model_a.predict_proba(x)
        p2 = self.model_b.predict_proba(x)
        return (p1 + p2) / 2.0

    def forward(self, x):
        return self.predict_proba(x)


class EnsembleClassificationModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x):
        p1 = self.model_a.predict_proba(x)
        p2 = self.model_b.predict_proba(x)
        return (p1 + p2) / 2.0

    def forward(self, x):
        return self.predict_proba(x)