import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE       = Path(__file__).parent.resolve()
MODEL_PATH = BASE / "models" / "best_model.pth"
CONFIG_PATH= BASE / "models" / "inference_config.json"
IMG_SIZE   = 224
DEVICE     = torch.device("cpu")  # backend runs on CPU

# ─────────────────────────────────────────────
# MODEL DEFINITION (must match train.py)
# ─────────────────────────────────────────────
class BreastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,     # no download needed
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logit    = self.classifier(features)
        return logit.squeeze(1)


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# ─────────────────────────────────────────────
def load_model():
    model = BreastCancerModel().to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model loaded  (val AUC = {ckpt['val_auc']:.4f})")
    return model

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    # fallback defaults from 3-epoch run
    return {"best_threshold": 0.014, "test_auc": 0.7625}


# ─────────────────────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    # Decode image bytes → numpy
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not decode image. Send PNG or JPG.")

    # Normalize (same as phase1.py)
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = (img * 255).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Grayscale → RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0

    # Apply transforms
    tensor = val_transforms(img)            # shape: (3, 224, 224)
    tensor = tensor.unsqueeze(0).to(DEVICE) # shape: (1, 3, 224, 224)
    return tensor


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(model, config, image_bytes: bytes) -> dict:
    tensor    = preprocess_image(image_bytes)
    threshold = config["best_threshold"]

    with torch.no_grad():
        logit       = model(tensor)
        probability = torch.sigmoid(logit).item()

    prediction = "Malignant" if probability >= threshold else "Benign"
    confidence = probability if prediction == "Malignant" else (1 - probability)

    # Confidence label for doctors
    if confidence >= 0.85:
        confidence_label = "High confidence"
    elif confidence >= 0.60:
        confidence_label = "Moderate confidence"
    else:
        confidence_label = "Low confidence — recommend manual review"

    return {
        "prediction"       : prediction,
        "probability_cancer": round(probability, 4),
        "probability_benign": round(1 - probability, 4),
        "confidence"       : round(confidence, 4),
        "confidence_label" : confidence_label,
        "threshold_used"   : round(threshold, 4),
        "model_auc"        : config.get("test_auc", "N/A"),
        "recommendation"   : (
            "⚠️  Refer for biopsy or further imaging."
            if prediction == "Malignant"
            else "✅ No malignancy detected. Routine follow-up recommended."
        )
    }