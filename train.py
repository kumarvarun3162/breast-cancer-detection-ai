import os
import random
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIG — change TEST_RUN to False for overnight training
# ─────────────────────────────────────────────────────────
BASE       = Path(r"C:\Users\kumar\Downloads\Breast-Cancer-Detection")
LABELS_CSV = BASE / "vindr_labels.csv"
MODEL_DIR  = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

TEST_RUN   = True    # ← True = 3 epochs (test), False = 15 epochs (full)
EPOCHS     = 3  if TEST_RUN else 15
BATCH_SIZE = 8  if TEST_RUN else 16
IMG_SIZE   = 224
LR         = 1e-4
SEED       = 42

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)
DEVICE = torch.device("cpu")

print("=" * 55)
print(f"  Breast Cancer Detection — {'TEST RUN' if TEST_RUN else 'FULL TRAINING'}")
print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Device: {DEVICE}")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class MammogramDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["png_path"]), cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["cancer"], dtype=torch.float32)
        return img, label

# ─────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────
def prepare_data():
    df       = pd.read_csv(LABELS_CSV)
    train_df = df[df["split"] == "training"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_df, test_size=0.1,
        random_state=SEED,
        stratify=train_df["cancer"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"\n── Data split ──────────────────────────────────")
    print(f"  Train : {len(train_df):>6,}  "
          f"(cancer: {train_df.cancer.sum():,} = {train_df.cancer.mean()*100:.1f}%)")
    print(f"  Val   : {len(val_df):>6,}  "
          f"(cancer: {val_df.cancer.sum():,} = {val_df.cancer.mean()*100:.1f}%)")
    print(f"  Test  : {len(test_df):>6,}  "
          f"(cancer: {test_df.cancer.sum():,} = {test_df.cancer.mean()*100:.1f}%)")
    return train_df, val_df, test_df

# ─────────────────────────────────────────────────────────
# WEIGHTED SAMPLER
# ─────────────────────────────────────────────────────────
def make_sampler(df):
    labels      = df["cancer"].values.astype(int)
    class_count = np.bincount(labels)
    weights     = 1.0 / class_count
    sample_wts  = weights[labels]
    return WeightedRandomSampler(
        weights     = sample_wts,
        num_samples = len(sample_wts),
        replacement = True
    )

# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────
class BreastCancerModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
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

# ─────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────
def get_loss(train_df):
    n_neg      = (train_df.cancer == 0).sum()
    n_pos      = (train_df.cancer == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"\n── Loss setup ──────────────────────────────────")
    print(f"  Benign   : {n_neg:,}")
    print(f"  Malignant: {n_pos:,}")
    print(f"  pos_weight = {pos_weight.item():.1f}x  "
          f"(missing a cancer costs {pos_weight.item():.1f}x more)")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ─────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []

    bar = tqdm(loader, desc=f"  Epoch {epoch} [train]", leave=True,
               bar_format="{l_bar}{bar:25}{r_bar}")

    for imgs, labels in bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc

# ─────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, epoch, split="val"):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    bar = tqdm(loader, desc=f"  Epoch {epoch} [{split}] ",
               leave=True,
               bar_format="{l_bar}{bar:25}{r_bar}")

    with torch.no_grad():
        for imgs, labels in bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc, np.array(all_labels), np.array(all_probs)

# ─────────────────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────
def plot_curves(history, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Training curves — {label}", fontsize=13, fontweight="bold")

    axes[0].plot(history["train_loss"], marker="o", label="Train", color="#2a78d6")
    axes[0].plot(history["val_loss"],   marker="o", label="Val",   color="#e34948")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_auc"], marker="o", label="Train AUC", color="#2a78d6")
    axes[1].plot(history["val_auc"],   marker="o", label="Val AUC",   color="#e34948")
    axes[1].set_title("AUC-ROC per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    name = "curves_test.png" if TEST_RUN else "curves_full.png"
    plt.savefig(MODEL_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: models/{name}")

# ─────────────────────────────────────────────────────────
# FINAL TEST EVALUATION
# ─────────────────────────────────────────────────────────
def final_evaluation(model, test_df, criterion):
    print(f"\n── Test set evaluation ─────────────────────────")
    test_ds     = MammogramDataset(test_df, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    test_loss, test_auc, all_labels, all_probs = evaluate(
        model, test_loader, criterion, epoch="final", split="test"
    )

    # Best threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_idx       = np.argmax(tpr - fpr)
    best_thresh = float(thresholds[j_idx])
    all_preds   = (all_probs >= best_thresh).astype(int)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS")
    print(f"{'='*55}")
    print(f"  AUC-ROC     : {test_auc:.4f}")
    print(f"  Threshold   : {best_thresh:.3f}  (Youden's J)")
    print(f"  Sensitivity : {sensitivity:.4f}  ← recall for cancer")
    print(f"  Specificity : {specificity:.4f}  ← recall for benign")
    print(f"\n  Confusion matrix:")
    print(f"              Pred Benign  Pred Cancer")
    print(f"  Act Benign   {tn:>8,}     {fp:>8,}")
    print(f"  Act Cancer   {fn:>8,}     {tp:>8,}")
    print(f"\n  Classification report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=["Benign", "Malignant"]))

    # ROC + confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test set results", fontsize=13, fontweight="bold")

    axes[0].plot(fpr, tpr, color="#2a78d6", lw=2,
                 label=f"AUC = {test_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].scatter(fpr[j_idx], tpr[j_idx], color="red", s=90,
                    zorder=5, label=f"Threshold = {best_thresh:.2f}")
    axes[0].set_title("ROC curve — test set")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Benign", "Malignant"])
    axes[1].set_yticklabels(["Benign", "Malignant"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         fontsize=13,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "test_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: models/test_results.png")

    return best_thresh, test_auc, sensitivity, specificity

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    train_df, val_df, test_df = prepare_data()

    train_ds = MammogramDataset(train_df, transform=train_transforms)
    val_ds   = MammogramDataset(val_df,   transform=val_transforms)

    sampler      = make_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n── Model ───────────────────────────────────────")
    model     = BreastCancerModel(pretrained=True).to(DEVICE)
    criterion = get_loss(train_df)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {total_params:,}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Batch size       : {BATCH_SIZE}")

    history   = {"train_loss": [], "val_loss": [],
                 "train_auc":  [], "val_auc":  []}
    best_auc  = 0.0
    best_path = MODEL_DIR / "best_model.pth"

    print(f"\n── Training ────────────────────────────────────\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'─'*55}")
        print(f"  EPOCH {epoch}/{EPOCHS}")
        print(f"{'─'*55}")

        tr_loss, tr_auc              = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch)
        vl_loss, vl_auc, _, _        = evaluate(
            model, val_loader, criterion, epoch, split="val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        print(f"\n  Loss  → train: {tr_loss:.4f}  |  val: {vl_loss:.4f}")
        print(f"  AUC   → train: {tr_auc:.4f}  |  val: {vl_auc:.4f}")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_auc"    : best_auc,
                "optimizer"  : optimizer.state_dict(),
                "threshold"  : 0.5
            }, best_path)
            print(f"  ✅ Best model saved  (val AUC = {best_auc:.4f})")

    print(f"\n── Training complete ───────────────────────────")
    plot_curves(history, label=f"{'Test run' if TEST_RUN else 'Full'} — {EPOCHS} epochs")

    # Load best model for evaluation
    print(f"\n── Loading best checkpoint ─────────────────────")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded epoch {ckpt['epoch']} (val AUC = {ckpt['val_auc']:.4f})")

    best_thresh, test_auc, sensitivity, specificity = final_evaluation(
        model, test_df, criterion
    )

    # Save inference config
    config = {
        "model_arch"   : "efficientnet_b0",
        "img_size"     : IMG_SIZE,
        "best_threshold": best_thresh,
        "val_auc"      : float(best_auc),
        "test_auc"     : float(test_auc),
        "sensitivity"  : float(sensitivity),
        "specificity"  : float(specificity),
        "epochs_trained": EPOCHS,
        "test_run"     : TEST_RUN
    }
    with open(MODEL_DIR / "inference_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Best val AUC  : {best_auc:.4f}")
    print(f"  Test AUC      : {test_auc:.4f}")
    print(f"  Sensitivity   : {sensitivity:.4f}")
    print(f"  Specificity   : {specificity:.4f}")
    print(f"  Threshold     : {best_thresh:.3f}")
    print(f"  Config saved  : models/inference_config.json")
    print(f"  Model saved   : models/best_model.pth")
    if TEST_RUN:
        print(f"\n  ✅ Test run done. Now set TEST_RUN = False")
        print(f"     and run again overnight for full training.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()