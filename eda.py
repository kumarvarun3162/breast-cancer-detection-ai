import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
from collections import Counter

BASE      = Path(r"C:\Users\kumar\Downloads\Breast-Cancer-Detection")
LABELS    = BASE / "vindr_labels.csv"
OUT_DIR   = BASE / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(LABELS)
print(f"Total images : {len(df)}")
print(f"Columns      : {list(df.columns)}\n")

# ── 1. Class distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("VinDr-Mammo — Dataset Overview", fontsize=14, fontweight="bold")

cancer_counts = df["cancer"].value_counts().sort_index()
axes[0].bar(["Benign (0)", "Malignant (1)"],
            cancer_counts.values,
            color=["#2a78d6", "#e34948"], width=0.5, edgecolor="white")
axes[0].set_title("Cancer label distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(cancer_counts.values):
    axes[0].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=11)

# ── 2. BI-RADS distribution ──────────────────────────────────────────
birads_counts = df["breast_birads"].value_counts().sort_index()
axes[1].bar(birads_counts.index, birads_counts.values,
            color="#7F77DD", edgecolor="white")
axes[1].set_title("BI-RADS distribution")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=30)
for i, (k, v) in enumerate(birads_counts.items()):
    axes[1].text(i, v + 50, str(v), ha="center", fontsize=9)

# ── 3. Breast density distribution ──────────────────────────────────
density_counts = df["breast_density"].value_counts().sort_index()
axes[2].bar(density_counts.index, density_counts.values,
            color="#1D9E75", edgecolor="white")
axes[2].set_title("Breast density distribution")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: 01_dataset_overview.png")

# ── 4. View position breakdown ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("View position & split breakdown", fontsize=13, fontweight="bold")

view_counts = df["view_position"].value_counts()
axes[0].pie(view_counts.values, labels=view_counts.index,
            autopct="%1.1f%%", colors=["#2a78d6", "#eb6834"],
            startangle=90)
axes[0].set_title("CC vs MLO views")

split_counts = df["split"].value_counts()
axes[1].bar(split_counts.index, split_counts.values,
            color=["#2a78d6", "#e34948"], width=0.4)
axes[1].set_title("Train / test split")
axes[1].set_ylabel("Count")
for i, v in enumerate(split_counts.values):
    axes[1].text(i, v + 50, str(v), ha="center")

plt.tight_layout()
plt.savefig(OUT_DIR / "02_view_split.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: 02_view_split.png")

# ── 5. Sample images — benign vs malignant ───────────────────────────
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Sample mammograms — benign (top) vs malignant (bottom)",
             fontsize=13, fontweight="bold")

benign_samples    = df[df["cancer"] == 0].sample(4, random_state=42)
malignant_samples = df[df["cancer"] == 1].sample(4, random_state=42)

for i, (_, row) in enumerate(benign_samples.iterrows()):
    ax = fig.add_subplot(2, 4, i + 1)
    img = cv2.imread(str(row["png_path"]), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Benign\n{row['view_position']} | {row['breast_birads']}",
                 fontsize=9)
    ax.axis("off")

for i, (_, row) in enumerate(malignant_samples.iterrows()):
    ax = fig.add_subplot(2, 4, i + 5)
    img = cv2.imread(str(row["png_path"]), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Malignant\n{row['view_position']} | {row['breast_birads']}",
                 fontsize=9, color="red")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "03_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: 03_sample_images.png")

# ── 6. Pixel intensity distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Pixel intensity distribution", fontsize=13, fontweight="bold")

for label, color, ax in [(0, "#2a78d6", axes[0]), (1, "#e34948", axes[1])]:
    subset = df[df["cancer"] == label].sample(min(30, len(df[df["cancer"]==label])),
                                               random_state=42)
    all_pixels = []
    for _, row in subset.iterrows():
        img = cv2.imread(str(row["png_path"]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            all_pixels.extend(img.flatten()[::100])  # sample every 100th pixel
    ax.hist(all_pixels, bins=50, color=color, alpha=0.8, edgecolor="white")
    title = "Benign" if label == 0 else "Malignant"
    ax.set_title(f"{title} pixel distribution")
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Frequency")
    mean_val = np.mean(all_pixels)
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:.0f}")
    ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "04_pixel_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: 04_pixel_distribution.png\n")

# ── 7. Summary stats ─────────────────────────────────────────────────
print("=" * 50)
print("PHASE 2 EDA SUMMARY")
print("=" * 50)
print(f"Total images      : {len(df):,}")
print(f"Benign (0)        : {(df.cancer==0).sum():,} ({(df.cancer==0).mean()*100:.1f}%)")
print(f"Malignant (1)     : {(df.cancer==1).sum():,} ({(df.cancer==1).mean()*100:.1f}%)")
print(f"Imbalance ratio   : {(df.cancer==0).sum() / (df.cancer==1).sum():.1f}:1")
print(f"\nView positions    : {dict(df.view_position.value_counts())}")
print(f"Train images      : {(df.split=='training').sum():,}")
print(f"Test images       : {(df.split=='test').sum():,}")
print(f"\nBI-RADS breakdown:")
print(df.groupby("breast_birads")["cancer"].agg(["count","sum"])
        .rename(columns={"count":"total","sum":"malignant"}))
print("\n✅ EDA complete — charts saved to eda_output/")