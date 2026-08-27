import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG — all paths in one place
# ============================================================
BASE        = Path(r"C:\Users\kumar\Downloads\Breast-Cancer-Detection")
IMAGES_DIR  = BASE / "images_png"
CSV_IN      = BASE / "breast-level_annotations.csv"
OUTPUT_DIR  = BASE / "images_processed"
LABELS_OUT  = BASE / "vindr_labels.csv"
TARGET_SIZE = (512, 512)   # 512x512 is fine for CPU; saves space too
# ============================================================

def birads_to_cancer(birads_str):
    """
    BI-RADS 1,2,3 → benign (0)
    BI-RADS 4,5   → malignant (1)
    BI-RADS 0,6   → uncertain (-1) — we'll drop these
    """
    birads_str = str(birads_str).strip()
    if "4" in birads_str or "5" in birads_str:
        return 1
    elif "1" in birads_str or "2" in birads_str or "3" in birads_str:
        return 0
    else:
        return -1   # BI-RADS 0 or 6 — ambiguous, drop later


def preprocess_png(src_path, dst_path, target_size):
    """
    Read PNG → resize → CLAHE → save
    Already PNGs so no DICOM conversion needed.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # Normalize to 0-255 cleanly
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = (img * 255).astype(np.uint8)

    # CLAHE — enhances local contrast (critical for mammograms)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(str(dst_path), img)
    return True


def build_labels_and_process():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_IN)
    print(f"Loaded CSV: {df.shape[0]} rows")

    # Derive cancer label from BI-RADS
    df["cancer"] = df["breast_birads"].apply(birads_to_cancer)

    # Drop ambiguous BI-RADS 0 / 6
    dropped = (df["cancer"] == -1).sum()
    df = df[df["cancer"] != -1].reset_index(drop=True)
    print(f"Dropped {dropped} ambiguous BI-RADS 0/6 rows")
    print(f"Remaining: {len(df)} images\n")

    # Show class distribution
    print("Cancer label distribution:")
    print(df["cancer"].value_counts())
    print(f"\nPositive rate: {df['cancer'].mean()*100:.1f}%\n")

    records = []
    failed  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        study_id = row["study_id"]
        image_id = row["image_id"]

        src = IMAGES_DIR / study_id / f"{image_id}.png"
        if not src.exists():
            failed.append(str(src))
            continue

        # Mirror folder structure in output
        out_dir = OUTPUT_DIR / study_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{image_id}.png"

        ok = preprocess_png(src, dst, TARGET_SIZE)
        if not ok:
            failed.append(str(src))
            continue

        records.append({
            "png_path"      : str(dst),
            "study_id"      : study_id,
            "image_id"      : image_id,
            "laterality"    : row["laterality"],
            "view_position" : row["view_position"],
            "breast_birads" : row["breast_birads"],
            "breast_density": row["breast_density"],
            "split"         : row["split"],
            "cancer"        : int(row["cancer"])
        })

    # Save labels CSV
    labels_df = pd.DataFrame(records)
    labels_df.to_csv(LABELS_OUT, index=False)

    print(f"\n✅ Done!")
    print(f"   Processed : {len(records)} images")
    print(f"   Failed    : {len(failed)} images")
    print(f"   Labels CSV: {LABELS_OUT}")

    if failed:
        pd.DataFrame(failed, columns=["path"]).to_csv(
            BASE / "failed_images.csv", index=False
        )
        print(f"   Failed log: {BASE / 'failed_images.csv'}")

    # Final distribution in output
    print(f"\nFinal label distribution in output CSV:")
    print(labels_df["cancer"].value_counts())
    print(f"\nSample rows:")
    print(labels_df.head(3).to_string())


if __name__ == "__main__":
    build_labels_and_process()