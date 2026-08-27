import os
import pandas as pd
from pathlib import Path

# ✅ Change this to YOUR actual dataset path
DATASET_PATH = r"C:\Users\kumar\Downloads\Breast-Cancer-Detection"

images_dir = Path(DATASET_PATH) / "images_png"
csv_path = Path(DATASET_PATH) / "breast-level_annotations.csv"

# Count studies and images
studies = [f for f in images_dir.iterdir() if f.is_dir()]
images = list(images_dir.rglob("*.png"))

print(f"✅ Dataset path found: {DATASET_PATH}")
print(f"📁 Total study folders: {len(studies)}")
print(f"🖼️  Total PNG images   : {len(images)}")

# Load CSV
df = pd.read_csv(csv_path)
print(f"\n📊 CSV shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")
print(f"\n🎯 Cancer label distribution:")
print(df["cancer"].value_counts())
print(f"\n📌 Sample rows:")
print(df.head(3))