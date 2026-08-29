import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\kumar\Downloads\Breast-Cancer-Detection")
df = pd.read_csv(BASE / "vindr_labels.csv")

filename = "581b0e4d471a5c0adc6888cb038fa722"

row = df[df["image_id"] == filename]

if len(row) == 0:
    print("Image ID not found")
else:
    actual = row.iloc[0]["cancer"]
    print("Actual output:", "Malignant" if actual == 1 else "Benign")