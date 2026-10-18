"""Load and clean the BRFSS heart-disease health indicators CSV."""
from pathlib import Path

import pandas as pd

TARGET_SOURCE = "HeartDiseaseorAttack"


def find_csv(raw_dir):
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV in {raw_dir}. Download it from Kaggle first "
            "(see data/README.md)."
        )
    return files[0]


def load_brfss(raw_dir):
    df = pd.read_csv(find_csv(raw_dir))
    if TARGET_SOURCE not in df.columns:
        raise ValueError(
            f"Expected column '{TARGET_SOURCE}', got: {list(df.columns)}"
        )
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={TARGET_SOURCE: "target"})
    df["target"] = df["target"].astype(int)
    return df