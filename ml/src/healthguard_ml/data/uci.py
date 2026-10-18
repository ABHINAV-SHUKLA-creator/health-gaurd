"""Load and clean the four UCI Heart Disease files."""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def load_site(raw_dir, site):
    path = Path(raw_dir) / SITES[site]
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["site"] = site
    return df


def load_uci(raw_dir):
    """Return all four hospitals as one cleaned DataFrame."""
    df = pd.concat([load_site(raw_dir, s) for s in SITES], ignore_index=True)
    # zeros in these columns are placeholders, not real measurements
    for col in ("chol", "trestbps"):
        df.loc[df[col] == 0, col] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])