"""Download the raw UCI Heart Disease files into data/raw/uci_heart/."""
import urllib.request
from pathlib import Path

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]
OUT = Path(__file__).resolve().parent / "raw" / "uci_heart"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT / name
        if dest.exists():
            print(f"skip   {name} (already there)")
            continue
        print(f"fetch  {name}")
        urllib.request.urlretrieve(BASE + name, dest)
    print("\nUCI done.")
    print("BRFSS (Model B) must be downloaded manually from Kaggle:")
    print("  see data/README.md -> put the CSV in data/raw/brfss/")


if __name__ == "__main__":
    main()