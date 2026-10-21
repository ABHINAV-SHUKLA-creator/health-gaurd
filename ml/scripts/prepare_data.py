"""Clean raw data into data/processed/. Run from the project root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from healthguard_ml.data.brfss import load_brfss  # noqa: E402
from healthguard_ml.data.uci import load_uci  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def summarize(name, df):
    print(f"\n== {name}: {len(df)} rows, {df.shape[1]} columns")
    print("target balance:\n", df["target"].value_counts(normalize=True).round(3))
    miss = df.isna().mean().round(3)
    print("missing share (non-zero only):\n", miss[miss > 0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    uci = load_uci(RAW / "uci_heart")
    uci.to_csv(OUT / "uci_heart.csv", index=False)
    summarize("UCI (Model A)", uci)
    print("rows per site:\n", uci["site"].value_counts())
    print("disease rate per site:\n", uci.groupby("site")["target"].mean().round(3))

    try:
        brfss = load_brfss(RAW / "brfss")
    except FileNotFoundError as err:
        print(f"\nBRFSS skipped: {err}")
        return
    brfss.to_csv(OUT / "brfss_heart.csv", index=False)
    summarize("BRFSS (Model B)", brfss)


if __name__ == "__main__":
    main()
