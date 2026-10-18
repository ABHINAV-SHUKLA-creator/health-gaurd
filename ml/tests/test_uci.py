import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from healthguard_ml.data.uci import SITES, load_uci  # noqa: E402

ROW = "63,1,1,145,233,1,2,150,0,2.3,3,0,6,{num}\n"
ROW_MISSING = "54,0,3,130,0,0,0,160,0,0.0,?,?,?,{num}\n"


def test_load_uci_cleans(tmp_path):
    for i, fname in enumerate(SITES.values()):
        (tmp_path / fname).write_text(
            ROW.format(num=i) + ROW_MISSING.format(num=0)
        )
    df = load_uci(tmp_path)
    assert len(df) == 8
    assert set(df["site"]) == set(SITES)
    assert df["chol"].isna().sum() == 4
    assert df["ca"].isna().sum() == 4
    assert set(df["target"]) == {0, 1}