
"""Settings. Artifact location can be overridden with HEALTHGUARD_ARTIFACTS."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("HEALTHGUARD_ARTIFACTS", ROOT / "artifacts"))

MODEL_A_DIR = ARTIFACTS / "model_a" / "v1"
MODEL_A_VERSION = "model_a/v1"
MODEL_B_DIR = ARTIFACTS / "model_b" / "v1"
MODEL_B_VERSION = "model_b/v1"

CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]

# Model A: provisional cut-offs on the calibrated score.
BAND_THRESHOLDS = (0.35, 0.65)
# Model B: from observed rates on the holdout (see artifacts/model_b/v1/bands.csv).
BAND_THRESHOLDS_B = (0.05, 0.15)

DISCLAIMER = (
    "Educational project, not a medical device and not medical advice. "
    "The score compares you with patients in a few historical hospital "
    "datasets; it is not your absolute chance of heart disease."
)
DISCLAIMER_B = (
    "Educational project, not a medical device and not medical advice. "
    "This estimate compares you with survey respondents from 2015 who "
    "reported whether they had ever had heart disease or a heart attack. "
    "It does not predict your future risk."
)
