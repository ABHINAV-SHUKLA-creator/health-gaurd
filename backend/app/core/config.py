"""Settings. Artifact location can be overridden with HEALTHGUARD_ARTIFACTS."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("HEALTHGUARD_ARTIFACTS", ROOT / "artifacts"))
MODEL_A_DIR = ARTIFACTS / "model_a" / "v1"
MODEL_A_VERSION = "model_a/v1"

CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]

# Provisional cut-offs on the calibrated score. To be derived from data later.
BAND_THRESHOLDS = (0.35, 0.65)

DISCLAIMER = (
    "Educational project, not a medical device and not medical advice. "
    "The score compares you with patients in a few historical hospital "
    "datasets; it is not your absolute chance of heart disease."
)
