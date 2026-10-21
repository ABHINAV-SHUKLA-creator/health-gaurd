"""Clinical Check (Model A) routes."""
from fastapi import APIRouter, HTTPException

from app.core import config
from app.schemas.clinical import ClinicalInput, ClinicalResult
from app.services.model_a import FEATURES, get_model_a

router = APIRouter()


def load_or_503():
    try:
        return get_model_a()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model A artifacts not found. Run ml/scripts/train_model_a.py.",
        )


@router.post("/predict/clinical", response_model=ClinicalResult)
def predict_clinical(payload: ClinicalInput):
    return load_or_503().predict(payload)


@router.get("/models/clinical/info")
def clinical_info():
    model = load_or_503()
    return {
        "model_version": config.MODEL_A_VERSION,
        "features": FEATURES,
        "band_thresholds": {"lower_below": config.BAND_THRESHOLDS[0],
                            "higher_from": config.BAND_THRESHOLDS[1],
                            "provisional": True},
        "metrics": model.metrics,
        "disclaimer": config.DISCLAIMER,
    }
