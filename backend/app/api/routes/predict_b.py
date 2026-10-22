"""Lifestyle Check (Model B) routes."""
from fastapi import APIRouter, HTTPException

from app.core import config
from app.schemas.lifestyle import LifestyleInput, LifestyleResult
from app.services.model_b import get_model_b

router = APIRouter()


def load_or_503():
    try:
        return get_model_b()
    except (FileNotFoundError, RuntimeError) as err:
        detail = ("Model B artifacts not found. Run ml/scripts/train_model_b.py."
                  if isinstance(err, FileNotFoundError) else str(err))
        raise HTTPException(status_code=503, detail=detail)


@router.post("/predict/lifestyle", response_model=LifestyleResult)
def predict_lifestyle(payload: LifestyleInput):
    return load_or_503().predict(payload)


@router.get("/models/lifestyle/info")
def lifestyle_info():
    model = load_or_503()
    return {
        "model_version": config.MODEL_B_VERSION,
        "features": model.feature_names,
        "band_thresholds": {"lower_below": config.BAND_THRESHOLDS_B[0],
                            "higher_from": config.BAND_THRESHOLDS_B[1]},
        "metrics": model.metrics,
        "disclaimer": config.DISCLAIMER_B,
    }
