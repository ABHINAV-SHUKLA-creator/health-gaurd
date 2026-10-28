"""Data behind the Dataset Explorer and Model Lab tabs."""
import json

from fastapi import APIRouter, HTTPException

from app.core import config

router = APIRouter()


@router.get("/site-data")
def site_data():
    path = config.ARTIFACTS / "site_data.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Site data not built. Run ml/scripts/make_site_data.py.",
        )
    return json.loads(path.read_text())
