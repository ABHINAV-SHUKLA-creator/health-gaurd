"""HealthGuard API. Run from the backend folder: uvicorn app.main:app --reload"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import predict_a
from app.core import config

app = FastAPI(title="HealthGuard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict_a.router)


@app.get("/health")
def health():
    return {"status": "ok"}

