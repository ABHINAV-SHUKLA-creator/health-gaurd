"""HealthGuard API + website. Run from the backend folder:
uvicorn app.main:app --reload   ->  http://127.0.0.1:8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import predict_a, predict_b, site
from app.core import config

app = FastAPI(title="HealthGuard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict_a.router)
app.include_router(predict_b.router)
app.include_router(site.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# The website is served by the same app, so no cross-origin requests are needed.
# This mount must stay last: it catches every path the API routes did not.
FRONTEND_DIR = config.ROOT / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="site")
