# 
# export OLLAMA_BASE_URL="https://XXXXXX-11434.usw3.devtunnels.ms/"
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
#

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config import FRONTEND_ORIGIN
from db.sqlite_reader import init_db                        # FIX: import init_db
from api.routes_receipts import router as receipts_router
from api.routes_validation import router as validations_router
from api.routes_upload import router as upload_router
from api.routes_process import router as process_router
from api.routes_auto_fix import router as auto_fix_router
from api.routes_llm import router as llm_router
from api.routes_approve import router as approve_router

app = FastAPI(title="FinScanAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")                                    # FIX: add startup hook
def on_startup() -> None:
    """
    Guarantee the full SQLite schema exists before the first
    request arrives. Safe to call on every restart — all DDL
    uses CREATE TABLE IF NOT EXISTS.
    """
    init_db()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

app.include_router(receipts_router)
app.include_router(validations_router)
app.include_router(upload_router)
app.include_router(process_router)
app.include_router(auto_fix_router)
app.include_router(llm_router)
app.include_router(approve_router)