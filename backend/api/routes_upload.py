# backend/api/routes_upload.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4
import imghdr

from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter(prefix="/api", tags=["upload"])

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "uploads" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_FILES = 10


def make_job_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid4().hex[:8].upper()
    return f"JOB-{ts}-{short_id}"


def sanitize_suffix(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return ".jpg"
    return ext


@router.post("/upload")
async def upload_receipts(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} images allowed per upload",
        )

    job_id = make_job_id()
    job_dir = RAW_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for idx, file in enumerate(files, start=1):
        if not file.filename:
            raise HTTPException(status_code=400, detail=f"File #{idx} has no filename")

        ext = sanitize_suffix(file.filename)
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail=f"File #{idx} is empty")

        detected = imghdr.what(None, h=content)
        if detected is None:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a valid image",
            )

        save_name = f"RSCAN-{idx:06d}{ext}"
        save_path = job_dir / save_name
        save_path.write_bytes(content)

        saved_files.append(
            {
                "filename": save_name,
                "original_filename": file.filename,
                "image_path": str(save_path),
            }
        )

    return {
        "ok": True,
        "job_id": job_id,
        "raw_dir": str(job_dir),
        "count": len(saved_files),
        "files": saved_files,
    }