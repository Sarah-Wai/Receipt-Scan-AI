# backend/api/routes_process.py
# Purpose:
# - take uploaded images from raw/<job_id>/
# - run main receipt pipeline
# - save extraction results into SQLite
# - return the real integer receipt_id from the DB

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DB_PATH
from db.sqlite_reader import get_conn
from db.sqlite_writer import init_sqlite_schema, save_geo_fusion_to_sqlite
from pipelines.receipt_pipeline import (
    ReceiptPipeline,
    ReceiptPipelineConfig,
)

router = APIRouter(prefix="/api/receipts", tags=["process"])

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "uploads" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "uploads" / "processed"

YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "best.onnx"
CORD_MODEL_PATH = BASE_DIR / "models" / "cord_layoutlmv3"
SROIE_MODEL_PATH = BASE_DIR / "models" / "sroie_layoutlmv3"


class ProcessReceiptRequest(BaseModel):
    job_id: str


@router.post("/{receipt_id}/process")
def process_receipt(receipt_id: int, payload: ProcessReceiptRequest):
    try:
        job_id = payload.job_id.strip()
        print("Job ID:", job_id)

        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        raw_job_dir = RAW_DIR / job_id
        if not raw_job_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded job folder not found: {raw_job_dir}",
            )

        cfg = ReceiptPipelineConfig(
            base_dir=BASE_DIR,
            raw_upload_dir=RAW_DIR,
            processed_dir=PROCESSED_DIR,
            yolo_model_path=YOLO_MODEL_PATH,
            yolo_device="cpu",
            yolo_imgsz=1024,
            yolo_conf=0.25,
            yolo_iou=0.70,
            yolo_save_masks=False,
            close_kernel=(9, 9),
            close_iters=2,
            approx_eps_frac=0.02,
            min_output_side=2,
            ocr_device="cpu",
            cord_model_path=CORD_MODEL_PATH,
            sroie_model_path=SROIE_MODEL_PATH,
        )

        pipeline = ReceiptPipeline(cfg)
        result = pipeline.run(job_id=job_id)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error or "Receipt pipeline failed",
            )

        extraction_results = result.final_output.get("extraction_results", {})
        results_list = extraction_results.get("results", [])

        real_receipt_id: int | None = None
        saved_ids: list[int] = []

        if results_list:
            init_sqlite_schema(Path(DB_PATH))

            with sqlite3.connect(str(DB_PATH)) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")

                for idx, item in enumerate(results_list):
                    geo_result        = item.get("geo_result") or {}
                    validation_report = item.get("validation_report") or {}

                    # FIX: always build a job-scoped source_id regardless of
                    # what the pipeline set in geo_result["id"].
                    # geo_result["id"] is just the filename stem (e.g.
                    # "RSCAN-000001") which is identical across every upload,
                    # causing every new receipt to overwrite the previous one.
                    # Prefixing with job_id makes it globally unique.
                    raw_image_id = (
                        geo_result.get("id")
                        or geo_result.get("source_id")
                        or item.get("image_path", "")
                    )
                    image_stem = Path(str(raw_image_id)).stem if raw_image_id else f"img-{idx}"
                    geo_result["id"]        = f"{job_id}-{image_stem}"
                    geo_result["source_id"] = f"{job_id}-{image_stem}"
                    geo_result["job_id"]    = job_id

                    sroie = geo_result.get("sroie_fields") or {}
                    summary_report_dict = {
                        "fields": {
                            "vendor":   sroie.get("vendor",   ""),
                            "phone":    sroie.get("phone",    ""),
                            "address":  sroie.get("address",  ""),
                            "date":     sroie.get("date",     ""),
                            "subtotal": sroie.get("subtotal", ""),
                            "tax":      sroie.get("tax",      ""),
                            "total":    sroie.get("total",    ""),
                        }
                    }

                    db_receipt_id = save_geo_fusion_to_sqlite(
                        conn=conn,
                        out=geo_result,
                        validation_report=validation_report,
                        summary_report=summary_report_dict,
                        receipt_name_fallback=job_id,
                        default_currency="CAD",
                        min_confidence_filter=0.70,
                    )
                    saved_ids.append(db_receipt_id)

            real_receipt_id = saved_ids[0] if saved_ids else None

        # Fallback: pipeline may have already saved and surfaced receipt_id
        if real_receipt_id is None:
            real_receipt_id = result.final_output.get("receipt_id")

        # Last resort: look up by source_id (job_id-scoped)
        if real_receipt_id is None:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT receipt_id FROM receipts WHERE source_id LIKE ?",
                    (f"{job_id}%",),
                ).fetchone()
            if row:
                real_receipt_id = int(row[0])

        if real_receipt_id is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Pipeline succeeded but receipt was not saved to the "
                    "database. Check extraction step logs."
                ),
            )

        return {
            "ok":         True,
            "receipt_id": real_receipt_id,
            "job_id":     job_id,
            "result":     result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))