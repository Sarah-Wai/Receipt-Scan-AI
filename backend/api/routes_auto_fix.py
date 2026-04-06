from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import GROQ_TEXT_MODEL, GROQ_VISION_MODEL, GROQ_API_KEY

from db.sqlite_reader import get_conn
from db.sqlite_writer import (
    load_ocr_text,
    save_llm_result_to_sqlite,
    get_llm_run_with_items,
)
from llm.llm_router import run_auto_fix_for_receipt, LLMRouterConfig

router = APIRouter(prefix="/api/receipts", tags=["auto-fix"])

logger = logging.getLogger(__name__)  # FIX #9: use logging instead of print()


class AutoFixRequest(BaseModel):
    force_route: Optional[str] = None
    prompt_version: str = "v1"  # FIX #8: not actually Optional — never None
    use_active_image_path: bool = True


def _normalize_image_path_for_backend(full_path: Optional[str]) -> Optional[str]:
    if not full_path:
        return None
    p = Path(str(full_path)).expanduser()
    return str(p)


def _extract_image_path_from_receipt_row(receipt_row) -> Optional[str]:
    raw_json_str = receipt_row["raw_json"] if "raw_json" in receipt_row.keys() else None
    if not raw_json_str:
        return None
    try:
        raw = json.loads(raw_json_str)
    except Exception:
        return None
    image_path = raw.get("image_path") or raw.get("source_image_path")
    return _normalize_image_path_for_backend(image_path)


def _extract_ocr_text(receipt_row) -> str:
    source_id = receipt_row["source_id"] if "source_id" in receipt_row.keys() else None

    if source_id:
        # NOTE #3: load_ocr_text is imported from sqlite_writer — consider moving
        # it to sqlite_reader for cleaner separation of concerns.
        ocr_text = load_ocr_text(source_id)
        if ocr_text and ocr_text.strip():
            return ocr_text.strip()

    ocr_json_str = receipt_row["ocr_json"] if "ocr_json" in receipt_row.keys() else None
    if not ocr_json_str:
        return ""

    try:
        ocr_obj = json.loads(ocr_json_str)
    except Exception:
        return ""

    text = ocr_obj.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    rows = ocr_obj.get("rows") or []
    if isinstance(rows, list):
        lines = []
        for r in rows:
            if isinstance(r, dict):
                line = r.get("text") or r.get("line_text")
                if line:
                    lines.append(str(line).strip())
            elif isinstance(r, str):
                lines.append(r.strip())
        return "\n".join([x for x in lines if x])

    return ""


def _extract_ocr_confidence(receipt_row) -> Optional[float]:
    try:
        conf = receipt_row["confidence"] if "confidence" in receipt_row.keys() else None
        if conf is None:
            return None
        conf = float(conf)
        if conf > 1.0:
            conf = conf / 100.0
        return max(0.0, min(1.0, conf))
    except Exception:
        return None


def _determine_validation_status(validation: dict) -> str:
    """
    FIX #6: Three clear states instead of collapsing unknown into needs_review.

    - "valid"        → math checks passed
    - "needs_review" → math checks ran but failed
    - "unvalidated"  → not enough data to run checks (total/subtotal were None)
    """
    total_ok = validation.get("total_ok")
    subtotal_ok = validation.get("subtotal_ok")

    if total_ok is None and subtotal_ok is None:
        return "unvalidated"
    if total_ok is True and (subtotal_ok is True or subtotal_ok is None):
        return "valid"
    return "needs_review"


# The original /auto-fix path is kept for backward compatibility.
@router.post("/{receipt_id}/auto-fix")
@router.post("/{receipt_id}/autofix-llm")
def auto_fix_receipt(receipt_id: int, payload: AutoFixRequest):

    # FIX #1 + #2: conn declared before try so finally can safely reference it.
    conn = None
    try:
        conn = get_conn()  # FIX #1: was outside try — NameError risk in finally

        receipt_row = conn.execute(
            """
            SELECT
              receipt_id, source_id, receipt_name, vendor, receipt_date,
              subtotal, tax, total, status, confidence,
              raw_json, ocr_json, predictionlog_json,
              active_llm_run_id
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        if receipt_row is None:
            raise HTTPException(status_code=404, detail="Receipt not found")

        image_path = _extract_image_path_from_receipt_row(receipt_row)
        ocr_text = _extract_ocr_text(receipt_row)
        ocr_confidence = _extract_ocr_confidence(receipt_row)

        # FIX #9: replaced print() with logger
        logger.info("[autofix] receipt_id   : %s", receipt_id)
        logger.info("[autofix] source_id    : %s", receipt_row["source_id"])
        logger.info("[autofix] ocr_text len : %s", len(ocr_text))
        logger.info("[autofix] ocr_text snip: %r", ocr_text[:120])
        logger.info("[autofix] image_path   : %s", image_path)

        if not ocr_text and not image_path:
            raise HTTPException(
                status_code=400,
                detail="Neither OCR text nor image path is available for Auto Fix",
            )

        # cfg = LLMRouterConfig(
        # text_model="qwen2.5:3b",
        # vision_model="qwen2.5vl:7b",
        # timeout_sec=600,
        # )

        cfg = LLMRouterConfig(
            api_key=GROQ_API_KEY,
            text_model=GROQ_TEXT_MODEL,
            vision_model=GROQ_VISION_MODEL,
            timeout_sec=120,
        )

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            if payload.force_route == "ocr_text":
                if not ocr_text:
                    raise HTTPException(
                        status_code=400,
                        detail="force_route=ocr_text but no OCR text is available",
                    )
                llm_out = run_auto_fix_for_receipt(
                    ocr_text=ocr_text,
                    image_path=None,
                    ocr_confidence=ocr_confidence,
                    config=cfg,
                )

            elif payload.force_route == "vision_image":
                # FIX #4 + #10: guard against missing image before forcing vision
                if not image_path:
                    raise HTTPException(
                        status_code=400,
                        detail="force_route=vision_image but no image path is available",
                    )
                llm_out = run_auto_fix_for_receipt(
                    ocr_text=None,
                    image_path=image_path,
                    ocr_confidence=None,
                    config=cfg,
                )

            else:
                llm_out = run_auto_fix_for_receipt(
                    ocr_text=ocr_text,
                    image_path=image_path,
                    ocr_confidence=ocr_confidence,
                    config=cfg,
                )

        except HTTPException:
            raise  # re-raise our own 400s cleanly
        except Exception as llm_exc:
            traceback.print_exc()  # FIX #5: traceback imported at top
            raise HTTPException(status_code=500, detail=f"LLM call crashed: {llm_exc}")

        logger.info("[autofix] llm success  : %s", llm_out["success"])
        logger.info("[autofix] route_used   : %s", llm_out.get("route_used"))
        logger.info("[autofix] model_used   : %s", llm_out.get("model_used"))
        logger.info("[autofix] error        : %s", llm_out.get("error"))

        if not llm_out["success"]:
            raise HTTPException(
                status_code=500,
                detail=llm_out.get("error") or "Auto Fix failed",
            )

        validation = llm_out.get("validation") or {}
        validation_status = _determine_validation_status(validation)  # FIX #6

        # ── DB save ───────────────────────────────────────────────────────────
        try:
            llm_run_id = save_llm_result_to_sqlite(
                conn,
                receipt_id=receipt_id,
                llm_result=llm_out["parsed_output"],
                raw_llm_response=llm_out.get("raw_response_text"),
                route_used=llm_out["route_used"],
                llm_provider="ollama",
                llm_model=llm_out["model_used"],
                prompt_version=payload.prompt_version,
                source_image_path=image_path,
                raw_ocr_text=ocr_text,
                validation_status=validation_status,
                set_active=True,
            )
            conn.commit()  # FIX #7: commit inside the DB try so failures are caught

        except HTTPException:
            raise
        except Exception as save_exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"DB save crashed: {save_exc}")

        llm_run = get_llm_run_with_items(conn, llm_run_id)

        return {
            "ok": True,
            "receipt_id": receipt_id,
            "llm_run_id": llm_run_id,
            "route_used": llm_out["route_used"],
            "model_used": llm_out["model_used"],
            "validation": validation,
            "llm_run": llm_run,
        }

    finally:
        if conn:  # FIX #2: guard — conn is None if get_conn() raised
            conn.close()
