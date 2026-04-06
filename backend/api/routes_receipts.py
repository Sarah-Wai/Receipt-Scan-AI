from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from db.sqlite_reader import get_conn
from db.sqlite_writer import (
    approve_llm_run,
    reject_llm_run,
    get_llm_run_with_items,
)
from pathlib import Path
from schemas.receipt_models import ReceiptStatusUpdate

router = APIRouter(prefix="/api/receipts", tags=["receipts"])


AI_REVIEW_STATUSES = {"not_requested", "pending", "approved", "rejected"}
ALLOWED_RECEIPT_STATUSES = {"Pending", "Processed", "Failed", "Error"}


# =========================================================
# Request models
# =========================================================

class ReceiptEditItem(BaseModel):
    line_no: int
    item_name: str
    currency: str = "CAD"
    unit_price: float = 0.0
    confidence: float = 0.0


class ReceiptApprovePayload(BaseModel):
    vendor: Optional[str] = None
    receipt_date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    summary_vendor: Optional[str] = None
    summary_phone: Optional[str] = None
    summary_address: Optional[str] = None
    summary_receipt_date: Optional[str] = None

    items: List[ReceiptEditItem] = Field(default_factory=list)


# =========================================================
# Helpers
# =========================================================

@contextmanager
def _conn() -> Generator:
    conn = get_conn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _loads_or_none(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    

def _normalize_image_path(full_path: Optional[str]) -> Optional[str]:
    if not full_path:
        return None

    full_path = str(full_path).replace("\\", "/").strip()

    # Best case: keep full relative path from "processed/"
    key = "processed/"
    idx = full_path.lower().find(key)
    if idx != -1:
        return full_path[idx:]

    # Next best: keep from "best_images/"
    key = "best_images/"
    idx = full_path.lower().rfind(key)
    if idx != -1:
        return full_path[idx:]

    # Fallback: filename only
    return Path(full_path).name if full_path else None


def _build_image_fields(receipt: Dict[str, Any], raw_json: Optional[Dict[str, Any]]) -> None:
    job_id = (receipt.get("job_id") or "").strip()

    # IMPORTANT:
    # source_id is DB unique key, not filesystem stem
    # receipt_name is the correct display/file stem like RSCAN-000001
    stem = (
        (receipt.get("receipt_name") or "").strip()
        or (receipt.get("source_id") or "").strip()
    )

    # Clean possible suffixes if any path/file value slipped in
    if stem:
        stem = Path(stem).stem.strip()
        for suffix in ("_best_view", "_best", "_view"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

    if job_id and stem:
        receipt["image_url"] = (
            f"/uploads/processed/{job_id}/ocr/{stem}/best_images/{stem}_best_view.png"
        )
    else:
        receipt["image_url"] = None

    raw_path = (
        (raw_json or {}).get("source_image_path")
        or (raw_json or {}).get("image_path")
        or (raw_json or {}).get("ocr_json_path")
        or receipt.get("source_image_path")
        or receipt.get("image_path")
    )

    receipt["image_path"] = _normalize_image_path(raw_path)


def _safe_currency(cur: Optional[str]) -> str:
    c = (cur or "CAD").strip().upper()[:3]
    return c if len(c) == 3 else "CAD"


def _save_ui_edits_to_main_tables(
    conn,
    *,
    receipt_id: int,
    payload: ReceiptApprovePayload,
    ai_review_status: str,
    active_llm_run_id: Optional[int],
) -> None:
    """
    Save the currently displayed UI values into main tables.

    Rule:
    - Current UI values are the source of truth on approve
    - If ai_review_status == pending, mark it approved after saving
    """

    conn.execute(
        """
        UPDATE receipts
        SET
            vendor = ?,
            receipt_date = ?,
            subtotal = ?,
            tax = ?,
            total = ?,
            extraction_source = CASE
                WHEN ? = 'pending' THEN 'ocr+llm'
                ELSE extraction_source
            END,
            ai_review_status = CASE
                WHEN ? = 'pending' THEN 'approved'
                ELSE ai_review_status
            END,
            status = 'Processed',
            updated_at = datetime('now')
        WHERE receipt_id = ?
        """,
        (
            payload.vendor,
            payload.receipt_date,
            payload.subtotal,
            payload.tax,
            payload.total,
            ai_review_status,
            ai_review_status,
            receipt_id,
        ),
    )

    conn.execute("DELETE FROM receipt_items WHERE receipt_id = ?", (receipt_id,))

    for item in payload.items:
        currency = _safe_currency(item.currency)
        conn.execute(
            """
            INSERT INTO receipt_items (
                receipt_id,
                line_no,
                item_name,
                currency,
                unit_price,
                confidence,
                name_conf,
                price_conf,
                is_outlier,
                item_status,
                status_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'OK', ?)
            """,
            (
                receipt_id,
                item.line_no,
                item.item_name.strip() if item.item_name else "",
                currency,
                item.unit_price,
                item.confidence,
                0.0,
                item.confidence,
                "approved_from_ui_edit",
            ),
        )

    summary_json = {
        "edited": True,
        "vendor": payload.summary_vendor,
        "phone": payload.summary_phone,
        "address": payload.summary_address,
        "receipt_date": payload.summary_receipt_date,
        "subtotal": payload.subtotal,
        "tax": payload.tax,
        "total": payload.total,
    }

    conn.execute(
        """
        INSERT INTO receipt_summary (
            receipt_id,
            subtotal,
            tax,
            total,
            vendor,
            phone,
            address,
            receipt_date,
            summary_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(receipt_id) DO UPDATE SET
            subtotal = excluded.subtotal,
            tax = excluded.tax,
            total = excluded.total,
            vendor = excluded.vendor,
            phone = excluded.phone,
            address = excluded.address,
            receipt_date = excluded.receipt_date,
            summary_json = excluded.summary_json
        """,
        (
            receipt_id,
            payload.subtotal,
            payload.tax,
            payload.total,
            payload.summary_vendor,
            payload.summary_phone,
            payload.summary_address,
            payload.summary_receipt_date,
            json.dumps(summary_json, ensure_ascii=False),
        ),
    )

    if ai_review_status == "pending" and active_llm_run_id:
        conn.execute(
            """
            UPDATE receipt_llm_runs
            SET
                approval_status = 'approved',
                approved_by = ?,
                approved_at = datetime('now'),
                updated_at = datetime('now')
            WHERE llm_run_id = ?
            """,
            ("ui_user", active_llm_run_id),
        )


# =========================================================
# Routes
# =========================================================

@router.get("")
def list_receipts() -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT
                receipt_id,
                job_id,
                source_id,
                receipt_name,
                vendor,
                receipt_date,
                subtotal,
                tax,
                total,
                status,
                confidence,
                extraction_source,
                ai_review_status,
                active_llm_run_id,
                ocr_json,
                predictionlog_json
            FROM receipts
            ORDER BY receipt_id DESC
            LIMIT 200
            """
        ).fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        row["ocr_json"] = _loads_or_none(row.get("ocr_json"))
        row["predictionlog_json"] = _loads_or_none(row.get("predictionlog_json"))

        ai_review_status = (row.get("ai_review_status") or "").strip().lower()
        row["data_source"] = "llm" if (
            row.get("active_llm_run_id") and ai_review_status == "pending"
        ) else "main"

        results.append(row)

    return results


@router.get("/{receipt_id}")
def receipt_detail(
    receipt_id: int,
    include_validation: bool = Query(False),
) -> Dict[str, Any]:
    with _conn() as conn:
        receipt_row = conn.execute(
            """
            SELECT
                receipt_id,
                job_id,
                source_id,
                receipt_name,
                vendor,
                receipt_date,
                subtotal,
                tax,
                total,
                status,
                confidence,
                extraction_source,
                ai_review_status,
                active_llm_run_id,
                raw_json,
                ocr_json,
                predictionlog_json,
                created_at,
                updated_at
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        if receipt_row is None:
            raise HTTPException(status_code=404, detail="Receipt not found")

        receipt = dict(receipt_row)
        ai_review_status = (receipt.get("ai_review_status") or "").strip().lower()
        active_llm_run_id = receipt.get("active_llm_run_id")

        if ai_review_status not in AI_REVIEW_STATUSES:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected ai_review_status in DB: {ai_review_status}",
            )

        use_llm_staging = bool(active_llm_run_id) and ai_review_status == "pending"

        items_rows = conn.execute(
            """
            SELECT
                item_id,
                receipt_id,
                line_no,
                item_name,
                currency,
                unit_price,
                confidence,
                name_conf,
                price_conf,
                is_outlier,
                item_status,
                status_reason
            FROM receipt_items
            WHERE receipt_id = ?
            ORDER BY line_no ASC, item_id ASC
            """,
            (receipt_id,),
        ).fetchall()

        summary_row = conn.execute(
            """
            SELECT
                summary_id,
                receipt_id,
                subtotal,
                tax,
                total,
                vendor,
                phone,
                address,
                receipt_date,
                summary_json
            FROM receipt_summary
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        llm_run_obj: Optional[Dict[str, Any]] = None
        validation_obj: Optional[Dict[str, Any]] = None

        if use_llm_staging:
            llm_run_obj = get_llm_run_with_items(conn, int(active_llm_run_id))
            if llm_run_obj:
                receipt["vendor"] = llm_run_obj.get("vendor")
                receipt["receipt_date"] = llm_run_obj.get("receipt_date")
                receipt["subtotal"] = llm_run_obj.get("subtotal")
                receipt["tax"] = llm_run_obj.get("tax")
                receipt["total"] = llm_run_obj.get("total")
                receipt["confidence"] = llm_run_obj.get("confidence")
                receipt["extraction_source"] = "ocr+llm"

                items_rows = []
                for x in llm_run_obj.get("items", []):
                    items_rows.append(
                        {
                            "item_id": x.get("llm_item_id"),
                            "receipt_id": receipt_id,
                            "line_no": x.get("line_no"),
                            "item_name": x.get("item_name"),
                            "currency": x.get("currency"),
                            "unit_price": x.get("line_price"),
                            "confidence": x.get("item_confidence"),
                            "name_conf": None,
                            "price_conf": None,
                            "is_outlier": 0,
                            "item_status": "OK",
                            "status_reason": f"llm_staging_run:{active_llm_run_id}",
                            "qty": x.get("qty"),
                            "line_price": x.get("line_price"),
                        }
                    )

                summary_row = {
                    "summary_id": None,
                    "receipt_id": receipt_id,
                    "subtotal": llm_run_obj.get("subtotal"),
                    "tax": llm_run_obj.get("tax"),
                    "total": llm_run_obj.get("total"),
                    "vendor": llm_run_obj.get("vendor"),
                    "phone": llm_run_obj.get("phone"),
                    "address": llm_run_obj.get("address"),
                    "receipt_date": llm_run_obj.get("receipt_date"),
                    "summary_json": llm_run_obj.get("parsed_json"),
                }

        if include_validation and not use_llm_staging:
            v = conn.execute(
                """
                SELECT
                    validation_id,
                    receipt_id,
                    subtotal_status,
                    subtotal_discrepancy,
                    subtotal_discrepancy_pct,
                    outliers_count,
                    name_quality_issues,
                    price_range_warnings,
                    validation_json,
                    created_at
                FROM receipt_validation
                WHERE receipt_id = ?
                """,
                (receipt_id,),
            ).fetchone()
            if v:
                validation_obj = dict(v)
                validation_obj["validation_json"] = _loads_or_none(
                    validation_obj.get("validation_json")
                )

    raw_json = _loads_or_none(receipt.pop("raw_json", None))
    receipt["ocr_json"] = _loads_or_none(receipt.pop("ocr_json", None))
    receipt["predictionlog_json"] = _loads_or_none(
        receipt.pop("predictionlog_json", None)
    )

    _build_image_fields(receipt, raw_json)

    summary_obj: Optional[Dict[str, Any]] = None
    if summary_row:
        summary_obj = dict(summary_row) if not isinstance(summary_row, dict) else summary_row
        summary_obj["summary_json"] = _loads_or_none(summary_obj.get("summary_json"))

    return {
        "receipt": receipt,
        "items": [dict(i) if not isinstance(i, dict) else i for i in items_rows],
        "validation": validation_obj,
        "summary": summary_obj,
        "llm_run": llm_run_obj,
        "data_source": "llm_staging" if use_llm_staging else "main",
        "validation_skipped": bool(use_llm_staging),
    }


@router.put("/{receipt_id}/approve")
def approve_receipt(
    receipt_id: int,
    payload: ReceiptApprovePayload,
) -> Dict[str, Any]:
    """
    Combined approve action:
    - saves current edited UI values into main tables
    - if ai_review_status == pending, marks LLM run approved
    - sets status = Processed
    """
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT
                receipt_id,
                source_id,
                status,
                ai_review_status,
                active_llm_run_id
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Receipt not found")

        db_row = dict(row)
        status = db_row.get("status")
        ai_review_status = (db_row.get("ai_review_status") or "not_requested").strip().lower()
        active_llm_run_id = db_row.get("active_llm_run_id")

        if status == "Processed":
            raise HTTPException(
                status_code=400,
                detail="Processed receipt is read-only",
            )

        if ai_review_status not in AI_REVIEW_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ai_review_status in DB: {ai_review_status}",
            )

        _save_ui_edits_to_main_tables(
            conn,
            receipt_id=receipt_id,
            payload=payload,
            ai_review_status=ai_review_status,
            active_llm_run_id=active_llm_run_id,
        )

        conn.commit()

        updated = conn.execute(
            """
            SELECT
                receipt_id,
                job_id,
                source_id,
                receipt_name,
                vendor,
                receipt_date,
                subtotal,
                tax,
                total,
                status,
                confidence,
                extraction_source,
                ai_review_status,
                active_llm_run_id
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        return {
            "ok": True,
            "receipt": dict(updated),
        }


@router.put("/{receipt_id}/status")
def update_receipt_status(
    receipt_id: int,
    payload: ReceiptStatusUpdate,
) -> Dict[str, Any]:
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT
                receipt_id,
                source_id,
                ai_review_status,
                active_llm_run_id,
                status
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Receipt not found")

        db_row = dict(row)
        ai_review_status = (db_row.get("ai_review_status") or "not_requested").strip().lower()
        current_status = db_row.get("status")

        if ai_review_status not in AI_REVIEW_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ai_review_status in DB: {ai_review_status}",
            )

        if payload.status not in ALLOWED_RECEIPT_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {payload.status}",
            )

        if current_status == "Processed" and payload.status != "Processed":
            raise HTTPException(
                status_code=400,
                detail="Processed receipt is read-only",
            )

        conn.execute(
            """
            UPDATE receipts
            SET status = ?,
                updated_at = datetime('now')
            WHERE receipt_id = ?
            """,
            (payload.status, receipt_id),
        )

        conn.commit()

        updated = conn.execute(
            """
            SELECT
                receipt_id,
                job_id,
                source_id,
                receipt_name,
                vendor,
                receipt_date,
                subtotal,
                tax,
                total,
                status,
                confidence,
                extraction_source,
                ai_review_status,
                active_llm_run_id
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        return {
            "ok": True,
            "receipt": dict(updated),
        }


@router.post("/{receipt_id}/reject-llm")
def reject_llm_receipt(receipt_id: int) -> Dict[str, Any]:
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT
                receipt_id,
                source_id,
                active_llm_run_id,
                ai_review_status,
                status
            FROM receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Receipt not found")

        db_row = dict(row)
        active_llm_run_id = db_row.get("active_llm_run_id")
        ai_review_status = (db_row.get("ai_review_status") or "").strip().lower()
        status = db_row.get("status")

        if status == "Processed":
            raise HTTPException(
                status_code=400,
                detail="Processed receipt is read-only",
            )

        if ai_review_status != "pending":
            raise HTTPException(
                status_code=400,
                detail="Only pending LLM results can be rejected",
            )

        if not active_llm_run_id:
            raise HTTPException(
                status_code=400,
                detail="No active_llm_run_id found for pending LLM receipt",
            )

        try:
            reject_llm_run(
                conn,
                llm_run_id=int(active_llm_run_id),
                rejected_by="ui_user",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        conn.commit()

        return {
            "ok": True,
            "receipt_id": receipt_id,
            "ai_review_status": "rejected",
        }