# backend/api/routes_dashboard.py
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List

from fastapi import APIRouter

from db.sqlite_reader import get_conn

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


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


def _fmt_month(ym: str) -> str:
    """'2026-01'  →  'Jan'"""
    try:
        return datetime.strptime(ym, "%Y-%m").strftime("%b")
    except Exception:
        return ym


# =========================================================
# GET /api/dashboard/summary
# =========================================================

@router.get("/summary")
def get_summary() -> Dict[str, Any]:
    """
    Aggregate counts and totals from receipts, receipt_items,
    receipt_validation tables.

    Returns:
        total_receipts      — COUNT(*) from receipts
        total_spend         — SUM(total) from receipts
        avg_confidence      — AVG(confidence) from receipts
        vendor_count        — COUNT(DISTINCT vendor)
        pending_review      — receipts WHERE ai_review_status = 'pending'
        flagged_items       — receipt_items WHERE item_status IN (OUTLIER, FLAGGED, LOW_CONF, ERROR)
        new_this_week       — receipts created in last 7 days
        ocr_only_count      — receipts WHERE extraction_source = 'ocr'
        ocr_llm_count       — receipts WHERE extraction_source = 'ocr+llm'
        approved_count      — receipts WHERE ai_review_status = 'approved'
        rejected_count      — receipts WHERE ai_review_status = 'rejected'
        failed_count        — receipts WHERE status IN ('Failed', 'Error')
    """
    week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    with _conn() as conn:

        totals_row = conn.execute(
            """
            SELECT
                COUNT(*)                            AS total_receipts,
                COALESCE(SUM(total),      0.0)      AS total_spend,
                COALESCE(AVG(confidence), 0.0)      AS avg_confidence,
                COUNT(DISTINCT vendor)              AS vendor_count,

                SUM(CASE WHEN ai_review_status = 'pending'
                         THEN 1 ELSE 0 END)         AS pending_review,
                SUM(CASE WHEN ai_review_status = 'approved'
                         THEN 1 ELSE 0 END)         AS approved_count,
                SUM(CASE WHEN ai_review_status = 'rejected'
                         THEN 1 ELSE 0 END)         AS rejected_count,

                SUM(CASE WHEN extraction_source = 'ocr'
                         THEN 1 ELSE 0 END)         AS ocr_only_count,
                SUM(CASE WHEN extraction_source = 'ocr+llm'
                         THEN 1 ELSE 0 END)         AS ocr_llm_count,

                SUM(CASE WHEN status IN ('Failed', 'Error')
                         THEN 1 ELSE 0 END)         AS failed_count,

                SUM(CASE WHEN date(created_at) >= ?
                         THEN 1 ELSE 0 END)         AS new_this_week
            FROM receipts
            """,
            (week_ago,),
        ).fetchone()

        flagged_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM receipt_items
            WHERE item_status IN ('OUTLIER', 'FLAGGED', 'LOW_CONF', 'ERROR')
            """
        ).fetchone()[0]

    r = dict(totals_row)

    return {
        "total_receipts":  r["total_receipts"]  or 0,
        "total_spend":     round(r["total_spend"] or 0.0, 2),
        "avg_confidence":  round(r["avg_confidence"] or 0.0, 1),
        "vendor_count":    r["vendor_count"]    or 0,
        "pending_review":  r["pending_review"]  or 0,
        "approved_count":  r["approved_count"]  or 0,
        "rejected_count":  r["rejected_count"]  or 0,
        "ocr_only_count":  r["ocr_only_count"]  or 0,
        "ocr_llm_count":   r["ocr_llm_count"]   or 0,
        "failed_count":    r["failed_count"]    or 0,
        "new_this_week":   r["new_this_week"]   or 0,
        "flagged_items":   flagged_count        or 0,
    }


# =========================================================
# GET /api/dashboard/validation-summary
# =========================================================

@router.get("/validation-summary")
def get_validation_summary() -> Dict[str, Any]:
    """
    Aggregate totals from receipt_validation and receipt_items.

    Returns:
        price_outliers       — SUM(outliers_count) from receipt_validation
        subtotal_mismatch    — COUNT receipts WHERE subtotal_status != 'ok'
        name_quality_issues  — SUM(name_quality_issues) from receipt_validation
        price_range_warnings — SUM(price_range_warnings) from receipt_validation
        low_confidence_items — COUNT receipt_items WHERE confidence < 60
        missing_vendor       — COUNT receipts WHERE vendor IS NULL OR vendor = ''
    """
    with _conn() as conn:

        val_row = conn.execute(
            """
            SELECT
                COALESCE(SUM(outliers_count),       0) AS price_outliers,
                COALESCE(SUM(
                    CASE WHEN subtotal_status IS NOT NULL
                              AND subtotal_status != 'ok'
                         THEN 1 ELSE 0 END
                ), 0)                                  AS subtotal_mismatch,
                COALESCE(SUM(name_quality_issues),  0) AS name_quality_issues,
                COALESCE(SUM(price_range_warnings), 0) AS price_range_warnings
            FROM receipt_validation
            """
        ).fetchone()

        low_conf = conn.execute(
            """
            SELECT COUNT(*)
            FROM receipt_items
            WHERE confidence < 60
            """
        ).fetchone()[0]

        missing_vendor = conn.execute(
            """
            SELECT COUNT(*)
            FROM receipts
            WHERE vendor IS NULL OR trim(vendor) = ''
            """
        ).fetchone()[0]

    v = dict(val_row)

    return {
        "price_outliers":       v["price_outliers"]       or 0,
        "subtotal_mismatch":    v["subtotal_mismatch"]    or 0,
        "name_quality_issues":  v["name_quality_issues"]  or 0,
        "price_range_warnings": v["price_range_warnings"] or 0,
        "low_confidence_items": low_conf                  or 0,
        "missing_vendor":       missing_vendor            or 0,
    }


# =========================================================
# GET /api/dashboard/monthly-trend
# =========================================================

@router.get("/monthly-trend")
def get_monthly_trend() -> List[Dict[str, Any]]:
    """
    Monthly spend and receipt count for the last 6 months.
    Groups by strftime('%Y-%m', receipt_date).
    Receipts with NULL receipt_date are excluded.

    Returns list of:
        month  — short month label e.g. 'Jan'
        spend  — SUM(total) for that month
        count  — COUNT(*) for that month
    """
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT
                strftime('%Y-%m', receipt_date)     AS ym,
                COALESCE(SUM(total), 0.0)           AS spend,
                COUNT(*)                            AS count
            FROM receipts
            WHERE receipt_date IS NOT NULL
              AND trim(receipt_date) != ''
              AND date(receipt_date) >= date('now', '-6 months')
            GROUP BY ym
            ORDER BY ym ASC
            """
        ).fetchall()

    return [
        {
            "month": _fmt_month(r["ym"]),
            "spend": round(r["spend"], 2),
            "count": r["count"],
        }
        for r in rows
    ]


# =========================================================
# GET /api/dashboard/confidence-distribution
# =========================================================

@router.get("/confidence-distribution")
def get_confidence_distribution() -> Dict[str, Any]:
    """
    Bucket receipts by confidence score into High / Medium / Low,
    split by extraction_source (ocr vs ocr+llm).

    Returns:
        ocr: [high, medium, low]   — counts for extraction_source = 'ocr'
        llm: [high, medium, low]   — counts for extraction_source = 'ocr+llm'
    """
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT
                extraction_source,
                SUM(CASE WHEN confidence >= 80 THEN 1 ELSE 0 END) AS high,
                SUM(CASE WHEN confidence >= 60
                          AND confidence <  80 THEN 1 ELSE 0 END) AS medium,
                SUM(CASE WHEN confidence <  60 THEN 1 ELSE 0 END) AS low
            FROM receipts
            WHERE extraction_source IN ('ocr', 'ocr+llm')
            GROUP BY extraction_source
            """
        ).fetchall()

    ocr = [0, 0, 0]
    llm = [0, 0, 0]

    for r in rows:
        bucket = [r["high"] or 0, r["medium"] or 0, r["low"] or 0]
        if r["extraction_source"] == "ocr":
            ocr = bucket
        elif r["extraction_source"] == "ocr+llm":
            llm = bucket

    return {"ocr": ocr, "llm": llm}


# =========================================================
# GET /api/dashboard/activity
# =========================================================

@router.get("/activity")
def get_activity() -> List[Dict[str, Any]]:
    """
    Last 15 receipt events ordered by updated_at DESC.
    Derives activity type from ai_review_status and extraction_source.

    Returns list of:
        type  — 'upload' | 'llm' | 'flag' | 'reject'
        icon  — single char label for the UI icon
        text  — HTML string with <strong> vendor/job label
    """
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT
                receipt_id,
                job_id,
                vendor,
                ai_review_status,
                extraction_source,
                status,
                created_at,
                updated_at
            FROM receipts
            ORDER BY updated_at DESC
            LIMIT 15
            """
        ).fetchall()

    feed: List[Dict[str, Any]] = []

    for r in rows:
        receipt_id      = r["receipt_id"]
        job_id          = r["job_id"] or ""
        vendor          = r["vendor"] or ""
        ai_status       = (r["ai_review_status"] or "").strip().lower()
        status          = (r["status"] or "").strip()

        label = vendor or job_id or f"Receipt #{receipt_id}"
        strong = f"<strong>{label}</strong>"

        if ai_status == "approved":
            feed.append({
                "type": "llm",
                "icon": "✓",
                "text": f"LLM run approved — {strong}",
            })
        elif ai_status == "rejected":
            feed.append({
                "type": "reject",
                "icon": "✕",
                "text": f"LLM run rejected — {strong}",
            })
        elif ai_status == "pending":
            feed.append({
                "type": "flag",
                "icon": "!",
                "text": f"Pending AI review — {strong}",
            })
        elif status in ("Failed", "Error"):
            feed.append({
                "type": "reject",
                "icon": "✕",
                "text": f"Processing {status.lower()} — {strong}",
            })
        else:
            feed.append({
                "type": "upload",
                "icon": "↑",
                "text": f"Uploaded — {strong}",
            })

    return feed