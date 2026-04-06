# api/routes_validation.py
from fastapi import APIRouter, HTTPException

from db.sqlite_reader import get_conn

router = APIRouter(prefix="/api/receipts", tags=["validation"])


@router.get("/{receipt_id}/validation")
def get_receipt_validation(receipt_id: int):
    conn = get_conn()

    try:
        row = conn.execute(
            """
            SELECT validation_id, receipt_id, subtotal_status,
                   subtotal_discrepancy, subtotal_discrepancy_pct,
                   outliers_count, name_quality_issues, price_range_warnings,
                   validation_json, created_at
            FROM receipt_validation
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="validation not found")

        return dict(row)

    finally:
        conn.close()