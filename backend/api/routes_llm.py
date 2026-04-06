# backend/api/routes_llm.py
# Get current LLM staging result for review UI

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from db.sqlite_reader import get_conn
from db.sqlite_writer import get_receipt_for_ui, get_llm_run_with_items

router = APIRouter(prefix="/api/receipts", tags=["llm"])


@router.get("/{receipt_id}/llm")
def get_receipt_llm_view(receipt_id: int):
    """
    Main UI endpoint for review page.

    Behavior:
    - If receipt has active_llm_run_id, return LLM staging data
    - Otherwise return original receipts + receipt_items data
    """
    conn = get_conn()
    try:
        result = get_receipt_for_ui(conn, receipt_id)
        return {
            "ok": True,
            **result,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load receipt UI data: {e}")
    finally:
        conn.close()


@router.get("/llm/runs/{llm_run_id}")
def get_single_llm_run(llm_run_id: int):
    """
    Fetch one specific LLM run with items.
    Useful if UI wants to reopen a specific staging result.
    """
    conn = get_conn()
    try:
        llm_run = get_llm_run_with_items(conn, llm_run_id)
        if llm_run is None:
            raise HTTPException(status_code=404, detail="LLM run not found")

        return {
            "ok": True,
            "source": "llm_staging",
            "llm_run": llm_run,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM run: {e}")
    finally:
        conn.close()