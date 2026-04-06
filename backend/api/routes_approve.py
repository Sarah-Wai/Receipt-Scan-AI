# backend/api/routes_approve.py
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from db.sqlite_reader import get_conn
from db.sqlite_writer import approve_llm_run, reject_llm_run

router = APIRouter(prefix="/api/receipts/llm", tags=["approve"])


class ApproveRequest(BaseModel):
    approved_by: Optional[str] = None


class RejectRequest(BaseModel):
    rejected_by: Optional[str] = None


@router.post("/{llm_run_id}/approve")
def approve_llm(llm_run_id: int, payload: ApproveRequest):
    conn = get_conn()
    try:
        receipt_id = approve_llm_run(
            conn,
            llm_run_id=llm_run_id,
            approved_by=payload.approved_by,
        )
        conn.commit()
        return {"ok": True, "receipt_id": receipt_id, "llm_run_id": llm_run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/{llm_run_id}/reject")
def reject_llm(llm_run_id: int, payload: RejectRequest):
    conn = get_conn()
    try:
        receipt_id = reject_llm_run(
            conn,
            llm_run_id=llm_run_id,
            rejected_by=payload.rejected_by,
        )
        conn.commit()
        return {"ok": True, "receipt_id": receipt_id, "llm_run_id": llm_run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()