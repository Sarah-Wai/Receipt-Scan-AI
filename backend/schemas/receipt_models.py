# schemas/receipt_models.py
from pydantic import BaseModel


class ReceiptStatusUpdate(BaseModel):
    status: str