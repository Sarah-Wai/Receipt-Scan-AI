from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

WORKER_URL = "https://3878-34-125-208-88.ngrok-free.app"
API_KEY = "demo-secret-key"


def encode_image_base64(
    image_path: str | Path,
    *,
    max_side: int = 1024,
    jpeg_quality: int = 65,
) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")

    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size

        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_worker(
    ocr_text: str,
    *,
    request_id: str = "req-001",
    image_path: Optional[str | Path] = None,
    debug: bool = True,
) -> dict:
    payload = {
        "request_id": request_id,
        "ocr_text": ocr_text,
        "debug": debug,
    }

    if image_path:
        payload["image_base64"] = encode_image_base64(image_path)

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
    }

    r = requests.post(
        f"{WORKER_URL.rstrip('/')}/infer",
        json=payload,
        headers=headers,
        timeout=180,
    )

    print("STATUS:", r.status_code)
    print("RESPONSE TEXT:")
    print(r.text)

    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    # arg1 = ocr txt path
    # arg2 = optional image path
    if len(sys.argv) > 1:
        ocr_path = Path(sys.argv[1])
    else:
        ocr_path = Path(
            "/Users/waiphu/FinScanAI/backend/data/uploads/processed/JOB-20260319_131141-D077E547/ocr/RSCAN-000001/best_texts/RSCAN-000001_best.txt"
        )

    if len(sys.argv) > 2:
        image_path = Path(sys.argv[2])
    else:
        image_path = Path(
            "/Users/waiphu/FinScanAI/backend/data/uploads/processed/JOB-20260319_131141-D077E547/yolo/rectified/RSCAN-000001.png"
        )

    if not ocr_path.exists():
        print(f"ERROR: OCR file not found: {ocr_path}")
        sys.exit(1)

    if image_path and not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    try:
        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"ERROR: Failed to read OCR file: {ocr_path}\n{e}")
        sys.exit(1)

    try:
        result = call_worker(
            ocr_text,
            request_id="receipt-demo-001",
            image_path=image_path,
            debug=True,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: Worker call failed\n{type(e).__name__}: {e}")
        sys.exit(1)