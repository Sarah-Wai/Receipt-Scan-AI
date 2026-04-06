from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types


API_KEY = "AIzaSyAMe-yvU-0taj9ZJK1p_l6QUTUJH4kPaqw"
MODEL_NAME = "gemini-2.5-flash-lite"


def extract_json_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise ValueError("No JSON object found in model response")


def compress_ocr_text(text: str, max_lines: int = 80, max_chars: int = 3500) -> str:
    if not text:
        return ""

    lines = []
    seen = set()

    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)

    def score(line: str) -> int:
        s = 0
        low = line.lower()
        if any(k in low for k in ["total", "subtotal", "tax", "gst", "pst", "hst"]):
            s += 5
        if any(ch.isdigit() for ch in line):
            s += 2
        if any(ch.isalpha() for ch in line):
            s += 1
        return s

    ranked = sorted(lines, key=score, reverse=True)

    keep = []
    used = set()

    for line in ranked[:25]:
        if line not in used:
            keep.append(line)
            used.add(line)

    for line in lines:
        if line not in used:
            keep.append(line)
            used.add(line)
        if len(keep) >= max_lines:
            break

    out = "\n".join(keep)
    return out[:max_chars]


def build_prompt(ocr_text: str) -> str:
    return f"""Return JSON only:
{{"vendor":null,"phone":null,"address":null,"date":null,"items":[{{"name":null,"price":null}}],"subtotal":null,"tax":null,"total":null,"currency":"CAD","confidence":null}}

Rules:
- no markdown
- no extra text
- date YYYY-MM-DD
- price = line total
- numbers as numbers
- null if unclear
- default currency CAD
- keep item order

OCR:
{ocr_text}
"""


def call_gemini(
    ocr_text: str,
    *,
    request_id: str = "req-001",
    image_path: Optional[str | Path] = None,
    debug: bool = True,
) -> dict:
    client = genai.Client(api_key=API_KEY)

    compact_ocr = compress_ocr_text(ocr_text)
    prompt = build_prompt(compact_ocr)

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=180,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    if debug:
        try:
            token_info = client.models.count_tokens(
                model=MODEL_NAME,
                contents=prompt,
            )
            print("PROMPT TOKENS:", getattr(token_info, "total_tokens", None))
        except Exception as e:
            print("WARNING: count_tokens failed:", e)

    if image_path:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        with Image.open(img_path) as im:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, im],
                config=config,
            )
    else:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )

    raw_text = getattr(response, "text", "") or ""

    print("REQUEST ID:", request_id)
    print("MODEL:", MODEL_NAME)
    print("RAW RESPONSE:")
    print(raw_text)

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {}
    if usage_metadata:
        usage = {
            "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
            "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
            "total_token_count": getattr(usage_metadata, "total_token_count", None),
            "thoughts_token_count": getattr(usage_metadata, "thoughts_token_count", None),
        }

    print("USAGE:")
    print(json.dumps(usage, indent=2, ensure_ascii=False))

    try:
        parsed = json.loads(raw_text)
    except Exception:
        try:
            parsed = json.loads(extract_json_block(raw_text))
        except Exception:
            parsed = None

    return {
        "request_id": request_id,
        "success": parsed is not None,
        "model_used": MODEL_NAME,
        "image_used": bool(image_path),
        "compressed_ocr": compact_ocr,
        "raw_response_text": raw_text,
        "parsed_output": parsed,
        "usage": usage,
    }


if __name__ == "__main__":
    # arg1 = OCR txt path
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
        result = call_gemini(
            ocr_text,
            request_id="receipt-demo-001",
            image_path=image_path,
            debug=True,
        )
        print("FINAL RESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: Gemini call failed\n{type(e).__name__}: {e}")
        sys.exit(1)