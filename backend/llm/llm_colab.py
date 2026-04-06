from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


DEFAULT_TIMEOUT_SEC = 180


@dataclass
class ColabLLMConfig:
    worker_url: str
    api_key: str
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    default_currency: str = "CAD"


def call_colab_receipt_llm(
    *,
    ocr_text: str,
    request_id: str,
    config: ColabLLMConfig,
    image_base64: Optional[str] = None,
    debug: bool = True,
) -> Dict[str, Any]:
    if not config.worker_url.strip():
        raise ValueError("worker_url is required")

    if not config.api_key.strip():
        raise ValueError("api_key is required")

    if not (ocr_text or "").strip():
        raise ValueError("ocr_text is empty")

    url = config.worker_url.rstrip("/") + "/infer"

    payload = {
        "request_id": request_id,
        "ocr_text": ocr_text.strip(),
        "image_base64": image_base64,
        "debug": debug,
    }

    headers = {
        "x-api-key": config.api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=config.timeout_sec,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to call Colab worker: {type(e).__name__}: {e}") from e

    print(f"COLAB WORKER URL: {url}")
    print(f"COLAB WORKER STATUS: {resp.status_code}")
    print("COLAB WORKER RESPONSE TEXT:")
    print(resp.text)

    if not resp.ok:
        try:
            err_json = resp.json()
        except Exception:
            err_json = None

        if isinstance(err_json, dict):
            detail = err_json.get("detail") or err_json
            raise RuntimeError(
                f"Colab worker failed with status={resp.status_code}, detail={detail}"
            )

        raise RuntimeError(
            f"Colab worker failed with status={resp.status_code}, body={resp.text}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise ValueError(
            f"Colab worker returned non-JSON response: {type(e).__name__}: {e} | body={resp.text}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(f"Invalid response from Colab worker: {data}")

    result = data.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Missing or invalid 'result' in response: {data}")

    normalized = normalize_receipt_json(
        result,
        default_currency=config.default_currency,
    )

    worker_validation = data.get("validation")
    if not isinstance(worker_validation, dict):
        worker_validation = validate_receipt_math(normalized)

    return {
        "success": bool(data.get("success", True)),
        "route_used": "ocr_plus_image" if image_base64 else "ocr_text_only",
        "model_used": data.get("model_used", "colab-qwen-worker"),
        "prompt_used": {
            "header": data.get("header_prompt_used", ""),
            "items": data.get("items_prompt_used", ""),
            "totals": data.get("totals_prompt_used", ""),
        },
        "raw_response_text": {
            "header": data.get("header_raw_response_text", ""),
            "items": data.get("items_raw_response_text", ""),
            "totals": data.get("totals_raw_response_text", ""),
        },
        "parsed_output": normalized,
        "validation": worker_validation,
        "clean_ocr_text": data.get("clean_ocr_text", ""),
        "header_block": data.get("header_block", ""),
        "item_block": data.get("item_block", ""),
        "totals_block": data.get("totals_block", ""),
        "full_response": data,
        "parse_error": data.get("parse_error"),
        "error": None,
    }


def call_colab_receipt_llm_from_file(
    *,
    ocr_text_path: str | Path,
    request_id: str,
    config: ColabLLMConfig,
    image_path: Optional[str | Path] = None,
    debug: bool = True,
) -> Dict[str, Any]:
    p = Path(ocr_text_path)
    if not p.exists():
        raise FileNotFoundError(f"OCR text file not found: {p}")

    ocr_text = p.read_text(encoding="utf-8", errors="ignore")

    image_base64 = None
    if image_path:
        image_base64 = encode_image_to_base64(image_path)

    return call_colab_receipt_llm(
        ocr_text=ocr_text,
        request_id=request_id,
        config=config,
        image_base64=image_base64,
        debug=debug,
    )


def encode_image_to_base64(image_path: str | Path) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def normalize_receipt_json(
    data: Dict[str, Any],
    default_currency: str = "CAD",
) -> Dict[str, Any]:
    items_in = data.get("items") or data.get("receipt_items") or []

    normalized_items = []
    for idx, item in enumerate(items_in, start=1):
        if not isinstance(item, dict):
            continue

        name = clean_text(item.get("name") or item.get("item_name"))
        price = parse_money_or_none(
            item.get("price") or item.get("line_price") or item.get("line_total")
        )

        normalized_items.append(
            {
                "line_no": idx,
                "name": name,
                "price": price,
                "confidence": parse_confidence_or_none(item.get("confidence")),
            }
        )

    out = {
        "vendor": clean_text(data.get("vendor")),
        "phone": normalize_phone(data.get("phone")),
        "address": clean_text(data.get("address")),
        "date": normalize_date_text(data.get("date") or data.get("receipt_date")),
        "items": normalized_items,
        "subtotal": parse_money_or_none(data.get("subtotal")),
        "tax": parse_money_or_none(data.get("tax")),
        "total": parse_money_or_none(data.get("total")),
        "currency": normalize_currency(data.get("currency"), default=default_currency),
        "confidence": parse_confidence_or_none(data.get("confidence")),
    }

    if out["confidence"] is None:
        out["confidence"] = estimate_result_confidence(out)

    return out


def clean_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_phone(v: Any) -> Optional[str]:
    s = clean_text(v)
    if not s:
        return None

    digits = re.sub(r"\D+", "", s)
    if len(digits) == 10:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"{digits[1:4]}-{digits[4:7]}-{digits[7:11]}"
    return s


def normalize_date_text(v: Any) -> Optional[str]:
    s = clean_text(v)
    if not s:
        return None

    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    s_norm = re.sub(r"[/\\\.\s]+", "-", s.strip())

    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s_norm)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", s_norm)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    m = re.match(r"^(\d{2})-(\d{1,2})-(\d{1,2})$", s_norm)
    if m:
        yy = int(m.group(1))
        year = 2000 + yy if yy <= 50 else 1900 + yy
        return f"{year}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    month_map = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }

    m = re.match(r"^([A-Za-z]{3,9})[,\-\s]+(\d{1,2})[,\-\s]+(\d{4})$", s.strip())
    if m:
        mon = month_map.get(m.group(1)[:3].lower())
        if mon:
            return f"{m.group(3)}-{mon}-{int(m.group(2)):02d}"

    m = re.match(r"^(\d{1,2})[,\-\s]+([A-Za-z]{3,9})[,\-\s]+(\d{4})$", s.strip())
    if m:
        mon = month_map.get(m.group(2)[:3].lower())
        if mon:
            return f"{m.group(3)}-{mon}-{int(m.group(1)):02d}"

    return s


def normalize_currency(v: Any, default: str = "CAD") -> str:
    s = clean_text(v)
    if not s:
        return default

    su = s.upper().strip()

    if re.match(r"^[A-Z]{3}$", su):
        return su

    symbol_map = {
        "US$": "USD",
        "USD$": "USD",
        "CAD$": "CAD",
        "C$": "CAD",
        "CA$": "CAD",
        "AU$": "AUD",
        "A$": "AUD",
        "NZ$": "NZD",
        "HK$": "HKD",
        "SG$": "SGD",
        "£": "GBP",
        "€": "EUR",
        "¥": "JPY",
        "¥CN": "CNY",
        "CN¥": "CNY",
        "₹": "INR",
        "₩": "KRW",
        "฿": "THB",
        "₺": "TRY",
        "₽": "RUB",
        "R$": "BRL",
        "MX$": "MXN",
        "$": default,
    }

    if su in symbol_map:
        return symbol_map[su]

    stripped = su.lstrip("$").strip()
    if re.match(r"^[A-Z]{3}$", stripped):
        return stripped

    return default


def parse_money_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None

    if isinstance(v, (int, float)):
        return round(float(v), 2)

    s = str(v).strip()
    if not s or s.lower() in {"null", "none", "n/a"}:
        return None

    s = s.replace("O", "0").replace("o", "0")
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace(":", ".")
    s = re.sub(r"(-?\d+\.\d{2})[SN]\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\d\.\-]", "", s)

    if s in {"", ".", "-", "-.", ".-"}:
        return None

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None

    try:
        return round(float(m.group(0)), 2)
    except Exception:
        return None


def parse_confidence_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None

    try:
        fv = float(v)
        if 0.0 <= fv <= 1.0:
            fv *= 100.0
        return max(0.0, min(100.0, round(fv, 2)))
    except Exception:
        return None


def validate_receipt_math(
    result: Dict[str, Any],
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    items = result.get("items") or []
    subtotal = result.get("subtotal")
    tax = result.get("tax")
    total = result.get("total")

    item_sum = round(sum(float(it.get("price") or 0.0) for it in items), 2)

    subtotal_ok = None
    total_ok = None

    if subtotal is not None:
        subtotal_ok = abs(item_sum - float(subtotal)) <= tolerance

    if subtotal is not None and tax is not None and total is not None:
        total_ok = abs((float(subtotal) + float(tax)) - float(total)) <= tolerance
    elif subtotal is not None and total is not None:
        total_ok = abs(float(subtotal) - float(total)) <= tolerance

    return {
        "item_sum": item_sum,
        "subtotal_ok": subtotal_ok,
        "total_ok": total_ok,
    }


def estimate_result_confidence(result: Dict[str, Any]) -> float:
    score = 40.0

    if result.get("vendor"):
        score += 10.0
    if result.get("total") is not None:
        score += 15.0
    if result.get("subtotal") is not None:
        score += 8.0
    if result.get("tax") is not None:
        score += 5.0
    if result.get("date"):
        score += 5.0
    if result.get("phone"):
        score += 5.0

    items = result.get("items") or []
    if items:
        score += min(12.0, len(items) * 2.0)

    return max(0.0, min(100.0, round(score, 2)))


if __name__ == "__main__":
    cfg = ColabLLMConfig(
        worker_url="https://YOUR-NGROK-URL.ngrok-free.app",
        api_key="demo-secret-key",
    )

    base_dir = Path(__file__).resolve().parents[1]

    ocr_path = (
        base_dir
        / "data/uploads/processed/JOB-20260311_220334-F88AD73A/ocr/RSCAN-000001/best_texts/RSCAN-000001_best.txt"
    )

    image_path = (
        base_dir
        / "data/uploads/processed/JOB-20260311_220334-F88AD73A/ocr/RSCAN-000001/best_images/RSCAN-000001_best_view.png"
    )

    result = call_colab_receipt_llm_from_file(
        ocr_text_path=ocr_path,
        image_path=image_path,
        request_id="RSCAN-000001",
        config=cfg,
        debug=True,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))