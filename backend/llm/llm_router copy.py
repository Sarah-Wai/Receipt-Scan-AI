from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import os

# ============================================================
# Config
# ============================================================


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://1427lsqz-11434.usw3.devtunnels.ms")
DEFAULT_TEXT_MODEL = "qwen2.5:3b"
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"
DEFAULT_TIMEOUT_SEC = 180


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class LLMRouterConfig:
    ollama_base_url: str = OLLAMA_BASE_URL
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    default_currency: str = "CAD"

    # Routing rules
    min_ocr_confidence_for_text: float = 0.80
    min_item_lines_for_text: int = 2
    prefer_vision_for_long_receipt: bool = True
    long_receipt_line_threshold: int = 80
    high_broken_token_score_for_vision: float = 0.35

    # LLM generation options (0 = let Ollama use model default)
    # These are set dynamically per call based on OCR length;
    # set non-zero here only to override the adaptive logic globally.
    num_ctx: int = 0
    num_predict: int = 0


@dataclass
class LLMCallResult:
    success: bool
    route_used: str  # "ocr_text" | "vision_image"
    model_used: str
    prompt_used: str
    raw_response_text: str
    parsed_output: Dict[str, Any]
    error: Optional[str] = None


# ============================================================
# Adaptive Ollama options
# ============================================================


def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def get_ollama_options(
    ocr_text: str = "",
    config: Optional[LLMRouterConfig] = None,
) -> Dict[str, Any]:
    """
    Return Ollama `options` dict sized to the OCR input.
    Config overrides take priority over adaptive values.
    """
    cfg = config or LLMRouterConfig()
    ocr_tokens = _estimate_tokens(ocr_text)

    # Adaptive sizing based on OCR length
    if ocr_tokens < 300:          # small receipt
        num_ctx, num_predict = 1024, 300
    elif ocr_tokens < 700:        # medium receipt
        num_ctx, num_predict = 2048, 600
    elif ocr_tokens < 1500:       # long receipt
        num_ctx, num_predict = 4096, 1000
    else:                         # very long receipt
        num_ctx, num_predict = 8192, 1500

    # Config overrides
    if cfg.num_ctx > 0:
        num_ctx = cfg.num_ctx
    if cfg.num_predict > 0:
        num_predict = cfg.num_predict

    return {
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
    }


# ============================================================
# Prompt templates
# ============================================================

# FIX #1: Removed qty and unit_price from schema.
# FIX #11: Added currency inference hint.
TEXT_PROMPT_TEMPLATE = """You are a receipt extraction system.

Extract the following fields from the OCR text below and return a single valid JSON object.

Rules:
- Return ONLY the JSON object. No markdown, no explanation, no code fences.
- All monetary values must be numbers (e.g. 12.99), never strings.
- date must be normalized to YYYY-MM-DD format.
- currency must be an ISO 4217 code. Infer it from any currency symbol ($, €, £, ¥ etc.), \
the country in the address, the phone dial code (+1, +44 etc.), or the tax label \
(GST/HST/VAT/Sales Tax etc.) found in the receipt. Default to CAD if nothing matches.
- Keep items in reading order.
- "price" means the final line total for that item.
- Correct obvious OCR mistakes only if strongly supported by nearby text.
- If a field is missing or unreadable, use null. Do NOT invent values.
- If no items are found, return [] for items.
- confidence reflects your overall extraction confidence (0.0 = no confidence, 1.0 = fully confident).

Required JSON schema:
{{
  "vendor": null,
  "phone": null,
  "address": null,
  "date": null,
  "items": [
    {{
      "name": null,
      "price": null
    }}
  ],
  "subtotal": null,
  "tax": null,
  "total": null,
  "currency": "CAD",
  "confidence": null
}}

OCR text:
\"\"\"
{ocr_text}
\"\"\"
"""


# FIX #2: Removed qty and unit_price from vision schema as well.
VISION_PROMPT_TEMPLATE = """You are a receipt extraction system.

Read the receipt image and extract the fields below. Return a single valid JSON object.

Rules:
- Return ONLY the JSON object. No markdown, no explanation, no code fences.
- All monetary values must be numbers (e.g. 12.99), never strings.
- date must be normalized to YYYY-MM-DD format.
- currency must be an ISO 4217 code. Infer it from any currency symbol, address country, \
phone dial code, or tax label (GST/HST/VAT/Sales Tax etc.) visible on the receipt. \
Default to CAD if nothing matches.
- Keep items in reading order.
- "price" means the final line total for that item.
- Prefer values visibly present in the image.
- If a field is missing or unclear, use null. Do NOT invent values.
- If no items are found, return [] for items.
- confidence reflects your overall extraction confidence (0.0 = no confidence, 1.0 = fully confident).

Required JSON schema:
{{
  "vendor": null,
  "phone": null,
  "address": null,
  "date": null,
  "items": [
    {{
      "name": null,
      "price": null
    }}
  ],
  "subtotal": null,
  "tax": null,
  "total": null,
  "currency": "CAD",
  "confidence": null
}}
"""


# ============================================================
# Main public function used by routes_auto_fix.py
# ============================================================


def run_auto_fix_for_receipt(
    *,
    ocr_text: Optional[str],
    image_path: Optional[str | Path],
    ocr_confidence: Optional[float],
    config: Optional[LLMRouterConfig] = None,
) -> Dict[str, Any]:
    """
    Main convenience wrapper for your pipeline / API route.

    Returns a plain dict:
    {
      "success": bool,
      "route_used": "...",
      "model_used": "...",
      "prompt_used": "...",
      "raw_response_text": "...",
      "parsed_output": {...},
      "validation": {...} | None,
      "error": str | None
    }
    """
    cfg = config or LLMRouterConfig()

    result = route_and_extract_receipt(
        ocr_text=ocr_text,
        image_path=image_path,
        ocr_confidence=ocr_confidence,
        config=cfg,
    )

    validation = validate_receipt_math(result.parsed_output) if result.success else None

    return {
        "success": result.success,
        "route_used": result.route_used,
        "model_used": result.model_used,
        "prompt_used": result.prompt_used,
        "raw_response_text": result.raw_response_text,
        "parsed_output": result.parsed_output,
        "validation": validation,
        "error": result.error,
    }


# ============================================================
# Router logic
# ============================================================


def route_and_extract_receipt(
    *,
    ocr_text: Optional[str] = None,
    image_path: Optional[str | Path] = None,
    ocr_confidence: Optional[float] = None,
    config: Optional[LLMRouterConfig] = None,
) -> LLMCallResult:
    cfg = config or LLMRouterConfig()

    route = choose_route(
        ocr_text=ocr_text,
        image_path=image_path,
        ocr_confidence=ocr_confidence,
        config=cfg,
    )

    if route == "ocr_text":
        if not (ocr_text or "").strip():
            if image_path:
                return call_vision_llm(image_path=image_path, config=cfg)
            return LLMCallResult(
                success=False,
                route_used="ocr_text",
                model_used=cfg.text_model,
                prompt_used="",
                raw_response_text="",
                parsed_output={},
                error="OCR text route selected but OCR text is empty",
            )
        return call_text_llm(ocr_text=ocr_text or "", config=cfg)

    if route == "vision_image":
        if not image_path:
            if (ocr_text or "").strip():
                return call_text_llm(ocr_text=ocr_text or "", config=cfg)
            return LLMCallResult(
                success=False,
                route_used="vision_image",
                model_used=cfg.vision_model,
                prompt_used="",
                raw_response_text="",
                parsed_output={},
                error="Vision route selected but image_path is missing",
            )
        return call_vision_llm(image_path=image_path, config=cfg)

    return LLMCallResult(
        success=False,
        route_used=str(route),
        model_used="",
        prompt_used="",
        raw_response_text="",
        parsed_output={},
        error=f"Unknown route selected: {route}",
    )


def choose_route(
    *,
    ocr_text: Optional[str],
    image_path: Optional[str | Path],
    ocr_confidence: Optional[float],
    config: LLMRouterConfig,
) -> str:
    """
    Rule-based route chooser.

    Returns:
      - "ocr_text"
      - "vision_image"
    """
    text = (ocr_text or "").strip()

    if not text:
        return "vision_image" if image_path else "ocr_text"

    num_lines = len([ln for ln in text.splitlines() if ln.strip()])
    item_line_count = estimate_item_line_count(text)
    has_total_keywords = contains_total_keywords(text)
    broken_token_score = estimate_broken_token_score(text)
    conf01 = normalize_confidence_to_0_1(ocr_confidence)

    if config.prefer_vision_for_long_receipt and image_path:
        if (
            num_lines >= config.long_receipt_line_threshold
            and broken_token_score > 0.20
        ):
            return "vision_image"

    if conf01 is not None and conf01 < config.min_ocr_confidence_for_text:
        return "vision_image" if image_path else "ocr_text"

    if item_line_count < config.min_item_lines_for_text and image_path:
        return "vision_image"

    if not has_total_keywords and image_path:
        return "vision_image"

    if broken_token_score > config.high_broken_token_score_for_vision and image_path:
        return "vision_image"

    return "ocr_text"


# ============================================================
# Actual LLM calls
# ============================================================


def call_text_llm(
    *,
    ocr_text: str,
    config: Optional[LLMRouterConfig] = None,
) -> LLMCallResult:
    cfg = config or LLMRouterConfig()
    prompt = TEXT_PROMPT_TEMPLATE.format(ocr_text=ocr_text.strip())

    # FIX #4: Pass adaptive options based on OCR length
    options = get_ollama_options(ocr_text=ocr_text, config=cfg)

    try:
        resp_json = _ollama_generate(
            base_url=cfg.ollama_base_url,
            model=cfg.text_model,
            prompt=prompt,
            options=options,
            timeout_sec=cfg.timeout_sec,
        )
        raw_text = resp_json.get("response", "") or ""
        parsed = parse_and_normalize_llm_json(
            raw_text, default_currency=cfg.default_currency
        )

        return LLMCallResult(
            success=True,
            route_used="ocr_text",
            model_used=cfg.text_model,
            prompt_used=prompt,
            raw_response_text=raw_text,
            parsed_output=parsed,
            error=None,
        )
    except Exception as e:
        return LLMCallResult(
            success=False,
            route_used="ocr_text",
            model_used=cfg.text_model,
            prompt_used=prompt,
            raw_response_text="",
            parsed_output={},
            error=str(e),
        )


def call_vision_llm(
    *,
    image_path: str | Path,
    config: Optional[LLMRouterConfig] = None,
) -> LLMCallResult:
    cfg = config or LLMRouterConfig()
    img_path = Path(image_path)

    if not img_path.exists():
        return LLMCallResult(
            success=False,
            route_used="vision_image",
            model_used=cfg.vision_model,
            prompt_used=VISION_PROMPT_TEMPLATE,
            raw_response_text="",
            parsed_output={},
            error=f"Image not found: {img_path}",
        )

    try:
        image_b64 = _encode_image_base64(img_path)

        # FIX #6: Removed full base64 image dump from print (huge + slow)
        print(
            f"Calling vision LLM with image: {img_path} "
            f"| base64 size: {len(image_b64)} bytes"
        )

        # FIX #5: Pass options to vision call too
        options = get_ollama_options(ocr_text="", config=cfg)

        resp_json = _ollama_generate(
            base_url=cfg.ollama_base_url,
            model=cfg.vision_model,
            prompt=VISION_PROMPT_TEMPLATE,
            images=[image_b64],
            options=options,
            timeout_sec=cfg.timeout_sec,
        )
        raw_text = resp_json.get("response", "") or ""
        parsed = parse_and_normalize_llm_json(
            raw_text, default_currency=cfg.default_currency
        )

        return LLMCallResult(
            success=True,
            route_used="vision_image",
            model_used=cfg.vision_model,
            prompt_used=VISION_PROMPT_TEMPLATE,
            raw_response_text=raw_text,
            parsed_output=parsed,
            error=None,
        )
    except Exception as e:
        return LLMCallResult(
            success=False,
            route_used="vision_image",
            model_used=cfg.vision_model,
            prompt_used=VISION_PROMPT_TEMPLATE,
            raw_response_text="",
            parsed_output={},
            error=str(e),
        )


# ============================================================
# Ollama HTTP client
# ============================================================


def _ollama_generate(
    *,
    base_url: str,
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,      # FIX #3: Added options param
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/generate"

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if images:
        payload["images"] = images

    if options:                                     # FIX #3: Inject options into payload
        payload["options"] = options

    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()


def _encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# Parsing + normalization
# ============================================================


def parse_and_normalize_llm_json(
    raw_text: str, default_currency: str = "CAD"
) -> Dict[str, Any]:
    candidate = extract_json_block(raw_text)
    parsed = json.loads(candidate)
    return normalize_receipt_json(parsed, default_currency=default_currency)


def extract_json_block(text: str) -> str:
    """
    Supports:
    - plain JSON
    - fenced ```json blocks
    - extra explanation before/after JSON
    """
    text = text.strip()

    m = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    raise ValueError("Could not extract JSON object from LLM response")


def normalize_receipt_json(
    data: Dict[str, Any], default_currency: str = "CAD"
) -> Dict[str, Any]:
    items_in = data.get("items") or data.get("receipt_items") or []

    normalized_items: List[Dict[str, Any]] = []
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


# ============================================================
# Value parsers
# ============================================================


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
    """
    FIX #8: Normalize date to YYYY-MM-DD.
    Handles common formats: MM/DD/YYYY, DD/MM/YYYY, YYYY/MM/DD,
    Month DD YYYY, DD Month YYYY, etc.
    """
    s = clean_text(v)
    if not s:
        return None

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    # Normalize separators
    s_norm = re.sub(r"[/\\\.\s]+", "-", s.strip())

    # YYYY-MM-DD or YYYY-M-D
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s_norm)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # MM-DD-YYYY or DD-MM-YYYY (ambiguous — assume MM-DD-YYYY for CAD/US)
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", s_norm)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    # Month name: "Jan 05 2024" or "05 Jan 2024" or "January 5, 2024"
    month_map = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    m = re.match(
        r"^([A-Za-z]{3,9})[,\-\s]+(\d{1,2})[,\-\s]+(\d{4})$", s.strip()
    )
    if m:
        mon = month_map.get(m.group(1)[:3].lower())
        if mon:
            return f"{m.group(3)}-{mon}-{int(m.group(2)):02d}"

    m = re.match(
        r"^(\d{1,2})[,\-\s]+([A-Za-z]{3,9})[,\-\s]+(\d{4})$", s.strip()
    )
    if m:
        mon = month_map.get(m.group(2)[:3].lower())
        if mon:
            return f"{m.group(3)}-{mon}-{int(m.group(1)):02d}"

    # Fallback: return cleaned string as-is
    return s


def normalize_currency(v: Any, default: str = "CAD") -> str:
    """
    FIX #7: Expanded currency normalization.
    Handles ISO codes, symbols, and common OCR variants.
    """
    s = clean_text(v)
    if not s:
        return default

    su = s.upper().strip()

    # Already a valid ISO 4217 3-letter code
    if re.match(r"^[A-Z]{3}$", su):
        return su

    # Symbol / prefix map
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
        "ZAR": "ZAR",
        "R": "ZAR",
        "$": default,   # bare $ → use default (CAD for Canada)
    }

    if su in symbol_map:
        return symbol_map[su]

    # Strip leading $ and try again
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
    s = re.sub(r"[^\d\.\-]", "", s)

    if s in {"", ".", "-", "-.", ".-"}:
        return None

    try:
        return round(float(s), 2)
    except Exception:
        return None


def parse_number_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None

    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    if not s or s.lower() in {"null", "none", "n/a"}:
        return None

    s = s.replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)

    if s in {"", ".", "-", "-.", ".-"}:
        return None

    try:
        return float(s)
    except Exception:
        return None


def parse_confidence_or_none(v: Any) -> Optional[float]:
    """
    FIX #9: Corrected scaling logic.
    LLM returns 0.0–1.0. We store as 0.0–100.0 internally.
    Only scale up if value is clearly in 0–1 range (exclusive of exactly 0 or 1
    which are valid boundary values but ambiguous — we treat <=1.0 as 0–1 scale).
    Guard against already-scaled values (e.g. 85.0) being doubled.
    """
    if v is None:
        return None

    try:
        fv = float(v)
        # If value is in 0–1 range, scale to 0–100
        if 0.0 <= fv <= 1.0:
            fv = fv * 100.0
        # Clamp to valid range
        return max(0.0, min(100.0, round(fv, 2)))
    except Exception:
        return None


# ============================================================
# Validation helpers
# ============================================================


def validate_receipt_math(
    result: Dict[str, Any], tolerance: float = 0.05
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


# ============================================================
# Routing helpers
# ============================================================


def normalize_confidence_to_0_1(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
        if fv > 1.0:
            fv = fv / 100.0
        return max(0.0, min(1.0, fv))
    except Exception:
        return None


def estimate_item_line_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _line_looks_like_item_price(line):
            count += 1
    return count


def contains_total_keywords(text: str) -> bool:
    keys = ["total", "subtotal", "sub total", "tax", "gst", "hst", "pst"]
    t = text.lower()
    return any(k in t for k in keys)


def estimate_broken_token_score(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0

    bad = 0
    total = 0

    patterns = [
        r"\d+\$\d",
        r"[A-Za-z]{1,2}\d{3,}",
        r"\d+[A-Za-z]+\d+",
        r"[^\w\s\.\,\-\$\:/]{2,}",
    ]

    for ln in lines:
        total += 1
        hit = False
        for p in patterns:
            if re.search(p, ln):
                hit = True
                break
        if hit:
            bad += 1

    return bad / max(total, 1)


def _line_looks_like_item_price(line: str) -> bool:
    price_patterns = [
        r"\d+\.\d{2}$",
        r"\$\s*\d+\.\d{2}$",
        r"\d+\.\d{2}\s+[A-Za-z]?$",
    ]
    has_price = any(re.search(p, line) for p in price_patterns)
    has_letters = bool(re.search(r"[A-Za-z]", line))
    return has_price and has_letters


# ============================================================
# Local test
# ============================================================

if __name__ == "__main__":
    sample_ocr = """
    REAL CANADIAN SUPERSTORE
    306-585-8890

    BANANA 2.99
    MILK 4.49
    BREAD 3.29

    SUBTOTAL 10.77
    TAX 0.54
    TOTAL 11.31
    """

    cfg = LLMRouterConfig()

    out = run_auto_fix_for_receipt(
        ocr_text=sample_ocr,
        image_path=None,
        ocr_confidence=0.92,
        config=cfg,
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))