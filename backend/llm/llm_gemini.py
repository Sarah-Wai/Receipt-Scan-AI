from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from google import genai
from google.genai import types
import typing_extensions as typing

from config import (
    GEMINI_API_KEY,
    GEMINI_TEXT_MODEL,
    GEMINI_VISION_MODEL,
)

# ============================================================
# Structured output schemas
# ============================================================

class ReceiptItemSchema(typing.TypedDict):
    name: Optional[str]
    price: Optional[float]


class ReceiptOutputSchema(typing.TypedDict):
    vendor: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    date: Optional[str]
    items: List[ReceiptItemSchema]
    subtotal: Optional[float]
    tax: Optional[float]
    total: Optional[float]
    currency: str
    confidence: Optional[float]


class ReceiptSummarySchema(typing.TypedDict):
    vendor: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    date: Optional[str]
    subtotal: Optional[float]
    tax: Optional[float]
    total: Optional[float]
    currency: str
    confidence: Optional[float]


class ReceiptItemsOnlyItemSchema(typing.TypedDict):
    name: Optional[str]
    price: Optional[float]


class ReceiptItemsOnlySchema(typing.TypedDict):
    items: List[ReceiptItemsOnlyItemSchema]
    currency: str
    confidence: Optional[float]


# ============================================================
# Config
# ============================================================

DEFAULT_TIMEOUT_SEC = 180


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class LLMRouterConfig:
    provider: str = "gemini"
    api_key: str = GEMINI_API_KEY
    text_model: str = GEMINI_TEXT_MODEL or "gemini-2.5-flash-lite"
    vision_model: str = GEMINI_VISION_MODEL or "gemini-2.5-flash-lite"
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    default_currency: str = "CAD"

    # Auto-route rules (optional; you can override with force_route)
    min_ocr_confidence_for_text: float = 0.45
    min_item_lines_for_text: int = 1
    prefer_vision_for_long_receipt: bool = False
    long_receipt_line_threshold: int = 200
    high_broken_token_score_for_vision: float = 0.95

    # Output caps
    text_max_output_tokens: int = 320
    vision_max_output_tokens: int = 600
    text_summary_max_output_tokens: int = 220
    vision_summary_max_output_tokens: int = 260
    text_items_max_output_tokens: int = 420
    vision_items_max_output_tokens: int = 900

    temperature: float = 0.0

    # Free-tier optimization
    force_json_output: bool = True
    free_tier_mode: bool = True
    text_max_prompt_tokens: int = 1200
    vision_max_prompt_tokens: int = 1200
    compress_ocr_max_lines: int = 80
    compress_ocr_max_chars: int = 3500

    # Fallback behavior
    allow_vision_fallback: bool = False
    enable_two_pass_fallback_for_text: bool = True
    use_two_pass_for_vision: bool = True

    # Retry behavior
    retry_attempts: int = 3
    retry_default_wait_sec: float = 8.0
    retry_max_wait_sec: float = 20.0


@dataclass
class LLMCallResult:
    success: bool
    route_used: str
    model_used: str
    prompt_used: str
    raw_response_text: str
    parsed_output: Dict[str, Any]
    error: Optional[str] = None


# ============================================================
# Prompt templates
# ============================================================

_OCR_PLACEHOLDER = "__OCR_TEXT__"

TEXT_PROMPT_TEMPLATE = """Return JSON only:
{"vendor":null,"phone":null,"address":null,"date":null,"items":[{"name":null,"price":null}],"subtotal":null,"tax":null,"total":null,"currency":"CAD","confidence":null}

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
__OCR_TEXT__
"""

TEXT_SUMMARY_PROMPT_TEMPLATE = """Return JSON only:
{"vendor":null,"phone":null,"address":null,"date":null,"subtotal":null,"tax":null,"total":null,"currency":"CAD","confidence":null}

Rules:
- no markdown
- no extra text
- date YYYY-MM-DD
- numbers as numbers
- null if unclear
- default currency CAD

OCR:
__OCR_TEXT__
"""

TEXT_ITEMS_PROMPT_TEMPLATE = """Return JSON only:
{"items":[{"name":null,"price":null}],"currency":"CAD","confidence":null}

Rules:
- no markdown
- no extra text
- each item must include final line price
- price = line total
- numbers as numbers
- null if unclear
- keep item order
- do not include subtotal, tax, total in items
- default currency CAD

OCR:
__OCR_TEXT__
"""

VISION_PROMPT_TEMPLATE = """Read this receipt image and return JSON only:
{"vendor":null,"phone":null,"address":null,"date":null,"items":[{"name":null,"price":null}],"subtotal":null,"tax":null,"total":null,"currency":"CAD","confidence":null}

Rules:
- no markdown
- no extra text
- date YYYY-MM-DD
- price = line total
- numbers as numbers
- null if unclear
- default currency CAD
- keep item order
"""

VISION_SUMMARY_PROMPT_TEMPLATE = """Read this receipt image and return JSON only:
{"vendor":null,"phone":null,"address":null,"date":null,"subtotal":null,"tax":null,"total":null,"currency":"CAD","confidence":null}

Rules:
- no markdown
- no extra text
- date YYYY-MM-DD
- numbers as numbers
- null if unclear
- default currency CAD
"""

VISION_ITEMS_PROMPT_TEMPLATE = """Read this receipt image and return JSON only:
{"items":[{"name":null,"price":null}],"currency":"CAD","confidence":null}

Rules:
- no markdown
- no extra text
- each item must include final line price
- price = line total
- numbers as numbers
- null if unclear
- keep item order
- do not include subtotal, tax, total in items
- default currency CAD
"""


def _inject_ocr_text(template: str, ocr_text: str) -> str:
    return template.replace(_OCR_PLACEHOLDER, ocr_text)


# ============================================================
# Retry helpers
# ============================================================

def _is_retryable_gemini_error(msg: str) -> bool:
    s = (msg or "").upper()
    return (
        "503" in s
        or "UNAVAILABLE" in s
        or "429" in s
        or "RESOURCE_EXHAUSTED" in s
        or "RATE LIMIT" in s
        or "QUOTA" in s
        or "RETRYDELAY" in s
    )


def _sleep_seconds_from_error(msg: str, default_sec: float = 8.0, max_sec: float = 20.0) -> float:
    s = msg or ""

    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", s, re.IGNORECASE)
    if m:
        try:
            return min(max(1.0, float(m.group(1))), max_sec)
        except Exception:
            pass

    m = re.search(r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"', s, re.IGNORECASE)
    if m:
        try:
            return min(max(1.0, float(m.group(1))), max_sec)
        except Exception:
            pass

    return min(max(1.0, default_sec), max_sec)


def _call_with_retry(fn, *, attempts: int = 3, default_wait_sec: float = 8.0, max_wait_sec: float = 20.0):
    last_err = None

    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e)

            if not _is_retryable_gemini_error(msg):
                raise

            if i == attempts - 1:
                raise

            wait_sec = _sleep_seconds_from_error(
                msg,
                default_sec=min(default_wait_sec + i * 4.0, max_wait_sec),
                max_sec=max_wait_sec,
            )
            time.sleep(wait_sec)

    raise last_err


# ============================================================
# Main public function
# ============================================================

def run_auto_fix_for_receipt(
    *,
    ocr_text: Optional[str],
    image_path: Optional[str | Path],
    ocr_confidence: Optional[float],
    config: Optional[LLMRouterConfig] = None,
    force_route: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = config or LLMRouterConfig()

    result = route_and_extract_receipt(
        ocr_text=ocr_text,
        image_path=image_path,
        ocr_confidence=ocr_confidence,
        config=cfg,
        force_route=force_route,
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
    force_route: Optional[str] = None,
) -> LLMCallResult:
    cfg = config or LLMRouterConfig()

    if force_route is not None:
        if force_route not in {"ocr_text", "vision_image"}:
            return LLMCallResult(
                success=False,
                route_used=str(force_route),
                model_used="",
                prompt_used="",
                raw_response_text="",
                parsed_output={},
                error=f"Invalid force_route: {force_route}",
            )
        route = force_route
    else:
        route = choose_route(
            ocr_text=ocr_text,
            image_path=image_path,
            ocr_confidence=ocr_confidence,
            config=cfg,
        )

    if route == "ocr_text":
        if not (ocr_text or "").strip():
            return LLMCallResult(
                success=False,
                route_used="ocr_text",
                model_used=cfg.text_model,
                prompt_used="",
                raw_response_text="",
                parsed_output={},
                error="OCR text route selected but OCR text is empty",
            )

        primary = call_text_llm(ocr_text=ocr_text, config=cfg)

        if (
            not primary.success
            and cfg.allow_vision_fallback
            and image_path
            and Path(image_path).exists()
        ):
            fallback = call_vision_llm(image_path=image_path, config=cfg)
            if fallback.success:
                return fallback

        return primary

    if route == "vision_image":
        if not image_path:
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
    text = (ocr_text or "").strip()

    if not text:
        return "vision_image" if image_path else "ocr_text"

    num_lines = len([ln for ln in text.splitlines() if ln.strip()])
    item_line_count = estimate_item_line_count(text)
    has_total_keywords = contains_total_keywords(text)
    broken_token_score = estimate_broken_token_score(text)
    conf01 = normalize_confidence_to_0_1(ocr_confidence)

    if config.free_tier_mode:
        if image_path and (
            num_lines < 3
            or (conf01 is not None and conf01 < 0.20)
            or broken_token_score > 0.95
        ):
            return "vision_image"
        return "ocr_text"

    if config.prefer_vision_for_long_receipt and image_path:
        if num_lines >= config.long_receipt_line_threshold and broken_token_score > 0.20:
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
# OCR compression
# ============================================================

def compress_ocr_text_for_llm(
    text: str,
    max_lines: int = 80,
    max_chars: int = 3500,
) -> str:
    if not text:
        return ""

    cleaned_lines: List[str] = []
    seen = set()

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue

        if re.fullmatch(r"[-_=*~:.]{2,}", line):
            continue

        alnum_count = sum(ch.isalnum() for ch in line)
        if alnum_count < 2:
            continue

        if line in seen:
            continue
        seen.add(line)

        cleaned_lines.append(line)

    def score(line: str) -> int:
        s = 0
        low = line.lower()

        if any(k in low for k in ["total", "subtotal", "sub total", "tax", "gst", "pst", "hst", "balance"]):
            s += 6
        if re.search(r"\d+\.\d{2}", line):
            s += 4
        if re.search(r"[a-zA-Z]", line):
            s += 1
        if re.search(r"\d{2,4}[-/]\d{1,2}[-/]\d{1,4}", line):
            s += 2
        if re.search(r"\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}", line):
            s += 2

        return s

    ranked = sorted(cleaned_lines, key=score, reverse=True)

    keep: List[str] = []
    used = set()

    for line in ranked[:25]:
        if line not in used:
            keep.append(line)
            used.add(line)

    for line in cleaned_lines:
        if line not in used:
            keep.append(line)
            used.add(line)
        if len(keep) >= max_lines:
            break

    out = "\n".join(keep)
    return out[:max_chars]


# ============================================================
# Merge helper
# ============================================================

def merge_summary_and_items(
    summary_data: Dict[str, Any],
    items_data: Dict[str, Any],
    default_currency: str = "CAD",
) -> Dict[str, Any]:
    merged = {
        "vendor": summary_data.get("vendor"),
        "phone": summary_data.get("phone"),
        "address": summary_data.get("address"),
        "date": summary_data.get("date"),
        "items": items_data.get("items") or [],
        "subtotal": summary_data.get("subtotal"),
        "tax": summary_data.get("tax"),
        "total": summary_data.get("total"),
        "currency": summary_data.get("currency") or items_data.get("currency") or default_currency,
        "confidence": summary_data.get("confidence") or items_data.get("confidence"),
    }
    return normalize_receipt_json(merged, default_currency=default_currency)


# ============================================================
# Actual LLM calls
# ============================================================

def call_text_llm(
    *,
    ocr_text: str,
    config: Optional[LLMRouterConfig] = None,
) -> LLMCallResult:
    cfg = config or LLMRouterConfig()

    compact_ocr = compress_ocr_text_for_llm(
        ocr_text.strip(),
        max_lines=cfg.compress_ocr_max_lines,
        max_chars=cfg.compress_ocr_max_chars,
    )
    prompt = _inject_ocr_text(TEXT_PROMPT_TEMPLATE, compact_ocr)

    try:
        client = _build_gemini_client(cfg.api_key)

        prompt_tokens = count_prompt_tokens(
            client=client,
            model=cfg.text_model,
            prompt=prompt,
        )

        if cfg.free_tier_mode and prompt_tokens > cfg.text_max_prompt_tokens:
            return LLMCallResult(
                success=False,
                route_used="ocr_text",
                model_used=cfg.text_model,
                prompt_used=prompt,
                raw_response_text="",
                parsed_output={},
                error=f"Text prompt too large for free-tier mode: {prompt_tokens} tokens",
            )

        raw_text, usage = _gemini_generate_text(
            client=client,
            model=cfg.text_model,
            prompt=prompt,
            timeout_sec=cfg.timeout_sec,
            temperature=cfg.temperature,
            max_output_tokens=cfg.text_max_output_tokens,
            force_json_output=cfg.force_json_output,
            response_schema=ReceiptOutputSchema,
            retry_attempts=cfg.retry_attempts,
            retry_default_wait_sec=cfg.retry_default_wait_sec,
            retry_max_wait_sec=cfg.retry_max_wait_sec,
        )

        try:
            parsed = parse_and_normalize_llm_json(
                raw_text,
                default_currency=cfg.default_currency,
            )
            parsed["_usage"] = usage
            parsed["_compressed_ocr"] = compact_ocr
            parsed["_mode"] = "single_pass_full"

            return LLMCallResult(
                success=True,
                route_used="ocr_text",
                model_used=cfg.text_model,
                prompt_used=prompt,
                raw_response_text=raw_text,
                parsed_output=parsed,
                error=None,
            )
        except Exception as parse_err:
            if not cfg.enable_two_pass_fallback_for_text:
                return LLMCallResult(
                    success=False,
                    route_used="ocr_text",
                    model_used=cfg.text_model,
                    prompt_used=prompt,
                    raw_response_text=raw_text,
                    parsed_output={},
                    error=f"Parse failed: {parse_err}",
                )

            summary_result = _run_text_summary_pass(
                client=client,
                model=cfg.text_model,
                compact_ocr=compact_ocr,
                cfg=cfg,
            )
            items_result = _run_text_items_pass(
                client=client,
                model=cfg.text_model,
                compact_ocr=compact_ocr,
                cfg=cfg,
            )

            merged = merge_summary_and_items(
                summary_result["parsed"],
                items_result["parsed"],
                default_currency=cfg.default_currency,
            )
            merged["_usage"] = {
                "summary": summary_result["usage"],
                "items": items_result["usage"],
            }
            merged["_compressed_ocr"] = compact_ocr
            merged["_mode"] = "two_pass_text"
            merged["_full_raw_response_text"] = raw_text
            merged["_summary_raw_response_text"] = summary_result["raw_text"]
            merged["_items_raw_response_text"] = items_result["raw_text"]

            summary_prompt = _inject_ocr_text(TEXT_SUMMARY_PROMPT_TEMPLATE, compact_ocr)
            items_prompt = _inject_ocr_text(TEXT_ITEMS_PROMPT_TEMPLATE, compact_ocr)

            return LLMCallResult(
                success=True,
                route_used="ocr_text",
                model_used=cfg.text_model,
                prompt_used=summary_prompt + "\n\n---\n\n" + items_prompt,
                raw_response_text=json.dumps(
                    {
                        "summary_raw_text": summary_result["raw_text"],
                        "items_raw_text": items_result["raw_text"],
                    },
                    ensure_ascii=False,
                ),
                parsed_output=merged,
                error=f"Single-pass parse failed, used two-pass text fallback: {parse_err}",
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
        client = _build_gemini_client(cfg.api_key)

        prompt_tokens = count_prompt_tokens_with_image(
            client=client,
            model=cfg.vision_model,
            prompt=VISION_SUMMARY_PROMPT_TEMPLATE if cfg.use_two_pass_for_vision else VISION_PROMPT_TEMPLATE,
            image_path=img_path,
        )

        if cfg.free_tier_mode and prompt_tokens > cfg.vision_max_prompt_tokens:
            return LLMCallResult(
                success=False,
                route_used="vision_image",
                model_used=cfg.vision_model,
                prompt_used=VISION_PROMPT_TEMPLATE,
                raw_response_text="",
                parsed_output={},
                error=f"Vision prompt too large for free-tier mode: {prompt_tokens} tokens",
            )

        if cfg.use_two_pass_for_vision:
            summary_result = _run_vision_summary_pass(
                client=client,
                model=cfg.vision_model,
                image_path=img_path,
                cfg=cfg,
            )
            items_result = _run_vision_items_pass(
                client=client,
                model=cfg.vision_model,
                image_path=img_path,
                cfg=cfg,
            )

            merged = merge_summary_and_items(
                summary_result["parsed"],
                items_result["parsed"],
                default_currency=cfg.default_currency,
            )
            merged["_usage"] = {
                "summary": summary_result["usage"],
                "items": items_result["usage"],
            }
            merged["_mode"] = "two_pass_vision"
            merged["_summary_raw_response_text"] = summary_result["raw_text"]
            merged["_items_raw_response_text"] = items_result["raw_text"]

            return LLMCallResult(
                success=True,
                route_used="vision_image",
                model_used=cfg.vision_model,
                prompt_used=VISION_SUMMARY_PROMPT_TEMPLATE + "\n\n---\n\n" + VISION_ITEMS_PROMPT_TEMPLATE,
                raw_response_text=json.dumps(
                    {
                        "summary_raw_text": summary_result["raw_text"],
                        "items_raw_text": items_result["raw_text"],
                    },
                    ensure_ascii=False,
                ),
                parsed_output=merged,
                error=None,
            )

        raw_text, usage = _gemini_generate_vision(
            client=client,
            model=cfg.vision_model,
            prompt=VISION_PROMPT_TEMPLATE,
            image_path=img_path,
            timeout_sec=cfg.timeout_sec,
            temperature=cfg.temperature,
            max_output_tokens=cfg.vision_max_output_tokens,
            force_json_output=cfg.force_json_output,
            response_schema=ReceiptOutputSchema,
            retry_attempts=cfg.retry_attempts,
            retry_default_wait_sec=cfg.retry_default_wait_sec,
            retry_max_wait_sec=cfg.retry_max_wait_sec,
        )

        parsed = parse_and_normalize_llm_json(
            raw_text,
            default_currency=cfg.default_currency,
        )
        parsed["_usage"] = usage
        parsed["_mode"] = "single_pass_full"

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
# Two-pass helpers
# ============================================================

def _run_text_summary_pass(
    *,
    client: genai.Client,
    model: str,
    compact_ocr: str,
    cfg: LLMRouterConfig,
) -> Dict[str, Any]:
    prompt = _inject_ocr_text(TEXT_SUMMARY_PROMPT_TEMPLATE, compact_ocr)
    raw_text, usage = _gemini_generate_text(
        client=client,
        model=model,
        prompt=prompt,
        timeout_sec=cfg.timeout_sec,
        temperature=cfg.temperature,
        max_output_tokens=cfg.text_summary_max_output_tokens,
        force_json_output=cfg.force_json_output,
        response_schema=ReceiptSummarySchema,
        retry_attempts=cfg.retry_attempts,
        retry_default_wait_sec=cfg.retry_default_wait_sec,
        retry_max_wait_sec=cfg.retry_max_wait_sec,
    )
    parsed = parse_and_normalize_llm_json(raw_text, default_currency=cfg.default_currency)
    return {"prompt": prompt, "raw_text": raw_text, "usage": usage, "parsed": parsed}


def _run_text_items_pass(
    *,
    client: genai.Client,
    model: str,
    compact_ocr: str,
    cfg: LLMRouterConfig,
) -> Dict[str, Any]:
    prompt = _inject_ocr_text(TEXT_ITEMS_PROMPT_TEMPLATE, compact_ocr)
    raw_text, usage = _gemini_generate_text(
        client=client,
        model=model,
        prompt=prompt,
        timeout_sec=cfg.timeout_sec,
        temperature=cfg.temperature,
        max_output_tokens=cfg.text_items_max_output_tokens,
        force_json_output=cfg.force_json_output,
        response_schema=ReceiptItemsOnlySchema,
        retry_attempts=cfg.retry_attempts,
        retry_default_wait_sec=cfg.retry_default_wait_sec,
        retry_max_wait_sec=cfg.retry_max_wait_sec,
    )
    parsed = parse_and_normalize_items_only_json(raw_text, default_currency=cfg.default_currency)
    return {"prompt": prompt, "raw_text": raw_text, "usage": usage, "parsed": parsed}


def _run_vision_summary_pass(
    *,
    client: genai.Client,
    model: str,
    image_path: Path,
    cfg: LLMRouterConfig,
) -> Dict[str, Any]:
    raw_text, usage = _gemini_generate_vision(
        client=client,
        model=model,
        prompt=VISION_SUMMARY_PROMPT_TEMPLATE,
        image_path=image_path,
        timeout_sec=cfg.timeout_sec,
        temperature=cfg.temperature,
        max_output_tokens=cfg.vision_summary_max_output_tokens,
        force_json_output=cfg.force_json_output,
        response_schema=ReceiptSummarySchema,
        retry_attempts=cfg.retry_attempts,
        retry_default_wait_sec=cfg.retry_default_wait_sec,
        retry_max_wait_sec=cfg.retry_max_wait_sec,
    )
    parsed = parse_and_normalize_llm_json(raw_text, default_currency=cfg.default_currency)
    return {"prompt": VISION_SUMMARY_PROMPT_TEMPLATE, "raw_text": raw_text, "usage": usage, "parsed": parsed}


def _run_vision_items_pass(
    *,
    client: genai.Client,
    model: str,
    image_path: Path,
    cfg: LLMRouterConfig,
) -> Dict[str, Any]:
    raw_text, usage = _gemini_generate_vision(
        client=client,
        model=model,
        prompt=VISION_ITEMS_PROMPT_TEMPLATE,
        image_path=image_path,
        timeout_sec=cfg.timeout_sec,
        temperature=cfg.temperature,
        max_output_tokens=cfg.vision_items_max_output_tokens,
        force_json_output=cfg.force_json_output,
        response_schema=ReceiptItemsOnlySchema,
        retry_attempts=cfg.retry_attempts,
        retry_default_wait_sec=cfg.retry_default_wait_sec,
        retry_max_wait_sec=cfg.retry_max_wait_sec,
    )
    parsed = parse_and_normalize_items_only_json(raw_text, default_currency=cfg.default_currency)
    return {"prompt": VISION_ITEMS_PROMPT_TEMPLATE, "raw_text": raw_text, "usage": usage, "parsed": parsed}


# ============================================================
# Gemini SDK client
# ============================================================

def _build_gemini_client(api_key: str) -> genai.Client:
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def _build_generation_config(
    *,
    temperature: float,
    max_output_tokens: int,
    force_json_output: bool,
    response_schema: Optional[Any] = None,
) -> types.GenerateContentConfig:
    kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }

    if force_json_output:
        kwargs["response_mime_type"] = "application/json"

    if response_schema is not None:
        kwargs["response_schema"] = response_schema

    return types.GenerateContentConfig(**kwargs)


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = candidates[0].content.parts
            merged = []
            for p in parts:
                part_text = getattr(p, "text", None)
                if part_text:
                    merged.append(part_text)
            if merged:
                return "\n".join(merged)
    except Exception:
        pass

    return ""


def count_prompt_tokens(
    *,
    client: genai.Client,
    model: str,
    prompt: str,
) -> int:
    try:
        resp = client.models.count_tokens(
            model=model,
            contents=prompt,
        )
        return int(getattr(resp, "total_tokens", 0) or 0)
    except Exception:
        return 0


def count_prompt_tokens_with_image(
    *,
    client: genai.Client,
    model: str,
    prompt: str,
    image_path: Path,
) -> int:
    try:
        with Image.open(image_path) as img:
            resp = client.models.count_tokens(
                model=model,
                contents=[prompt, img],
            )
        return int(getattr(resp, "total_tokens", 0) or 0)
    except Exception:
        return 0


def _gemini_generate_text(
    *,
    client: genai.Client,
    model: str,
    prompt: str,
    timeout_sec: int,
    temperature: float,
    max_output_tokens: int,
    force_json_output: bool,
    response_schema: Optional[Any] = None,
    retry_attempts: int = 3,
    retry_default_wait_sec: float = 8.0,
    retry_max_wait_sec: float = 20.0,
):
    config = _build_generation_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        force_json_output=force_json_output,
        response_schema=response_schema,
    )

    def _do_call():
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

    response = _call_with_retry(
        _do_call,
        attempts=retry_attempts,
        default_wait_sec=retry_default_wait_sec,
        max_wait_sec=retry_max_wait_sec,
    )

    usage = _extract_usage(response)
    return _extract_response_text(response), usage


def _gemini_generate_vision(
    *,
    client: genai.Client,
    model: str,
    prompt: str,
    image_path: Path,
    timeout_sec: int,
    temperature: float,
    max_output_tokens: int,
    force_json_output: bool,
    response_schema: Optional[Any] = None,
    retry_attempts: int = 3,
    retry_default_wait_sec: float = 8.0,
    retry_max_wait_sec: float = 20.0,
):
    config = _build_generation_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        force_json_output=force_json_output,
        response_schema=response_schema,
    )

    def _do_call():
        with Image.open(image_path) as img:
            return client.models.generate_content(
                model=model,
                contents=[prompt, img],
                config=config,
            )

    response = _call_with_retry(
        _do_call,
        attempts=retry_attempts,
        default_wait_sec=retry_default_wait_sec,
        max_wait_sec=retry_max_wait_sec,
    )

    usage = _extract_usage(response)
    return _extract_response_text(response), usage


def _extract_usage(response: Any) -> Dict[str, Any]:
    usage = {}
    meta = getattr(response, "usage_metadata", None)
    if meta:
        usage = {
            "prompt_token_count": getattr(meta, "prompt_token_count", None),
            "candidates_token_count": getattr(meta, "candidates_token_count", None),
            "total_token_count": getattr(meta, "total_token_count", None),
            "thoughts_token_count": getattr(meta, "thoughts_token_count", None),
        }
    return usage


# ============================================================
# Parsing + normalization
# ============================================================

def parse_and_normalize_llm_json(
    raw_text: str,
    default_currency: str = "CAD",
) -> Dict[str, Any]:
    candidate = extract_json_block(raw_text)
    parsed = json.loads(candidate)
    return normalize_receipt_json(parsed, default_currency=default_currency)


def parse_and_normalize_items_only_json(
    raw_text: str,
    default_currency: str = "CAD",
) -> Dict[str, Any]:
    candidate = extract_json_block(raw_text)
    parsed = json.loads(candidate)
    return normalize_items_only_json(parsed, default_currency=default_currency)


def extract_json_block(text: str) -> str:
    text = (text or "").strip()

    if not text:
        raise ValueError("Empty LLM response")

    m = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    raise ValueError(f"Could not extract JSON object from LLM response. Raw text: {text[:500]}")


def normalize_items_only_json(
    data: Dict[str, Any],
    default_currency: str = "CAD",
) -> Dict[str, Any]:
    items_in = data.get("items") or []

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

    return {
        "items": normalized_items,
        "currency": normalize_currency(data.get("currency"), default=default_currency),
        "confidence": parse_confidence_or_none(data.get("confidence")),
    }


def normalize_receipt_json(
    data: Dict[str, Any],
    default_currency: str = "CAD",
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
        "ZAR": "ZAR",
        "R": "ZAR",
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
    s = re.sub(r"[^\d\.\-]", "", s)

    if s in {"", ".", "-", "-.", ".-"}:
        return None

    try:
        return round(float(s), 2)
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


# ============================================================
# Validation helpers
# ============================================================

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

    cfg = LLMRouterConfig(
        text_model="gemini-2.5-flash-lite",
        vision_model="gemini-2.5-flash-lite",
        free_tier_mode=True,
        use_two_pass_for_vision=True,
        enable_two_pass_fallback_for_text=True,
        retry_attempts=3,
        retry_default_wait_sec=8.0,
        retry_max_wait_sec=20.0,
    )

    out = run_auto_fix_for_receipt(
        ocr_text=sample_ocr,
        image_path=None,
        ocr_confidence=0.92,
        config=cfg,
        force_route="ocr_text",
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))