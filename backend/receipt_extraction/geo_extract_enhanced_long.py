from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
import re

try:
    from .geo_extract_enhanced_v4 import (
        safe_parse_price,
        group_tokens_by_label_sequence,
        extract_menu_items_adaptive,
        detect_item_pattern,
    )
except ImportError:
    from geo_extract_enhanced_v4 import (
        safe_parse_price,
        group_tokens_by_label_sequence,
        extract_menu_items_adaptive,
        detect_item_pattern,
    )


# ============================================================
# HELPERS
# ============================================================
def _safe_text(x: Any) -> str:
    return str(x or "").strip()

def _norm_upper(x: Any) -> str:
    return _safe_text(x).upper()

def _alpha_ratio(text: str) -> float:
    t = _safe_text(text)
    if not t:
        return 0.0
    alpha = sum(ch.isalpha() for ch in t)
    return alpha / max(1, len(t))

def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0

def _dedupe_float_values(vals: List[float], eps: float = 0.005) -> List[float]:
    out: List[float] = []
    for v in vals:
        if not any(abs(v - x) < eps for x in out):
            out.append(float(v))
    return out

def _contains_any(text: str, words: List[str]) -> bool:
    u = _norm_upper(text)
    return any(w in u for w in words)


# ============================================================
# KEYWORDS / REGEX
# ============================================================
LONG_SECTION_KEYWORDS = {
    "GROCERY", "PRODUCE", "MEATS", "MEAT", "BAKERY", "DELI",
    "FROZEN", "DAIRY", "SEAFOOD", "PHARMACY", "HOME", "HEALTH",
    "BEAUTY", "FLORAL", "INSTORE", "NATURAL", "ORGANIC"
}

LONG_SUMMARY_KEYWORDS = {
    "SUBTOTAL", "TOTAL", "TAX", "GST", "PST", "HST", "QST",
    "DEBIT", "PURCHASE", "INTERAC", "CHANGE", "TENDERED",
    "CUSTOMER COPY", "VERIFIED BY PIN", "CARD TYPE", "REF#",
    "ACCOUNT", "CHEQUING", "SAVINGS", "STORE MANAGER",
    "TRANS. TYPE", "AUTH", "DATE/TIME", "DATETIME"
}

LONG_PROMO_KEYWORDS = {
    "PC OPTIMUM", "OPTIMUM", "APPLE TV", "APPLE MUSIC", "APPLE FITNESS",
    "SURVEY", "FREE TRIAL", "MEMBERS GET", "THANK YOU", "POINTS",
    "CODE:", "CUSTOMER COPY", "RETAIN THIS COPY", "VALIDATION",
    "STOREOPINION", "FULL CONTEST RULES", "OFFER VALID", "NEW YEAR",
    "EARN", "CLAIM YOUR FREE TRIAL"
}

# Module-level summary words — used by block builder and flush
# FIX #11: moved from local scope to module level to avoid fragile closure
BLOCK_SUMMARY_WORDS = {
    "SUBTOTAL", "TOTAL", "TAX", "GST", "PST", "HST", "QST",
    "PURCHASE", "DEBIT", "INTERAC", "CHANGE", "TENDERED",
    "AMOUNT", "BALANCE", "CLOSING", "CREDIT",
}

WEAK_ITEM_NAMES = {
    "EA", "KG", "LB", "LBS", "IS", "NT", "PT", "RP",
    "MRJ", "HRJ", "ARCP", "DS", "INT", "UNIT", "PK"
}

KNOWN_JUNK_TOKENS = {
    "GPHRJ", "GPRQ", "GPRO", "GPHR)", "HRJ", "MRJ", "NRJ", "PRJ",
    "GPHRJ.", "HRJ.", "MRJ.", "GPRO)", "GPRD", "GPRQ)", "GPHRD", "GPMRJ"
}

UNIT_PATTERNS_RE = re.compile(r"\b(?:kg|lb|lbs|ea|each|/kg|/lb)\b", re.I)
WEIGHT_ONLY_RE_1 = re.compile(r"^\d+(?:[.,]\d+)?\s*(kg|lb|lbs|g|gm|ea|each)$", re.I)
WEIGHT_ONLY_RE_2 = re.compile(r"^\d+(kg|lb|lbs|g|gm|ea)$", re.I)
UNIT_ONLY_RE = re.compile(r"^(kg|lb|lbs|ea|each|pk|unit|int)$", re.I)

MONEY_RE = re.compile(r"[-+]?\$?\d+(?:[.,]\d{2})")
UNIT_PRICE_RE = re.compile(r"\$\s*\d+(?:[.,]\d{2})\s*/\s*(kg|lb|lbs)", re.I)
WEIGHT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(kg|lb|lbs)\b", re.I)

PAYMENT_LINE_RE = re.compile(
    r"\b(debit|purchase|interac|verified|pin|chequing|account|ref#|auth|card|mastercard|visa|credit)\b",
    re.I,
)
PROMO_LINE_RE = re.compile(
    r"\b(optimum|survey|apple|trial|members|get up to|points|code:|offer valid|new year|earn)\b",
    re.I,
)
SUMMARY_LINE_RE = re.compile(
    r"\b(subtotal|total|tax|gst|pst|hst|qst|amount|tendered|change)\b",
    re.I,
)
LONG_TAX_RE = re.compile(
    r"\b(TAX|GST|PST|HST|QST|VAT|G5T|P5T|H5T|Q5T|6ST|PSD|HSD|QSD)\b", re.I
)
LONG_DISCOUNT_RE = re.compile(
    r"\b(SAVE|SAVINGS|DISCOUNT|COUPON|PROMO|ARCP|POINTS REDEEMED)\b", re.I
)
LONG_TOTAL_RE = re.compile(r"\bTOTAL\b", re.I)
LONG_SUBTOTAL_RE = re.compile(r"\bSUBTOTAL\b", re.I)
SECTION_CODE_RE = re.compile(r"^\d{1,2}\s*-\s*[A-Z]+", re.I)
UPC_RE = re.compile(r"^\d{5,}$")
RATE_ONLY_RE = re.compile(r"^(5|6|7|13|15)(?:[.,]0+)?$")


# ============================================================
# SECTION / FOOTER / WEAK NAME CHECKS
# ============================================================
def _looks_like_section_header(text: str) -> bool:
    t = _norm_upper(text)
    if not t:
        return False
    if t in LONG_SECTION_KEYWORDS:
        return True
    if SECTION_CODE_RE.match(t):
        return True
    for kw in LONG_SECTION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return True
    if re.search(
        r"\b(HOME|HEALTH|BEAUTY|FROZEN|DAIRY|PHARMACY|BAKERY|PRODUCE|GROCERY|MEAT|MEATS)\b", t
    ):
        if len(t.split()) <= 4:
            return True
    return False


def _looks_like_footer_or_payment(text: str) -> bool:
    t = _safe_text(text)
    tu = t.upper()
    if any(k in tu for k in LONG_PROMO_KEYWORDS):
        return True
    if any(
        k in tu for k in LONG_SUMMARY_KEYWORDS
        if k not in {"SUBTOTAL", "TOTAL", "TAX", "GST", "PST", "HST", "QST"}
    ):
        return True
    if PAYMENT_LINE_RE.search(t):
        return True
    if PROMO_LINE_RE.search(t):
        return True
    return False


def _looks_like_summary_anchor(text: str) -> bool:
    return bool(SUMMARY_LINE_RE.search(_safe_text(text)))


def _looks_like_tax_rate_only(text: str, val: Optional[float]) -> bool:
    if val is None:
        return False
    t = _safe_text(text)
    tu = t.upper()
    if "%" in t:
        return True
    if RATE_ONLY_RE.fullmatch(t.replace("$", "").strip()):
        return True
    if abs(float(val) - round(float(val))) < 1e-6 and float(val) in {5.0, 6.0, 7.0, 13.0, 15.0}:
        return True
    if re.search(r"(5\.?0+%|6\.?0+%|7\.?0+%|13\.?0+%|15\.?0+%)", tu):
        return True
    return False


# FIX #6 + #7: tightened digit rule and vowel check scoped to single-word tokens
def _looks_like_weak_name(text: str) -> bool:
    t = _safe_text(text)
    u = _norm_upper(text)

    if not u:
        return True
    if u in WEAK_ITEM_NAMES:
        return True
    if u in KNOWN_JUNK_TOKENS:
        return True
    if _looks_like_section_header(u):
        return True
    if _looks_like_footer_or_payment(u):
        return True
    if t.startswith("$"):
        return True
    if UNIT_PATTERNS_RE.fullmatch(t.lower()):
        return True
    if UNIT_ONLY_RE.fullmatch(t):
        return True
    if WEIGHT_ONLY_RE_1.fullmatch(t):
        return True
    if WEIGHT_ONLY_RE_2.fullmatch(u):
        return True
    if re.fullmatch(r"(INT|EA|EACH|PK)\s*\d*", u):
        return True
    if re.fullmatch(r"\d+\s*(EA|EACH|PK)", u):
        return True
    if UPC_RE.fullmatch(t):
        return True

    parts = [p for p in re.split(r"\s+", u) if p]
    parts_nonjunk = [
        p for p in parts
        if p not in WEAK_ITEM_NAMES
        and p not in KNOWN_JUNK_TOKENS
        and not re.fullmatch(r"[\d\W]+", p or "")
    ]
    if len(parts_nonjunk) >= 2:
        alpha_total = sum(ch.isalpha() for ch in " ".join(parts_nonjunk))
        if alpha_total >= 5:
            if not any(
                p in {"EA", "INT", "EACH", "KG", "LB", "LBS", "PK", "UNIT"}
                for p in parts_nonjunk
            ):
                return False

    if _alpha_ratio(u) < 0.20:
        return True

    alpha_count = sum(ch.isalpha() for ch in t)
    if alpha_count < 3:
        return True
    if len(u) <= 2 and u not in {"XL", "BB"}:
        return True

    digit_count = sum(ch.isdigit() for ch in t)
    if digit_count >= alpha_count and digit_count >= 2:
        return True

    if re.search(r"\d", t):
        punct_count = sum(not ch.isalnum() and not ch.isspace() for ch in t)
        if digit_count >= 2 and punct_count >= 1 and alpha_count <= 8:
            return True

    weak_parts = {
        "EA", "KG", "LB", "LBS", "PT", "NT", "IS", "RP",
        "MRJ", "HRJ", "DS", "INT", "PK"
    }
    if parts and all(
        p in weak_parts or re.fullmatch(r"[\d\W]+", p or "") for p in parts
    ):
        return True

    if any(part in {"EA", "INT", "EACH"} for part in parts) and alpha_count <= 4:
        return True

    # FIX #6: was `if re.search(r"\d", t) and alpha_count <= 6` — too aggressive.
    # Now requires digit_count >= alpha_count so pure abbreviations like
    # "COM BDS", "DR SMITHS" are not dropped.
    if re.search(r"\d", t) and digit_count >= alpha_count and alpha_count <= 6:
        return True

    letters_only = re.sub(r"[^A-Z]", "", u)

    # FIX #7: vowel check now only applies to single-word tokens (no spaces).
    # Multi-word abbreviated names like "COM BDS", "DR SMITHS" are legitimate.
    if len(letters_only) >= 4 and " " not in u.strip():
        vowel_count = sum(ch in "AEIOU" for ch in letters_only)
        if vowel_count == 0:
            return True

    if len(letters_only) >= 5 and len(parts_nonjunk) <= 1 and " " not in u.strip():
        vowel_ratio = sum(ch in "AEIOU" for ch in letters_only) / len(letters_only)
        if vowel_ratio < 0.15:
            return True

    return False


# ============================================================
# MONEY / ROW PARSING HELPERS
# ============================================================
def _extract_money_spans(text: str) -> List[Tuple[int, int, float, str]]:
    out: List[Tuple[int, int, float, str]] = []
    for m in re.finditer(r"[-+]?\$?\d+(?:[.,]\d{2})", text):
        raw = m.group(0)
        val = safe_parse_price(raw, 0.0, 2_000_000.0)
        if val is not None:
            out.append((m.start(), m.end(), float(val), raw))
    return out


def _choose_rightmost_nonrate_money(text: str) -> Optional[float]:
    spans = _extract_money_spans(text)
    candidates: List[float] = []
    for start, end, val, raw in spans:
        tail = text[end:end + 3]
        if "%" in tail:
            continue
        if _looks_like_tax_rate_only(raw, val):
            continue
        candidates.append(float(val))
    return candidates[-1] if candidates else None


def _good_name_from_row_text(text: str) -> Optional[str]:
    t = _safe_text(text)
    if not t:
        return None
    t = re.sub(r"^\d{1,2}\s*-\s*[A-Z]+\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^\d{5,}\s+", "", t).strip()
    t = re.sub(
        r"\b(MRJ|HRJ|NRJ|GRJ|PRJ|GPHRJ|GPRQ|GPRO|GPMRJ)\b", "", t, flags=re.I
    ).strip()
    t = re.sub(r"[-+]?\$?\d+(?:[.,]\d{2})", "", t).strip()
    t = re.sub(
        r"\b\d+(?:[.,]\d+)?\s*(kg|lb|lbs|ea|each)\b", "", t, flags=re.I
    ).strip()
    t = re.sub(
        r"\b(GROSS|NET|TARE|INT|EA|KG|LB|LBS|PT|NT|RP|IS|UNIT|PK)\b",
        "", t, flags=re.I
    ).strip()
    t = re.sub(r"\s+", " ", t).strip(" -:;,.")
    if not t:
        return None
    if _looks_like_weak_name(t):
        return None
    return t


def _is_junk_only_row(row: Dict[str, Any]) -> bool:
    text = _safe_text(row.get("text"))
    if not text:
        return True
    stripped = re.sub(r"[-+]?\$?\d+(?:[.,]\d{2,})", "", text).strip()
    stripped = re.sub(r"\b\d+(?:[.,]\d+)?\s*%", "", stripped).strip()
    for junk in KNOWN_JUNK_TOKENS:
        stripped = re.sub(rf"\b{re.escape(junk)}\b", "", stripped, flags=re.I)
    stripped = stripped.strip(" -:;,./")
    return len(stripped) == 0 or _looks_like_weak_name(stripped)


def _build_rows_from_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    words   = example.get("words",   []) or []
    labels  = example.get("labels",  []) or []
    confs   = example.get("confs",   []) or []
    bboxes  = example.get("bboxes",  []) or []
    row_ids = example.get("row_ids", []) or list(range(len(words)))

    n = min(len(words), len(labels), len(confs), len(bboxes), len(row_ids))
    rows_map: Dict[int, Dict[str, Any]] = {}

    for i in range(n):
        rid = int(row_ids[i])
        if rid not in rows_map:
            rows_map[rid] = {
                "row_id": rid,
                "words": [], "labels": [], "confs": [], "bboxes": [],
                "text": "",
                "x_left": 9999, "x_right": -1,
                "y_top": 9999,  "y_bottom": -1,
                "money_tokens": [], "all_prices": [], "money_spans": [],
                "unit_prices": [], "final_prices": [],
                "has_weight": False, "has_unit_price": False,
                "has_summary_anchor": False, "has_footer_or_payment": False,
                "is_section_header": False,
            }
        row = rows_map[rid]
        w = _safe_text(words[i])
        row["words"].append(w)
        row["labels"].append(_safe_text(labels[i]))
        row["confs"].append(float(confs[i] or 0.0))
        row["bboxes"].append(bboxes[i])
        bb = bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]
        if len(bb) >= 4:
            row["x_left"]  = min(row["x_left"],  int(bb[0]))
            row["x_right"] = max(row["x_right"], int(bb[2]))
            row["y_top"]   = min(row["y_top"],   int(bb[1]))
            row["y_bottom"]= max(row["y_bottom"],int(bb[3]))

    rows = [rows_map[k] for k in sorted(rows_map.keys())]

    for row in rows:
        row["text"] = " ".join(w for w in row["words"] if w).strip()
        text = row["text"]

        row["has_weight"]            = bool(WEIGHT_RE.search(text))
        row["has_unit_price"]        = bool(UNIT_PRICE_RE.search(text))
        row["has_summary_anchor"]    = _looks_like_summary_anchor(text)
        row["has_footer_or_payment"] = _looks_like_footer_or_payment(text)
        row["is_section_header"]     = _looks_like_section_header(text.upper())

        money_spans = _extract_money_spans(text)
        row["money_spans"]  = money_spans
        row["money_tokens"] = [raw for _, _, _, raw in money_spans]
        row["all_prices"]   = [val for _, _, val, _ in money_spans]

        unit_prices: List[float] = []
        if row["has_unit_price"]:
            for m in re.finditer(r"\$\s*\d+(?:[.,]\d{2})\s*/\s*(kg|lb|lbs)", text, re.I):
                unit_str = m.group(0).split("/")[0].strip()
                unit_val = safe_parse_price(unit_str, 0.0, 2_000_000.0)
                if unit_val is not None:
                    unit_prices.append(float(unit_val))

        final_prices: List[float] = []
        for _, _, val, raw in money_spans:
            if any(abs(val - up) < 0.005 for up in unit_prices):
                continue
            if _looks_like_tax_rate_only(raw, val):
                continue
            final_prices.append(float(val))

        row["unit_prices"]  = unit_prices
        row["final_prices"] = final_prices

    return rows


# ============================================================
# RECEIPT-LINE GROUPING BY Y-COORDINATE
# ============================================================
def _group_rows_into_receipt_lines(
    rows: List[Dict[str, Any]],
    y_tolerance: int = 15,          # FIX #9: raised from 10 → 15
) -> List[List[int]]:
    """
    Groups row indices into physical receipt lines by y_top proximity.
    Rows within y_tolerance pixels of each other = same physical line.
    """
    if not rows:
        return []

    indexed = sorted(enumerate(rows), key=lambda x: x[1].get("y_top", 0))
    lines: List[List[int]] = []
    current_line: List[int] = [indexed[0][0]]
    current_y = rows[indexed[0][0]].get("y_top", 0)

    for orig_idx, row in indexed[1:]:
        row_y = row.get("y_top", 0)
        if abs(row_y - current_y) <= y_tolerance:
            current_line.append(orig_idx)
        else:
            lines.append(sorted(current_line))
            current_line = [orig_idx]
            current_y = row_y

    if current_line:
        lines.append(sorted(current_line))

    return lines


# ============================================================
# VERSION SENTINEL
# ============================================================
_LONG_EXTRACTOR_VERSION = "sequential_v4"


# ============================================================
# SEQUENTIAL NAME↔PRICE PAIRING
# ============================================================
def _resolve_all_items_by_receipt_order(
    rows: List[Dict[str, Any]],
    used_rows: Set[int],
    cfg,
) -> List[Dict[str, Any]]:
    """
    Sequential name<->price pairing in receipt order. v4.
    FIX #5: pending name is only replaced if new candidate has more
    alpha characters — prevents silent discard of valid pending names.
    """
    if cfg.debug_print:
        print(f"  [SEQ v4] Running on {len(rows)} rows")

    items: List[Dict[str, Any]] = []
    n = len(rows)

    def _prices(row: Dict[str, Any]) -> List[float]:
        out = []
        for v in (row.get("final_prices") or []):
            fv = float(v)
            if row.get("has_weight") and 0 < fv < 3.5:
                continue
            out.append(fv)
        return out

    def _name(text: str) -> Optional[str]:
        t = _good_name_from_row_text(text)
        return t if (t and not _looks_like_weak_name(t)) else None

    def _skippable(row: Dict[str, Any]) -> bool:
        return bool(
            row.get("is_section_header")
            or row.get("has_footer_or_payment")
            or row.get("has_summary_anchor")
        )

    def _multi_split(row: Dict[str, Any], ridx: int) -> List[Dict[str, Any]]:
        prices = _prices(row)
        if len(prices) < 2:
            return []

        words_u = [_norm_upper(w) for w in (row.get("words") or [])]
        segments: List[str] = []
        cur: List[str] = []

        for w in words_u:
            is_boundary = (
                w in KNOWN_JUNK_TOKENS
                or w in WEAK_ITEM_NAMES
                or bool(re.fullmatch(r"[\d.,\$%\-\+/\\]+", w))
                or _looks_like_section_header(w)
                or bool(UPC_RE.fullmatch(w))
            )
            if is_boundary:
                if cur:
                    segments.append(" ".join(cur))
                    cur = []
            else:
                cur.append(w)
        if cur:
            segments.append(" ".join(cur))

        clean: List[str] = []
        for seg in segments:
            nm = _good_name_from_row_text(seg)
            if nm and not _looks_like_weak_name(nm):
                clean.append(nm)

        if len(clean) >= 2 and len(prices) >= len(clean):
            result = []
            for nm, pr in zip(clean, prices):
                result.append({
                    "name": nm,
                    "price": float(pr),
                    "price_confidence": _mean(row.get("confs", [])),
                    "source": "long_row_seq_multi_split",
                })
            used_rows.add(ridx)
            if cfg.debug_print:
                for it in result:
                    print(
                        f"  [SEQ v4 MULTI-SPLIT] '{it['name']}'"
                        f" → ${it['price']:.2f}"
                    )
            return result

        return []

    pending_name: Optional[str] = None
    pending_ridx: Optional[int] = None

    i = 0
    while i < n:
        if i in used_rows:
            i += 1
            continue

        row  = rows[i]
        text = _safe_text(row.get("text", ""))

        if _skippable(row):
            if cfg.debug_print:
                print(f"  [SEQ v4 SKIP] row={i} '{text[:40]}'")
            i += 1
            continue

        prices  = _prices(row)
        nm      = _name(text)
        is_junk = _is_junk_only_row(row)

        if cfg.debug_print:
            print(
                f"  [SEQ v4 ROW {i:02d}] "
                f"nm={nm!r:25s} "
                f"prices={[round(p,2) for p in prices]} "
                f"junk={is_junk}"
            )

        # E: multi-name + multi-price
        split = _multi_split(row, i)
        if split:
            items.extend(split)
            pending_name = None
            pending_ridx = None
            i += 1
            continue

        # A/B: name + price on same row
        if nm and prices and not row.get("has_unit_price"):
            next_price: Optional[float] = None
            next_ridx:  Optional[int]  = None

            j = i + 1
            while j < n and j in used_rows:
                j += 1

            if j < n:
                nxt = rows[j]
                if not _skippable(nxt) and _is_junk_only_row(nxt):
                    nxt_p = _prices(nxt)
                    if nxt_p and abs(nxt_p[-1] - prices[-1]) > 0.005:
                        next_price = nxt_p[-1]
                        next_ridx  = j

            chosen     = next_price if next_price is not None else prices[-1]
            price_ridx = next_ridx  if next_ridx  is not None else i

            items.append({
                "name": nm,
                "price": float(chosen),
                "price_confidence": _mean(rows[price_ridx].get("confs", [])),
                "source": "long_row_seq_direct",
            })
            used_rows.add(i)
            if next_ridx is not None:
                used_rows.add(next_ridx)

            if cfg.debug_print:
                tag = " (next-row-price)" if next_price is not None else ""
                print(f"  [SEQ v4 DIRECT{tag}] '{nm}' → ${chosen:.2f}")

            pending_name = None
            pending_ridx = None
            i = (next_ridx + 1) if next_ridx is not None else (i + 1)
            continue

        # C: name-only → pending
        # FIX #5: only replace pending if new name has more alpha chars
        if nm and not prices:
            if pending_name is None:
                pending_name = nm
                pending_ridx = i
                used_rows.add(i)
                if cfg.debug_print:
                    print(f"  [SEQ v4 PENDING] '{nm}' row={i}")
            else:
                old_alpha = sum(c.isalpha() for c in pending_name)
                new_alpha = sum(c.isalpha() for c in nm)
                if new_alpha > old_alpha:
                    if cfg.debug_print:
                        print(
                            f"  [SEQ v4] Replacing pending '{pending_name}'"
                            f" with better '{nm}' (+{new_alpha - old_alpha} alpha)"
                        )
                    pending_name = nm
                    pending_ridx = i
                    used_rows.add(i)
                else:
                    if cfg.debug_print:
                        print(
                            f"  [SEQ v4] Keeping pending '{pending_name}'"
                            f" over '{nm}' (not better)"
                        )
            i += 1
            continue

        # D: junk-only price row
        if is_junk and prices:
            if pending_name is not None:
                items.append({
                    "name": pending_name,
                    "price": float(prices[-1]),
                    "price_confidence": _mean(row.get("confs", [])),
                    "source": "long_row_seq_junk",
                })
                used_rows.add(i)
                if cfg.debug_print:
                    print(
                        f"  [SEQ v4 JUNK→NAME] '{pending_name}'"
                        f" → ${prices[-1]:.2f} row={i}"
                    )
                pending_name = None
                pending_ridx = None
            else:
                if cfg.debug_print:
                    print(
                        f"  [SEQ v4 ORPHAN] ${prices[-1]:.2f}"
                        f" row={i} (no pending)"
                    )
            i += 1
            continue

        # F: price-only non-junk with pending name
        if prices and not nm and pending_name is not None:
            items.append({
                "name": pending_name,
                "price": float(prices[-1]),
                "price_confidence": _mean(row.get("confs", [])),
                "source": "long_row_seq_price",
            })
            used_rows.add(i)
            if cfg.debug_print:
                print(
                    f"  [SEQ v4 PRICE→NAME] '{pending_name}'"
                    f" → ${prices[-1]:.2f} row={i}"
                )
            pending_name = None
            pending_ridx = None
            i += 1
            continue

        i += 1

    if cfg.debug_print:
        print(f"  [SEQ v4] Finished. {len(items)} items extracted.")

    return items


# ============================================================
# LAYOUT-AWARE JUNK PAIRING
# ============================================================
def _resolve_junk_price_rows_by_layout(
    rows: List[Dict[str, Any]],
    used_rows: Set[int],
    cfg,
) -> List[Dict[str, Any]]:
    """
    Pair junk-only price rows with name rows using physical receipt layout.
    Unchanged logic — only called correctly now via
    _extract_weighted_or_direct_items_from_rows (FIX #1).
    """
    pre_items: List[Dict[str, Any]] = []
    receipt_lines = _group_rows_into_receipt_lines(rows, y_tolerance=15)

    if cfg.debug_print:
        print(f"\n[LAYOUT] {len(receipt_lines)} physical receipt lines detected")

    def _line_names(line_ridxs: List[int]) -> List[Tuple[int, str]]:
        result = []
        for ridx in line_ridxs:
            if ridx in used_rows:
                continue
            row = rows[ridx]
            if (
                row.get("is_section_header")
                or row.get("has_footer_or_payment")
                or row.get("has_summary_anchor")
            ):
                continue
            nm = _good_name_from_row_text(row.get("text", ""))
            if nm and not _looks_like_weak_name(nm):
                result.append((ridx, nm))
        return result

    def _line_junk_prices(line_ridxs: List[int]) -> List[Tuple[int, float]]:
        result = []
        for ridx in line_ridxs:
            if ridx in used_rows:
                continue
            row = rows[ridx]
            if not _is_junk_only_row(row):
                continue
            if (
                row.get("is_section_header")
                or row.get("has_footer_or_payment")
                or row.get("has_summary_anchor")
            ):
                continue
            for p in (row.get("final_prices") or []):
                result.append((ridx, float(p)))
        return result

    def _line_has_own_price(line_ridxs: List[int]) -> bool:
        for ridx in line_ridxs:
            row = rows[ridx]
            if _is_junk_only_row(row):
                continue
            if row.get("final_prices"):
                return True
        return False

    for line_idx, line_ridxs in enumerate(receipt_lines):
        junk_prices = _line_junk_prices(line_ridxs)
        if not junk_prices:
            continue

        same_line_names = _line_names(line_ridxs)
        if same_line_names:
            for (price_ridx, price_val), (name_ridx, nm) in zip(
                junk_prices, same_line_names
            ):
                pre_items.append({
                    "name": nm,
                    "price": price_val,
                    "price_confidence": _mean(rows[price_ridx].get("confs", [])),
                    "source": "long_row_layout_same_line",
                })
                used_rows.add(price_ridx)
                used_rows.add(name_ridx)
                if cfg.debug_print:
                    print(
                        f"  [LAYOUT SAME-LINE] line[{line_idx}] "
                        f"'{nm}' → ${price_val:.2f}"
                    )
            continue

        name_found = False
        if line_idx > 0:
            prev_line_ridxs = receipt_lines[line_idx - 1]
            if not _line_has_own_price(prev_line_ridxs):
                prev_names = _line_names(prev_line_ridxs)
                if prev_names:
                    for (price_ridx, price_val), (name_ridx, nm) in zip(
                        junk_prices, prev_names
                    ):
                        pre_items.append({
                            "name": nm,
                            "price": price_val,
                            "price_confidence": _mean(rows[price_ridx].get("confs", [])),
                            "source": "long_row_layout_prev_line",
                        })
                        used_rows.add(price_ridx)
                        used_rows.add(name_ridx)
                        if cfg.debug_print:
                            print(
                                f"  [LAYOUT PREV-LINE] line[{line_idx}] "
                                f"'{nm}' → ${price_val:.2f} "
                                f"(name from line[{line_idx-1}])"
                            )
                    name_found = True

        if name_found:
            continue

        if line_idx + 1 < len(receipt_lines):
            next_line_ridxs = receipt_lines[line_idx + 1]
            if not _line_has_own_price(next_line_ridxs):
                next_names = _line_names(next_line_ridxs)
                if next_names:
                    for (price_ridx, price_val), (name_ridx, nm) in zip(
                        junk_prices, next_names
                    ):
                        pre_items.append({
                            "name": nm,
                            "price": price_val,
                            "price_confidence": _mean(rows[price_ridx].get("confs", [])),
                            "source": "long_row_layout_next_line",
                        })
                        used_rows.add(price_ridx)
                        used_rows.add(name_ridx)
                        if cfg.debug_print:
                            print(
                                f"  [LAYOUT NEXT-LINE] line[{line_idx}] "
                                f"'{nm}' → ${price_val:.2f} "
                                f"(name from line[{line_idx+1}])"
                            )

    return pre_items


# FIX #1: layout pass now runs as pre-pass before sequential scan
def _extract_weighted_or_direct_items_from_rows(
    rows: List[Dict[str, Any]],
    cfg,
) -> List[Dict[str, Any]]:
    """
    Entry point for row-based item extraction.
    FIX #1: runs layout pre-pass first, then sequential scan on remaining rows.
    """
    if cfg.debug_print:
        print(f"\n[ROW EXTRACTOR] version={_LONG_EXTRACTOR_VERSION}")

    used_rows: Set[int] = set()

    # Pre-pass: layout-aware junk pairing (RCS-style split name/price lines)
    pre_items = _resolve_junk_price_rows_by_layout(rows, used_rows, cfg)

    # Sequential pass: handles everything not already claimed
    seq_items = _resolve_all_items_by_receipt_order(rows, used_rows, cfg)

    return pre_items + seq_items


# ============================================================
# SUMMARY EXTRACTION
# ============================================================
def _extract_summary_items_long(
    grouped_items: List[Dict[str, Any]],
    cfg,
) -> Dict[str, Any]:
    summary = {
        "subtotal": None, "subtotal_tokens": [], "subtotal_confidence": 0.0,
        "tax":      None, "tax_tokens":      [], "tax_confidence":      0.0,
        "total":    None, "total_tokens":    [], "total_confidence":    0.0,
        "discount": None, "discount_tokens": [], "discount_confidence": 0.0,
    }
    tax_components: List[float] = []
    tax_token_bank: List[Any]   = []

    if cfg.debug_print:
        print(f"\n[LONG SUMMARY EXTRACTION]")

    n = len(grouped_items)

    def get_text(i):   return _safe_text(grouped_items[i].get("text"))
    def get_tokens(i): return grouped_items[i].get("tokens", []) or []
    def get_val(i):    return safe_parse_price(get_text(i), 0.0, 2_000_000.0)

    def _is_probable_tax_amount(text_: str, val_: Optional[float]) -> bool:
        if val_ is None:
            return False
        fv = float(val_)
        if "%" in _safe_text(text_):
            return False
        if abs(fv - round(fv)) < 1e-6 and fv in {5.0, 6.0, 7.0, 13.0, 15.0}:
            return False
        if fv > 10.0 or fv < 0.0:
            return False
        return True

    for idx, item in enumerate(grouped_items):
        label  = _safe_text(item.get("label", "O"))
        text   = _safe_text(item.get("text"))
        tu     = text.upper()
        conf   = float(item.get("avg_confidence", 0.0) or 0.0)
        tokens = item.get("tokens", []) or []
        val    = safe_parse_price(text, 0.0, 2_000_000.0)

        is_discount_line = bool(LONG_DISCOUNT_RE.search(tu))
        is_total_savings = "TOTAL" in tu and "SAVINGS" in tu

        if label in ("B-SUM.SUBTOTAL", "I-SUM.SUBTOTAL"):
            if val is not None and (
                summary["subtotal"] is None or conf >= summary["subtotal_confidence"]
            ):
                summary["subtotal"] = float(val)
                summary["subtotal_confidence"] = conf
                summary["subtotal_tokens"] = tokens

        if label in ("B-SUM.TOTAL", "I-SUM.TOTAL"):
            if val is not None and not is_total_savings:
                if summary["total"] is None or conf >= summary["total_confidence"]:
                    summary["total"] = float(val)
                    summary["total_confidence"] = conf
                    summary["total_tokens"] = tokens

        if label in ("B-TAX", "I-TAX", "B-SUM.TAX", "I-SUM.TAX"):
            if not is_discount_line and _is_probable_tax_amount(text, val):
                tax_components.append(float(val))
                tax_token_bank.extend(tokens)

        # FIX #4: check SUBTOTAL and TOTAL independently on the same item
        # (no early continue so both can fire if both keywords present)
        if LONG_SUBTOTAL_RE.search(tu) and val is not None:
            if summary["subtotal"] is None or conf >= summary["subtotal_confidence"]:
                summary["subtotal"] = float(val)
                summary["subtotal_confidence"] = conf
                summary["subtotal_tokens"] = tokens

        if (
            LONG_TOTAL_RE.search(tu)
            and "SUBTOTAL" not in tu
            and not is_total_savings
            and val is not None
        ):
            if summary["total"] is None or conf >= summary["total_confidence"]:
                summary["total"] = float(val)
                summary["total_confidence"] = conf
                summary["total_tokens"] = tokens

        if LONG_TAX_RE.search(tu) and not is_discount_line and not is_total_savings:
            if _is_probable_tax_amount(text, val):
                tax_components.append(float(val))
                tax_token_bank.extend(tokens)
            else:
                for j in range(idx + 1, min(idx + 3, n)):
                    nxt_val = get_val(j)
                    if _is_probable_tax_amount(get_text(j), nxt_val):
                        tax_components.append(float(nxt_val))
                        tax_token_bank.extend(get_tokens(j))
                        break

        if is_discount_line and val is not None:
            summary["discount"] = abs(float(val)) if val < 0 else float(val)
            if "SAVINGS" in tu and val >= 0:
                summary["discount"] = float(val)
            summary["discount_confidence"] = conf
            summary["discount_tokens"] = tokens

    if tax_components:
        deduped = _dedupe_float_values(tax_components)
        summary["tax"] = round(sum(deduped), 2)
        summary["tax_confidence"] = max(float(summary.get("tax_confidence", 0.0)), 0.85)
        if tax_token_bank:
            summary["tax_tokens"] = tax_token_bank

    return summary


# FIX #4: _extract_summary_from_rows — removed early `continue` so SUBTOTAL
# rows can also be checked for tax, and rows with both keywords are handled.
def _extract_summary_from_rows(rows: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    out = {
        "subtotal": None, "subtotal_confidence": 0.0,
        "tax":      None, "tax_confidence":      0.0,
        "total":    None, "total_confidence":    0.0,
        "discount": None, "discount_confidence": 0.0,
    }
    tax_vals: List[float] = []

    for row in rows:
        text = _safe_text(row.get("text"))
        tu   = text.upper()
        if not text:
            continue
        money_spans = _extract_money_spans(text)
        if not money_spans:
            continue

        matched_subtotal = False
        matched_total    = False

        if "SUBTOTAL" in tu:
            val = _choose_rightmost_nonrate_money(text)
            if val is not None:
                out["subtotal"] = float(val)
                out["subtotal_confidence"] = 0.95
            matched_subtotal = True

        if "TOTAL" in tu and "SAVINGS" not in tu and not matched_subtotal:
            # FIX #4: only skip TOTAL check when this row is purely a SUBTOTAL row
            val = _choose_rightmost_nonrate_money(text)
            if val is not None:
                out["total"] = float(val)
                out["total_confidence"] = 0.95
            matched_total = True

        if matched_subtotal or matched_total:
            continue

        if "SAVE" in tu or "SAVINGS" in tu or "ARCP" in tu or "POINTS REDEEMED" in tu:
            last_val = float(money_spans[-1][2])
            out["discount"] = abs(last_val) if last_val < 0 else last_val
            out["discount_confidence"] = 0.85
            continue

        if re.search(r"\b(GST|PST|HST|QST)\b", tu):
            taxable_base_positions: set = set()
            for base_match in re.finditer(
                r"(\d+(?:[.,]\d+)?)\s*@\s*\d+(?:[.,]\d+)?\s*%", text, re.I
            ):
                base_str = base_match.group(1)
                for s, e, sv, sr in money_spans:
                    try:
                        if abs(float(base_str.replace(",", ".")) - sv) < 0.005:
                            taxable_base_positions.add(s)
                    except ValueError:
                        pass
            for rate_match in re.finditer(r"@\s*(\d+(?:[.,]\d+)?)\s*%", text, re.I):
                rate_str = rate_match.group(1)
                for s, e, sv, sr in money_spans:
                    try:
                        if abs(float(rate_str.replace(",", ".")) - sv) < 0.005:
                            taxable_base_positions.add(s)
                    except ValueError:
                        pass
            for s, e, sv, sr in reversed(money_spans):
                if s in taxable_base_positions:
                    continue
                if "%" in text[e:e + 3]:
                    continue
                if _looks_like_tax_rate_only(sr, sv):
                    continue
                if 0.01 <= sv <= 20.0:
                    tax_vals.append(sv)
                    break
            continue

        if re.search(r"\bTAX\b", tu):
            last_val = float(money_spans[-1][2])
            if 0.0 <= last_val <= 20.0:
                tax_vals.append(last_val)

    if tax_vals:
        tax_vals = _dedupe_float_values(tax_vals)
        out["tax"] = round(sum(tax_vals), 2)
        out["tax_confidence"] = 0.90

    return out


def _merge_summary_sources(
    grouped_summary: Dict[str, Any], row_summary: Dict[str, Any]
) -> Dict[str, Any]:
    summary = dict(grouped_summary)
    for key in ("subtotal", "tax", "total", "discount"):
        conf_key = f"{key}_confidence"
        row_val  = row_summary.get(key)
        row_conf = float(row_summary.get(conf_key, 0.0) or 0.0)
        grp_conf = float(summary.get(conf_key, 0.0) or 0.0)
        if row_val is not None:
            if summary.get(key) is None or row_conf >= grp_conf:
                summary[key] = row_val
                summary[conf_key] = row_conf
    return summary


def _summary_keyword_fallback(
    summary: Dict[str, Any], rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    out = dict(summary)
    for row in rows:
        text = _safe_text(row.get("text"))
        tu   = text.upper()
        if not (row.get("all_prices") or []):
            continue
        if out.get("subtotal") is None and "SUBTOTAL" in tu:
            val = _choose_rightmost_nonrate_money(text)
            if val is not None:
                out["subtotal"] = float(val)
                out["subtotal_confidence"] = max(
                    float(out.get("subtotal_confidence", 0.0)), 0.80
                )
        if out.get("total") is None and "TOTAL" in tu and "SAVINGS" not in tu:
            val = _choose_rightmost_nonrate_money(text)
            if val is not None:
                out["total"] = float(val)
                out["total_confidence"] = max(
                    float(out.get("total_confidence", 0.0)), 0.80
                )
    return out


def _rescue_summary_from_all_rows(
    summary: Dict[str, Any], all_rows: List[Dict[str, Any]], cfg
) -> Dict[str, Any]:
    out = dict(summary)
    for row in all_rows:
        text = _safe_text(row.get("text"))
        tu   = text.upper()
        if not text:
            continue
        if out.get("subtotal") is None and "SUBTOTAL" in tu:
            val = _choose_rightmost_nonrate_money(text)
            if val is not None and val > 0:
                out["subtotal"] = float(val)
                out["subtotal_confidence"] = 0.92
                if cfg.debug_print:
                    print(f"  [RESCUE SUBTOTAL] '{text}' → ${val:.2f}")
        if (
            out.get("total") is None
            and "TOTAL" in tu
            and "SUBTOTAL" not in tu
            and "SAVINGS" not in tu
        ):
            val = _choose_rightmost_nonrate_money(text)
            if val is not None and val > 0:
                out["total"] = float(val)
                out["total_confidence"] = 0.92
                if cfg.debug_print:
                    print(f"  [RESCUE TOTAL] '{text}' → ${val:.2f}")
    return out


def _repair_summary_from_total_delta(summary: Dict[str, Any], cfg) -> Dict[str, Any]:
    out = dict(summary)
    sub = out.get("subtotal")
    tax = out.get("tax")
    tot = out.get("total")
    if sub is not None and tot is not None:
        delta = round(float(tot) - float(sub), 2)
        if 0.0 <= delta <= 20.0:
            if tax is None or abs(float(tax) - delta) > 1.0:
                out["tax"] = delta
                out["tax_confidence"] = max(
                    float(out.get("tax_confidence", 0.0)), 0.92
                )
                if cfg.debug_print:
                    print(
                        f"  ✓ Repaired tax from total-subtotal delta: ${delta:.2f}"
                    )
    return out


def validate_and_correct_summary_long(
    summary: Dict[str, Any], items: List[Dict[str, Any]], cfg
) -> Dict[str, Any]:
    if cfg.debug_print:
        print(f"\n[LONG SUMMARY VALIDATION]")
        for k in ("subtotal", "tax", "total"):
            v = summary.get(k)
            print(
                f"  {k.capitalize()}: ${v:.2f}"
                if v is not None
                else f"  {k.capitalize()}: None"
            )
        items_sum = sum(float(it.get("price", 0.0) or 0.0) for it in items)
        print(f"  Items sum: ${items_sum:.2f}")

    subtotal_explicit = (
        bool(summary.get("subtotal_tokens")) or summary.get("subtotal") is not None
    )
    tax_explicit = (
        bool(summary.get("tax_tokens")) or summary.get("tax") is not None
    )

    if summary["subtotal"] is None and cfg.debug_print:
        print("  ⊘ Subtotal remains missing")

    if all(summary.get(k) is not None for k in ("subtotal", "tax", "total")):
        expected = float(summary["subtotal"]) + float(summary["tax"])
        actual   = float(summary["total"])
        if abs(expected - actual) > 0.25:
            if cfg.debug_print:
                print(
                    f"  ⚠ Mismatch: subtotal+tax={expected:.2f}, total={actual:.2f}"
                )
            if subtotal_explicit and not tax_explicit:
                rec_tax = actual - float(summary["subtotal"])
                if rec_tax >= 0:
                    summary["tax"] = round(rec_tax, 2)
            elif subtotal_explicit and tax_explicit:
                maybe_tax = round(
                    float(summary["total"]) - float(summary["subtotal"]), 2
                )
                if 0.0 <= maybe_tax <= 20.0:
                    summary["tax"] = maybe_tax

    if (
        summary["total"] is None
        and summary["subtotal"] is not None
        and summary["tax"] is not None
    ):
        summary["total"] = round(
            float(summary["subtotal"]) + float(summary["tax"]), 2
        )
        summary["total_confidence"] = min(
            float(summary.get("subtotal_confidence", 0.0)),
            float(summary.get("tax_confidence", 0.0)),
        )

    if cfg.debug_print:
        print(f"\n[CORRECTED LONG SUMMARY]")
        for k in ("subtotal", "tax", "total"):
            v = summary.get(k)
            print(
                f"  {k.capitalize()}: ${v:.2f}"
                if v is not None
                else f"  {k.capitalize()}: None"
            )

    return summary


# ============================================================
# ITEM FILTER / DEDUPE
# ============================================================
def _filter_long_receipt_items(
    items: List[Dict[str, Any]], cfg
) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        name  = _safe_text(item.get("name"))
        price = item.get("price")
        if price is None:
            continue
        if _looks_like_weak_name(name):
            if cfg.debug_print:
                print(f"  [DROP WEAK NAME] '{name}' -> {price}")
            continue
        if _looks_like_footer_or_payment(name):
            if cfg.debug_print:
                print(f"  [DROP FOOTER ITEM] '{name}' -> {price}")
            continue
        out.append(item)
    return out


def _dedupe_same_name_price(
    items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for item in items:
        name  = _norm_upper(item.get("name", ""))
        price = item.get("price")
        key   = (name, round(float(price), 2) if price is not None else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _drop_undetected_items(
    items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return [
        it for it in items
        if _norm_upper(it.get("name", "")) != "ITEM_NOT_DETECTED"
    ]


def _normalize_item_name_for_merge(name: str) -> str:
    t = _norm_upper(name)
    t = re.sub(r"\b(MRJ|HRJ|NRJ|GRJ|PRJ|GPHRJ|GPRQ|GPRO|GPMRJ)\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _merge_items_preferring_row_parser(
    base_items: List[Dict[str, Any]], row_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set = set()

    def item_key(it):
        name  = _normalize_item_name_for_merge(_safe_text(it.get("name", "")))
        price = it.get("price")
        return (name, round(float(price), 2) if price is not None else None)

    for it in row_items + base_items:
        key = item_key(it)
        if key in seen:
            continue
        seen.add(key)
        merged.append(it)
    return merged


# ============================================================
# FORWARD SCAN (direct + block + weighted)
# ============================================================
def _extract_direct_items_forward_scan(
    rows: List[Dict[str, Any]], used_rows: Set[int], cfg
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    n = len(rows)

    def final_prices(row):
        vals = row.get("final_prices", []) or []
        out  = []
        for v in vals:
            fv = float(v)
            if row.get("has_weight") and 0 < fv < 3.5:
                continue
            out.append(fv)
        return out

    def strong_name(text):
        t = _good_name_from_row_text(text)
        return t if t and not _looks_like_weak_name(t) else None

    i = 0
    while i < n:
        if i in used_rows:
            i += 1
            continue

        row  = rows[i]
        text = row.get("text", "")

        if (
            row.get("is_section_header")
            or row.get("has_footer_or_payment")
            or row.get("has_summary_anchor")
        ):
            i += 1
            continue

        name   = strong_name(text)
        prices = final_prices(row)
        if name and prices and not row.get("has_unit_price") and not row.get("has_weight"):
            items.append({
                "name": name,
                "price": float(prices[-1]),
                "price_confidence": _mean(row.get("confs", [])),
                "source": "long_row_direct",
            })
            used_rows.add(i)
            i += 1
            continue

        block_rows = []
        j = i
        while j < n and len(block_rows) < 6:
            r = rows[j]
            if (
                r.get("is_section_header")
                or r.get("has_footer_or_payment")
                or r.get("has_summary_anchor")
            ):
                break
            block_rows.append(j)
            j += 1

        name_rows:  List[Tuple[int, str]]   = []
        price_rows: List[Tuple[int, float]] = []
        for ridx in block_rows:
            if ridx in used_rows:
                continue
            r  = rows[ridx]
            nm = strong_name(r.get("text", ""))
            if nm:
                name_rows.append((ridx, nm))
            for p in final_prices(r):
                price_rows.append((ridx, p))

        if name_rows and price_rows:
            block_added = 0
            for ridx_name, nm in name_rows:
                if ridx_name in used_rows:
                    continue
                chosen_price     = None
                chosen_price_row = None
                for ridx_price, pr in price_rows:
                    if ridx_price in used_rows:
                        continue
                    if ridx_price < ridx_name:
                        continue
                    if ridx_price - ridx_name > 2:
                        continue
                    if any(
                        nr > ridx_name and nr < ridx_price and nr not in used_rows
                        for nr, _ in name_rows
                    ):
                        continue
                    chosen_price     = pr
                    chosen_price_row = ridx_price
                    break
                if chosen_price is not None:
                    items.append({
                        "name": nm,
                        "price": float(chosen_price),
                        "price_confidence": _mean(
                            rows[chosen_price_row].get("confs", [])
                        ),
                        "source": "long_row_block_ordered",
                    })
                    used_rows.add(ridx_name)
                    used_rows.add(chosen_price_row)
                    block_added += 1
            if block_added > 0:
                i = block_rows[-1] + 1
                continue

        if row.get("has_weight") or row.get("has_unit_price"):
            window = rows[i:i + 4]
            best_name      = None
            best_name_row  = None
            best_price     = None
            best_price_row = None
            for k, r in enumerate(window):
                nm = strong_name(r.get("text", ""))
                if nm:
                    best_name     = nm
                    best_name_row = i + k
                    break
            for k, r in enumerate(window):
                vals = final_prices(r)
                if vals:
                    best_price     = float(vals[-1])
                    best_price_row = i + k
                    break
            if best_name and best_price is not None:
                items.append({
                    "name": best_name,
                    "price": best_price,
                    "price_confidence": _mean(
                        rows[best_price_row].get("confs", [])
                    ),
                    "source": "long_row_weighted",
                })
                used_rows.add(best_name_row)
                used_rows.add(best_price_row)
                i = max(best_name_row, best_price_row) + 1
                continue

        i += 1

    return items


# ============================================================
# GROUPED ITEM BOUNDARY
# ============================================================
def _find_first_summary_or_footer_group(
    grouped_items: List[Dict[str, Any]]
) -> Optional[int]:
    candidates: List[int] = []
    for i, g in enumerate(grouped_items):
        text  = _safe_text(g.get("text"))
        label = _safe_text(g.get("label", "O"))
        conf  = float(g.get("avg_confidence", 0.0) or 0.0)
        if _looks_like_summary_anchor(text):
            candidates.append(i)
            continue
        if label in {"B-SUM.SUBTOTAL", "B-SUM.TOTAL", "B-SUM.TAX", "B-TAX"}:
            candidates.append(i)
            continue
        if _looks_like_footer_or_payment(text) and conf >= 0.50:
            candidates.append(i)
            continue
    return min(candidates) if candidates else None


def _trim_grouped_items_for_long_receipt(
    grouped_items: List[Dict[str, Any]], cfg
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    stop_idx = _find_first_summary_or_footer_group(grouped_items)
    if stop_idx is None:
        return grouped_items, None
    if cfg.debug_print:
        print(f"\n[LONG BOUNDARY]")
        print(
            f"  Stopping grouped item extraction at group index {stop_idx}"
        )
    return grouped_items[:stop_idx], stop_idx


def _find_summary_start_row(rows: List[Dict[str, Any]]) -> Optional[int]:
    candidates: List[int] = []
    for i, row in enumerate(rows):
        tu = row.get("text", "").upper()
        if row.get("has_summary_anchor"):
            candidates.append(i)
            continue
        if "TOTAL" in tu and "SAVINGS" not in tu:
            candidates.append(i)
            continue
        if row.get("has_footer_or_payment"):
            candidates.append(i)
            continue
    return min(candidates) if candidates else None


def _trim_rows_before_summary(
    rows: List[Dict[str, Any]], cfg
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    stop = _find_summary_start_row(rows)
    if stop is None:
        return rows, None
    if cfg.debug_print:
        print(f"\n[LONG ROW BOUNDARY]")
        print(f"  Stopping row parsing at row index {stop}")
    return rows[:stop], stop


# ============================================================
# TOKEN-BLOCK ITEM EXTRACTION
# FIX #2: UPC gate replaced — blocks now start on B-MENU.NM label
#         OR a UPC token, whichever comes first.
# FIX #3: price cap raised from $20 → $500.
# FIX #10: honours raw_stop_token_idx so summary tokens never enter blocks.
# FIX #11: BLOCK_SUMMARY_WORDS moved to module level.
# ============================================================
def _build_item_blocks_from_tokens(
    example: Dict[str, Any],
    cfg,
    stop_token_idx: Optional[int] = None,   # FIX #10
) -> List[Dict[str, Any]]:
    words   = example.get("words",   []) or []
    labels  = example.get("labels",  []) or []
    confs   = example.get("confs",   []) or []
    bboxes  = example.get("bboxes",  []) or []
    row_ids = example.get("row_ids", []) or list(range(len(words)))

    n = min(len(words), len(labels), len(confs))

    # FIX #10: honour caller-supplied stop index
    if stop_token_idx is not None:
        n = min(n, stop_token_idx)

    if cfg.debug_print:
        print(f"\n[BLOCK BUILDER v6] Processing {n} tokens")

    tok: List[Dict[str, Any]] = []
    for i in range(n):
        w    = _safe_text(words[i])
        lbl  = _safe_text(labels[i])
        conf = float(confs[i] or 0.0)
        wu   = _norm_upper(w)
        bb   = bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]
        rid  = int(row_ids[i]) if i < len(row_ids) else i

        is_upc      = bool(UPC_RE.fullmatch(wu))
        is_junk     = wu in KNOWN_JUNK_TOKENS
        is_price    = bool(re.fullmatch(r"[-+]?\$?\d+[.,]\d{2}", w))
        is_section  = _looks_like_section_header(wu)
        is_footer   = _looks_like_footer_or_payment(wu)
        is_summary  = bool(re.search(r"\b(SUBTOTAL|TOTAL|TAX|GST|PST)\b", wu))
        is_subtotal = "SUBTOTAL" in wu
        is_total    = "TOTAL" in wu and "SUBTOTAL" not in wu
        # FIX #11: use module-level BLOCK_SUMMARY_WORDS
        is_name_forbidden = wu in BLOCK_SUMMARY_WORDS or is_footer or is_section

        # FIX #2: also flag B-MENU.NM label as a block-start signal
        is_menu_nm_begin = lbl == "B-MENU.NM"

        tok.append({
            "idx": i, "word": w, "word_u": wu, "label": lbl,
            "conf": conf, "bb": bb, "rid": rid,
            "is_upc": is_upc, "is_junk": is_junk, "is_price": is_price,
            "is_section": is_section, "is_footer": is_footer,
            "is_summary": is_summary, "is_subtotal": is_subtotal,
            "is_total": is_total, "is_name_forbidden": is_name_forbidden,
            "is_menu_nm_begin": is_menu_nm_begin,
        })

    # ── Find summary boundary ──────────────────────────────────────
    stop_idx = len(tok)
    for i, t in enumerate(tok):
        if not (t["is_subtotal"] or t["is_total"]):
            continue
        j = i + 1
        while j < len(tok) and (tok[j]["is_junk"] or not tok[j]["word_u"].strip()):
            j += 1
        if j < len(tok):
            nxt = tok[j]
            if nxt["is_upc"]:
                if cfg.debug_print:
                    print(
                        f"  [BLOCK] Skipping boundary token {i}: "
                        f"'{t['word']}' (next UPC '{nxt['word']}')"
                    )
                continue
            if nxt["is_price"] or nxt["is_summary"] or nxt["is_footer"]:
                stop_idx = i
                if cfg.debug_print:
                    print(
                        f"  [BLOCK] Boundary at token {i}: "
                        f"'{t['word']}' (next='{nxt['word']}')"
                    )
                break
            if nxt["is_name_forbidden"]:
                stop_idx = i
                if cfg.debug_print:
                    print(
                        f"  [BLOCK] Boundary at token {i}: "
                        f"'{t['word']}' (next forbidden '{nxt['word']}')"
                    )
                break
            if cfg.debug_print:
                print(
                    f"  [BLOCK] Skipping boundary token {i}: "
                    f"'{t['word']}' (next='{nxt['word']}' looks like product)"
                )
        else:
            stop_idx = i
            break

    if stop_idx == len(tok):
        for i, t in enumerate(tok):
            if t["is_footer"]:
                stop_idx = i
                if cfg.debug_print:
                    print(
                        f"  [BLOCK] Footer boundary at token {i}: '{t['word']}'"
                    )
                break

    item_tokens = tok[:stop_idx]

    # ── Build blocks ───────────────────────────────────────────────
    blocks: List[Dict[str, Any]] = []
    current_block: Optional[Dict[str, Any]] = None

    def _new_block() -> Dict[str, Any]:
        return {
            "name_words":  [],
            "name_confs":  [],
            "price_vals":  [],
            "price_confs": [],
        }

    # FIX #11: use module-level BLOCK_SUMMARY_WORDS in _flush_block
    def _flush_block(b: Optional[Dict[str, Any]]) -> None:
        if b is None:
            return
        if not b["name_words"] and not b["price_vals"]:
            return
        name_parts = [
            w for w in b["name_words"]
            if w not in KNOWN_JUNK_TOKENS
            and w not in WEAK_ITEM_NAMES
            and not re.fullmatch(r"[\d.,\$%\-\+/\\]+", w)
            and not _looks_like_section_header(w)
            and _norm_upper(w) not in BLOCK_SUMMARY_WORDS
        ]
        b["name_raw"]   = " ".join(name_parts).strip()
        b["name_clean"] = _good_name_from_row_text(b["name_raw"]) or ""
        b["avg_conf"]   = _mean(b["name_confs"]) if b["name_confs"] else 0.0
        blocks.append(b)

    i = 0
    while i < len(item_tokens):
        t = item_tokens[i]

        if t["is_section"]:
            i += 1
            continue

        # FIX #2: start a new block on UPC token OR B-MENU.NM label
        if t["is_upc"] or t["is_menu_nm_begin"]:
            _flush_block(current_block)
            current_block = _new_block()
            # If this token is B-MENU.NM (not a UPC), add its word as a name token
            if t["is_menu_nm_begin"] and not t["is_upc"]:
                current_block["name_words"].append(t["word_u"])
                current_block["name_confs"].append(t["conf"])
            i += 1
            continue

        # Only process tokens after a block has been started
        if current_block is None:
            i += 1
            continue

        # FIX #11: summary/forbidden words never enter name_words
        if t["is_name_forbidden"]:
            i += 1
            continue

        if t["is_junk"]:
            j = i + 1
            while j < len(item_tokens) and item_tokens[j]["is_junk"]:
                j += 1
            if j < len(item_tokens) and item_tokens[j]["is_price"]:
                price_val = safe_parse_price(
                    item_tokens[j]["word"], 0.0, 2_000_000.0
                )
                if price_val is not None:
                    fv = float(price_val)
                    # FIX #3: raised cap from $20 → $500
                    if len(current_block["price_vals"]) < 2 and fv < 500.0:
                        current_block["price_vals"].append(fv)
                        current_block["price_confs"].append(item_tokens[j]["conf"])
                i = j + 1
                continue
            i += 1
            continue

        if t["is_price"]:
            price_val = safe_parse_price(t["word"], 0.0, 2_000_000.0)
            if price_val is not None:
                fv = float(price_val)
                # FIX #3: raised cap from $20 → $500
                if len(current_block["price_vals"]) < 2 and fv < 500.0:
                    current_block["price_vals"].append(fv)
                    current_block["price_confs"].append(t["conf"])
            i += 1
            continue

        # Regular name word
        current_block["name_words"].append(t["word_u"])
        current_block["name_confs"].append(t["conf"])
        i += 1

    _flush_block(current_block)

    if cfg.debug_print:
        print(f"  [BLOCK] Built {len(blocks)} raw blocks")
        for bi, b in enumerate(blocks):
            print(
                f"    block[{bi:02d}] "
                f"name='{b.get('name_clean', '')}' "
                f"prices={[round(p, 2) for p in b.get('price_vals', [])]}"
            )

    return blocks


def _extract_items_from_blocks(
    blocks: List[Dict[str, Any]],
    cfg,
) -> List[Dict[str, Any]]:
    """
    Convert item blocks into extracted items.

    Cases handled:
    1. 2 segments + 2 prices  → multi-split, emit both directly
    2. 2 segments + 1 price   → seg[0] gets price (FIX #8), seg[1] looks forward
    3. 1 segment  + price     → direct (takes prices[-1])
    4. 1 segment  + no price  → look forward for price-only block
    5. no segment + price     → look backward for name-only block
    6. no segment + no price  → skip
    """
    items: List[Dict[str, Any]] = []
    n = len(blocks)

    if cfg.debug_print:
        print(f"\n[BLOCK EXTRACTOR v5] Processing {n} blocks")

    used: Set[int] = set()

    # ── Name segmentation ─────────────────────────────────────────
    def _split_name_into_segments(name_words: List[str]) -> List[str]:
        """
        Split a raw name_words list into at most 2 product name segments.
        Never splits 1- or 2-word names (e.g. "FISH SAUCE", "NH BACON LS").
        """
        if not name_words:
            return []

        FORBIDDEN = {
            "SUBTOTAL", "TOTAL", "TAX", "GST", "PST", "HST",
            "BALANCE", "CLOSING", "CREDIT", "PURCHASE",
        }
        GENERIC_SUFFIXES = {
            "ULTRA", "PLUS", "EXTRA", "LIGHT", "DARK", "MINI",
            "MAXI", "MEGA", "SUPER", "ZERO", "MAX", "PRO",
        }

        clean = [
            w for w in name_words
            if w not in KNOWN_JUNK_TOKENS
            and w not in WEAK_ITEM_NAMES
            and not re.fullmatch(r"[\d.,\$%\-\+/\\]+", w)
            and not _looks_like_section_header(w)
            and _norm_upper(w) not in FORBIDDEN
            and len(w) >= 2
            and w[0].isalpha()
        ]

        if not clean:
            return []

        if len(clean) == 1:
            nm = _good_name_from_row_text(clean[0])
            return [nm] if nm and not _looks_like_weak_name(nm) else []

        # PASS 1: strict — both sides >= 2 words AND >= 4 alpha chars
        for split_at in range(2, len(clean)):
            left_w  = clean[:split_at]
            right_w = clean[split_at:]

            if len(left_w) < 2 or len(right_w) < 2:
                continue

            left_nm  = _good_name_from_row_text(" ".join(left_w))
            right_nm = _good_name_from_row_text(" ".join(right_w))

            if not left_nm or _looks_like_weak_name(left_nm):
                continue
            if not right_nm or _looks_like_weak_name(right_nm):
                continue
            if sum(c.isalpha() for c in " ".join(left_w)) < 4:
                continue
            if sum(c.isalpha() for c in " ".join(right_w)) < 4:
                continue

            return [left_nm, right_nm]

        # PASS 2: relaxed — left >= 2 words, right >= 1 word (not a suffix)
        if len(clean) >= 3:
            for split_at in range(2, len(clean)):
                left_w  = clean[:split_at]
                right_w = clean[split_at:]

                if len(left_w) < 2 or not right_w:
                    continue
                if _norm_upper(" ".join(right_w)) in GENERIC_SUFFIXES:
                    continue

                left_nm  = _good_name_from_row_text(" ".join(left_w))
                right_nm = _good_name_from_row_text(" ".join(right_w))

                if not left_nm or _looks_like_weak_name(left_nm):
                    continue
                if not right_nm or _looks_like_weak_name(right_nm):
                    continue

                left_alpha  = sum(c.isalpha() for c in " ".join(left_w))
                right_alpha = sum(c.isalpha() for c in " ".join(right_w))

                if left_alpha < 4 or right_alpha < 4:
                    continue

                return [left_nm, right_nm]

        # No valid split — return as single segment
        nm = _good_name_from_row_text(" ".join(clean))
        return [nm] if nm and not _looks_like_weak_name(nm) else []

    # ── Main extraction loop ───────────────────────────────────────
    for i, block in enumerate(blocks):
        if i in used:
            continue

        name    = block.get("name_clean", "")
        prices  = block.get("price_vals",  []) or []
        conf    = block.get("avg_conf",    0.0)
        p_confs = block.get("price_confs", []) or []
        nw      = block.get("name_words",  []) or []

        has_name  = bool(name) and not _looks_like_weak_name(name)
        has_price = len(prices) > 0

        segments = _split_name_into_segments(nw)

        # ── Case 1: 2 segments + 2 prices → multi-split ───────────
        if len(segments) >= 2 and len(prices) >= 2:
            for seg, price, pc in zip(
                segments,
                prices,
                p_confs + [0.0] * len(prices),
            ):
                items.append({
                    "name": seg,
                    "price": float(price),
                    "price_confidence": float(pc),
                    "source": "long_block_multi_split",
                })
                if cfg.debug_print:
                    print(f"  [BLOCK MULTI] '{seg}' → ${price:.2f}")
            used.add(i)
            continue

        # ── Case 2: 2 segments + 1 price ──────────────────────────
        # FIX #8: seg[0] (first item in receipt order) gets the current price.
        #         seg[1] (second item) looks forward for its price.
        if len(segments) == 2 and len(prices) == 1:
            # seg[0] → current price
            items.append({
                "name": segments[0],
                "price": float(prices[-1]),
                "price_confidence": float(p_confs[-1]) if p_confs else conf,
                "source": "long_block_direct",
            })
            if cfg.debug_print:
                print(
                    f"  [BLOCK DIRECT (seg0)] '{segments[0]}'"
                    f" → ${prices[-1]:.2f}"
                )

            # seg[1] → look forward for price
            orphan_name = segments[1]
            found_price = False

            j = i + 1
            while j < n and j in used:
                j += 1

            if j < n:
                nb          = blocks[j]
                nb_pr       = nb.get("price_vals", []) or []
                nb_nm       = nb.get("name_clean", "")
                nb_has_name = bool(nb_nm) and not _looks_like_weak_name(nb_nm)

                if nb_pr and not nb_has_name:
                    # Price-only block → consume it
                    items.append({
                        "name": orphan_name,
                        "price": float(nb_pr[-1]),
                        "price_confidence": float(
                            (nb.get("price_confs") or [0.0])[-1]
                        ),
                        "source": "long_block_split_fwd",
                    })
                    used.add(j)
                    found_price = True
                    if cfg.debug_print:
                        print(
                            f"  [BLOCK SPLIT FWD] '{orphan_name}'"
                            f" → ${nb_pr[-1]:.2f} (consumed block {j})"
                        )

                elif nb_pr and nb_has_name:
                    # Next block is self-contained — borrow without consuming
                    items.append({
                        "name": orphan_name,
                        "price": float(nb_pr[-1]),
                        "price_confidence": float(
                            (nb.get("price_confs") or [0.0])[-1]
                        ),
                        "source": "long_block_split_borrowed",
                    })
                    found_price = True
                    if cfg.debug_print:
                        print(
                            f"  [BLOCK SPLIT BORROWED] '{orphan_name}'"
                            f" → ${nb_pr[-1]:.2f} (block {j} kept intact)"
                        )

            if not found_price and cfg.debug_print:
                print(f"  [BLOCK NO-PRICE] '{orphan_name}' (split orphan)")

            used.add(i)
            continue

        # ── Case 3: single segment + price → direct ───────────────
        if has_name and has_price:
            next_price: Optional[float] = None
            next_conf:  float           = 0.0
            next_i:     Optional[int]   = None

            j = i + 1
            while j < n and j in used:
                j += 1

            if j < n:
                nb          = blocks[j]
                nb_nm       = nb.get("name_clean", "")
                nb_pr       = nb.get("price_vals", []) or []
                nb_has_name = bool(nb_nm) and not _looks_like_weak_name(nb_nm)

                if not nb_has_name and nb_pr:
                    candidate = nb_pr[-1]
                    if abs(candidate - prices[-1]) > 0.005:
                        next_price = candidate
                        next_conf  = (nb.get("price_confs") or [0.0])[-1]
                        next_i     = j

            chosen      = next_price if next_price is not None else prices[-1]
            chosen_conf = (
                next_conf if next_price is not None
                else (p_confs[-1] if p_confs else conf)
            )

            items.append({
                "name": name,
                "price": float(chosen),
                "price_confidence": float(chosen_conf),
                "source": "long_block_direct",
            })
            used.add(i)
            if next_i is not None:
                used.add(next_i)

            if cfg.debug_print:
                tag = " (next-block)" if next_price is not None else ""
                print(f"  [BLOCK DIRECT{tag}] '{name}' → ${chosen:.2f}")
            continue

        # ── Case 4: name only → look forward for price ─────────────
        if has_name and not has_price:
            j = i + 1
            while j < n and j in used:
                j += 1

            if j < n:
                nb          = blocks[j]
                nb_pr       = nb.get("price_vals", []) or []
                nb_nm       = nb.get("name_clean", "")
                nb_has_name = bool(nb_nm) and not _looks_like_weak_name(nb_nm)

                if nb_pr and not nb_has_name:
                    items.append({
                        "name": name,
                        "price": float(nb_pr[-1]),
                        "price_confidence": float(
                            (nb.get("price_confs") or [0.0])[-1]
                        ),
                        "source": "long_block_name_fwd",
                    })
                    used.add(i)
                    used.add(j)
                    if cfg.debug_print:
                        print(
                            f"  [BLOCK NAME→PRICE] '{name}'"
                            f" → ${nb_pr[-1]:.2f}"
                        )
                    continue

            if cfg.debug_print:
                print(f"  [BLOCK NO-PRICE] '{name}'")
            used.add(i)
            continue

        # ── Case 5: price only → look backward for name ────────────
        if has_price and not has_name:
            j = i - 1
            while j >= 0 and j in used:
                j -= 1

            if j >= 0:
                pb          = blocks[j]
                pb_nm       = pb.get("name_clean", "")
                pb_pr       = pb.get("price_vals", []) or []
                pb_has_name = bool(pb_nm) and not _looks_like_weak_name(pb_nm)

                if pb_has_name and not pb_pr:
                    items.append({
                        "name": pb_nm,
                        "price": float(prices[-1]),
                        "price_confidence": float(
                            p_confs[-1] if p_confs else 0.0
                        ),
                        "source": "long_block_price_bwd",
                    })
                    used.add(i)
                    used.add(j)
                    if cfg.debug_print:
                        print(
                            f"  [BLOCK PRICE→NAME] '{pb_nm}'"
                            f" → ${prices[-1]:.2f}"
                        )
                    continue

            if cfg.debug_print:
                print(f"  [BLOCK NO-NAME] orphan ${prices[-1]:.2f}")
            used.add(i)
            continue

        # ── Case 6: no name, no price → skip ──────────────────────
        used.add(i)

    if cfg.debug_print:
        print(f"  [BLOCK EXTRACTOR] Done. {len(items)} items.")

    return items


# ============================================================
# MAIN LONG EXTRACTOR
# ============================================================
def cord_plus_geo_extract_long(
    example: Dict[str, Any],
    *,
    cfg,
) -> Dict[str, Any]:
    words  = example.get("words",  []) or []
    labels = example.get("labels", []) or []
    confs  = example.get("confs",  []) or []
    bboxes = example.get("bboxes", []) or []

    if not (words and labels and confs):
        return {
            "id":         example.get("id"),
            "image_path": example.get("image_path"),
            "SUBTOTAL": None, "SUBTOTAL_CONFIDENCE": 0.0,
            "TAX":      None, "TAX_CONFIDENCE":      0.0,
            "TOTAL":    None, "TOTAL_CONFIDENCE":    0.0,
            "DISCOUNT": None, "DISCOUNT_CONFIDENCE": 0.0,
            "ITEMS": [],
            "PATTERN_DETECTED": "none",
            "debug": {"error": "Missing required fields"},
        }

    if cfg.debug_print:
        print(f"\n{'='*80}")
        print("CORD EXTRACTION LONG - CANADIAN GROCERY MODE")
        print(f"{'='*80}")
        print(f"Processing {len(words)} tokens")

    # --------------------------------------------------------
    # STEP 1: Group tokens into label sequences
    # --------------------------------------------------------
    grouped_items = group_tokens_by_label_sequence(
        words,
        labels,
        confs,
        bboxes if len(bboxes) == len(words) else [[0, 0, 0, 0]] * len(words),
    )

    if cfg.debug_print:
        print(f"Grouped into {len(grouped_items)} label sequences")

    # --------------------------------------------------------
    # STEP 2: Extract summary fields from grouped sequences
    # --------------------------------------------------------
    grouped_summary = _extract_summary_items_long(grouped_items, cfg)

    # --------------------------------------------------------
    # STEP 3: Trim grouped items — stop at first summary/footer
    # --------------------------------------------------------
    trimmed_grouped_items, stop_idx = _trim_grouped_items_for_long_receipt(
        grouped_items, cfg
    )

    raw_stop_token_idx = None
    if stop_idx is not None and stop_idx < len(grouped_items):
        raw_stop_token_idx = sum(
            len(g.get("tokens", []) or []) for g in grouped_items[:stop_idx]
        )

    trimmed_words  = words[:raw_stop_token_idx]  if raw_stop_token_idx is not None else words
    trimmed_labels = labels[:raw_stop_token_idx] if raw_stop_token_idx is not None else labels
    trimmed_confs  = confs[:raw_stop_token_idx]  if raw_stop_token_idx is not None else confs

    # --------------------------------------------------------
    # STEP 4A: Adaptive base extractor on trimmed token window
    # --------------------------------------------------------
    base_items = extract_menu_items_adaptive(
        trimmed_grouped_items,
        trimmed_words,
        trimmed_labels,
        trimmed_confs,
        cfg,
    )
    base_items = _drop_undetected_items(base_items)

    if cfg.debug_print:
        print(f"\n[BASE ITEMS after ITEM_NOT_DETECTED drop]: {len(base_items)}")
        for it in base_items:
            print(f"  {it.get('name')} → ${it.get('price', 0.0):.2f}")

    # --------------------------------------------------------
    # STEP 4B: Build all rows (used for summary + row extractor)
    # --------------------------------------------------------
    all_rows = _build_rows_from_example(example)

    # --------------------------------------------------------
    # STEP 4C: Row-based summary extraction (all rows)
    # --------------------------------------------------------
    row_summary = _extract_summary_from_rows(all_rows, cfg)
    summary     = _merge_summary_sources(grouped_summary, row_summary)

    trimmed_rows, stop_row_idx = _trim_rows_before_summary(all_rows, cfg)

    # --------------------------------------------------------
    # STEP 4D: Token-block item extraction
    # FIX #10: pass raw_stop_token_idx so block builder never
    #          sees summary/footer tokens.
    # --------------------------------------------------------
    if cfg.debug_print:
        print(f"\n[STEP 4D] Token-block item extraction (v6)")

    item_blocks = _build_item_blocks_from_tokens(
        example,
        cfg,
        stop_token_idx=raw_stop_token_idx,   # FIX #10
    )
    row_items = _extract_items_from_blocks(item_blocks, cfg)

    # --------------------------------------------------------
    # STEP 4E: Row-based extraction on trimmed rows
    # FIX #1: _extract_weighted_or_direct_items_from_rows now
    #         runs layout pre-pass + sequential scan.
    # --------------------------------------------------------
    if cfg.debug_print:
        print(f"\n[STEP 4E] Row-based extraction (layout + sequential)")

    row_direct_items = _extract_weighted_or_direct_items_from_rows(
        trimmed_rows, cfg
    )

    if cfg.debug_print:
        print(f"\n[LONG ITEM PARSER]")
        print(f"  Built rows          : {len(all_rows)}")
        print(f"  Trimmed rows        : {len(trimmed_rows)}")
        print(f"  Token blocks built  : {len(item_blocks)}")
        print(f"  Block-derived items : {len(row_items)}")
        print(f"  Row-direct items    : {len(row_direct_items)}")

    # --------------------------------------------------------
    # STEP 5: Merge — block and row items preferred over base
    # --------------------------------------------------------
    combined_row_items = _merge_items_preferring_row_parser(
        row_direct_items, row_items
    )
    items = _merge_items_preferring_row_parser(base_items, combined_row_items)
    items = _filter_long_receipt_items(items, cfg)
    items = _dedupe_same_name_price(items)

    # --------------------------------------------------------
    # STEP 6: Summary pipeline
    # --------------------------------------------------------
    summary = _summary_keyword_fallback(summary, all_rows)
    summary = _rescue_summary_from_all_rows(summary, all_rows, cfg)
    summary = _repair_summary_from_total_delta(summary, cfg)
    summary = validate_and_correct_summary_long(summary, items, cfg)

    # --------------------------------------------------------
    # STEP 7: Conservative subtotal fallback from item sum
    # --------------------------------------------------------
    if summary.get("subtotal") is None and summary.get("total") is not None:
        items_sum = sum(float(it.get("price", 0.0) or 0.0) for it in items)
        total_val = float(summary["total"])
        if abs(items_sum - total_val) <= 1.00:
            summary["subtotal"]            = round(items_sum, 2)
            summary["subtotal_confidence"] = 0.40
            if cfg.debug_print:
                print(f"  [SUBTOTAL FALLBACK] item sum=${items_sum:.2f}")
        else:
            if cfg.debug_print:
                print(
                    f"  [NO SUBTOTAL FALLBACK] "
                    f"items_sum=${items_sum:.2f} ≠ total=${total_val:.2f}"
                )

    # --------------------------------------------------------
    # STEP 8: Build source-count debug metadata
    # --------------------------------------------------------
    src_counts: Dict[str, int] = {}
    for item in items:
        src = item.get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    if cfg.debug_print:
        print(f"\n{'='*80}")
        print("LONG EXTRACTION SUMMARY")
        print(f"{'='*80}")
        print(f"Items extracted: {len(items)}")

        sub  = summary.get("subtotal")
        tx   = summary.get("tax")
        tot  = summary.get("total")
        disc = summary.get("discount")

        print(f"Subtotal : ${sub:.2f}"  if sub  is not None else "Subtotal : None")
        print(f"Tax      : ${tx:.2f}"   if tx   is not None else "Tax      : None")
        print(f"Total    : ${tot:.2f}"  if tot  is not None else "Total    : None")
        print(f"Discount : ${disc:.2f}" if disc is not None else "Discount : None")

        if src_counts:
            print("\nItems by source:")
            for src, count in sorted(src_counts.items()):
                print(f"  {src}: {count}")

        print("\nFinal items:")
        for it in items:
            print(
                f"  [{it.get('source', '?')}] "
                f"{it.get('name', '?')} → "
                f"${it.get('price', 0.0):.2f} "
                f"(conf: {it.get('price_confidence', 0.0):.2f})"
            )

    # --------------------------------------------------------
    # STEP 9: Return structured output
    # --------------------------------------------------------
    return {
        "id":         example.get("id"),
        "image_path": example.get("image_path"),

        "SUBTOTAL":            summary.get("subtotal"),
        "SUBTOTAL_CONFIDENCE": float(summary.get("subtotal_confidence", 0.0)),

        "TAX":            summary.get("tax"),
        "TAX_CONFIDENCE": float(summary.get("tax_confidence", 0.0)),

        "TOTAL":            summary.get("total"),
        "TOTAL_CONFIDENCE": float(summary.get("total_confidence", 0.0)),

        "DISCOUNT":            summary.get("discount"),
        "DISCOUNT_CONFIDENCE": float(summary.get("discount_confidence", 0.0)),

        "ITEMS": items,

        "PATTERN_DETECTED": (
            detect_item_pattern(trimmed_grouped_items, cfg)
            if trimmed_grouped_items else "none"
        ),

        "debug": {
            "mode":                    "long_grocery_block_v4",
            "n_words":                 len(words),
            "n_grouped_items":         len(grouped_items),
            "n_trimmed_grouped_items": len(trimmed_grouped_items),
            "n_rows":                  len(all_rows),
            "n_trimmed_rows":          len(trimmed_rows),
            "n_item_blocks":           len(item_blocks),
            "n_base_items":            len(base_items),
            "n_block_items":           len(row_items),
            "n_row_direct_items":      len(row_direct_items),
            "n_extracted_items":       len(items),
            "stop_group_idx":          stop_idx,
            "stop_row_idx":            stop_row_idx,
            "stop_token_idx":          raw_stop_token_idx,
            "subtotal_tokens":         summary.get("subtotal_tokens", []),
            "tax_tokens":              summary.get("tax_tokens",      []),
            "total_tokens":            summary.get("total_tokens",    []),
            "discount_tokens":         summary.get("discount_tokens", []),
            "extraction_sources":      src_counts,
        },
    }