from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import context
from typing import Any, Dict, List, Optional, Tuple, Set
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# FUZZY MATCHING
# ============================================================
def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)

    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[-1]


def normalize_token(s: str) -> str:
    """Normalize token for fuzzy matching."""
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


@dataclass
class FuzzyMatch:
    keyword: str
    distance: int
    normalized_line: str
    normalized_keyword: str


class FuzzyKeywordEngine:
    """Fuzzy keyword matcher with configurable distance threshold."""
    def __init__(self, keywords: List[str], max_distance: int = 2):
        self.raw_keywords = keywords
        self.max_distance = int(max_distance)
        self.norm_keywords = {kw: normalize_token(kw) for kw in keywords}

    def best_match(self, line: str) -> Optional[FuzzyMatch]:
        norm_line = normalize_token(line)
        if not norm_line:
            return None
        
        # ✅ NEW: prevent fuzzy matching on very short tokens like "TOFU", "PAD", "IN"
        # These cause false matches to "TOT", "TAX", etc.
        if len(norm_line) <= 4:
        # exact-only for short tokens
            for kw, norm_kw in self.norm_keywords.items():
                if norm_line == norm_kw:
                    return FuzzyMatch(kw, 0, norm_line, norm_kw)
            return None

        best: Optional[FuzzyMatch] = None
        for kw, norm_kw in self.norm_keywords.items():
            d = levenshtein(norm_line, norm_kw)
            if d <= self.max_distance:
                if best is None or d < best.distance:
                    best = FuzzyMatch(
                        keyword=kw,
                        distance=d,
                        normalized_line=norm_line,
                        normalized_keyword=norm_kw,
                    )
        return best

    def any_match(self, line: str) -> bool:
        return self.best_match(line) is not None


# ============================================================
# CONFIG
# ============================================================
@dataclass(frozen=True)
class CordGeoConfig:
    max_y_gap_merge: int = 55
    max_y_gap_itemline: int = 50
    price_col_tol: int = 80
    cap_tol: float = 2.00
    min_price: float = 0.01
    max_price: float = 2_000_000.0
    attach_gap_mult: float = 4.0
    allow_price_only_items: bool = False
    default_item_name: str = "ITEM"
    debug_print: bool = True
    
    min_label_confidence: float = 0.50
    fuzzy_distance_threshold: int = 2
    use_fuzzy_fallback: bool = True
    
    # Multi-pattern handling
    region_y_gap_threshold: int = 30
    price_name_search_window: int = 3
    
    # Unlabeled item extraction
    extract_unlabeled_items: bool = True
    unlabeled_min_price: float = 0.01
    unlabeled_max_price: float = 500.0


# ============================================================
# REGEX PATTERNS
# ============================================================
MONEY_FIND_RE_LOCAL = re.compile(r"(?<![A-Za-z])\$?\s*\d{1,7}(?:[.,]\d{1,4})?")
TOTAL_TAX_RE = re.compile(r"\btotal\s*tax\b", re.I)
SUBTOTAL_RE = re.compile(r"\b(?:sub\s*total|subtotal|s[j1i]b\s*total|s[j1i]btotal)\b", re.I)
TOTAL_RE = re.compile(r"(?:^|[^A-Za-z])(?:to?tal|ro?tal|t0tal)(?:\b|:)", re.I)
TOTAL_SALE_RE = re.compile(r"\btotal\s+sale\b", re.I)
TAX_RE = re.compile(r"\btax\b|\bgst\b|\bpst\b|\bhst\b|\bvat\b|\btps\b|\btvq\b", re.I)

_OCR_NOISE_RE = re.compile(r"^[xX]{6,}$|x{6,}|xxxxxxx")
_LEADING_NOISE_RE = re.compile(r"^[\s\*\:\;\,\.\+\=]+")
_TRAILING_NOISE_RE = re.compile(r"[\s\*\:\;\,\.\+\=]+$")

SUBTOTAL_KEYWORDS = ["subtotal", "sub total", "subtot"]
TOTAL_KEYWORDS = ["total", "tot"]
TAX_KEYWORDS = ["tax", "gst", "pst", "hst", "vat"]
DISCOUNT_KEYWORDS = ["discount", "disc", "promo", "coupon", "void", "comp"]

TAX_KEYWORD_RE = re.compile(r"\b(tax|vat|gst|hst|pst|qst|sales\s*tax|tps|tvq)\b",re.I)

TIP_FEE_KEYWORD_RE = re.compile(r"\b(tip|gratuity|service\s*charge|svc|fee)\b",re.I)
# NEW: Post-total keywords to exclude
TIP_SUGGESTION_KEYWORDS = [
    "tip", "gratuity", "suggested", "suggestions", "cash", "change", 
    "tendered", "thank you", "percent", "%", "18%", "20%", "25%"
]

## 26 Feb

SUMMARY_CONTEXT_RE = re.compile(
    r"\b(total|subtotal|amount|tax|gst|pst|hst|vat|tendered|change|tip|gratuity)\b",
    re.I
)

# ============================================================
# CORD LABEL PARSER
# ============================================================
class CordLabelParser:
    """Parse and categorize CORD model labels."""
    
    MENU_ITEM_START = {"B-MENU.NM"}
    MENU_ITEM_CONT = {"I-MENU.NM"}
    MENU_ITEM_LABELS = MENU_ITEM_START | MENU_ITEM_CONT
    
    MENU_PRICE_START = {"B-MENU.PRICE"}
    MENU_PRICE_LABELS = MENU_PRICE_START
    
    SUM_SUBTOTAL_START = {"B-SUM.SUBTOTAL"}
    SUM_SUBTOTAL_CONT = {"I-SUM.SUBTOTAL"}
    SUM_SUBTOTAL_LABELS = SUM_SUBTOTAL_START | SUM_SUBTOTAL_CONT
    
    SUM_TAX_START = {"B-TAX", "B-SUM.TAX"}
    SUM_TAX_CONT = {"I-TAX", "I-SUM.TAX"}
    SUM_TAX_LABELS = SUM_TAX_START | SUM_TAX_CONT
    
    SUM_TOTAL_START = {"B-SUM.TOTAL"}
    SUM_TOTAL_CONT = {"I-SUM.TOTAL"}
    SUM_TOTAL_LABELS = SUM_TOTAL_START | SUM_TOTAL_CONT

    @staticmethod
    def get_label_type(label: str) -> str:
        """Extract label type from B-/I- prefix."""
        if label.startswith("B-") or label.startswith("I-"):
            return label[2:]
        return label

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def clean_ocr_noise(text: str) -> str:
    """Clean common OCR artifacts."""
    if not text:
        return text
    
    text = str(text).strip()
    
    if _OCR_NOISE_RE.match(text):
        return ""
    
    text = _LEADING_NOISE_RE.sub("", text)
    text = _TRAILING_NOISE_RE.sub("", text)
    
    replacements = {
        "O0": "00", "l1": "11", "S5": "55", "B8": "88", "Z2": "22",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def safe_parse_price(
    text: str,
    min_price: float = 0.01,
    max_price: float = 2_000_000.0
) -> Optional[float]:
    """
    Safely parse price from text.

    Fixes:
    - Rejects real clock times like "12:56" so they don't become "12.56"
    - Still allows OCR money separator mistakes like "40:00" -> "40.00"
    - Handles negative prices (discount/credit indicators like "9.95-")
    - Enforces min/max range strictly (no "val>0 return val" bypass)
    """
    if not text:
        return None

    try:
        raw = str(text).strip()
        if not raw:
            return None

        # ================================================================
        # Reject clock times like "12:56" / "7:03"
        # But keep OCR mistakes like "40:00" (hour > 23) convertible to price.
        # ================================================================
        m_time = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
        if m_time:
            hh = int(m_time.group(1))
            mm = int(m_time.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return None

        # ================================================================
        # Normalize OCR money separator mistakes: "40:00" -> "40.00"
        # ================================================================
        if re.fullmatch(r"\d{1,7}:\d{2}", raw):
            raw = raw.replace(":", ".")
        else:
            raw = re.sub(r"(?<=\d):(?=\d{2}\b)", ".", raw)

        raw = clean_ocr_noise(raw)

        # ================================================================
        # CHECK FOR NEGATIVE INDICATOR (trailing - or leading -)
        # ================================================================
        is_negative = False

        if raw.rstrip().endswith("-"):
            is_negative = True
            raw = raw.rstrip().rstrip("-").strip()
        elif raw.startswith("-"):
            is_negative = True
            raw = raw.lstrip("-").strip()

        # extra guard (kept)
        if raw.startswith("-"):
            is_negative = True
            raw = raw[1:].strip()

        if not raw:
            return None

        # ================================================================
        # PARSE NUMERIC VALUE
        # ================================================================
        price_patterns = [
            r"(\d+[.,]\d{2})",   # 23.99 or 23,99
            r"(\.[0-9]{2})",     # .99
            r"(,[0-9]{2})",      # ,99
            r"(\d+)",            # 123
        ]

        val: Optional[float] = None

        for pattern in price_patterns:
            matches = re.findall(pattern, raw)
            if not matches:
                continue

            if pattern == r"(\d+)":
                price_str = max(matches, key=len)
                try:
                    val = float(price_str)
                    break
                except ValueError:
                    continue
            else:
                price_str = matches[-1].replace(",", ".")
                try:
                    val = float(price_str)
                    break
                except ValueError:
                    continue

        if val is None:
            return None

        # ================================================================
        # APPLY NEGATIVE FLAG
        # ================================================================
        if is_negative:
            val = -val

        # ================================================================
        #  VALIDATE RANGE (STRICT)
        # ================================================================
        if val == 0.0:
            return val

        # Negative allowed within -max_price
        if val < 0:
            return val if val >= -max_price else None

        # Positive must be within [min_price, max_price]
        if min_price <= val <= max_price:
            return val

        return None

    except (ValueError, TypeError):
        return None

def get_y_center(item: Dict[str, Any]) -> float:
    """Extract Y-center from grouped item."""
    bboxes = item.get("bboxes", [])
    if not bboxes:
        return 0.0
    ys = [b[1] if len(b) > 1 else 0 for b in bboxes]
    if not ys:
        return 0.0
    return float((min(ys) + max(ys)) / 2.0)

def is_item_price_candidate(
    words: List[str],
    labels: List[str],
    confs: List[float],
    idx: int,
    total_boundary: Optional[int],
    cfg: CordGeoConfig
) -> bool:

    word = words[idx]
    label = labels[idx] or "O"
    conf = float(confs[idx] or 0.0)

    if not looks_like_money(word):
        return False

    # Must be before summary block
    if total_boundary is not None and idx > total_boundary:
        return False

    # Reject if near summary keywords
    context_text = " ".join(
        words[max(0, idx-4):min(len(words), idx+4)]
    )
    if SUMMARY_CONTEXT_RE.search(context_text):
        return False

    # Accept if truly unlabeled
    if label == "O":
        return True

    # Accept if mislabeled as summary but low confidence
    if label in SUMMARYISH_LABELS and conf < cfg.min_label_confidence:
        return True

    return False

def demote_fake_tax_groups_to_O(
    words: List[str],
    labels: List[str],
    confs: List[float],
    cfg: CordGeoConfig,
) -> List[str]:
    """
    Demote mis-labeled TAX groups into 'O' so item extraction can catch them.

    Targets cases like:
      Bev (B-SUM.TAX), Bar (B-SUM.TAX), 3.00 (I-SUM.TAX)
    Because it has NO real tax keyword.

    Safety:
    - Do NOT demote pure numeric tax amounts like "0.61" alone.
    - Only demote when group contains alphabetic text AND looks like an item line.
    """
    n = min(len(words), len(labels), len(confs))
    new_labels = list(labels)

    taxish = {"B-TAX", "I-TAX", "B-SUM.TAX", "I-SUM.TAX"}

    i = 0
    while i < n:
        lab = (labels[i] or "O")

        if lab in taxish:
            # collect a contiguous taxish span
            j = i
            span_words: List[str] = []
            span_has_alpha = False
            span_has_money = False

            while j < n and ((labels[j] or "O") in taxish):
                w = (words[j] or "").strip()
                span_words.append(w)

                if re.search(r"[A-Za-z]", w):
                    span_has_alpha = True
                if looks_like_money_strict(w) or looks_like_money(w):
                    span_has_money = True

                j += 1

            span_text = " ".join([x for x in span_words if x]).strip()

            #  demote only if:
            # - no real tax keyword in the span text
            # - contains alphabetic tokens (Bev/Bar/etc)
            # - and has a money-looking token (3.00)
            if span_text and (not TAX_KEYWORD_RE.search(span_text)) and span_has_alpha and span_has_money:
                for k in range(i, j):
                    new_labels[k] = "O"

                if cfg.debug_print:
                    print(f"  [DEMOTE FAKE TAX] '{span_text}' -> relabeled as O")

            i = j
            continue

        i += 1

    return new_labels

def is_real_tax_line(text: str) -> bool:
    t = (text or "").lower()
    if TAX_KEYWORD_RE.search(t):
        return True
    if TIP_FEE_KEYWORD_RE.search(t):
        return False
    return False
# ============================================================
# FIND SUMMARY BOUNDARIES (NEW)
# ============================================================
def find_summary_boundary(
    labels: List[str],
) -> Optional[int]:
    """
    Find the index of the TOTAL label.
    Items after this index should be skipped (tips, change, etc.)
    
    Returns:
        Index of the last TOTAL-related label, or None if not found
    """
    
    last_total_idx = None
    
    for idx, label in enumerate(labels):
        label = label or "O"
        if label in ("B-SUM.TOTAL", "I-SUM.TOTAL"):
            last_total_idx = idx
    
    return last_total_idx

# ============================================================
# UNLABELED ITEM EXTRACTION (RAW TOKENS)
# ============================================================
SUMMARYISH_LABELS = {
    "B-SUM.SUBTOTAL","I-SUM.SUBTOTAL",
    "B-SUM.TAX","I-SUM.TAX",
    "B-SUM.TOTAL","I-SUM.TOTAL",
    "B-TAX","I-TAX",
}

SUMMARY_START_RE = re.compile(r"^(amount|subtotal|sub\s*total|tax|total)\b", re.I)

def find_summary_start_old(words: List[str], labels: List[str]) -> Optional[int]:
    for i, (w, lab) in enumerate(zip(words, labels)):
        w = (w or "").strip()
        lab = lab or "O"
        if lab.startswith("B-SUM") or lab in ("B-TAX",):
            return i
        if SUMMARY_START_RE.search(w):
            return i
    return None

def find_summary_start(
    words: List[str],
    labels: List[str],
    confs: List[float],
    cfg: CordGeoConfig
) -> Optional[int]:
    """
    Find the start index of the real summary block.
    Avoid false early stops caused by mislabels like: "Guest" -> B-SUM.TAX (low conf).
    """
    n = min(len(words), len(labels), len(confs))

    def has_money_near(i: int, win: int = 2) -> bool:
        a = max(0, i - win)
        b = min(n, i + win + 1)
        return any(looks_like_money(words[k]) for k in range(a, b))

    for i in range(n):
        w = (words[i] or "").strip()
        lab = (labels[i] or "O").strip()
        c = float(confs[i] or 0.0)

        # Strong text signal always wins
        if SUMMARY_START_RE.search(w):
            return i

        # Label-based stop, but only if it's credible
        if lab.startswith("B-SUM") or lab in ("B-TAX",):
            text_has_summary_signal = bool(SUMMARY_CONTEXT_RE.search(w))

            # confident label => accept
            if c >= cfg.min_label_confidence:
                return i

            # low confidence => accept only if text looks summary-ish
            if text_has_summary_signal:
                return i

            # special case for tax: accept only if word looks like tax OR money nearby
            if lab in ("B-TAX", "B-SUM.TAX") and (TAX_RE.search(w) or has_money_near(i)):
                return i

            # otherwise ignore (e.g., "Guest" mislabeled)
            continue

    return None

MONEY_TOKEN_RE = re.compile(r"^\$?\s*\d{1,7}(?:[.,]\d{2})\s*$")

def looks_like_money(tok: str) -> bool:
    return bool(MONEY_TOKEN_RE.fullmatch((tok or "").strip()))

def is_effectively_O(label: str, conf: float, cfg) -> bool:
    """Treat low-confidence summary-ish labels as O (mislabeled prices)."""
    label = label or "O"
    if label == "O":
        return True
    if cfg.use_fuzzy_fallback and (label in SUMMARYISH_LABELS) and (float(conf) < cfg.min_label_confidence):
        return True
    return False

def safe_skip_keyword(engine, token: str) -> bool:
    """
    Avoid fuzzy false positives on short tokens like TOFU/PAD/IN.
    Use exact match for short tokens; fuzzy for longer strings.
    """
    t = (token or "").strip()
    norm = normalize_token(t)
    if not norm:
        return False
    if len(norm) <= 4:
        # exact only
        return any(norm == normalize_token(k) for k in engine.raw_keywords)
    # fuzzy for longer strings
    return engine.any_match(t)


STRICT_MONEY_TOKEN_RE = re.compile(r"^\$?\s*\d{1,7}(?:[.,]\d{2})\s*$")

def looks_like_money_strict(tok: str) -> bool:
    return bool(STRICT_MONEY_TOKEN_RE.fullmatch((tok or "").strip()))

def extract_unlabeled_items_from_raw_tokens(
    words: List[str],
    labels: List[str],
    confs: List[float],
    cfg: CordGeoConfig,
    labeled_token_indices: set,
) -> List[Dict[str, Any]]:

    items: List[Dict[str, Any]] = []

    summary_start = find_summary_start(words, labels, confs, cfg)
    total_boundary = find_summary_boundary(labels)

    stop_at: Optional[int] = None
    if total_boundary is not None:
        stop_at = total_boundary + 1
    elif summary_start is not None:
        stop_at = summary_start

    menu_keywords = FuzzyKeywordEngine(
        SUBTOTAL_KEYWORDS + TOTAL_KEYWORDS + TAX_KEYWORDS + DISCOUNT_KEYWORDS,
        max_distance=cfg.fuzzy_distance_threshold,
    )

    post_total_keywords = FuzzyKeywordEngine(
        TIP_SUGGESTION_KEYWORDS,
        max_distance=cfg.fuzzy_distance_threshold,
    )

    if cfg.debug_print:
        print(f"\n[UNLABELED EXTRACTION] Processing {len(words)} raw tokens...")
        if stop_at is not None:
            print(
                f"  ⊘ Stopping at stop_at index {stop_at} "
                f"(summary_start={summary_start}, total_boundary={total_boundary})"
            )

    processed_token_indices: Set[int] = set(labeled_token_indices)

    # ================================================================
    # PASS 1: embedded price tokens like "Tofu$0.00"
    # ================================================================
    if cfg.debug_print:
        print("  [PASS 1] Scanning for tokens with embedded prices...")

    for idx, (word, label, conf) in enumerate(zip(words, labels, confs)):
        if stop_at is not None and idx >= stop_at:
            continue
        if idx in processed_token_indices:
            continue

        label = label or "O"
        if not is_effectively_O(label, float(conf), cfg):
            continue

        if safe_skip_keyword(menu_keywords, word):
            if cfg.debug_print:
                print(f"    ⊘ {str(word):30} (menu keyword, skipped)")
            continue

        if safe_skip_keyword(post_total_keywords, word):
            if cfg.debug_print:
                print(f"    ⊘ {str(word):30} (tip/post-total keyword, skipped)")
            continue

        money_matches = MONEY_FIND_RE_LOCAL.findall(word or "")
        if not money_matches:
            continue

        price_split_patterns = [
            r"^(.+?)\$\s*(\d+[.,]\d{2})$",
            r"^(.+?)\$\s*(\d+)$",
        ]

        extracted_name = None
        extracted_price = None

        for pattern in price_split_patterns:
            match = re.match(pattern, (word or "").strip())
            if not match:
                continue

            name_part = clean_ocr_noise(match.group(1).strip())
            price_part = match.group(2).strip()

            if not name_part:
                continue

            price_val = safe_parse_price(price_part, cfg.unlabeled_min_price, cfg.unlabeled_max_price)
            # ✅ skip 0.00 in unlabeled mode (prevents "PM → 0.00")
            if price_val == 0.0 and cfg.unlabeled_min_price > 0:
                price_val = None

            if price_val is not None:
                extracted_name = name_part
                extracted_price = price_val
                break

        if extracted_price is not None and extracted_name:
            items.append({
                "name": extracted_name,
                "price": extracted_price,
                "price_tokens": [word],
                "price_confidence": float(conf),
                "source": "unlabeled_embedded",
            })
            if cfg.debug_print:
                print(f"    ✓ {extracted_name:25} → ${extracted_price:.2f} (embedded)")
            processed_token_indices.add(idx)

    # ================================================================
    # PASS 2: name tokens followed by a price token
    # ================================================================
    if cfg.debug_print:
        print("  [PASS 2] Scanning for multi-word items with separate price token...")

    i = 0
    while i < len(words):
        if stop_at is not None and i >= stop_at:
            break
        if i in processed_token_indices:
            i += 1
            continue

        word = words[i]
        label = labels[i] or "O"
        conf = float(confs[i] or 0.0)

        if safe_skip_keyword(menu_keywords, word) or safe_skip_keyword(post_total_keywords, word):
            i += 1
            continue

        # ✅ NEW: strict money-shape gate (kills "1" tokens -> BeachB/Db/All Star Meal)
        if not looks_like_money_strict(word):
            i += 1
            continue

        # parse price with strict range
        price_val = safe_parse_price(word, cfg.unlabeled_min_price, cfg.unlabeled_max_price)
        # ✅ skip 0.00 in unlabeled mode
        if price_val == 0.0 and cfg.unlabeled_min_price > 0:
            price_val = None

        if price_val is not None and is_effectively_O(label, conf, cfg):
            context_left = " ".join(words[max(0, i-5):i])
            context_right = " ".join(words[i+1:i+6])
            context = (context_left + " " + context_right).lower()

            if SUMMARY_CONTEXT_RE.search(context):
                if cfg.debug_print:
                    print(f"    ⊘ Skipping {word} (near summary context)")
                i += 1
                continue

            name_parts: List[str] = []
            j = i - 1
            MAX_BACKTRACK = 6
            backtrack_count = 0

            while j >= 0 and j not in processed_token_indices and backtrack_count < MAX_BACKTRACK:
                prev_word = words[j]
                prev_label = labels[j] or "O"
                prev_conf = float(confs[j] or 0.0)
                pw = (prev_word or "").strip()

                # strong header/metadata stops
                if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", pw): break
                if re.search(r"\d{1,2}:\d{2}(?::\d{2})?", pw): break
                if re.search(r"\d{3}[-]\d{3}[-]\d{4}", pw): break
                if re.search(r"\b(cashier|order|ticket|thank|visit|restaurant)\b", pw, re.I): break
                if "#" in pw: break
                if sum(c.isdigit() for c in pw) >= 4: break

                if not is_effectively_O(prev_label, prev_conf, cfg): break
                if safe_skip_keyword(menu_keywords, pw): break
                if safe_skip_keyword(post_total_keywords, pw): break
                if looks_like_money_strict(pw): break  # stop at previous money token

                name_parts.insert(0, pw)
                j -= 1
                backtrack_count += 1

            name = clean_ocr_noise(" ".join(name_parts).strip()) if name_parts else ""
            if name:
                items.append({
                    "name": name,
                    "price": price_val,
                    "price_tokens": [word],
                    "price_confidence": conf,
                    "source": "unlabeled_separate",
                })
                if cfg.debug_print:
                    print(f"    ✓ {name:25} → ${price_val:.2f} (separate)")

                processed_token_indices.add(i)
                for k in range(j + 1, i):
                    processed_token_indices.add(k)

        i += 1

    return items

# ============================================================
# PATTERN DETECTION
# ============================================================
def detect_item_pattern(
    grouped_items: List[Dict[str, Any]],
    cfg: CordGeoConfig
) -> str:
    """
    Detect dominant item pattern across entire receipt.
    
    Returns:
        "name_then_price" - NAME label appears before PRICE label
        "price_then_name" - PRICE label appears before NAME label
        "mixed" - Both patterns present significantly
    """
    
    name_before_price = 0
    price_before_name = 0
    
    for i, item in enumerate(grouped_items):
        label = item.get("label", "O")
        
        if label == "B-MENU.PRICE":
            for j in range(max(0, i - cfg.price_name_search_window), i):
                prev_label = grouped_items[j].get("label", "O")
                if prev_label in ("B-MENU.NM", "I-MENU.NM"):
                    name_before_price += 1
                    break
            
            for j in range(i + 1, min(len(grouped_items), i + cfg.price_name_search_window + 1)):
                next_label = grouped_items[j].get("label", "O")
                if next_label in ("B-MENU.NM", "I-MENU.NM"):
                    price_before_name += 1
                    break
    
    if cfg.debug_print:
        print(f"\n[PATTERN DETECTION]")
        print(f"  Name→Price occurrences: {name_before_price}")
        print(f"  Price→Name occurrences: {price_before_name}")
    
    if name_before_price > price_before_name * 1.5:
        pattern = "name_then_price"
        if cfg.debug_print:
            print(f"  ✓ Detected pattern: NAME→PRICE")
    elif price_before_name > name_before_price * 1.5:
        pattern = "price_then_name"
        if cfg.debug_print:
            print(f"  ✓ Detected pattern: PRICE→NAME")
    else:
        pattern = "mixed"
        if cfg.debug_print:
            print(f"  ⚠ Detected pattern: MIXED (both present)")
    
    return pattern


def detect_regions(
    grouped_items: List[Dict[str, Any]],
    cfg: CordGeoConfig
) -> List[Tuple[int, int]]:
    """
    Detect region boundaries based on Y-gap.
    
    Returns:
        List of (start_idx, end_idx) tuples for each region
    """
    
    if not grouped_items:
        return []
    
    y_centers = [get_y_center(item) for item in grouped_items]
    
    regions = []
    current_region_start = 0
    
    for i in range(1, len(y_centers)):
        gap = y_centers[i] - y_centers[i-1]
        
        if gap > cfg.region_y_gap_threshold:
            regions.append((current_region_start, i))
            current_region_start = i
    
    regions.append((current_region_start, len(grouped_items)))
    
    if cfg.debug_print:
        print(f"\n[REGION DETECTION] Found {len(regions)} regions:")
        for reg_idx, (start, end) in enumerate(regions):
            print(f"  Region {reg_idx}: items {start}-{end-1}")
    
    return regions


def detect_region_pattern(
    region_items: List[Dict[str, Any]],
    cfg: CordGeoConfig
) -> str:
    """Detect pattern within a single region."""
    
    name_before_price = 0
    price_before_name = 0
    
    for i, item in enumerate(region_items):
        label = item.get("label", "O")
        
        if label == "B-MENU.PRICE":
            for j in range(max(0, i - 2), i):
                if region_items[j].get("label", "O") in ("B-MENU.NM", "I-MENU.NM"):
                    name_before_price += 1
                    break
            
            for j in range(i + 1, min(len(region_items), i + 3)):
                if region_items[j].get("label", "O") in ("B-MENU.NM", "I-MENU.NM"):
                    price_before_name += 1
                    break
    
    if name_before_price > price_before_name:
        return "name_then_price"
    elif price_before_name > name_before_price:
        return "price_then_name"
    else:
        return "undecided"

# ============================================================
# EXTRACTION: PATTERN 1 - NAME → PRICE
# ============================================================
def extract_name_then_price_1(
    region_items: List[Dict[str, Any]],
    cfg: CordGeoConfig,
    processed_indices: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], set]:
    """
    Extract items following NAME→PRICE pattern.

    Updates / Fixes:
    - Correct handling of 2 consecutive prices:
        A) Line total + unit price + "each/ea/@"
           Koobideh x7   $115.50   $16.50 each
           -> price=$115.50, unit_price=$16.50
        B) Unit price + line total (no "each"), simple multiple relationship
           $55.00  $110.00  (qty=2)
           -> price=$110.00, unit_price=$55.00   (so your case becomes 110.00 ✅)
    - Prevents the 2nd price from becoming an orphan later (consumes it)
    - Skips orphan prices that are gratuity/tip suggestions (e.g., 18% Gratuity $22.40)
    - Skips "each" unit prices from being added as ITEM_NOT_DETECTED in PASS 2
    """

    if processed_indices is None:
        processed_indices = set()

    items: List[Dict[str, Any]] = []
    processed_price_indices: Set[int] = set()

    EACH_RE = re.compile(r"\b(each|ea\.?|@)\b", re.I)
    TIP_RE = re.compile(r"\b(gratuity|grat|tip|tipping|service)\b|\b\d{1,2}\s*%\b", re.I)
    QTY_RE = re.compile(r"(?:^|\s)(?:x|×)\s*(\d{1,2})(?:\b|$)", re.I)

    def group_text(i: int) -> str:
        return (clean_ocr_noise(region_items[i].get("text", "")) or "").strip()

    def group_tokens(i: int) -> List[str]:
        return region_items[i].get("tokens", []) or []

    def window_ctx(i: int, win: int = 2) -> str:
        a = max(0, i - win)
        b = min(len(region_items), i + win + 1)
        parts: List[str] = []
        for k in range(a, b):
            t = group_text(k)
            if t:
                parts.append(t)
            parts.extend([x for x in group_tokens(k) if x])
        return " ".join(parts).strip()

    def near_tip_or_gratuity(price_group_idx: int, win: int = 2) -> bool:
        return bool(TIP_RE.search(window_ctx(price_group_idx, win=win)))

    def is_each_context(after_price2_text: str, price2_group_idx: int) -> bool:
        # "each" may appear as its own O-group right after price2, or within nearby ctx
        if after_price2_text and EACH_RE.search(after_price2_text):
            return True
        return bool(EACH_RE.search(window_ctx(price2_group_idx, win=2)))

    def get_qty_from_name(name: str) -> Optional[int]:
        """
        Try to infer quantity from name tokens like 'x7', '×7', 'Koobideh x 7'
        """
        m = QTY_RE.search(name or "")
        if not m:
            return None
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except Exception:
            return None
        return None

    def choose_two_prices(
        price1: float,
        price2: float,
        after_price2_text: str,
        price1_group_idx: int,
        price2_group_idx: int,
        name_for_qty: str,
    ) -> Tuple[float, Optional[float]]:
        """
        Decide which one is the item price and optionally unit price.

        Return:
          (final_price, unit_price_or_none)

        Priority:
        1) If there is clear "each/ea/@": line total = larger, unit = smaller (usually)
           but if price1 is already clearly total (e.g., much larger), keep it.
        2) If qty is detected (e.g., x7): prefer total ≈ unit*qty (within tolerance).
        3) Else if one is a clean multiple of the other (2..12): treat larger as total.
        4) Else default to price1.
        """
        hi = max(price1, price2)
        lo = min(price1, price2)

        # 1) explicit EACH context => second price is unit most of the time
        if is_each_context(after_price2_text, price2_group_idx):
            # If price2 is the unit, and price1 is the total, that's perfect.
            # If they came reversed, still choose hi as total.
            return hi, lo

        # 2) quantity hint from name
        qty = get_qty_from_name(name_for_qty)
        if qty and lo > 0:
            # Check if either matches unit*qty
            # If lo*qty ≈ hi => hi is total, lo is unit
            if abs((lo * qty) - hi) <= max(0.05 * hi, 0.10):
                return hi, lo
            # If hi*qty ≈ lo => lo is total, hi is unit (rare, but handle)
            if abs((hi * qty) - lo) <= max(0.05 * lo, 0.10):
                return lo, hi

        # 3) multiple relationship (your $55 + $110 case hits this)
        if lo > 0:
            ratio = hi / lo
            nearest = int(round(ratio))
            if 2 <= nearest <= 12 and abs(ratio - nearest) <= 0.05:
                return hi, lo

        # 4) default
        return price1, None

    # ------------------------------------------------------------
    # PASS 1: Standard NAME→PRICE extraction (+ two-price logic)
    # ------------------------------------------------------------
    for idx, item in enumerate(region_items):
        label = item.get("label", "O")

        if label != "B-MENU.NM":
            continue
        if label.startswith("B-SUM.") or label.startswith("B-TAX"):
            continue

        text = group_text(idx)
        name_parts = [text] if text else []

        j = idx + 1
        while j < len(region_items):
            next_label = region_items[j].get("label", "O")
            next_text = group_text(j)

            if next_label == "I-MENU.NM":
                if next_text:
                    name_parts.append(next_text)
                j += 1
                continue

            if next_label == "B-MENU.PRICE":
                # skip if this price is clearly gratuity/tip context
                if near_tip_or_gratuity(j):
                    break

                price1 = safe_parse_price(next_text, 0.0, cfg.max_price)
                if price1 is None:
                    break

                # second consecutive price?
                price2 = None
                after_price2_text = ""
                price2_idx = None

                if j + 1 < len(region_items) and region_items[j + 1].get("label", "O") == "B-MENU.PRICE":
                    t2 = group_text(j + 1)
                    price2 = safe_parse_price(t2, 0.0, cfg.max_price)
                    price2_idx = j + 1
                    if j + 2 < len(region_items):
                        after_price2_text = group_text(j + 2)

                name = " ".join([p for p in name_parts if p]).strip()
                if not name:
                    name = cfg.default_item_name

                final_price = price1
                unit_price = None

                if price2 is not None and price2_idx is not None:
                    final_price, unit_price = choose_two_prices(
                        price1=price1,
                        price2=price2,
                        after_price2_text=after_price2_text,
                        price1_group_idx=j,
                        price2_group_idx=price2_idx,
                        name_for_qty=name,
                    )

                    # consume 2nd price so it won't become orphan later
                    processed_price_indices.add(price2_idx)

                out: Dict[str, Any] = {
                    "name": name,
                    "price": final_price,
                    "price_tokens": region_items[j].get("tokens", []) or [],
                    "price_confidence": float(region_items[j].get("avg_confidence", 0.0) or 0.0),
                    "source": "labeled_name_price",
                }
                if unit_price is not None:
                    out["unit_price"] = unit_price

                items.append(out)

                if cfg.debug_print:
                    if unit_price is not None:
                        print(f"    ✓ {name:30} → ${final_price:.2f} (unit=${unit_price:.2f})")
                    else:
                        print(f"    ✓ {name:30} → ${final_price:.2f}")

                processed_price_indices.add(j)
                break

            if next_label == "B-MENU.NM" or next_label.startswith("B-SUM."):
                break

            j += 1

    # ------------------------------------------------------------
    # PASS 1b: Orphan I-MENU.NM handling
    # ------------------------------------------------------------
    if cfg.debug_print:
        print("    [PASS 1b] Scanning for orphan I-MENU.NM tokens...")

    for idx, item in enumerate(region_items):
        if idx in processed_price_indices:
            continue

        label = item.get("label", "O")
        if label != "I-MENU.NM":
            continue

        has_preceding_b_menu = False
        for j in range(max(0, idx - 5), idx):
            if region_items[j].get("label", "O") == "B-MENU.NM":
                between = region_items[j + 1:idx]
                if any(x.get("label", "O") == "B-MENU.PRICE" for x in between):
                    continue
                has_preceding_b_menu = True
                break

        if has_preceding_b_menu:
            continue

        text = group_text(idx)
        name_parts = [text] if text else []

        # prepend previous O token with letters (e.g., "1.CHEESE")
        if idx - 1 >= 0:
            prev = group_text(idx - 1)
            prev_alpha = re.sub(r"[^A-Za-z]+", "", prev)
            if prev_alpha and not SUMMARY_CONTEXT_RE.search(prev):
                prev_clean = re.sub(r"^\d+\W*", "", prev).strip()
                if prev_clean:
                    name_parts.insert(0, prev_clean)

        j = idx + 1
        while j < len(region_items):
            next_label = region_items[j].get("label", "O")
            next_text = group_text(j)

            if next_label == "I-MENU.NM":
                if next_text:
                    name_parts.append(next_text)
                j += 1
                continue

            if next_label == "B-MENU.PRICE":
                if near_tip_or_gratuity(j):
                    break

                price = safe_parse_price(next_text, 0.0, cfg.max_price)
                if price is not None:
                    name = " ".join([p for p in name_parts if p]).strip() or cfg.default_item_name
                    items.append({
                        "name": name,
                        "price": price,
                        "price_tokens": region_items[j].get("tokens", []) or [],
                        "price_confidence": float(region_items[j].get("avg_confidence", 0.0) or 0.0),
                        "source": "labeled_orphan_i_menu",
                    })
                    if cfg.debug_print:
                        print(f"    ✓ {name:30} → ${price:.2f} (orphan I-MENU.NM)")
                    processed_price_indices.add(j)
                break

            if next_label == "B-MENU.NM" or next_label.startswith("B-SUM."):
                break

            j += 1

    # ------------------------------------------------------------
    # PASS 2: Orphan prices (skip tip/gratuity AND skip unit-price 'each')
    # ------------------------------------------------------------
    if cfg.debug_print:
        print("    [PASS 2] Scanning for orphan prices (no name found)...")

    for idx, item in enumerate(region_items):
        if idx in processed_price_indices:
            continue

        label = item.get("label", "O")
        if label != "B-MENU.PRICE":
            continue
        if label.startswith("B-SUM.") or label.startswith("B-TAX"):
            continue

        # ✅ skip gratuity/tip prices like $22.40
        if near_tip_or_gratuity(idx):
            if cfg.debug_print:
                print(f"    ⊘ {group_text(idx)} (tip/gratuity context, skipped)")
            processed_price_indices.add(idx)
            continue

        # skip "each/ea/@": unit-price token shouldn't become ITEM_NOT_DETECTED
        # (it might appear nearby even if it's not in a grouped text)
        if EACH_RE.search(window_ctx(idx, win=2)):
            if cfg.debug_print:
                print(f"    ⊘ {group_text(idx)} (each/unit-price context, skipped)")
            processed_price_indices.add(idx)
            continue

        text = group_text(idx)
        price = safe_parse_price(text, 0.0, cfg.max_price)
        if price is None:
            continue

        items.append({
            "name": "ITEM_NOT_DETECTED",
            "price": price,
            "price_tokens": item.get("tokens", []) or [],
            "price_confidence": float(item.get("avg_confidence", 0.0) or 0.0),
            "source": "labeled_orphan_price",
        })

        if cfg.debug_print:
            print(f"    ! ${price:.2f} → ITEM_NOT_DETECTED (orphan price, no name found)")

        processed_price_indices.add(idx)

    processed_indices.update(processed_price_indices)
    return items, processed_indices

def extract_name_then_price(
    region_items: List[Dict[str, Any]],
    cfg: CordGeoConfig,
    processed_indices: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], set]:
    """
    Extract items following NAME→PRICE pattern.

    Updates / Fixes:
    - Correct handling of 2 consecutive prices:
        A) Line total + unit price + "each/ea/@"
           Koobideh x7   $115.50   $16.50 each
           -> price=$115.50, unit_price=$16.50
        B) Unit price + line total (no "each"), simple multiple relationship
           $55.00  $110.00  (qty=2)
           -> price=$110.00, unit_price=$55.00
    - NEW: Pending-name pairing for:
           NAME1 (no price) + NAME2 + PRICE2 + PRICE1
           -> NAME2 gets PRICE2, NAME1 gets PRICE1  (fixes your 11.99 case)
    - Prevents the 2nd price from becoming an orphan later (consumes it)
    - Skips orphan prices that are gratuity/tip suggestions (e.g., 18% Gratuity $22.40)
    - Skips "each" unit prices from being added as ITEM_NOT_DETECTED in PASS 2
    """

    if processed_indices is None:
        processed_indices = set()

    items: List[Dict[str, Any]] = []
    processed_price_indices: Set[int] = set()

    EACH_RE = re.compile(r"\b(each|ea\.?|@)\b", re.I)
    TIP_RE = re.compile(r"\b(gratuity|grat|tip|tipping|service)\b|\b\d{1,2}\s*%\b", re.I)
    QTY_RE = re.compile(r"(?:^|\s)(?:x|×)\s*(\d{1,2})(?:\b|$)", re.I)

    def group_text(i: int) -> str:
        return (clean_ocr_noise(region_items[i].get("text", "")) or "").strip()

    def group_tokens(i: int) -> List[str]:
        return region_items[i].get("tokens", []) or []

    def window_ctx(i: int, win: int = 2) -> str:
        a = max(0, i - win)
        b = min(len(region_items), i + win + 1)
        parts: List[str] = []
        for k in range(a, b):
            t = group_text(k)
            if t:
                parts.append(t)
            parts.extend([x for x in group_tokens(k) if x])
        return " ".join(parts).strip()

    def near_tip_or_gratuity(price_group_idx: int, win: int = 2) -> bool:
        return bool(TIP_RE.search(window_ctx(price_group_idx, win=win)))

    def is_each_context(after_price2_text: str, price2_group_idx: int) -> bool:
        if after_price2_text and EACH_RE.search(after_price2_text):
            return True
        return bool(EACH_RE.search(window_ctx(price2_group_idx, win=2)))

    def get_qty_from_name(name: str) -> Optional[int]:
        m = QTY_RE.search(name or "")
        if not m:
            return None
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except Exception:
            return None
        return None

    def choose_two_prices(
        price1: float,
        price2: float,
        after_price2_text: str,
        price1_group_idx: int,
        price2_group_idx: int,
        name_for_qty: str,
    ) -> Tuple[float, Optional[float]]:
        hi = max(price1, price2)
        lo = min(price1, price2)

        # 1) explicit EACH context => total=hi, unit=lo
        if is_each_context(after_price2_text, price2_group_idx):
            return hi, lo

        # 2) qty from name: if lo*qty ≈ hi
        qty = get_qty_from_name(name_for_qty)
        if qty and lo > 0:
            if abs((lo * qty) - hi) <= max(0.05 * hi, 0.10):
                return hi, lo
            if abs((hi * qty) - lo) <= max(0.05 * lo, 0.10):
                return lo, hi

        # 3) simple multiple relationship
        if lo > 0:
            ratio = hi / lo
            nearest = int(round(ratio))
            if 2 <= nearest <= 12 and abs(ratio - nearest) <= 0.05:
                return hi, lo

        # 4) default
        return price1, None

    def looks_like_unit_total_case(
        price1: float,
        price2: float,
        after_price2_text: str,
        price1_group_idx: int,
        price2_group_idx: int,
        name_for_qty: str,
    ) -> bool:
        """
        Conservative gate: return True if we should treat the two prices as (total, unit)
        rather than (current, pending).
        """
        # explicit each
        if is_each_context(after_price2_text, price2_group_idx):
            return True

        hi = max(price1, price2)
        lo = min(price1, price2)

        # qty hint
        qty = get_qty_from_name(name_for_qty)
        if qty and lo > 0:
            if abs((lo * qty) - hi) <= max(0.05 * hi, 0.10):
                return True
            if abs((hi * qty) - lo) <= max(0.05 * lo, 0.10):
                return True

        # multiple relationship
        if lo > 0:
            ratio = hi / lo
            nearest = int(round(ratio))
            if 2 <= nearest <= 12 and abs(ratio - nearest) <= 0.05:
                return True

        return False

    # ------------------------------------------------------------
    # NEW: Pending-name buffer for NAME1(no price) -> NAME2 + 2 prices
    # ------------------------------------------------------------
    pending_name: Optional[str] = None
    pending_name_conf: float = 0.0  # optional, keep simple

    # ------------------------------------------------------------
    # PASS 1: Standard NAME→PRICE extraction (+ two-price logic + pending fix)
    # ------------------------------------------------------------
    for idx, item in enumerate(region_items):
        label = item.get("label", "O")

        if label != "B-MENU.NM":
            continue
        if label.startswith("B-SUM.") or label.startswith("B-TAX"):
            continue

        text = group_text(idx)
        name_parts = [text] if text else []

        j = idx + 1
        saw_price = False

        while j < len(region_items):
            next_label = region_items[j].get("label", "O")
            next_text = group_text(j)

            if next_label == "I-MENU.NM":
                if next_text:
                    name_parts.append(next_text)
                j += 1
                continue

            if next_label == "B-MENU.PRICE":
                saw_price = True

                # skip if this price is clearly gratuity/tip context
                if near_tip_or_gratuity(j):
                    break

                price1 = safe_parse_price(next_text, 0.0, cfg.max_price)
                if price1 is None:
                    break

                # second consecutive price?
                price2 = None
                after_price2_text = ""
                price2_idx = None

                if j + 1 < len(region_items) and region_items[j + 1].get("label", "O") == "B-MENU.PRICE":
                    t2 = group_text(j + 1)
                    price2 = safe_parse_price(t2, 0.0, cfg.max_price)
                    price2_idx = j + 1
                    if j + 2 < len(region_items):
                        after_price2_text = group_text(j + 2)

                name = " ".join([p for p in name_parts if p]).strip()
                if not name:
                    name = cfg.default_item_name

                final_price = price1
                unit_price = None

                # --------------------------
                # Two consecutive prices logic
                # --------------------------
                if price2 is not None and price2_idx is not None:
                    # Decide if this is unit/total case OR pending-name case
                    if pending_name and not looks_like_unit_total_case(
                        price1=price1,
                        price2=price2,
                        after_price2_text=after_price2_text,
                        price1_group_idx=j,
                        price2_group_idx=price2_idx,
                        name_for_qty=name,
                    ):
                        # ✅ pending-name case:
                        # current item gets price1, pending item gets price2
                        # (this fixes Margherita + Coke + 2.29 + 11.99)
                        items.append({
                            "name": name,
                            "price": price1,
                            "price_tokens": region_items[j].get("tokens", []) or [],
                            "price_confidence": float(region_items[j].get("avg_confidence", 0.0) or 0.0),
                            "source": "labeled_name_price",
                        })
                        items.append({
                            "name": pending_name,
                            "price": float(price2),
                            "price_tokens": region_items[price2_idx].get("tokens", []) or [],
                            "price_confidence": float(region_items[price2_idx].get("avg_confidence", 0.0) or 0.0),
                            "source": "labeled_pending_name_price",
                        })

                        if cfg.debug_print:
                            print(f"    ✓ {name:30} → ${price1:.2f}")
                            print(f"    ✓ {pending_name:30} → ${float(price2):.2f}  [pending-name-fix]")

                        # consume both prices
                        processed_price_indices.add(j)
                        processed_price_indices.add(price2_idx)

                        # clear pending after use
                        pending_name = None
                        pending_name_conf = 0.0
                        break

                    # Otherwise: treat as unit/total (your original behavior)
                    final_price, unit_price = choose_two_prices(
                        price1=price1,
                        price2=price2,
                        after_price2_text=after_price2_text,
                        price1_group_idx=j,
                        price2_group_idx=price2_idx,
                        name_for_qty=name,
                    )
                    processed_price_indices.add(price2_idx)

                out: Dict[str, Any] = {
                    "name": name,
                    "price": final_price,
                    "price_tokens": region_items[j].get("tokens", []) or [],
                    "price_confidence": float(region_items[j].get("avg_confidence", 0.0) or 0.0),
                    "source": "labeled_name_price",
                }
                if unit_price is not None:
                    out["unit_price"] = unit_price

                items.append(out)

                if cfg.debug_print:
                    if unit_price is not None:
                        print(f"    ✓ {name:30} → ${final_price:.2f} (unit=${unit_price:.2f})")
                    else:
                        print(f"    ✓ {name:30} → ${final_price:.2f}")

                processed_price_indices.add(j)
                break

            # If we hit a new name before a price: store pending (only when no price for this name)
            if next_label == "B-MENU.NM":
                break

            if next_label.startswith("B-SUM."):
                break

            j += 1

        # If name ended without price before next name/summary → set pending (conservative)
        if not saw_price:
            name = " ".join([p for p in name_parts if p]).strip()
            if name:
                pending_name = name
                pending_name_conf = float(item.get("avg_confidence", 0.0) or 0.0)
                if cfg.debug_print:
                    print(f"    · pending name stored: {pending_name}")

    # ------------------------------------------------------------
    # PASS 1b: Orphan I-MENU.NM handling
    # ------------------------------------------------------------
    if cfg.debug_print:
        print("    [PASS 1b] Scanning for orphan I-MENU.NM tokens...")

    for idx, item in enumerate(region_items):
        if idx in processed_price_indices:
            continue

        label = item.get("label", "O")
        if label != "I-MENU.NM":
            continue

        has_preceding_b_menu = False
        for j in range(max(0, idx - 5), idx):
            if region_items[j].get("label", "O") == "B-MENU.NM":
                between = region_items[j + 1:idx]
                if any(x.get("label", "O") == "B-MENU.PRICE" for x in between):
                    continue
                has_preceding_b_menu = True
                break

        if has_preceding_b_menu:
            continue

        text = group_text(idx)
        name_parts = [text] if text else []

        # prepend previous O token with letters (e.g., "1.CHEESE")
        if idx - 1 >= 0:
            prev = group_text(idx - 1)
            prev_alpha = re.sub(r"[^A-Za-z]+", "", prev)
            if prev_alpha and not SUMMARY_CONTEXT_RE.search(prev):
                prev_clean = re.sub(r"^\d+\W*", "", prev).strip()
                if prev_clean:
                    name_parts.insert(0, prev_clean)

        j = idx + 1
        while j < len(region_items):
            next_label = region_items[j].get("label", "O")
            next_text = group_text(j)

            if next_label == "I-MENU.NM":
                if next_text:
                    name_parts.append(next_text)
                j += 1
                continue

            if next_label == "B-MENU.PRICE":
                if near_tip_or_gratuity(j):
                    break

                price = safe_parse_price(next_text, 0.0, cfg.max_price)
                if price is not None:
                    name = " ".join([p for p in name_parts if p]).strip() or cfg.default_item_name
                    items.append({
                        "name": name,
                        "price": price,
                        "price_tokens": region_items[j].get("tokens", []) or [],
                        "price_confidence": float(region_items[j].get("avg_confidence", 0.0) or 0.0),
                        "source": "labeled_orphan_i_menu",
                    })
                    if cfg.debug_print:
                        print(f"    ✓ {name:30} → ${price:.2f} (orphan I-MENU.NM)")
                    processed_price_indices.add(j)
                break

            if next_label == "B-MENU.NM" or next_label.startswith("B-SUM."):
                break

            j += 1

    # ------------------------------------------------------------
    # PASS 2: Orphan prices (skip tip/gratuity AND skip unit-price 'each')
    # ------------------------------------------------------------
    if cfg.debug_print:
        print("    [PASS 2] Scanning for orphan prices (no name found)...")

    for idx, item in enumerate(region_items):
        if idx in processed_price_indices:
            continue

        label = item.get("label", "O")
        if label != "B-MENU.PRICE":
            continue
        if label.startswith("B-SUM.") or label.startswith("B-TAX"):
            continue

        # skip gratuity/tip prices
        if near_tip_or_gratuity(idx):
            if cfg.debug_print:
                print(f"    ⊘ {group_text(idx)} (tip/gratuity context, skipped)")
            processed_price_indices.add(idx)
            continue

        # skip unit-price token in "each/ea/@"
        if EACH_RE.search(window_ctx(idx, win=2)):
            if cfg.debug_print:
                print(f"    ⊘ {group_text(idx)} (each/unit-price context, skipped)")
            processed_price_indices.add(idx)
            continue

        text = group_text(idx)
        price = safe_parse_price(text, 0.0, cfg.max_price)
        if price is None:
            continue

        items.append({
            "name": "ITEM_NOT_DETECTED",
            "price": price,
            "price_tokens": item.get("tokens", []) or [],
            "price_confidence": float(item.get("avg_confidence", 0.0) or 0.0),
            "source": "labeled_orphan_price",
        })

        if cfg.debug_print:
            print(f"    ! ${price:.2f} → ITEM_NOT_DETECTED (orphan price, no name found)")

        processed_price_indices.add(idx)

    processed_indices.update(processed_price_indices)
    return items, processed_indices
# ============================================================
# EXTRACTION: PATTERN 2 - PRICE → NAME
# ============================================================
def extract_price_then_name(
    region_items: List[Dict[str, Any]],
    cfg: CordGeoConfig,
    processed_indices: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], set]:
    """
    Extract items following PRICE→NAME pattern.

    Combined logic:
    - Keep your original "one price -> look forward for a name" behavior
    - NEW: handle "stacked prices then stacked names" by buffering prices and
      then attaching to subsequent names in reverse order (LIFO), which matches
      common receipt printing patterns.

    Returns:
        (items, processed_indices_set)
    """

    if processed_indices is None:
        processed_indices = set()

    items: List[Dict[str, Any]] = []
    processed_price_indices: Set[int] = set()

    def gtext(i: int) -> str:
        return (clean_ocr_noise(region_items[i].get("text", "")) or "").strip()

    def glabel(i: int) -> str:
        return (region_items[i].get("label", "O") or "O").strip()

    def gconf(i: int) -> float:
        try:
            return float(region_items[i].get("avg_confidence", 0.0) or 0.0)
        except Exception:
            return 0.0

    def is_summaryish_label(lb: str) -> bool:
        return lb.startswith("B-SUM.") or lb.startswith("I-SUM.") or lb.startswith("B-TAX") or lb.startswith("I-TAX")

    # ------------------------------------------------------------
    # NEW: price buffer for stacked PRICE→NAME patterns
    # Each entry: (price_group_idx, price_value)
    # ------------------------------------------------------------
    price_buffer: List[Tuple[int, float]] = []

    i = 0
    while i < len(region_items):
        lb = glabel(i)

        # stop on summary region
        if is_summaryish_label(lb):
            break

        # ----------------------------
        # If we see a price, try the old forward-search first.
        # If it fails, buffer it for later pairing.
        # ----------------------------
        if lb == "B-MENU.PRICE" and i not in processed_indices and i not in processed_price_indices:
            txt = gtext(i)
            price = safe_parse_price(txt, 0.0, cfg.max_price)
            if price is None:
                i += 1
                continue

            # ---- Existing logic: look ahead for a name ----
            name_parts: List[str] = []
            found_name = False

            for j in range(i + 1, min(len(region_items), i + cfg.price_name_search_window + 1)):
                nlb = glabel(j)

                if is_summaryish_label(nlb):
                    break

                ntxt = gtext(j)
                if nlb == "B-MENU.NM":
                    name_parts.append(ntxt)
                    found_name = True

                    # also grab trailing I-MENU.NM
                    k = j + 1
                    while k < len(region_items):
                        klb = glabel(k)
                        if klb == "I-MENU.NM":
                            ktxt = gtext(k)
                            if ktxt:
                                name_parts.append(ktxt)
                            k += 1
                            continue
                        break
                    break

                if nlb == "B-MENU.PRICE":
                    break

            if found_name:
                name = " ".join([p for p in name_parts if p]).strip() or "ITEM"
                items.append({
                    "name": name,
                    "price": price,
                    "price_tokens": region_items[i].get("tokens", []) or [],
                    "price_confidence": gconf(i),
                    "source": "labeled_price_name",
                })
                if cfg.debug_print:
                    print(f"    ✓ ${price:.2f} → {name}")

                processed_price_indices.add(i)
                i += 1
                continue

            # ---- NEW: buffer the price for later stacked matching ----
            price_buffer.append((i, price))
            processed_price_indices.add(i)  # consume it so it won't become orphan later
            i += 1
            continue

        # ----------------------------
        # If we see a name and we have buffered prices, pair them (LIFO)
        # ----------------------------
        if lb == "B-MENU.NM" and price_buffer and i not in processed_indices:
            # build full name (B + following I*)
            name_parts = [gtext(i)] if gtext(i) else []
            j = i + 1
            while j < len(region_items) and glabel(j) == "I-MENU.NM":
                t = gtext(j)
                if t:
                    name_parts.append(t)
                j += 1

            name = " ".join([p for p in name_parts if p]).strip() or "ITEM"

            # pop the most recent buffered price (closest price above this name)
            p_idx, p_val = price_buffer.pop()

            items.append({
                "name": name,
                "price": p_val,
                "price_tokens": region_items[p_idx].get("tokens", []) or [],
                "price_confidence": gconf(p_idx),
                "source": "labeled_price_name_buffered",
            })

            if cfg.debug_print:
                print(f"    ✓ ${p_val:.2f} → {name} (buffered)")

            # skip past the I-MENU.NM tokens we consumed
            i = j
            continue

        i += 1

    # ------------------------------------------------------------
    # Any remaining buffered prices become orphan prices
    # (same behavior as your old method)
    # ------------------------------------------------------------
    for p_idx, p_val in price_buffer:
        items.append({
            "name": "ITEM_NOT_DETECTED",
            "price": p_val,
            "price_tokens": region_items[p_idx].get("tokens", []) or [],
            "price_confidence": gconf(p_idx),
            "source": "labeled_price_name_orphan_buffer",
        })
        if cfg.debug_print:
            print(f"    ! ${p_val:.2f} → ITEM_NOT_DETECTED (buffered orphan)")

    processed_indices.update(processed_price_indices)
    return items, processed_indices
# ============================================================
# EXTRACTION: PATTERN 3 - MIXED/UNDECIDED
# ============================================================
def extract_mixed_heuristic(
    region_items: List[Dict[str, Any]],
    cfg: CordGeoConfig,
    processed_indices: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], set]:
    """
    Extract items from undecided region using token-by-token heuristics.
    
    Returns:
        (items, processed_indices_set)
    """
    
    if processed_indices is None:
        processed_indices = set()
    
    items = []
    
    # FIRST: Handle B-MENU.NM → NAME→PRICE patterns
    for idx, item in enumerate(region_items):
        if idx in processed_indices:
            continue
        
        label = item.get("label", "O")
        
        if label == "B-MENU.NM":
            text = clean_ocr_noise(item.get("text", "")) or ""
            name_parts = [text]
            processed_indices.add(idx)
            
            j = idx + 1
            while j < len(region_items):
                if j in processed_indices:
                    j += 1
                    continue
                
                next_item = region_items[j]
                next_label = next_item.get("label", "O")
                next_text = clean_ocr_noise(next_item.get("text", "")) or ""
                
                if next_label == "I-MENU.NM":
                    name_parts.append(next_text)
                    processed_indices.add(j)
                    j += 1
                elif next_label == "B-MENU.PRICE":
                    price = safe_parse_price(next_text, 0.0, cfg.max_price)
                    if price is not None:
                        name = " ".join(name_parts).strip()
                        items.append({
                            "name": name,
                            "price": price,
                            "price_tokens": next_item.get("tokens", []),
                            "price_confidence": float(next_item.get("avg_confidence", 0.0) or 0.0),
                            "source": "labeled_mixed_name_price",
                        })
                        if cfg.debug_print:
                            print(f"    ✓ {name:30} → ${price:.2f} (heuristic B-MENU.NM)")
                        processed_indices.add(j)
                    break
                else:
                    break
    
    # SECOND: Handle orphan I-MENU.NM tokens
    if cfg.debug_print:
        print(f"    [ORPHAN I-MENU.NM] Scanning for continuation tokens without beginning...")
    
    for idx, item in enumerate(region_items):
        if idx in processed_indices:
            continue
        
        label = item.get("label", "O")
        
        if label == "I-MENU.NM":
            has_preceding_b_menu = False
            for j in range(max(0, idx - 5), idx):
                if region_items[j].get("label") == "B-MENU.NM":
                    has_preceding_b_menu = True
                    break
            
            if not has_preceding_b_menu:
                text = clean_ocr_noise(item.get("text", "")) or ""
                name_parts = [text]
                processed_indices.add(idx)
                
                j = idx + 1
                while j < len(region_items):
                    if j in processed_indices:
                        j += 1
                        continue
                    
                    next_item = region_items[j]
                    next_label = next_item.get("label", "O")
                    next_text = clean_ocr_noise(next_item.get("text", "")) or ""
                    
                    if next_label == "I-MENU.NM":
                        name_parts.append(next_text)
                        processed_indices.add(j)
                        j += 1
                    elif next_label == "B-MENU.PRICE":
                        price = safe_parse_price(next_text, 0.0, cfg.max_price)
                        if price is not None:
                            name = " ".join(name_parts).strip()
                            items.append({
                                "name": name,
                                "price": price,
                                "price_tokens": next_item.get("tokens", []),
                                "price_confidence": float(next_item.get("avg_confidence", 0.0) or 0.0),
                                "source": "labeled_mixed_orphan_i",
                            })
                            if cfg.debug_print:
                                print(f"    ✓ {name:30} → ${price:.2f} (orphan I-MENU.NM)")
                            processed_indices.add(j)
                        break
                    else:
                        break
    
    # THIRD: Handle B-MENU.PRICE → PRICE→NAME patterns
    if cfg.debug_print:
        print(f"    [PRICE→NAME] Scanning for unmatched prices...")
    
    for idx, item in enumerate(region_items):
        if idx in processed_indices:
            continue
        
        label = item.get("label", "O")
        
        if label != "B-MENU.PRICE":
            continue
        
        text = clean_ocr_noise(item.get("text", "")) or ""
        price = safe_parse_price(text, 0.0, cfg.max_price)
        
        if price is None:
            continue
        
        has_preceding_name = False
        for j in range(max(0, idx - 3), idx):
            if j in processed_indices and region_items[j].get("label", "O") in ("B-MENU.NM", "I-MENU.NM"):
                has_preceding_name = True
                break
        
        if has_preceding_name:
            continue
        
        name_parts = []
        name_idx = None
        
        for j in range(idx + 1, min(len(region_items), idx + cfg.price_name_search_window + 1)):
            if j in processed_indices:
                continue
            
            next_item = region_items[j]
            next_label = next_item.get("label", "O")
            next_text = clean_ocr_noise(next_item.get("text", "")) or ""
            
            if next_label == "B-MENU.NM":
                name_parts.append(next_text)
                name_idx = j
                processed_indices.add(j)
                
                k = j + 1
                while k < len(region_items) and k not in processed_indices:
                    if region_items[k].get("label", "O") == "I-MENU.NM":
                        name_text = clean_ocr_noise(region_items[k].get("text", "")) or ""
                        name_parts.append(name_text)
                        processed_indices.add(k)
                        k += 1
                    else:
                        break
                break
            elif next_label == "B-MENU.PRICE":
                break
        
        if name_parts:
            name = " ".join(name_parts).strip()
            items.append({
                "name": name,
                "price": price,
                "price_tokens": item.get("tokens", []),
                "price_confidence": float(item.get("avg_confidence", 0.0) or 0.0),
                "source": "labeled_mixed_price_name",
            })
            
            if cfg.debug_print:
                print(f"    ✓ ${price:.2f} → {name} (PRICE→NAME, forward)")
            
            processed_indices.add(idx)
    
    # FOURTH: Remaining orphan prices
    if cfg.debug_print:
        print(f"    [ORPHAN PRICES] Scanning for unmatched prices...")
    
    for idx, item in enumerate(region_items):
        if idx in processed_indices:
            continue
        
        label = item.get("label", "O")
        
        if label == "B-MENU.PRICE":
            text = clean_ocr_noise(item.get("text", "")) or ""
            price = safe_parse_price(text, 0.0, cfg.max_price)
            
            if price is not None:
                items.append({
                    "name": "ITEM_NOT_DETECTED",
                    "price": price,
                    "price_tokens": item.get("tokens", []),
                    "price_confidence": float(item.get("avg_confidence", 0.0) or 0.0),
                    "source": "labeled_mixed_orphan_price",
                })
                
                if cfg.debug_print:
                    print(f"    ! ${price:.2f} → ITEM_NOT_DETECTED (orphan price)")
                processed_indices.add(idx)
    
    return items, processed_indices


# ============================================================
# MAIN EXTRACTION FUNCTION - HANDLES ALL PATTERNS
# ============================================================
def extract_menu_items_adaptive_1(
    grouped_items: List[Dict[str, Any]],
    words: List[str],
    labels: List[str],
    confs: List[float],
    cfg: CordGeoConfig
) -> List[Dict[str, Any]]:
    """
    Adaptive menu item extraction handling 4 patterns:
    1. NAME→PRICE
    2. PRICE→NAME
    3. MIXED (regions)
    4. UNLABELED (fallback)
    """

    if cfg.debug_print:
        print(f"\n[EXTRACT MENU ITEMS] Processing {len(grouped_items)} grouped tokens")

    global_processed_indices: set = set()

    # Track which raw tokens are part of labeled items (optional; your unlabeled extractor may use it)
    labeled_token_indices: set = set()

    # ----------------------------
    # Helpers for region merging
    # ----------------------------
    QTY_NOISE_RE = re.compile(r"^(x|×|x'|x’|x\.|x:|x;)\s*\d{0,2}$", re.I)
    JUST_X_RE = re.compile(r"^(x|×|x'|x’)$", re.I)

    def gtext(g: Dict[str, Any]) -> str:
        return (clean_ocr_noise(g.get("text", "")) or "").strip()

    def glabel(g: Dict[str, Any]) -> str:
        return g.get("label", "O") or "O"

    def gconf(g: Dict[str, Any]) -> float:
        try:
            return float(g.get("avg_confidence", 1.0) or 1.0)
        except Exception:
            return 1.0

    def is_noise_bridge_group(g: Dict[str, Any]) -> bool:
        """
        A small region/group that often breaks NAME→PRICE adjacency:
        - mis-labeled B-SUM.TAX with low confidence
        - text like x, x', × (qty indicator)
        """
        t = gtext(g)
        lb = glabel(g)

        # qty marker (OCR often produces x' instead of x7, etc.)
        if t and (QTY_NOISE_RE.match(t) or JUST_X_RE.match(t)):
            return True

        # tax/service labels but low confidence and short text
        if lb.startswith("B-SUM.") and gconf(g) < 0.70 and len(t) <= 6:
            return True

        return False

    def merge_regions_safely(
        items: List[Dict[str, Any]],
        regions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Merge tiny/noise regions into neighbors so NAME and PRICE stay connected.
        Strategy:
          - If a region is length 1-2 and looks like a bridge/noise, merge with next,
            or with prev if next doesn't exist.
          - Also, if prev region ends with B-MENU.NM and next region starts with B-MENU.PRICE,
            merge across the boundary (even if middle is tiny noise).
        """
        if not regions:
            return regions

        merged: List[Tuple[int, int]] = []
        i = 0
        while i < len(regions):
            s, e = regions[i]
            length = e - s

            # Decide if this whole region is "noise bridge"
            is_tiny = length <= 2
            all_noise = is_tiny and all(is_noise_bridge_group(items[k]) for k in range(s, e))

            if all_noise:
                # merge into next if possible, else into previous
                if i + 1 < len(regions):
                    ns, ne = regions[i + 1]
                    # merge current into next: extend next start backward
                    regions[i + 1] = (s, ne)
                    i += 1
                    continue
                else:
                    # merge into previous already in merged
                    if merged:
                        ps, pe = merged[-1]
                        merged[-1] = (ps, e)
                    else:
                        merged.append((s, e))
                    i += 1
                    continue

            # Extra: merge boundary if prev ends with name and next starts with price
            if merged:
                ps, pe = merged[-1]
                prev_last = items[pe - 1] if pe - 1 >= 0 else None
                curr_first = items[s] if s < len(items) else None

                if prev_last and curr_first:
                    if glabel(prev_last) == "B-MENU.NM" and glabel(curr_first) == "B-MENU.PRICE":
                        merged[-1] = (ps, e)
                        i += 1
                        continue

            merged.append((s, e))
            i += 1

        return merged

    # ----------------------------
    # STEP 1: DETECT GLOBAL PATTERN
    # ----------------------------
    global_pattern = detect_item_pattern(grouped_items, cfg)

    # ----------------------------
    # STEP 2: Handle based on global pattern
    # ----------------------------
    if global_pattern == "name_then_price":
        if cfg.debug_print:
            print("[EXTRACT] Using NAME→PRICE extraction")

        items, global_processed_indices = extract_name_then_price(
            grouped_items, cfg, global_processed_indices
        )

    elif global_pattern == "price_then_name":
        if cfg.debug_print:
            print("[EXTRACT] Using PRICE→NAME extraction")

        items, global_processed_indices = extract_price_then_name(
            grouped_items, cfg, global_processed_indices
        )

    else:  # MIXED pattern
        if cfg.debug_print:
            print("[EXTRACT] Detected MIXED pattern - segmenting by regions")

        items = []
        regions = detect_regions(grouped_items, cfg)

        # ✅ NEW: merge “noise bridge” regions so name + price stay together
        merged_regions = merge_regions_safely(grouped_items, regions)

        if cfg.debug_print and merged_regions != regions:
            print(f"\n[REGION MERGE] Merged {len(regions)} regions -> {len(merged_regions)} regions")
            # optional: print merged ranges
            for r_idx, (s, e) in enumerate(merged_regions):
                print(f"  MergedRegion {r_idx}: items {s}-{e-1}")

        for region_idx, (start, end) in enumerate(merged_regions):
            region_items = grouped_items[start:end]
            if not region_items:
                continue

            region_pattern = detect_region_pattern(region_items, cfg)

            if cfg.debug_print:
                print(f"  Region {region_idx} ({start}-{end-1}): {region_pattern}")

            if region_pattern == "name_then_price":
                region_items_extracted, _ = extract_name_then_price(region_items, cfg)
            elif region_pattern == "price_then_name":
                region_items_extracted, _ = extract_price_then_name(region_items, cfg)
            else:
                region_items_extracted, _ = extract_mixed_heuristic(region_items, cfg)

            items.extend(region_items_extracted)

    # ----------------------------
    # STEP 3: UNLABELED fallback
    # ----------------------------
    if cfg.extract_unlabeled_items:
        if cfg.debug_print:
            print("\n[UNLABELED ITEMS] Extracting items without CORD labels...")

        unlabeled_items = extract_unlabeled_items_from_raw_tokens(
            words, labels, confs, cfg, labeled_token_indices
        )
        items.extend(unlabeled_items)

    if cfg.debug_print:
        print(f"\n[EXTRACT] Extracted {len(items)} total items")

    return items

def extract_menu_items_adaptive(
    grouped_items: List[Dict[str, Any]],
    words: List[str],
    labels: List[str],
    confs: List[float],
    cfg: CordGeoConfig
) -> List[Dict[str, Any]]:
    """
    Adaptive menu item extraction handling 4 patterns:
    1. NAME→PRICE
    2. PRICE→NAME
    3. MIXED (regions)
    4. UNLABELED (fallback)

    Fixes:
    - Merge small "name-only" region into next region when the next region clearly contains
      name+price(s), to avoid losing the first item name (receipt_139: Margherita ...).
    - Keep bbox-x swap for (NAME, NAME, PRICE, PRICE) misalignment.
    """

    if cfg.debug_print:
        print(f"\n[EXTRACT MENU ITEMS] Processing {len(grouped_items)} grouped tokens")

    global_processed_indices: set = set()
    labeled_token_indices: set = set()

    QTY_NOISE_RE = re.compile(r"^(x|×|x'|x’|x\.|x:|x;)\s*\d{0,2}$", re.I)
    JUST_X_RE = re.compile(r"^(x|×|x'|x’)$", re.I)

    def gtext(g: Dict[str, Any]) -> str:
        return (clean_ocr_noise(g.get("text", "")) or "").strip()

    def glabel(g: Dict[str, Any]) -> str:
        return g.get("label", "O") or "O"

    def gconf(g: Dict[str, Any]) -> float:
        try:
            return float(g.get("avg_confidence", 1.0) or 1.0)
        except Exception:
            return 1.0

    def is_noise_bridge_group(g: Dict[str, Any]) -> bool:
        t = gtext(g)
        lb = glabel(g)

        if t and (QTY_NOISE_RE.match(t) or JUST_X_RE.match(t)):
            return True

        if lb.startswith("B-SUM.") and gconf(g) < 0.70 and len(t) <= 6:
            return True

        return False

    # ----------------------------
    # NEW: bbox-x based fix for price misalignment
    # ----------------------------
    def x_center_from_bboxes(g: Dict[str, Any]) -> Optional[float]:
        bxs = g.get("bboxes", []) or []
        if not bxs:
            return None
        xs: List[float] = []
        for b in bxs:
            if not b or len(b) < 4:
                continue
            xs.append(float(b[0]))
            xs.append(float(b[2]))
        if not xs:
            return None
        return float((min(xs) + max(xs)) / 2.0)

    def fix_two_prices_after_two_names_by_x(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(items)
        if n < 4:
            return items

        i = 0
        while i <= n - 4:
            a, b, p1, p2 = items[i], items[i + 1], items[i + 2], items[i + 3]

            if not (glabel(a) == "B-MENU.NM" and glabel(b) == "B-MENU.NM"):
                i += 1
                continue
            if not (glabel(p1) == "B-MENU.PRICE" and glabel(p2) == "B-MENU.PRICE"):
                i += 1
                continue

            ax = x_center_from_bboxes(a)
            bx = x_center_from_bboxes(b)
            p1x = x_center_from_bboxes(p1)
            p2x = x_center_from_bboxes(p2)

            if None in (ax, bx, p1x, p2x):
                i += 1
                continue

            # detect "cross" assignment
            p1_closer_to_b = abs(p1x - bx) < abs(p1x - ax)
            p2_closer_to_a = abs(p2x - ax) < abs(p2x - bx)

            if p1_closer_to_b and p2_closer_to_a:
                if cfg.debug_print:
                    print(f"  [X-REALIGN] Swapping consecutive prices at groups {i+2} and {i+3}")
                    print(f"    Names: '{gtext(a)}' | '{gtext(b)}'")
                    print(f"    Prices(before): '{gtext(p1)}' , '{gtext(p2)}'")
                items[i + 2], items[i + 3] = items[i + 3], items[i + 2]
                if cfg.debug_print:
                    print(f"    Prices(after):  '{gtext(items[i+2])}' , '{gtext(items[i+3])}'")
                i += 4
                continue

            i += 1

        return items

    # ----------------------------
    # UPDATED: safer region merging (adds "name-only tiny region" rule)
    # ----------------------------
    def region_is_name_only(items: List[Dict[str, Any]], s: int, e: int) -> bool:
        """All groups are MENU.NM (B/I) and not summary/price."""
        for k in range(s, e):
            lb = glabel(items[k])
            if lb not in ("B-MENU.NM", "I-MENU.NM"):
                return False
        return True

    def next_region_has_price_soon(items: List[Dict[str, Any]], s: int, e: int, look: int = 6) -> bool:
        """Within first few groups of next region, there is a MENU.PRICE."""
        for k in range(s, min(e, s + look)):
            if glabel(items[k]) == "B-MENU.PRICE":
                return True
        return False

    def merge_regions_safely(
        items: List[Dict[str, Any]],
        regions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not regions:
            return regions

        merged: List[Tuple[int, int]] = []
        i = 0
        while i < len(regions):
            s, e = regions[i]
            length = e - s

            is_tiny = length <= 2
            all_noise = is_tiny and all(is_noise_bridge_group(items[k]) for k in range(s, e))

            # Rule A: noise-bridge merge (your existing logic)
            if all_noise:
                if i + 1 < len(regions):
                    ns, ne = regions[i + 1]
                    regions[i + 1] = (s, ne)
                    i += 1
                    continue
                else:
                    if merged:
                        ps, pe = merged[-1]
                        merged[-1] = (ps, e)
                    else:
                        merged.append((s, e))
                    i += 1
                    continue

            # ✅ Rule B (NEW): tiny NAME-only region that should attach to next region
            # Example receipt_139: Region0 = "Margherita Calzcne" alone, Region1 starts with "Coke" and has prices.
            if is_tiny and region_is_name_only(items, s, e) and (i + 1 < len(regions)):
                ns, ne = regions[i + 1]
                if glabel(items[ns]) == "B-MENU.NM" and next_region_has_price_soon(items, ns, ne, look=6):
                    if cfg.debug_print:
                        print(f"  [REGION MERGE] Tiny NAME-only region {s}-{e-1} merged into next {ns}-{ne-1}")
                    regions[i + 1] = (s, ne)
                    i += 1
                    continue

            # Rule C: boundary merge if prev ends with NAME and current starts with PRICE
            if merged:
                ps, pe = merged[-1]
                prev_last = items[pe - 1] if pe - 1 >= 0 else None
                curr_first = items[s] if s < len(items) else None
                if prev_last and curr_first:
                    if glabel(prev_last) == "B-MENU.NM" and glabel(curr_first) == "B-MENU.PRICE":
                        merged[-1] = (ps, e)
                        i += 1
                        continue

            merged.append((s, e))
            i += 1

        return merged

    # ----------------------------
    # STEP 1: DETECT GLOBAL PATTERN
    # ----------------------------
    global_pattern = detect_item_pattern(grouped_items, cfg)

    # Safe no-op for most receipts
    grouped_items = fix_two_prices_after_two_names_by_x(grouped_items)

    # ----------------------------
    # STEP 2: Handle based on global pattern
    # ----------------------------
    if global_pattern == "name_then_price":
        if cfg.debug_print:
            print("[EXTRACT] Using NAME→PRICE extraction")
        items, global_processed_indices = extract_name_then_price(
            grouped_items, cfg, global_processed_indices
        )

    elif global_pattern == "price_then_name":
        if cfg.debug_print:
            print("[EXTRACT] Using PRICE→NAME extraction")
        items, global_processed_indices = extract_price_then_name(
            grouped_items, cfg, global_processed_indices
        )

    else:  # MIXED
        if cfg.debug_print:
            print("[EXTRACT] Detected MIXED pattern - segmenting by regions")

        items = []
        regions = detect_regions(grouped_items, cfg)

        merged_regions = merge_regions_safely(grouped_items, regions)

        if cfg.debug_print and merged_regions != regions:
            print(f"\n[REGION MERGE] Merged {len(regions)} regions -> {len(merged_regions)} regions")
            for r_idx, (rs, re_) in enumerate(merged_regions):
                print(f"  MergedRegion {r_idx}: items {rs}-{re_-1}")

        for region_idx, (start, end) in enumerate(merged_regions):
            region_items = grouped_items[start:end]
            if not region_items:
                continue

            # Apply swap after merges so Margherita+Coke+prices become contiguous
            region_items = fix_two_prices_after_two_names_by_x(region_items)

            region_pattern = detect_region_pattern(region_items, cfg)

            if cfg.debug_print:
                print(f"  Region {region_idx} ({start}-{end-1}): {region_pattern}")

            if region_pattern == "name_then_price":
                region_items_extracted, _ = extract_name_then_price(region_items, cfg)
            elif region_pattern == "price_then_name":
                region_items_extracted, _ = extract_price_then_name(region_items, cfg)
            else:
                region_items_extracted, _ = extract_mixed_heuristic(region_items, cfg)

            items.extend(region_items_extracted)

    # ----------------------------
    # STEP 3: UNLABELED fallback
    # ----------------------------
    if cfg.extract_unlabeled_items:
        if cfg.debug_print:
            print("\n[UNLABELED ITEMS] Extracting items without CORD labels...")

        unlabeled_items = extract_unlabeled_items_from_raw_tokens(
            words, labels, confs, cfg, labeled_token_indices
        )
        items.extend(unlabeled_items)

    if cfg.debug_print:
        print(f"\n[EXTRACT] Extracted {len(items)} total items")

    return items
# ============================================================
# GROUP TOKENS BY LABEL (from original code)
# ============================================================
def group_tokens_by_label_sequence(
    words: List[str],
    labels: List[str],
    confs: List[float],
    bboxes: List[List[int]],
) -> List[Dict[str, Any]]:
    """Group consecutive tokens by label type."""
    
    if not words:
        return []

    items: List[Dict[str, Any]] = []

    def finalize_group(g: Dict[str, Any]) -> None:
        if not g["tokens"]:
            return

        g["text"] = " ".join(g["tokens"]).strip()
        g["avg_confidence"] = float(np.mean(g["confidences"])) if g["confidences"] else 0.0

        label_type = g["label_type"] or ""
        b_lab = f"B-{label_type}" if label_type else None
        i_lab = f"I-{label_type}" if label_type else None

        lbls = g["labels"]
        if b_lab and (b_lab in lbls):
            g["label"] = b_lab
        elif i_lab and (i_lab in lbls):
            g["label"] = i_lab
        else:
            g["label"] = lbls[-1] if lbls else (g["label"] or "O")

        items.append(g)

    current_group = {
        "tokens": [],
        "labels": [],
        "label": None,
        "confidences": [],
        "bboxes": [],
        "text": "",
        "label_type": None,
        "avg_confidence": 0.0,
    }

    for word, label, conf, bbox in zip(words, labels, confs, bboxes):
        label = label or "O"
        label_type = CordLabelParser.get_label_type(label)
        is_beginning = label.startswith("B-")

        if is_beginning or (current_group["label_type"] and current_group["label_type"] != label_type):
            finalize_group(current_group)
            current_group = {
                "tokens": [],
                "labels": [],
                "label": label,
                "confidences": [],
                "bboxes": [],
                "text": "",
                "label_type": label_type,
                "avg_confidence": 0.0,
            }

        current_group["tokens"].append(word)
        current_group["labels"].append(label)
        current_group["confidences"].append(float(conf))
        current_group["bboxes"].append(bbox)

        if current_group["label"] is None:
            current_group["label"] = label
        if current_group["label_type"] is None:
            current_group["label_type"] = label_type

    finalize_group(current_group)
    return items


# ============================================================
# EXTRACT SUMMARY (TAX, SUBTOTAL, TOTAL)
# ============================================================
# ============================================================
# SUMMARY VALIDATION & CORRECTION (NEW)
# ============================================================
def validate_and_correct_summary(
    summary: Dict[str, Any],
    items: List[Dict[str, Any]],
    cfg: CordGeoConfig
) -> Dict[str, Any]:
    """
    Validate summary values and detect/fix common OCR/labeling errors.
    
    Common issues:
    1. Swapped Tax/Total values
    2. Missing Total (when total should equal subtotal + tax)
    3. Subtotal > Total (impossible)
    4. Tax value seems too large
    
    Returns:
        Corrected summary dict
    """
    
    if cfg.debug_print:
        print(f"\n[SUMMARY VALIDATION]")
        print(f"  Subtotal: {summary['subtotal']}")
        print(f"  Tax: {summary['tax']}")
        print(f"  Total: {summary['total']}")
    
    # Calculate expected total from items
    items_sum = sum(item.get("price", 0) for item in items if item.get("price"))
    
    if cfg.debug_print:
        print(f"  Items sum: ${items_sum:.2f}")
    
    # ================================================================
    # CHECK 1: Validate Subtotal
    # ================================================================
    # If subtotal is None but we have items, use items sum
    if summary["subtotal"] is None and items_sum > 0:
        summary["subtotal"] = items_sum
        if cfg.debug_print:
            print(f"  ✓ Subtotal missing, using items sum: ${items_sum:.2f}")
    
    # ================================================================
    # CHECK 2: Detect Swapped Tax/Total
    # ================================================================
    # If tax looks too large compared to subtotal, it might be swapped with total
    if summary["subtotal"] is not None and summary["tax"] is not None and summary["total"] is None:
        subtotal = summary["subtotal"]
        tax = summary["tax"]
        
        # Typical tax rate is 5-10% of subtotal
        expected_tax_range = (subtotal * 0.02, subtotal * 0.15)  # 2-15% range
        
        # If tax is outside normal range, it might be the total
        if tax > subtotal:
            if cfg.debug_print:
                print(f"  ⚠ Tax (${tax:.2f}) > Subtotal (${subtotal:.2f})")
                print(f"    This looks like Tax and Total are SWAPPED")
            
            # Swap them
            summary["total"] = tax
            summary["tax"] = tax - subtotal  # Calculate correct tax
            summary["tax_confidence"] = min(summary["tax_confidence"], 0.5)  # Lower confidence
            
            if cfg.debug_print:
                print(f"  ✓ Corrected: Tax=${summary['tax']:.2f}, Total=${summary['total']:.2f}")
        
        # If tax is missing but total exists, calculate it
        elif summary["total"] is not None:
            calculated_tax = summary["total"] - subtotal
            if calculated_tax >= 0:
                summary["tax"] = calculated_tax
                if cfg.debug_print:
                    print(f"  ✓ Calculated tax from total: ${summary['tax']:.2f}")
    
    # ================================================================
    # CHECK 3: Calculate missing Total from Subtotal + Tax
    # ================================================================
    if summary["total"] is None:
        if summary["subtotal"] is not None and summary["tax"] is not None:
            summary["total"] = summary["subtotal"] + summary["tax"]
            summary["total_confidence"] = min(summary["subtotal_confidence"], summary["tax_confidence"])
            
            if cfg.debug_print:
                print(f"  ✓ Calculated total from subtotal + tax: ${summary['total']:.2f}")
        elif summary["subtotal"] is not None and summary["subtotal"] > 0:
            # If only subtotal exists, use it as total (assume 0 tax)
            summary["total"] = summary["subtotal"]
            summary["tax"] = 0.0
            
            if cfg.debug_print:
                print(f"  ✓ Using subtotal as total (no tax): ${summary['total']:.2f}")
    
    # ================================================================
    # CHECK 4: Validate consistency
    # ================================================================
    if summary["subtotal"] and summary["tax"] and summary["total"]:
        expected_total = summary["subtotal"] + summary["tax"]
        actual_total = summary["total"]
        
        tolerance = 0.05  # Allow 5 cent difference due to rounding
        
        if abs(expected_total - actual_total) > tolerance:
            if cfg.debug_print:
                print(f"  ⚠ Total mismatch: Expected ${expected_total:.2f}, Got ${actual_total:.2f}")
                print(f"    Recalculating...")
            
            # Trust subtotal and total, recalculate tax
            summary["tax"] = summary["total"] - summary["subtotal"]
            if cfg.debug_print:
                print(f"  ✓ Recalculated tax: ${summary['tax']:.2f}")
    
    if cfg.debug_print:
        print(f"\n[CORRECTED SUMMARY]")
        print(f"  Subtotal: ${summary['subtotal']:.2f}" if summary['subtotal'] is not None else "  Subtotal: None")
        print(f"  Tax: ${summary['tax']:.2f}" if summary['tax'] is not None else "  Tax: None")
        print(f"  Total: ${summary['total']:.2f}" if summary['total'] is not None else "  Total: None")
    
    return summary

def extract_summary_items(
    grouped_items: List[Dict[str, Any]],
    cfg: CordGeoConfig
) -> Dict[str, Any]:
    """Extract tax, subtotal, total from grouped items."""
    
    summary = {
        "subtotal": None, "subtotal_tokens": [], "subtotal_confidence": 0.0,
        "tax": None, "tax_tokens": [], "tax_confidence": 0.0,
        "total": None, "total_tokens": [], "total_confidence": 0.0,
        "discount": None, "discount_tokens": [], "discount_confidence": 0.0,
    }

    expecting_tax_amount = False
    saw_labeled_tax = False
    subtotal_idx: Optional[int] = None
    total_idx: Optional[int] = None
    
    # NEW: Track all summary amounts for validation
    summary_amounts: List[Tuple[int, str, float]] = []  # (idx, type, value)

    for idx, item in enumerate(grouped_items):
        actual_label = item.get("label", "") or "O"
        tokens = item.get("tokens", []) or []
        confidence = float(item.get("avg_confidence", 0.0) or 0.0)
        cleaned_text = clean_ocr_noise(item.get("text", "")) or ""

        # SUBTOTAL
        if actual_label in ("B-SUM.SUBTOTAL", "I-SUM.SUBTOTAL"):
            subtotal_idx = idx
            val = safe_parse_price(cleaned_text, cfg.min_price, cfg.max_price)
            if val is not None:
                summary_amounts.append((idx, "SUBTOTAL", val))
                if summary["subtotal"] is None or confidence >= summary["subtotal_confidence"]:
                    summary["subtotal"] = val
                    summary["subtotal_confidence"] = confidence
                    summary["subtotal_tokens"] = tokens

        # TOTAL
        if actual_label in ("B-SUM.TOTAL", "I-SUM.TOTAL"):
            total_idx = idx
            val = safe_parse_price(cleaned_text, cfg.min_price, cfg.max_price)
            if val is not None:
                summary_amounts.append((idx, "TOTAL", val))
                if summary["total"] is None or confidence >= summary["total_confidence"]:
                    summary["total"] = val
                    summary["total_confidence"] = confidence
                    summary["total_tokens"] = tokens

        # TAX
        if actual_label in ("B-TAX", "B-SUM.TAX"):
            saw_labeled_tax = True
            expecting_tax_amount = True
            summary["tax_confidence"] = max(summary["tax_confidence"], confidence)
            summary["tax_tokens"].extend(tokens)

            if not is_real_tax_line(cleaned_text):
                expecting_tax_amount = False
                continue

            
            val_inline = safe_parse_price(cleaned_text, 0.0, cfg.max_price)
            if val_inline is not None and "%" not in cleaned_text:
                summary_amounts.append((idx, "TAX", val_inline))
                if summary["tax"] is None or confidence >= summary["tax_confidence"]:
                    summary["tax"] = val_inline
                    summary["tax_confidence"] = confidence

        elif actual_label in ("I-TAX", "I-SUM.TAX"):
            saw_labeled_tax = True

            if not is_real_tax_line(cleaned_text):
                expecting_tax_amount = False
                continue

            val = safe_parse_price(cleaned_text, 0.0, cfg.max_price)
            if val is not None:
                summary_amounts.append((idx, "TAX", val))
                if summary["tax"] is None or confidence >= summary["tax_confidence"]:
                    summary["tax"] = val
                    summary["tax_confidence"] = confidence
                    summary["tax_tokens"] = tokens
            expecting_tax_amount = False

        # TAX FALLBACK
        if not saw_labeled_tax:
            near_summary = False
            if subtotal_idx is not None and abs(idx - subtotal_idx) <= 3:
                near_summary = True
            if total_idx is not None and abs(idx - total_idx) <= 3:
                near_summary = True

            if near_summary:
                if TAX_RE.search(cleaned_text) and not TOTAL_TAX_RE.search(cleaned_text):
                    expecting_tax_amount = True

                if expecting_tax_amount:
                    val = safe_parse_price(cleaned_text, 0.0, cfg.max_price)
                    if val is not None and "%" not in cleaned_text:
                        summary["tax"] = val
                        summary["tax_confidence"] = max(summary["tax_confidence"], confidence)
                        summary["tax_tokens"].extend(tokens)
                        expecting_tax_amount = False
    
    # ================================================================
    # NEW: Detect if subtotal/tax values are swapped
    # ================================================================
    if cfg.debug_print and summary_amounts:
        print(f"\n[SUMMARY AMOUNT DETECTION]")
        for idx, typ, val in sorted(summary_amounts):
            print(f"  [{idx}] {typ}: ${val:.2f}")
    
    # If we have SUBTOTAL and TAX labels but values are swapped
    # (tax > subtotal), swap the values
    if summary["subtotal"] is not None and summary["tax"] is not None:
        if summary["tax"] > summary["subtotal"]:
            if cfg.debug_print:
                print(f"\n[SWAP DETECTION] Tax (${summary['tax']:.2f}) > Subtotal (${summary['subtotal']:.2f})")
                print(f"  Swapping Subtotal ↔ Tax")
            
            # Swap them
            summary["subtotal"], summary["tax"] = summary["tax"], summary["subtotal"]
            summary["subtotal_confidence"], summary["tax_confidence"] = \
                summary["tax_confidence"], summary["subtotal_confidence"]
            summary["subtotal_tokens"], summary["tax_tokens"] = \
                summary["tax_tokens"], summary["subtotal_tokens"]

    return summary

# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================
def cord_plus_geo_extract_v4(
    example: Dict[str, Any],
    *,
    cfg: CordGeoConfig = CordGeoConfig(),
) -> Dict[str, Any]:
    """
    Enhanced CORD extraction with adaptive multi-pattern support.
    
    Handles:
    1. NAME→PRICE pattern (name appears before price in labels)
    2. PRICE→NAME pattern (price appears before name in labels)
    3. MIXED pattern (both in same receipt, detected via regions)
    4. UNLABELED pattern (items without CORD labels - name$price format)
    
    KEY FIXES:
    - Stops extraction after TOTAL label (no tips/change)
    - Filters tip suggestion keywords (18%, 20%, 25%, gratuity, etc.)
    - Processes unlabeled items from RAW TOKENS before grouping
    """
    
    words = example.get("words", []) or []
    labels = example.get("labels", []) or []
    confs = example.get("confs", []) or []
    bboxes = example.get("bboxes", []) or []
    
    if not (words and labels and confs):
        return {
            "id": example.get("id"),
            "image_path": example.get("image_path"),
            "SUBTOTAL": None,
            "SUBTOTAL_CONFIDENCE": 0.0,
            "TAX": None,
            "TAX_CONFIDENCE": 0.0,
            "TOTAL": None,
            "TOTAL_CONFIDENCE": 0.0,
            "ITEMS": [],
            "PATTERN_DETECTED": "none",
            "debug": {"error": "Missing required fields"},
        }
    
    if cfg.debug_print:
        print(f"\n{'='*80}")
        print(f"CORD EXTRACTION V6 - Enhanced Multi-Pattern Support (Post-Total Filtering)")
        print(f"{'='*80}")
        print(f"Processing {len(words)} tokens")
    
    # STEP 1: Group tokens by label sequence
    grouped_items = group_tokens_by_label_sequence(
        words,
        labels,
        confs,
        bboxes if len(bboxes) == len(words) else [[0, 0, 0, 0]] * len(words)
    )
    
    if cfg.debug_print:
        print(f"Grouped into {len(grouped_items)} label sequences")
    
    # STEP 2: Extract summary
    summary = extract_summary_items(grouped_items, cfg)
    
    # STEP 3: Extract menu items (ADAPTIVE MULTI-PATTERN)
    items = extract_menu_items_adaptive(grouped_items, words, labels, confs, cfg)
    
    # STEP 4: Validate and correct summary (NEW - AFTER items extracted)
    summary = validate_and_correct_summary(summary, items, cfg)

    
    if cfg.debug_print:
        print(f"\n{'='*80}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*80}")
        print(f"Items extracted: {len(items)}")
        
        sub = summary.get('subtotal')
        sub_str = f"${sub:.2f}" if sub is not None else "None"
        print(f"Subtotal: {sub_str}")
        
        tx = summary.get('tax')
        tx_str = f"${tx:.2f}" if tx is not None else "None"
        print(f"Tax: {tx_str}")
        
        tot = summary.get('total')
        tot_str = f"${tot:.2f}" if tot is not None else "None"
        print(f"Total: {tot_str}")
        
        # Breakdown by source
        sources = {}
        for item in items:
            src = item.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        
        if sources:
            print(f"\nItems by source:")
            for src, count in sorted(sources.items()):
                print(f"  {src}: {count}")
    
    # RETURN
    return {
        "id": example.get("id"),
        "image_path": example.get("image_path"),
        "SUBTOTAL": summary.get("subtotal"),
        "SUBTOTAL_CONFIDENCE": float(summary.get("subtotal_confidence", 0.0)),
        "TAX": summary.get("tax"),
        "TAX_CONFIDENCE": float(summary.get("tax_confidence", 0.0)),
        "TOTAL": summary.get("total"),
        "TOTAL_CONFIDENCE": float(summary.get("total_confidence", 0.0)),
        "ITEMS": items,
        "PATTERN_DETECTED": detect_item_pattern(grouped_items, cfg),
        "debug": {
            "n_words": len(words),
            "n_grouped_items": len(grouped_items),
            "n_extracted_items": len(items),
            "subtotal_tokens": summary.get("subtotal_tokens", []),
            "tax_tokens": summary.get("tax_tokens", []),
            "total_tokens": summary.get("total_tokens", []),
            "extraction_sources": {item.get("source", "unknown"): 1 
                                   for item in items}
        }
    }