from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _is_time_like(s: str) -> bool:
    s = (s or "").strip().lower()
    # Examples: 6:35PM, 04:18, 21:31, 3.24pm
    if re.match(r"^\d{1,2}[:\.]\d{2}(\s*[ap]\.?m\.?)*$", s):
        return True
    # OCR-mushed times like 04:1831PM (treat as time-ish)
    if re.match(r"^\d{1,2}[:\.]\d{3,4}(\s*[ap]\.?m\.?)*$", s):
        return True
    return False


def _near(a: float, b: float, eps: float = 0.06) -> bool:
    return abs(float(a) - float(b)) <= eps


# ---------------------------------------------------------
# Levenshtein distance (iterative, memory-safe)
# ---------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
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
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[-1]


# ---------------------------------------------------------
# Global tax keywords (region-aware)
# ---------------------------------------------------------
GLOBAL_TAX_KEYWORDS: Dict[str, List[str]] = {
    "UNIVERSAL": [
        "TAX", "VAT", "GST", "HST", "PST", "QST",
        "SALES TAX", "SERVICE TAX", "SVC", "SERVICE",
        "IVA", "MWST", "TVA", "BTW", "GST/HST",
        "KDV", "MOMS", "IGIC", "ICMS", "ISS", "IPI",
        "TIP", "GRATUITY", "GRAT", "SERVICE CHARGE",
        "18% GRATUITY", "15% GRATUITY",
    ],
    "CANADA": ["GST", "HST", "PST", "QST", "GST/HST"],
    "USA": ["SALES TAX", "TAX", "STATE TAX", "CITY TAX"],
    "EU": ["VAT", "TVA", "MWST", "IVA", "BTW"],
}


# ---------------------------------------------------------
# Currency + numeric parsing
# ---------------------------------------------------------
_CURRENCY_PREFIX_RE = re.compile(
    r"^(CA\$|CAD\$|USD\$|AUD\$|SGD\$|NZD\$|HKD\$|CHF|SEK|NOK|DKK|NZD|AUD|SGD|HKD|THB|CLP|COP|ARS|ZAR|AED|SAR|ILS|R\$|\$|£|€|¥|₹)\s*",
    re.IGNORECASE,
)
_CURRENCY_SUFFIX_RE = re.compile(
    r"\s*(USD|CAD|AUD|SGD|NZD|HKD|CHF|SEK|NOK|DKK|THB|CLP|COP|ARS|ZAR|AED|SAR|ILS)\s*$",
    re.IGNORECASE,
)

# Accept: 27.36, $27.36, 1,234.56, -2.60, ($16.50
_NUM_RE = re.compile(r"[-(]?\$?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})")


def normalize_amount(s: str) -> Optional[float]:
    """
    Parse monetary amounts with:
      - optional currency symbols
      - comma thousand separators
      - decimal comma conversion
      - parentheses negative "(12.34" or "(12.34)"
    """
    if not s:
        return None
    t = s.strip()

    # Ignore pure percentages like "18.00%" (no currency hint)
    up = t.upper()
    if "%" in t and "$" not in t and "USD" not in up and "CAD" not in up:
        return None

    # Remove currency prefix/suffix
    t = _CURRENCY_PREFIX_RE.sub("", t)
    t = _CURRENCY_SUFFIX_RE.sub("", t)

    # Keep only relevant chars
    t = re.sub(r"[^0-9\-\(\)\.,]", "", t)
    if not t:
        return None

    neg = False
    if t.startswith("("):
        neg = True
        t = t[1:]
    if t.endswith(")"):
        neg = True
        t = t[:-1]

    # Replace comma decimal to dot if looks like "12,34"
    if re.match(r"^\d+,\d{2}$", t):
        t = t.replace(",", ".")

    # If both '.' and ',', assume last one is decimal marker
    if "." in t and "," in t:
        if t.rfind(".") > t.rfind(","):
            t = t.replace(",", "")
        else:
            t = t.replace(".", "").replace(",", ".")
    else:
        if "," in t and re.search(r",\d{2}$", t):
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")

    try:
        val = float(t)
    except Exception:
        return None

    if neg:
        val = -abs(val)
    return val


# ---------------------------------------------------------
# Fuzzy keyword matching engine
# ---------------------------------------------------------
@dataclass
class FuzzyMatch:
    keyword: str
    matched: str
    distance: int


class FuzzyKeywordEngine:
    def __init__(self, keywords: List[str], max_distance: int = 3):
        self.keywords = [k.upper().strip() for k in keywords if k and k.strip()]
        self.max_distance = int(max_distance)

    def best_match(self, text: str) -> Optional[FuzzyMatch]:
        if not text:
            return None
        t = text.upper()

        # Exact contains first (fast path)
        for kw in self.keywords:
            if kw in t:
                return FuzzyMatch(keyword=kw, matched=kw, distance=0)

        toks = re.findall(r"[A-Z0-9%]+", t)
        if not toks:
            return None

        best: Optional[FuzzyMatch] = None
        for kw in self.keywords:
            kw_toks = re.findall(r"[A-Z0-9%]+", kw)
            if not kw_toks:
                continue

            # Compare by joining tokens (helps "GRAND TOTAL" vs "GRANDTOTAL")
            cand = "".join(toks)
            target = "".join(kw_toks)
            d = levenshtein(cand, target)

            if d <= self.max_distance:
                m = FuzzyMatch(keyword=kw, matched=cand, distance=d)
                if best is None or m.distance < best.distance:
                    best = m

        return best


# ---------------------------------------------------------
# Extractor
# ---------------------------------------------------------
class ReceiptFieldExtractor:
    """Extract tax, subtotal, and total from receipt OCR text with global support."""

    def __init__(self, max_distance: int = 3, tax_regions: Optional[List[str]] = None):
        self.max_distance = int(max_distance)
        self.tax_regions = tax_regions or ["UNIVERSAL"]
        self._init_engines()

    def _init_engines(self) -> None:
        total_keywords = [
            "TOTAL", "GRAND TOTAL", "TOTAL AMOUNT", "AMOUNT DUE",
            "NET TOTAL", "FINAL TOTAL", "INVOICE TOTAL", "TOTAL DUE",
            "MONTANT TOTAL", "IMPORTE TOTAL", "TOTAAL", "TOTAL A PAGAR",
            "BALANCE DUE",
        ]
        subtotal_keywords = [
            "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL", "SUBTOTAL AMOUNT",
            "BEFORE TAX", "NET AMOUNT", "AMOUNT BEFORE TAX",
            "SOUS-TOTAL", "SUBTOTAL FINAL", "SUBTOTAL NETTO",
        ]

        tax_keywords: List[str] = []
        for region in self.tax_regions:
            if region in GLOBAL_TAX_KEYWORDS:
                tax_keywords.extend(GLOBAL_TAX_KEYWORDS[region])

        # unique preserve order
        tax_keywords = list(dict.fromkeys(tax_keywords))

        self.total_engine = FuzzyKeywordEngine(total_keywords, max_distance=self.max_distance)
        self.subtotal_engine = FuzzyKeywordEngine(subtotal_keywords, max_distance=self.max_distance)
        self.tax_engine = FuzzyKeywordEngine(tax_keywords, max_distance=self.max_distance)

    def detect_currency(self, text: str) -> Optional[str]:
        # multi-char first (prevents R$ matching as $)
        multi_char = [
            "CA$", "CAD$", "AUD$", "USD$", "SGD$", "NZD$", "HKD$",
            "CHF", "SEK", "NOK", "DKK", "NZD", "AUD", "SGD", "HKD",
            "THB", "CLP", "COP", "ARS", "ZAR", "AED", "SAR", "ILS", "R$",
        ]
        for currency in sorted(multi_char, key=len, reverse=True):
            if currency in (text or ""):
                return currency

        for currency in ["£", "€", "¥", "₹"]:
            if currency in (text or ""):
                return currency

        if "$" in (text or ""):
            return "$"
        return None

    def infer_region_from_text(self, text: str) -> List[str]:
        currency = self.detect_currency(text)
        t = (text or "").upper()

        # Heuristics: currency + regional tax acronyms
        if "GST" in t or "HST" in t or "PST" in t or "QST" in t:
            return ["CANADA", "UNIVERSAL"]
        if "VAT" in t or "TVA" in t or "MWST" in t or "BTW" in t:
            return ["EU", "UNIVERSAL"]

        if currency in ("CA$", "CAD", "CAD$"):
            return ["CANADA", "UNIVERSAL"]
        if currency in ("USD", "$", "USD$"):
            return ["USA", "UNIVERSAL"]
        if currency in ("€",):
            return ["EU", "UNIVERSAL"]

        return ["UNIVERSAL"]

    def cleanup_ocr_text(self, receipt_text_or_list) -> str:
        """
        If input is a list of OCR tokens (words), create pseudo-lines.
        This is a heuristic; it can sometimes produce patterns like:
          Soda 27.36
          Subtotal 2.60
        (we handle these later in repair passes)
        """
        if isinstance(receipt_text_or_list, list):
            items = [str(x).strip() for x in receipt_text_or_list if str(x).strip()]
            if not items:
                return ""

            lines: List[str] = []
            current: List[str] = []

            for tok in items:
                item_str = tok.strip()
                if not item_str:
                    continue

                current.append(item_str)

                has_currency = bool(_CURRENCY_PREFIX_RE.match(item_str)) or ("$" in item_str)
                is_numeric = normalize_amount(item_str) is not None

                if has_currency or is_numeric:
                    lines.append(" ".join(current))
                    current = []

            if current:
                lines.append(" ".join(current))

            return "\n".join(lines)

        text = str(receipt_text_or_list or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_from_line(self, line: str) -> Dict[str, Any]:
        """Extract matches and the best numeric value from a single line."""
        t = (line or "").strip()
        up = t.upper()

        total_match = self.total_engine.best_match(up)
        subtotal_match = self.subtotal_engine.best_match(up)
        tax_match = self.tax_engine.best_match(up)

        nums = _NUM_RE.findall(t)
        value = normalize_amount(nums[-1]) if nums else None

        # Extra safety: percent-only lines should not become money
        if value is not None and "%" in up and ("$" not in up and "USD" not in up and "CAD" not in up):
            value = None

        return {
            "line": t,
            "value": value,
            "total_match": total_match is not None,
            "subtotal_match": subtotal_match is not None,
            "tax_match": tax_match is not None,
            "total_match_info": total_match,
            "subtotal_match_info": subtotal_match,
            "tax_match_info": tax_match,
        }

    # ---------------------------
    # Math resolver helper
    # ---------------------------
    @staticmethod
    def _resolve_summary_by_math(
        subtotal_candidates: List[float],
        tax_candidates: List[float],
        total_candidates: List[float],
        gratuity_candidates: List[float],
        eps: float = 0.06,
    ) -> Dict[str, Optional[float]]:
        def near(a: float, b: float) -> bool:
            return abs(float(a) - float(b)) <= eps

        subs = [x for x in subtotal_candidates if x is not None and x > 0]
        taxes = [x for x in tax_candidates if x is not None and x >= 0]
        tots = [x for x in total_candidates if x is not None and x > 0]
        grats = [x for x in gratuity_candidates if x is not None and x > 0]

        if not tots:
            return {"subtotal": None, "tax": None, "total": None, "gratuity": None, "pre_total": None}

        # choose final total as the LARGEST total seen (often final due)
        final_total = max(tots)

        # Case 1: subtotal + tax == final_total (no gratuity)
        for s in subs:
            for t in taxes:
                if near(s + t, final_total):
                    return {"subtotal": s, "tax": t, "total": final_total, "gratuity": None, "pre_total": final_total}

        # Case 2: pre_total + gratuity == final_total AND subtotal + tax == pre_total
        for g in grats:
            pre_total = final_total - g
            if pre_total <= 0:
                continue
            for s in subs:
                for t in taxes:
                    if near(s + t, pre_total):
                        return {"subtotal": s, "tax": t, "total": final_total, "gratuity": g, "pre_total": pre_total}

        return {"subtotal": None, "tax": None, "total": final_total, "gratuity": None, "pre_total": None}

    def extract_from_receipt(self, receipt_text_or_list, auto_detect_region: bool = True) -> Dict[str, Any]:
        """Extract receipt fields with support for gratuity and pre-total."""
        cleaned_text = self.cleanup_ocr_text(receipt_text_or_list)

        if auto_detect_region:
            regions = self.infer_region_from_text(cleaned_text)
            self.tax_regions = regions
            self._init_engines()

        lines = cleaned_text.split("\n")

        totals: List[Dict[str, Any]] = []
        subtotals: List[Dict[str, Any]] = []
        taxes: List[Dict[str, Any]] = []
        gratuities: List[Dict[str, Any]] = []
        parsed_lines: List[Dict[str, Any]] = []

        def _last_value(bucket: List[Dict[str, Any]]) -> Optional[float]:
            vals = [x.get("value") for x in bucket if isinstance(x.get("value"), (int, float))]
            return float(vals[-1]) if vals else None

        def _values(bucket: List[Dict[str, Any]]) -> List[float]:
            out: List[float] = []
            for x in bucket:
                v = x.get("value")
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out

        # ---------------------------------------------------------
        # PASS 1: parse lines & collect keyword matches
        # ---------------------------------------------------------
        for line in lines:
            if not line.strip():
                continue

            r = self.extract_from_line(line)

            # Ignore time-only lines
            if r.get("value") is not None and _is_time_like(line):
                r["value"] = None

            # Ignore percent-only lines as money
            up = (line or "").upper()
            if "%" in up and ("$" not in up and "USD" not in up and "CAD" not in up):
                r["value"] = None

            parsed_lines.append(r)

            if r["total_match"]:
                totals.append(r)
            if r["subtotal_match"]:
                subtotals.append(r)
            if r["tax_match"]:
                taxes.append(r)

            # Gratuity/Tip amount (not percent)
            if any(k in up for k in ["GRATUITY", "TIP", "SERVICE CHARGE"]):
                if isinstance(r.get("value"), (int, float)):
                    gratuities.append(r)

        repaired_total = _last_value(totals)
        repaired_subtotal = _last_value(subtotals)
        repaired_tax = _last_value(taxes)

        repaired_pre_total: Optional[float] = None
        repaired_gratuity: Optional[float] = None

        # ---------------------------------------------------------
        # PASS 2A: repair "reverse subtotal/tax" layouts
        # If prev_numeric + subtotal_line_numeric ~= total -> swap
        # ---------------------------------------------------------
        if repaired_total is not None:
            for i, r in enumerate(parsed_lines):
                if not r.get("subtotal_match"):
                    continue
                cur_val = r.get("value")
                if cur_val is None:
                    continue

                prev_val: Optional[float] = None
                for j in range(i - 1, max(-1, i - 3), -1):
                    pr = parsed_lines[j]
                    pv = pr.get("value")
                    if pv is None:
                        continue
                    if pr.get("total_match") or pr.get("tax_match") or pr.get("subtotal_match"):
                        continue
                    prev_val = float(pv)
                    break

                if prev_val is None:
                    continue

                # subtotal line number is actually "tax" and prev numeric is subtotal
                if _near(prev_val + float(cur_val), float(repaired_total)) and prev_val >= float(cur_val):
                    repaired_subtotal = float(prev_val)
                    repaired_tax = float(cur_val)
                    break

        # ---------------------------------------------------------
        # PASS 2B: math resolver (supports gratuity + pre_total)
        #
        # Goal order:
        # 1) Find final_total
        # 2) If final_total = pre_total + gratuity => solve (pre_total, gratuity)
        # 3) Solve pre_total = subtotal + tax => solve (subtotal, tax)
        #
        # This fixes receipt_74 pattern:
        #   total=64.52, pre_total=55.51, gratuity=9.01, tax=5.46, subtotal=50.05
        # ---------------------------------------------------------
        subtotal_cands = _values(subtotals)
        tax_cands = _values(taxes)
        total_cands = _values(totals)
        grat_cands = _values(gratuities)

        # Add tail numeric candidates (very important for OCR weirdness)
        tail_vals: List[float] = []
        for rr in parsed_lines[-24:]:
            v = rr.get("value")
            if isinstance(v, (int, float)):
                tail_vals.append(float(v))

        # Candidate pools (unique)
        def _uniq(vals: List[float]) -> List[float]:
            # stable unique keeping order
            seen = set()
            out = []
            for x in vals:
                rx = round(float(x), 4)
                if rx not in seen:
                    seen.add(rx)
                    out.append(float(x))
            return out

        all_vals = _uniq([*tail_vals, *subtotal_cands, *tax_cands, *total_cands, *grat_cands])

        # Choose final total (prefer max total keyword match; fallback to max numeric in tail)
        final_total: Optional[float] = None
        if total_cands:
            final_total = float(max(total_cands))
        else:
            # fallback: choose largest positive number near the end
            pos_tail = [v for v in tail_vals if v is not None and v > 0]
            final_total = float(max(pos_tail)) if pos_tail else None

        if final_total is not None:
            repaired_total = float(final_total)

            # --- Step 1: solve (pre_total, gratuity) where pre_total + gratuity == final_total ---
            # Build candidates for pre_total and gratuity
            # NOTE: even if "gratuity" line has no money, 9.01 can be mislabeled as tax -> include all_vals
            pre_total_candidates = [v for v in all_vals if 0 < v < final_total]
            gratuity_candidates = _uniq([*grat_cands, *all_vals])  # allow mislabeled values

            best_pre_grat = None  # (pre_total, gratuity, err, score)
            for g in gratuity_candidates:
                if g is None:
                    continue
                g = float(g)
                if g <= 0:
                    continue
                if g > final_total * 0.40:  # very generous upper bound
                    continue

                pre = final_total - g
                if pre <= 0:
                    continue

                # Prefer if pre is actually present in OCR numbers
                # (receipt_74 has 55.51)
                present_bonus = 0.0
                for cand in pre_total_candidates:
                    if _near(cand, pre, eps=0.06):
                        present_bonus = 0.5
                        break

                # Also bonus if there is an explicit gratuity/tip keyword line anywhere
                has_grat_kw = any(
                    ("GRATUITY" in (pl.get("line", "") or "").upper())
                    or ("TIP" in (pl.get("line", "") or "").upper())
                    or ("SERVICE CHARGE" in (pl.get("line", "") or "").upper())
                    for pl in parsed_lines
                )
                kw_bonus = 0.2 if has_grat_kw else 0.0

                # error if pre is not close to any OCR number (we still allow it, but lower score)
                pre_err = 0.0
                if pre_total_candidates:
                    pre_err = min(abs(pre - cand) for cand in pre_total_candidates)
                else:
                    pre_err = 0.0

                err = pre_err
                score = present_bonus + kw_bonus - err  # higher is better

                # accept if pre is reasonably close to some number OR we have gratuity keyword
                if err <= 0.06 or has_grat_kw:
                    if best_pre_grat is None or score > best_pre_grat[3]:
                        best_pre_grat = (pre, g, err, score)

            if best_pre_grat is not None:
                repaired_pre_total = float(best_pre_grat[0])
                repaired_gratuity = float(best_pre_grat[1])

            # --- Step 2: solve (subtotal, tax) such that subtotal + tax == base_total ---
            base_total = repaired_pre_total if repaired_pre_total is not None else final_total

            # candidate values for splitting base_total
            split_vals = [v for v in all_vals if 0 < v <= base_total + 5.0]
            split_vals = _uniq(split_vals)

            best_st = None  # (subtotal, tax, err)
            best_err = 1e9

            for a in split_vals:
                for b in split_vals:
                    if a == b:
                        continue
                    s = max(a, b)
                    t = min(a, b)

                    if t > base_total * 0.40:  # tax rarely huge
                        continue
                    if s < t:  # force subtotal >= tax
                        continue

                    err = abs((s + t) - base_total)
                    if err <= 0.06:
                        # prefer larger subtotal, then smaller tax
                        if err < best_err or (abs(err - best_err) < 1e-9 and best_st and s > best_st[0]):
                            best_err = err
                            best_st = (s, t, err)

            if best_st is not None:
                repaired_subtotal = float(best_st[0])
                repaired_tax = float(best_st[1])

        # ---------------------------------------------------------
        # Return full repaired summary (with pre_total + gratuity)
        # ---------------------------------------------------------
        return {
            "totals": totals,
            "subtotals": subtotals,
            "taxes": taxes,
            "gratuities": gratuities,
            "detected_regions": self.tax_regions,
            "repaired_total": repaired_total,
            "repaired_pre_total": repaired_pre_total,
            "repaired_gratuity": repaired_gratuity,
            "repaired_subtotal": repaired_subtotal,
            "repaired_tax": repaired_tax,
        }


# ---------------------------------------------------------
# Public API (call this from geo_extract.py / pipeline)
# ---------------------------------------------------------
def extract_summary_fields(
    words_or_text,
    *,
    max_distance: int = 3,
    auto_detect_region: bool = True,
) -> Dict[str, Any]:
    """
    Input:
      - words_or_text: OCR words list OR raw receipt text string

    Output:
      {
        "total": float|None,
        "subtotal": float|None,
        "tax": float|None,
        "detected_regions": [...]
      }
    """
    extractor = ReceiptFieldExtractor(max_distance=max_distance)
    res = extractor.extract_from_receipt(words_or_text, auto_detect_region=auto_detect_region)

    def pick_last_value(bucket: List[Dict[str, Any]]) -> Optional[float]:
        if not bucket:
            return None
        vals = [x.get("value") for x in bucket if isinstance(x.get("value"), (int, float))]
        return float(vals[-1]) if vals else None

    return {
        "total": res.get("repaired_total") if res.get("repaired_total") is not None else pick_last_value(res["totals"]),
        "subtotal": res.get("repaired_subtotal") if res.get("repaired_subtotal") is not None else pick_last_value(res["subtotals"]),
        "tax": res.get("repaired_tax") if res.get("repaired_tax") is not None else pick_last_value(res["taxes"]),
        "detected_regions": res.get("detected_regions", ["UNIVERSAL"]),
    }