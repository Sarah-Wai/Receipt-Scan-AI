# ============================================================
# POST-PROCESSING — all label fixes without retraining
# Place this block BEFORE predict_word_labels_and_conf_chunked
# ============================================================
import re
from typing import Any, Dict, List, Optional, Tuple

# ── PP-D1: Regex helpers ─────────────────────────────────────────────────────
_PP_PRICE_RE    = re.compile(r"^\-?\$?\d{1,4}[.,]\d{2}$")
_PP_NUMBER_RE   = re.compile(r"^\d+$")
_PP_QUANTITY_RE = re.compile(r"^\d{1,2}$")

# Zone Y thresholds (0-1000 bbox scale)
_PP_HEADER_Y_MAX  = 120    # top 12%  = header
_PP_FOOTER_Y_MIN  = 900    # bottom 10% = footer
_PP_PRICE_COL_X   = 650    # x-center > this = right price column
_PP_LOW_CONF_THR  = 0.75

# Total / summary keywords
_PP_TOTAL_KW = re.compile(
    r"^(total|grand\s*total|amount\s*due|balance\s*due|balance|settled|tot\.?)$",
    re.IGNORECASE,
)
_PP_DUE_CONT    = re.compile(r"^(due|amount)$",                              re.IGNORECASE)
_PP_SUBTOTAL_KW = re.compile(r"^(subtotal|sub[\s\-]?total|sub|slbtotal)$",   re.IGNORECASE)
_PP_TAX_KW      = re.compile(r"^(tax|vat|gst|hst|pst|qst|sst|service\s*charge|srv\.?)$", re.IGNORECASE)
_PP_DISCOUNT_KW = re.compile(r"^(discount|disc\.?|saving|coupon|promo)$",    re.IGNORECASE)

# ── NEW: Payment / tender keywords — NEVER menu items ────────────────────────
_PP_PAYMENT_KW = re.compile(
    r"^(cash|card|visa|mastercard|amex|discover|"
    r"wells?\s*fargo|chase|debit|debtt|credit|tap|nfc|"
    r"apple\s*pay|google\s*pay|interac|interao|"
    r"change|tender|paid|payment|charged|approved|auth|"
    r"tip|gratuity|suggested|tip\s*\d|server\s*tip|"
    r"shopping\s*card|shop\s*card)$",
    re.IGNORECASE,
)

# ── NEW: Prices that are $0.00 — always O outside discount context ────────────
_PP_ZERO_PRICE_RE = re.compile(r"^\$?0\.00$")

# ── NEW: Tax modifier (region word before real tax keyword) ───────────────────
_PP_TAX_MODIFIER_RE = re.compile(
    r"^(ca|us|state|local|city|county|fed|federal|"
    r"provincial|national|regional|g|p)$",
    re.IGNORECASE,
)

# ── NEW: Barcode pattern — 8-13 digits, never a menu name ────────────────────
_PP_BARCODE_RE = re.compile(r"^\d{6,14}$")

# ── NEW: Real Canadian Superstore / grocery flag tokens ──────────────────────
_PP_GROCERY_FLAG_RE = re.compile(r"^(MRJ|HRJ|GPHRJ|GPMRJ|GPRQ|FP|Fp|PP|RP)$", re.IGNORECASE)

# ── NEW: Glued fuzzy keyword patterns (OCR noise on summary words) ────────────
_PP_FUZZY_SUBTOTAL_RE = re.compile(
    r"^(sub[\.\-]?tot|subt[oa]|slbtotal|s[uo]btotal)[\:\.]?$", re.IGNORECASE
)
_PP_FUZZY_TOTAL_RE = re.compile(
    r"^(tot[\.\-]?al|totol|t[o0]tal|debittend)[\:\.]?$", re.IGNORECASE
)
_PP_FUZZY_TAX_RE = re.compile(
    r"^(\.?tax|\.tax|taxtotal|p=pst\d*|g=gst\d*|pste?\d*[\.\%]*)[\:\.]?$",
    re.IGNORECASE,
)

# Gst guest-count disambiguation
_PP_GST_GUESTS_RE = re.compile(r"^Gst\d*$",  re.IGNORECASE)
_PP_GST_GLUED_RE  = re.compile(r"^Gst(\d+)$", re.IGNORECASE)
_PP_SMALL_INT_RE  = re.compile(r"^\d{1,2}$")

# Words that should NEVER carry menu entity labels
_PP_NEVER_MENU = re.compile(
    r"^(table|tbl|chk|gst|pst|hst|qst|station|server|date|time|receipt|order|"
    r"guests?|items?|subtotal|total|tax|vat|balance|due|thank|please|"
    r"visit|call|online|www\.|http|change|cash|debit|credit|card|"
    r"interac|approved|auth|terminal|ref|rrn|transaction|purchase|"
    r"savings|refund|amount|deposit)$",
    re.IGNORECASE,
)

# Words that are only header/metadata content
_PP_HEADER_WORDS = re.compile(
    r"^(tel|vat|reg|gst|abn|ein|tin|fax|email|web|www|http|"
    r"©|registered|steakhouse|grill|restaurant|bistro|cafe|bar)$",
    re.IGNORECASE,
)

# ── NEW: Post-total structural tokens — never items ───────────────────────────
_PP_POST_TOTAL_STRUCTURAL = re.compile(
    r"^(transaction|record|type|purchase|acct|chequing|account|"
    r"amount|card|number|date|time|reference|author|invoice|"
    r"interac|approved|approveli|approveei|uerified|verified|"
    r"retain|copy|customer|important|signature|required|"
    r"items|sold|savings|refund|change|cash|debit|credit|"
    r"f|ds|tn|tnd|nue|rr\.|rr|rf)$",
    re.IGNORECASE,
)


def _pp_cy(bbox: List[int]) -> float:
    return (bbox[1] + bbox[3]) / 2.0

def _pp_cx(bbox: List[int]) -> float:
    return (bbox[0] + bbox[2]) / 2.0

def _pp_is_price(word: str) -> bool:
    return bool(_PP_PRICE_RE.match(word.strip()))

def _pp_is_barcode(word: str) -> bool:
    """6-14 digit strings are barcodes — never item names or prices."""
    return bool(_PP_BARCODE_RE.match(word.strip()))

def _pp_is_grocery_flag(word: str) -> bool:
    """MRJ/HRJ/FP/Fp/PP flags on Canadian grocery receipts."""
    return bool(_PP_GROCERY_FLAG_RE.match(word.strip()))

def _pp_get_zone(bbox: List[int]) -> str:
    cy = _pp_cy(bbox)
    if cy < _PP_HEADER_Y_MAX: return "header"
    if cy > _PP_FOOTER_Y_MIN: return "footer"
    return "body"


# ── PP-D2: Zone derivation ────────────────────────────────────��───────────────
def _pp_resolve_zones(
    bboxes:     List[List[int]],
    json_zones: Optional[List[str]],
) -> List[str]:
    if json_zones and len(json_zones) == len(bboxes):
        return json_zones
    return [_pp_get_zone(b) for b in bboxes]


# ── PP-D3: row_ids from bbox y-centers ───────────────────────────────────────
def _derive_row_ids_from_boxes(
    boxes_1000: List[List[int]],
    y_tol: int = 15,
) -> List[int]:
    if not boxes_1000:
        return []
    row_ids = []
    cur_row = 0
    prev_cy = (boxes_1000[0][1] + boxes_1000[0][3]) / 2.0
    for b in boxes_1000:
        cy = (b[1] + b[3]) / 2.0
        if abs(cy - prev_cy) > y_tol:
            cur_row += 1
            prev_cy  = cy
        row_ids.append(cur_row)
    return row_ids


# ── PP-Fix 0: Low-confidence false positive correction ───────────────────────
def _pp_fix_low_conf_false_positives(
    words:  List[str],
    labels: List[str],
    confs:  List[float],
) -> Tuple[List[str], List[float]]:
    """
    Tokens with entity label but confidence < threshold AND
    the token looks like structural noise → demote to O.
    """
    labels = labels[:]
    confs  = confs[:]
    for i, (w, lab, conf) in enumerate(zip(words, labels, confs)):
        if lab == "O":
            continue
        if conf < _PP_LOW_CONF_THR:
            wt = w.strip()
            # Barcodes tagged as MENU.NM → O
            if _pp_is_barcode(wt) and "MENU.NM" in lab:
                labels[i] = "O"
                confs[i]  = 1.0
            # Grocery flags (MRJ/HRJ/FP) → O
            elif _pp_is_grocery_flag(wt) and "MENU" in lab:
                labels[i] = "O"
                confs[i]  = 1.0
    return labels, confs


# ── PP-Fix 1: Zone-aware override ────────────────────────────────────────────
def _pp_fix_zone_labels(
    words:  List[str],
    labels: List[str],
    confs:  List[float],
    bboxes: List[List[int]],
    zones:  List[str],
) -> Tuple[List[str], List[float]]:
    """
    Header zone → force all entity labels to O
    Footer zone → force MENU labels to O
    Structural words → force MENU labels to O
    Barcodes → force MENU.NM to O
    Grocery flags (MRJ/HRJ/FP) → force MENU to O
    """
    labels = labels[:]
    confs  = confs[:]
    for i, (w, lab, zone) in enumerate(zip(words, labels, zones)):
        wt = w.strip()
        if zone == "header" and lab != "O":
            labels[i] = "O"
            confs[i]  = 1.0
        elif zone == "footer" and "MENU" in lab:
            labels[i] = "O"
            confs[i]  = 1.0
        # Barcodes should never be MENU.NM
        if _pp_is_barcode(wt) and "MENU.NM" in lab:
            labels[i] = "O"
            confs[i]  = 1.0
        # Grocery price flags should never be MENU entities
        if _pp_is_grocery_flag(wt) and "MENU" in lab:
            labels[i] = "O"
            confs[i]  = 1.0
        if _PP_NEVER_MENU.match(wt) and "MENU" in lab:
            labels[i] = "O"
            confs[i]  = 1.0
        if _PP_HEADER_WORDS.match(wt) and lab != "O":
            labels[i] = "O"
            confs[i]  = 1.0
    return labels, confs


# ── PP-Fix 2: Total keyword promotion ────────────────────────────────────────
def _pp_fix_total_keywords(
    words:  List[str],
    labels: List[str],
    confs:  List[float],
    bboxes: List[List[int]],
    zones:  List[str],
) -> Tuple[List[str], List[float]]:
    """
    Keyword-anchored label promotion including fuzzy OCR variants:
      "SLBTOTAL" → B-SUM.SUBTOTAL
      "DEBITTEND" → treated as total context
      "P=PST6%" → B-SUM.TAX
      "G=GST 5%" → B-SUM.TAX
      "Gst 4" / "Gst4" → O O  (guest count, not tax)
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    i = 0
    while i < n:
        w    = words[i].strip()
        zone = zones[i]
        conf = confs[i]

        # ── Fuzzy subtotal ────────────────────────────────────────────────────
        if (_PP_SUBTOTAL_KW.match(w) or _PP_FUZZY_SUBTOTAL_RE.match(w)) \
                and zone in ("body", "footer"):
            labels[i] = "B-SUM.SUBTOTAL"
            confs[i]  = max(conf, 0.95)
            j = i + 1
            while j < n and (_pp_is_price(words[j]) or words[j].strip() in (":", "$")):
                if _pp_is_price(words[j]):
                    labels[j] = "I-SUM.SUBTOTAL"
                    confs[j]  = 0.95
                j += 1
            i = j
            continue

        # ── Fuzzy total ───────────────────────────────────────────────────────
        if (_PP_TOTAL_KW.match(w) or _PP_FUZZY_TOTAL_RE.match(w)) \
                and zone in ("body", "footer"):
            labels[i] = "B-SUM.TOTAL"
            confs[i]  = max(conf, 0.95)
            j = i + 1
            while j < n and _PP_DUE_CONT.match(words[j].strip()):
                labels[j] = "I-SUM.TOTAL"
                confs[j]  = 0.95
                j += 1
            while j < n and (_pp_is_price(words[j]) or words[j].strip() in (":", "$")):
                if _pp_is_price(words[j]):
                    labels[j] = "I-SUM.TOTAL"
                    confs[j]  = 0.95
                j += 1
            i = j
            continue

        # ── Fuzzy tax (P=PST6%, G=GST 5%, .Tax, etc.) ────────────────────────
        if (_PP_TAX_KW.match(w) or _PP_FUZZY_TAX_RE.match(w)) \
                and zone in ("body", "footer"):

            next_tok      = words[i + 1].strip() if i + 1 < n else ""
            is_gst_glued  = bool(_PP_GST_GLUED_RE.match(w))
            is_gst_spaced = (
                _PP_GST_GUESTS_RE.match(w)
                and _PP_SMALL_INT_RE.match(next_tok)
                and int(next_tok) <= 20
            )
            if is_gst_glued or is_gst_spaced:
                labels[i] = "O"
                confs[i]  = 1.0
                i += 1
                continue

            labels[i] = "B-SUM.TAX"
            confs[i]  = max(conf, 0.90)
            j = i + 1
            # Skip percentage tokens like "5%", "6%"
            while j < n and re.match(r"^\d+\.?\d*\s*%$", words[j].strip()):
                j += 1
            while j < n and (_pp_is_price(words[j]) or words[j].strip() in (":", "%")):
                if _pp_is_price(words[j]):
                    labels[j] = "I-SUM.TAX"
                    confs[j]  = 0.90
                j += 1
            i = j
            continue

        # ── Tax modifier: "CA Sales Tax", "G=GST", "P=PST" ───────────────────
        if labels[i] == "B-SUM.TAX" and _PP_TAX_MODIFIER_RE.match(w):
            found_real_tax = False
            for j in range(i + 1, min(i + 4, n)):
                wj = words[j].strip()
                if _PP_TAX_KW.match(wj) or _PP_FUZZY_TAX_RE.match(wj):
                    found_real_tax = True
                    break
            if found_real_tax:
                labels[i] = "O"
                confs[i]  = 1.0
            i += 1
            continue

        # ── Discount ──────────────────────────────────────────────────────────
        if _PP_DISCOUNT_KW.match(w) and zone == "body":
            labels[i] = "B-MENU.DISCOUNTPRICE"
            confs[i]  = max(conf, 0.88)
            j = i + 1
            while j < n and (_pp_is_price(words[j]) or words[j].strip() in (":", "-")):
                if _pp_is_price(words[j]):
                    labels[j] = "I-MENU.DISCOUNTPRICE"
                    confs[j]  = 0.88
                j += 1
            i = j
            continue

        # ── "Items" token → O ─────────────────────────────────────────────────
        if w.lower() in ("items", "items.sold:") and "SUM" in labels[i]:
            labels[i] = "O"
            confs[i]  = 1.0

        i += 1
    return labels, confs


# ── PP-Fix 2b: Two-column summary price redistribution ───────────────────────
def _pp_fix_summary_price_order(
    words:  List[str],
    labels: List[str],
    confs:  List[float],
    bboxes: List[List[int]],
    zones:  List[str],
) -> Tuple[List[str], List[float]]:
    """
    Fixes two-column summary misalignment where prices appear AFTER
    all keywords in token order:

      Token seq:  Subtotal | Tax | Total | 33.90 | 3.39 | 37.29
      Wrong:      B-SUB    | B-TAX | B-TOT | I-TAX | I-TAX | I-TOT
      Fixed:      B-SUB    | B-TAX | B-TOT | I-SUB | I-TAX | I-TOT
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    SUM_KW_TO_PRICE = {
        "B-SUM.SUBTOTAL": "I-SUM.SUBTOTAL",
        "B-SUM.TAX":      "I-SUM.TAX",
        "B-SUM.TOTAL":    "I-SUM.TOTAL",
    }

    def _is_skip(idx: int) -> bool:
        w   = words[idx].strip()
        lab = labels[idx]
        return (
            lab == "O"
            and not _pp_is_price(w)
            and w.lower() not in ("due", "amount")
        )

    i = 0
    while i < n:
        if labels[i] not in SUM_KW_TO_PRICE:
            i += 1
            continue

        kw_list: List[Tuple[int, str]] = []
        j = i
        while j < n:
            lab = labels[j]
            if lab in SUM_KW_TO_PRICE:
                kw_list.append((j, lab))
                j += 1
            elif _is_skip(j):
                j += 1
            else:
                break

        if len(kw_list) < 2:
            i = max(j, i + 1)
            continue

        price_list: List[int] = []
        while j < n:
            w   = words[j].strip()
            lab = labels[j]
            if _pp_is_price(w):
                price_list.append(j)
                j += 1
            elif _is_skip(j):
                j += 1
            else:
                break

        if len(price_list) == len(kw_list):
            for (kw_idx, kw_lab), price_idx in zip(kw_list, price_list):
                expected = SUM_KW_TO_PRICE[kw_lab]
                if labels[price_idx] != expected:
                    labels[price_idx] = expected
                    confs[price_idx]  = max(confs[price_idx], 0.90)

        i = max(j, i + 1)
    return labels, confs


# ── PP-Fix 2c: Fuzzy OCR-corrupted total keyword recovery ────────────────────
def _pp_fix_fuzzy_total_keywords(
    words:  List[str],
    labels: List[str],
    confs:  List[float],
    bboxes: List[List[int]],
    zones:  List[str],
) -> Tuple[List[str], List[float]]:
    """
    Handles additional OCR-corrupted summary keywords not caught by fix 2:
      "iotal::" → B-SUM.SUBTOTAL/TOTAL context
      "Taxable Total:" → subtotal context
      "ARCP30.00" → ARCP prefix stripped, treat price
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    for i in range(n):
        w    = words[i].strip()
        zone = zones[i]
        if zone not in ("body", "footer"):
            continue

        # "iotal::" or "iotal" → probably "Total"
        if re.match(r"^[iI][oO]tal[\:\.]?[\:\.]?$", w) and labels[i] == "O":
            labels[i] = "B-SUM.TOTAL"
            confs[i]  = 0.85

        # "Taxable" before "Total:" → treat as subtotal marker
        if w.lower() == "taxable" and i + 1 < n:
            nxt = words[i + 1].strip().lower()
            if nxt in ("total:", "total", "iotal::", "iotal"):
                labels[i]     = "B-SUM.SUBTOTAL"
                confs[i]      = 0.88
                labels[i + 1] = "O"   # suppress the following "Total:" token
                confs[i + 1]  = 1.0

        # "ARCP30.00" / "ARCP" → O (loyalty program prefix, not price)
        if re.match(r"^ARCP\d*[\.\d]*$", w, re.IGNORECASE):
            labels[i] = "O"
            confs[i]  = 1.0

    return labels, confs


# ── PP-Fix 3: BIO sequence repair ────────────────────────────────────────────
def _pp_fix_bio_sequence(
    words:  List[str],
    labels: List[str],
    confs:  List[float],
    bboxes: List[List[int]],
) -> Tuple[List[str], List[float]]:
    """
    Pass 1: orphan I-X with no prior B-X → promote to B-X
    Pass 2: digit-only tokens → not MENU.NM
    Pass 3: price token tagged as MENU.NM → fix to MENU.PRICE
    Pass 4: quantity digit before B-MENU.NM next token → force O
    Pass 5: barcodes tagged as MENU.NM → O
    Pass 6: grocery flags (MRJ/HRJ/FP) tagged as MENU → O
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    # Pass 1
    active = None
    for i in range(n):
        lab = labels[i]
        if lab == "O":
            active = None
            continue
        prefix, etype = (lab.split("-", 1) + [""])[:2]
        if prefix == "B":
            active = etype
        elif prefix == "I" and active != etype:
            labels[i] = f"B-{etype}"
            active     = etype

    # Pass 2
    for i in range(n):
        w = words[i].strip()
        if "MENU.NM" in labels[i] and _PP_NUMBER_RE.match(w):
            labels[i] = "O"
            confs[i]  = 1.0

    # Pass 3
    for i in range(n):
        if "MENU.NM" in labels[i] and _pp_is_price(words[i]):
            labels[i] = "B-MENU.PRICE"
            confs[i]  = max(confs[i], 0.88)

    # Pass 4
    for i in range(n):
        w = words[i].strip()
        if _PP_QUANTITY_RE.match(w) and "MENU" in labels[i]:
            if i + 1 < n and "MENU.NM" in labels[i + 1]:
                labels[i] = "O"
                confs[i]  = 1.0

    # Pass 5: barcodes → O
    for i in range(n):
        if _pp_is_barcode(words[i].strip()) and "MENU.NM" in labels[i]:
            labels[i] = "O"
            confs[i]  = 1.0

    # Pass 6: grocery flags → O
    for i in range(n):
        if _pp_is_grocery_flag(words[i].strip()) and "MENU" in labels[i]:
            labels[i] = "O"
            confs[i]  = 1.0

    return labels, confs


# ── PP-Fix 4: Menu name continuation ─────────────────────────────────────────
def _pp_fix_menu_continuation(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    zones:   List[str],
    row_ids: List[int],
) -> Tuple[List[str], List[float]]:
    """
    Forward extension:  "Haricots" B + "Vert" O → I-MENU.NM
    Backward extension: "BT" O + "TURNBULL" B   → BT becomes B, TURNBULL I
    
    Stops at:
      - barcodes
      - grocery flags (MRJ/HRJ/FP)
      - structural words
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    rows: Dict[int, List[int]] = {}
    for idx, rid in enumerate(row_ids):
        rows.setdefault(rid, []).append(idx)

    for rid, idxs in rows.items():

        menu_start = None
        for idx in idxs:
            if labels[idx] == "B-MENU.NM":
                menu_start = idx
                break
        if menu_start is None:
            continue

        # ── Forward extension ─────────────────────────────────────────────────
        for idx in idxs:
            if idx <= menu_start:
                continue
            w    = words[idx].strip()
            lab  = labels[idx]
            conf = confs[idx]
            if zones[idx] == "header":                   break
            if _pp_is_price(w):                          break
            if _pp_is_barcode(w):                        break   # NEW
            if _pp_is_grocery_flag(w):                   break   # NEW
            if lab.startswith("B-") and lab != "B-MENU.NM": break
            if lab == "O" and conf >= 0.92 and _PP_NEVER_MENU.match(w): break
            if lab == "I-MENU.NM":                       continue
            if (not _PP_NUMBER_RE.match(w)
                    and not _pp_is_price(w)
                    and not _PP_NEVER_MENU.match(w)
                    and not _pp_is_barcode(w)
                    and not _pp_is_grocery_flag(w)
                    and len(w) > 1):
                labels[idx] = "I-MENU.NM"
                confs[idx]  = max(conf, 0.80)

        # ── Backward extension ────────────────────────────────────────────────
        pre = [idx for idx in idxs if idx < menu_start]
        if pre and _PP_QUANTITY_RE.match(words[pre[0]].strip()):
            pre = pre[1:]

        new_b_set = False
        for idx in pre:
            w    = words[idx].strip()
            conf = confs[idx]
            if (_PP_NUMBER_RE.match(w) or _pp_is_price(w)
                    or _PP_NEVER_MENU.match(w)
                    or _pp_is_barcode(w)
                    or _pp_is_grocery_flag(w)
                    or len(w) <= 1):
                continue
            if not new_b_set:
                labels[idx]        = "B-MENU.NM"
                confs[idx]         = max(conf, 0.80)
                labels[menu_start] = "I-MENU.NM"
                new_b_set = True
            else:
                labels[idx] = "I-MENU.NM"
                confs[idx]  = max(conf, 0.80)

    return labels, confs


# ── PP-Fix 5: Two-column price alignment ─────────────────────────────────────
def _pp_fix_price_column(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: List[int],
) -> Tuple[List[str], List[float]]:
    """
    Right-column prices (x-center > _PP_PRICE_COL_X) in body zone
    that the model missed → tag as B-MENU.PRICE.
    
    Skips:
      - tokens already tagged as SUM.*
      - $0.00 prices in post-total rows
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    rows_with_price: set = set()

    for i in range(n):
        w    = words[i].strip()
        lab  = labels[i]
        bbox = bboxes[i]
        rid  = row_ids[i]

        if not _pp_is_price(w):
            continue

        # Never re-tag summary prices
        if "SUM." in lab:
            if lab in ("B-SUM.PRICE", "I-SUM.PRICE",
                       "B-SUM.TOTAL", "I-SUM.TOTAL",
                       "B-SUM.SUBTOTAL", "I-SUM.SUBTOTAL",
                       "B-SUM.TAX", "I-SUM.TAX"):
                rows_with_price.add(rid)
                continue

        cx   = _pp_cx(bbox)
        zone = _pp_get_zone(bbox)

        if cx > _PP_PRICE_COL_X and zone == "body":
            if lab == "O" or (
                lab in ("B-MENU.PRICE", "I-MENU.PRICE")
                and confs[i] < _PP_LOW_CONF_THR
            ):
                labels[i] = "I-MENU.PRICE" if rid in rows_with_price else "B-MENU.PRICE"
                confs[i]  = max(confs[i], 0.88)

        if labels[i] in ("B-MENU.PRICE", "I-MENU.PRICE"):
            rows_with_price.add(rid)

    return labels, confs


# ─�� PP-Fix 6: Consecutive B- cleanup ─────────────────────────────────────────
def _pp_fix_consecutive_b_tags(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    row_ids: List[int],
) -> Tuple[List[str], List[float]]:
    """
    Two consecutive B-MENU.NM on the SAME row → second becomes I-MENU.NM.
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    for i in range(1, n):
        if (labels[i]     == "B-MENU.NM"
                and labels[i - 1] == "B-MENU.NM"
                and row_ids[i]    == row_ids[i - 1]
                and not _pp_is_price(words[i].strip())):
            labels[i] = "I-MENU.NM"

    return labels, confs


# ── PP-Fix 7: Post-total zone enforcement ─────────────────────────────────────
def _pp_fix_post_total_zone(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    zones:   List[str],
) -> Tuple[List[str], List[float]]:
    """
    After the LAST B-SUM.TOTAL token, NOTHING should be MENU.NM or MENU.PRICE.
    
    Specifically:
      "Wells Fargo $64.43" → O O O
      "Change $0.00"       → O O
      "DEBIT $19.47"       → O O
      "Cash $8.00"         → O O
      "DS 74.74"           → O O  (debit tender)
      "DEBITTEND"          → already caught by fuzzy total
    
    Keeps:
      B-SUM.TOTAL, I-SUM.TOTAL    → unchanged
      B-SUM.TAX, I-SUM.TAX        → unchanged
      B-SUM.SUBTOTAL              → unchanged
    
    Also handles "CA Sales Tax" pattern (region modifier before tax keyword).
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    # ── Step A: Find last B-SUM.TOTAL index ──────────────────────────────────
    last_total_idx = -1
    for i in range(n):
        if labels[i] == "B-SUM.TOTAL":
            last_total_idx = i

    # ── Step B: Fix "CA Sales Tax" region modifier pattern ───────────────────
    for i in range(n):
        w   = words[i].strip()
        lab = labels[i]
        if lab == "B-SUM.TAX" and _PP_TAX_MODIFIER_RE.match(w):
            found_real_tax = False
            for j in range(i + 1, min(i + 4, n)):
                wj = words[j].strip()
                if _PP_TAX_KW.match(wj) or _PP_FUZZY_TAX_RE.match(wj):
                    found_real_tax = True
                    break
            if found_real_tax:
                labels[i] = "O"
                confs[i]  = 1.0

    # ── Step C: Post-total cleanup ────────────────────────────────────────────
    if last_total_idx < 0:
        return labels, confs

    # Find the I-SUM.TOTAL that follows (skip over it)
    post_start = last_total_idx + 1
    while post_start < n and labels[post_start] in (
        "I-SUM.TOTAL", "I-SUM.TAX", "I-SUM.SUBTOTAL"
    ):
        post_start += 1

    for i in range(post_start, n):
        w   = words[i].strip()
        lab = labels[i]

        # Keep summary labels intact
        if lab in (
            "B-SUM.TOTAL",   "I-SUM.TOTAL",
            "B-SUM.TAX",     "I-SUM.TAX",
            "B-SUM.SUBTOTAL","I-SUM.SUBTOTAL",
        ):
            continue

        # Demote any remaining MENU labels
        if "MENU" in lab:
            labels[i] = "O"
            confs[i]  = 1.0
            continue

        # Payment keywords always → O (catches "Wells Fargo", "DEBIT", "Cash", etc.)
        if _PP_PAYMENT_KW.match(w):
            labels[i] = "O"
            confs[i]  = 1.0
            continue

        # Post-total structural tokens → O
        if _PP_POST_TOTAL_STRUCTURAL.match(w):
            labels[i] = "O"
            confs[i]  = 1.0

    return labels, confs


# ── PP-Fix 7b: Discount line recovery ────────────────────────────────────────
def _pp_fix_discount_lines(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: List[int],
    zones:   List[str],
) -> Tuple[List[str], List[float]]:
    """
    Negative prices (e.g. -50.00, -$3.20) in body zone on a row that has
    a MENU item → tag as B-MENU.DISCOUNTPRICE.
    Also handles lines like "X off each -Y.YY".
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    _NEG_PRICE_RE = re.compile(r"^\-\$?\d{1,4}[.,]\d{2}$")

    rows_with_menu: set = set()
    for i in range(n):
        if "MENU.NM" in labels[i] or "MENU.PRICE" in labels[i]:
            rows_with_menu.add(row_ids[i])

    for i in range(n):
        w    = words[i].strip()
        lab  = labels[i]
        zone = zones[i]
        if zone != "body":
            continue
        if _NEG_PRICE_RE.match(w) and lab == "O":
            labels[i] = "B-MENU.DISCOUNTPRICE"
            confs[i]  = max(confs[i], 0.85)

    return labels, confs


# ── PP-Fix 8: Orphan I- price pairing ────────────────────────────────────────
def _pp_fix_orphan_i_with_price_pairing(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: List[int],
) -> Tuple[List[str], List[float]]:
    """
    I-MENU.PRICE with no prior B-MENU.PRICE on the same row →
    promote to B-MENU.PRICE.
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    row_has_b_price: Dict[int, bool] = {}
    for i in range(n):
        rid = row_ids[i]
        if labels[i] == "B-MENU.PRICE":
            row_has_b_price[rid] = True

    for i in range(n):
        if labels[i] == "I-MENU.PRICE":
            rid = row_ids[i]
            if not row_has_b_price.get(rid, False):
                labels[i] = "B-MENU.PRICE"
                row_has_b_price[rid] = True

    return labels, confs


# ── PP-Fix 9: Unanchored price recovery ──────────────────────────────────────
def _pp_fix_unanchored_prices(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: List[int],
    zones:   List[str],
) -> Tuple[List[str], List[float]]:
    """
    Prices in body zone that are O but sit in the right column
    AND no summary keyword has been seen recently → B-MENU.PRICE.
    
    Avoids touching post-total prices (already zeroed by fix 7).
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    # Don't tag unanchored prices after TOTAL
    last_total_idx = -1
    for i in range(n):
        if labels[i] == "B-SUM.TOTAL":
            last_total_idx = i

    for i in range(n):
        if i > last_total_idx and last_total_idx >= 0:
            break
        w    = words[i].strip()
        lab  = labels[i]
        zone = zones[i]
        if lab != "O" or zone != "body":
            continue
        if not _pp_is_price(w):
            continue
        cx = _pp_cx(bboxes[i])
        if cx > _PP_PRICE_COL_X:
            labels[i] = "B-MENU.PRICE"
            confs[i]  = max(confs[i], 0.82)

    return labels, confs


# ── PP-Fix 10: Colon-price context fix ───────────────────────────────────────
def _pp_fix_colon_prices(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: List[int],
) -> Tuple[List[str], List[float]]:
    """
    Tokens like "0:34" or "2:80" where the colon is OCR noise for a decimal:
    if the token is in the right column and not a valid time → convert to price.
    Mutates words in-place for downstream use.
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    _COLON_NUM_RE = re.compile(r"^(\d{1,4}):(\d{2})$")

    for i in range(n):
        w  = words[i].strip()
        m  = _COLON_NUM_RE.match(w)
        if not m:
            continue
        h, mm = int(m.group(1)), int(m.group(2))
        if h > 23 or mm > 59:
            # Not a valid time → it's a price with OCR colon
            fixed = f"{h}.{mm:02d}"
            words[i] = fixed
            if labels[i] == "O" and _pp_cx(bboxes[i]) > _PP_PRICE_COL_X:
                labels[i] = "B-MENU.PRICE"
                confs[i]  = max(confs[i], 0.82)

    return labels, confs


# ── PP-Fix 11: Real Canadian Superstore / grocery format fix ──────────────────
def _pp_fix_grocery_receipt_format(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: List[int],
    zones:   List[str],
) -> Tuple[List[str], List[float]]:
    """
    Handles the Canadian grocery format:
      [ITEM NAME]  [BARCODE]  [MRJ/HRJ flag]  [PRICE]

    On each row:
      1. Barcodes (6-14 digits) → O
      2. MRJ/HRJ/FP/Fp flags → O
      3. Category codes like "21-GROCERY", "27-PRODUCE", "31-NEATS" → O
      4. Price stays as MENU.PRICE (or gets tagged if missed)
      5. Only text tokens before barcode → MENU.NM

    Also handles:
      - "RECYCLING FEE DEPOSIT" → O (not a purchasable item line)
      - Per-unit pricing: "2 @ $0.99 ea", "3 for $1.00" → O for the formula tokens
      - Weight pricing: "1.015kg @ $6.55/kg" → O for weight tokens
    """
    n      = len(words)
    labels = labels[:]
    confs  = confs[:]

    # Category section header pattern
    _CATEGORY_RE = re.compile(r"^\d{2}-[A-Z]", re.IGNORECASE)
    # Per-unit pricing tokens
    _PER_UNIT_RE = re.compile(
        r"^(\d+\.?\d*\s*(kg|lb|g|oz|l|ml)[\@\/]?|"
        r"\d+\s*(for|@|at)\s*\$?\d|"
        r"[234]\.for\$\d|"
        r"\d+\.\d+kg|"
        r"Gross|tare=|Net[\d\w]*)$",
        re.IGNORECASE,
    )
    # "RECYCLING FEE", "DEPOSIT" — eco fees, not purchasable items
    _ECO_FEE_RE = re.compile(
        r"^(recycling|deposit|eco|crf|crv|bottle|bag\s*fee|fee)$",
        re.IGNORECASE,
    )

    rows: Dict[int, List[int]] = {}
    for idx, rid in enumerate(row_ids):
        rows.setdefault(rid, []).append(idx)

    for rid, idxs in rows.items():
        if not idxs:
            continue

        zone = zones[idxs[0]]
        if zone not in ("body",):
            continue

        # ── Step 1: Nullify barcodes, flags, category codes, per-unit tokens ──
        for idx in idxs:
            w = words[idx].strip()
            if _pp_is_barcode(w):
                labels[idx] = "O"
                confs[idx]  = 1.0
            elif _pp_is_grocery_flag(w):
                labels[idx] = "O"
                confs[idx]  = 1.0
            elif _CATEGORY_RE.match(w):
                labels[idx] = "O"
                confs[idx]  = 1.0
            elif _PER_UNIT_RE.match(w):
                labels[idx] = "O"
                confs[idx]  = 1.0
            elif _ECO_FEE_RE.match(w):
                # Eco fees: if they get B-MENU.NM, relabel as O
                # (they appear in item lines but aren't extractable items)
                pass  # keep — they may be valid line items with prices

        # ── Step 2: Find the price (rightmost price-like token) ───────────────
        price_idx = None
        for idx in reversed(idxs):
            w = words[idx].strip()
            if _pp_is_price(w) or re.match(r"^\d{1,4}[.,]\d{2}$", w):
                price_idx = idx
                break

        if price_idx is not None:
            # Ensure price is tagged
            if labels[price_idx] == "O":
                labels[price_idx] = "B-MENU.PRICE"
                confs[price_idx]  = max(confs[price_idx], 0.85)

        # ── Step 3: Name tokens = text tokens BEFORE any barcode ─────────────
        # Find first barcode position on this row
        first_barcode_idx = None
        for idx in idxs:
            if _pp_is_barcode(words[idx].strip()):
                first_barcode_idx = idx
                break

        # All non-structural, non-number text before the barcode → MENU.NM
        name_set_b = False
        for idx in idxs:
            if first_barcode_idx is not None and idx >= first_barcode_idx:
                break
            if idx == price_idx:
                continue
            w    = words[idx].strip()
            lab  = labels[idx]
            if (lab == "O"
                    and not _PP_NUMBER_RE.match(w)
                    and not _pp_is_price(w)
                    and not _pp_is_barcode(w)
                    and not _pp_is_grocery_flag(w)
                    and not _PP_NEVER_MENU.match(w)
                    and not _CATEGORY_RE.match(w)
                    and len(w) > 1):
                if not name_set_b:
                    labels[idx] = "B-MENU.NM"
                    confs[idx]  = max(confs[idx], 0.82)
                    name_set_b  = True
                else:
                    labels[idx] = "I-MENU.NM"
                    confs[idx]  = max(confs[idx], 0.80)

    return labels, confs


# ── MASTER: postprocess_predictions ──────────────────────────────────────────
def postprocess_predictions(
    words:   List[str],
    labels:  List[str],
    confs:   List[float],
    bboxes:  List[List[int]],
    row_ids: Optional[List[int]] = None,
    zones:   Optional[List[str]] = None,
) -> Tuple[List[str], List[float]]:
    """
    All post-processing fixes in correct order. No retraining needed.

    Step  0  Low-conf false positive correction
    Step  1  Zone override (+ barcode/grocery flag suppression)
    Step  2  Total keyword promotion  (dot-strip fix + fuzzy OCR variants)
    Step  2b Two-column summary price redistribution
    Step  2c Fuzzy OCR-corrupted total keyword recovery
    Step  3  BIO sequence repair     (+ barcode/flag passes)
    Step  4  Menu name continuation  (stops at barcodes/flags)
    Step  5  Two-column price alignment
    Step  6  Consecutive B- cleanup
    Step  7  Post-total zone enforcement
    Step  7b Discount line recovery
    Step  8  Orphan I- price pairing
    Step  9  Unanchored price recovery
    Step 10  Colon-price context fix
    Step 11  Grocery receipt format fix (Real Canadian Superstore, Walmart, etc.)
    """
    n = len(words)

    if not row_ids or len(row_ids) != n:
        row_ids = _derive_row_ids_from_boxes(bboxes)

    resolved_zones = _pp_resolve_zones(bboxes, zones)

    labels, confs = _pp_fix_low_conf_false_positives(words, labels, confs)
    labels, confs = _pp_fix_zone_labels(words, labels, confs, bboxes, resolved_zones)
    labels, confs = _pp_fix_total_keywords(words, labels, confs, bboxes, resolved_zones)
    labels, confs = _pp_fix_summary_price_order(words, labels, confs, bboxes, resolved_zones)
    labels, confs = _pp_fix_fuzzy_total_keywords(words, labels, confs, bboxes, resolved_zones)
    labels, confs = _pp_fix_bio_sequence(words, labels, confs, bboxes)
    labels, confs = _pp_fix_menu_continuation(words, labels, confs, bboxes, resolved_zones, row_ids)
    labels, confs = _pp_fix_price_column(words, labels, confs, bboxes, row_ids)
    labels, confs = _pp_fix_consecutive_b_tags(words, labels, confs, row_ids)
    labels, confs = _pp_fix_post_total_zone(words, labels, confs, bboxes, resolved_zones)
    labels, confs = _pp_fix_discount_lines(words, labels, confs, bboxes, row_ids, resolved_zones)
    labels, confs = _pp_fix_orphan_i_with_price_pairing(words, labels, confs, bboxes, row_ids)
    labels, confs = _pp_fix_unanchored_prices(words, labels, confs, bboxes, row_ids, resolved_zones)
    labels, confs = _pp_fix_colon_prices(words, labels, confs, bboxes, row_ids)
    labels, confs = _pp_fix_grocery_receipt_format(words, labels, confs, bboxes, row_ids, resolved_zones)

    return labels, confs