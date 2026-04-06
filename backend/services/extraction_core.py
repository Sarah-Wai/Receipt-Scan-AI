from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from receipt_extraction.item_flag_enricher import ItemFlagEnricher
from receipt_extraction.geo_extract_enhanced_v4 import cord_plus_geo_extract_v4, CordGeoConfig
from receipt_extraction.geo_extract_enhanced_long import cord_plus_geo_extract_long
from db.sqlite_writer import init_sqlite_schema, save_geo_fusion_to_sqlite,load_ocr_text
from receipt_extraction.sroie_extractor import extract_sroie_fields_from_ocr_json, resolve_image_path
from receipt_extraction.fuzzy_summary import ReceiptFieldExtractor
from receipt_extraction.item_validation_engine import ItemValidationEngine
from receipt_extraction.item_line_processor import reconstruct_items_from_sequence
from receipt_extraction.model_postprocess import postprocess_predictions

# ============================================================
# Base Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]   # backend/
MODELS_DIR = BASE_DIR / "models"

CORD_MODEL_PATH = str(MODELS_DIR / "cord_layout")
SROIE_MODEL_PATH = str(MODELS_DIR / "sorie_layout")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

OUT_DIR = DATA_DIR / "extraction_outputs"
PATTERN = "*.json"
MAX_FILES = None

# ------------------------------------------------------------
# Device Setup
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Inference Parameters
# ------------------------------------------------------------
MAX_LENGTH = 512
STRIDE = 128
WORD_CHUNK = 400

# ============================================================
# Phase-2 Model → CORD (Item Extraction)
# ============================================================
cord_processor = LayoutLMv3Processor.from_pretrained(
    str(CORD_MODEL_PATH),
    apply_ocr=False
)
cord_model = LayoutLMv3ForTokenClassification.from_pretrained(
    str(CORD_MODEL_PATH)
).to(DEVICE)
cord_model.eval()

CORD_ID2LABEL = {int(k): v for k, v in cord_model.config.id2label.items()}
CORD_LABEL2ID = {v: k for k, v in CORD_ID2LABEL.items()}

# ============================================================
# Phase-3 Model → SROIE (Summary / Metadata Extraction)
# ============================================================
sroie_processor = LayoutLMv3Processor.from_pretrained(
    str(SROIE_MODEL_PATH),
    apply_ocr=False
)
sroie_model = LayoutLMv3ForTokenClassification.from_pretrained(
    str(SROIE_MODEL_PATH)
).to(DEVICE)
sroie_model.eval()

SROIE_ID2LABEL = {int(k): v for k, v in sroie_model.config.id2label.items()}
SROIE_LABEL2ID = {v: k for k, v in SROIE_ID2LABEL.items()}

# ============================================================
# Utility
# ============================================================
def get_models():
    return {
        "device": DEVICE,
        "cord": {
            "model":     cord_model,
            "processor": cord_processor,
            "id2label":  CORD_ID2LABEL,
            "label2id":  CORD_LABEL2ID,
        },
        "sroie": {
            "model":     sroie_model,
            "processor": sroie_processor,
            "id2label":  SROIE_ID2LABEL,
            "label2id":  SROIE_LABEL2ID,
        },
        "params": {
            "max_length": MAX_LENGTH,
            "stride":     STRIDE,
            "word_chunk": WORD_CHUNK,
        },
    }

# ============================================================
# Initialize enricher
# ============================================================
enricher = ItemFlagEnricher()
print("✓ ItemFlagEnricher initialized")

# ============================================================
# Tax anchors
# ============================================================
TAX_ANCHORS = {
    "tax", "taxes", "vat", "gst", "pst", "hst", "qst", "sst", "cst",
    "cgst", "sgst", "igst",
    "ppn", "pph", "pajak",
    "iva", "tva", "btw", "moms", "mwst", "ust", "kdv", "dph",
    "pb1", "pb-1", "get",
    "txtl", "taxtotal", "totaltax", "salestax", "statetax",
}
NUM_LIKE_RE = re.compile(r"^\$?[-+]?\d[\d\.,]*$")

# ============================================================
# Currency Prefix Normalization
# ============================================================
CURRENCY_PREFIX_RE = re.compile(
    r"""
    ^
    (?P<cur>
        CA\$|CAD\$?|AUD\$?|USD\$?|SGD\$?|NZD\$?|HKD\$?|
        EUR|GBP|JPY|
        \$|€|£|¥
    )
    \s*
    (?P<amt>
        [-+]?\d{1,6}(?:[.,]\d{2})
    )
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

def normalize_currency_prefix_token(w: str) -> Tuple[str, Optional[str]]:
    t = (w or "").strip()
    if not t:
        return t, None
    t2 = t
    t2 = t2.replace("C A $", "CA$").replace("C A D", "CAD")
    t2 = t2.replace("A U D", "AUD").replace("U S D", "USD")
    t2 = t2.replace("€ ", "€").replace("$ ", "$").replace("£ ", "£")
    t2 = re.sub(r"\s+", "", t2)
    m = CURRENCY_PREFIX_RE.match(t2)
    if not m:
        return t, None
    cur = m.group("cur")
    amt = m.group("amt").replace(",", ".")
    return amt, cur


def _colon_price_fix(m: re.Match) -> str:
    """Convert colon-separated pair to price ONLY if not a valid time."""
    h, mm = int(m.group(1)), int(m.group(2))
    if h > 23 or mm > 59:
        return f"{h}.{mm:02d}"
    return m.group(0)

def normalize_word_for_layoutlm(w: str) -> str:
    t = (w or "").strip()
    t, _cur = normalize_currency_prefix_token(t)
    t = re.sub(r"(\d),(\d{2})\b",        r"\1.\2", t)
    t = re.sub(r"\b(\d{1,4}):(\d{2})\b", _colon_price_fix, t)
    t = re.sub(r"^:(\d)",                r"\1",    t)
    t = re.sub(r"(\d\.\d{2})[.:*]+$",    r"\1",    t)
    t = re.sub(r"(\d\.\d{2})[A-Za-z]+$", r"\1",    t)
    t = re.sub(r"(\d\.\d{2})-A\b",       r"\1",    t)
    t = re.sub(r"[<>\";]+",              "",       t)
    return t


# ============================================================
# Fuzzy token normalization
# ============================================================
def build_summary_extractor_for_words(
    words: List[str],
    *,
    max_distance: int = 2,
    auto_detect_region: bool = True,
):
    try:
        extractor = ReceiptFieldExtractor(max_distance=max_distance)
    except Exception:
        return None
    if auto_detect_region:
        try:
            text = " ".join([str(w) for w in words if w is not None])
            regions = extractor.infer_region_from_text(text)
            extractor.tax_regions = regions
            extractor._init_engines()
        except Exception:
            pass
    return extractor

def fuzzy_normalize_summary_tokens(
    words: List[str],
    *,
    extractor=None,
    max_distance: int = 2,
    auto_detect_region: bool = True,
    canonical_case: str = "UPPER",
) -> List[str]:
    if extractor is None:
        extractor = build_summary_extractor_for_words(
            words,
            max_distance=max_distance,
            auto_detect_region=auto_detect_region,
        )
    if extractor is None:
        return [str(w) if w is not None else "" for w in words]

    def canon(tok: str) -> str:
        return tok.upper() if canonical_case == "UPPER" else tok.title()

    out: List[str] = []
    for w in words:
        t = (str(w) if w is not None else "").strip()
        if not t:
            out.append(t)
            continue
        if re.fullmatch(r"[$€£¥₹]?\s*[-+]?\d[\d,\.]*", t):
            out.append(t)
            continue
        try:
            if extractor.subtotal_engine.best_match(t):
                out.append(canon("subtotal"))
            elif extractor.total_engine.best_match(t):
                out.append(canon("total"))
            elif extractor.tax_engine.best_match(t):
                out.append(canon("tax"))
            else:
                out.append(t)
        except Exception:
            out.append(t)
    return out

def normalize_words_keep_currency(
    words: List[str],
) -> Tuple[List[str], List[Optional[str]]]:
    words = fuzzy_normalize_summary_tokens(
        words,
        max_distance=2,
        auto_detect_region=True,
        canonical_case="UPPER",
    )
    norm_words: List[str] = []
    word_currency: List[Optional[str]] = []
    for w in words:
        amt, cur = normalize_currency_prefix_token(w)
        norm_words.append(normalize_word_for_layoutlm(amt))
        word_currency.append(cur)
    return norm_words, word_currency


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

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
            prev_cy = cy
        row_ids.append(cur_row)
    return row_ids


def load_cordlike_example(
    json_path: Path,
    *,
    dataset_root: Path,
) -> Dict[str, Any]:
    rec = json.loads(json_path.read_text(encoding="utf-8"))
    words = rec.get("words")
    bboxes = rec.get("bboxes")
    img_path = rec.get("image_path")

    if not isinstance(words, list) or not isinstance(bboxes, list):
        raise ValueError(f"Missing words/bboxes in {json_path.name}")
    if len(words) != len(bboxes):
        raise ValueError(
            f"Length mismatch: words={len(words)} bboxes={len(bboxes)}"
            f" in {json_path.name}"
        )

    img_file = resolve_image_path(
        str(img_path), dataset_root=dataset_root, json_path=json_path
    )
    image = Image.open(img_file).convert("RGB")
    W, H = image.size

    boxes = [[int(x) for x in bb] for bb in bboxes]
    max_x2 = max(b[2] for b in boxes) if boxes else 0
    max_y2 = max(b[3] for b in boxes) if boxes else 0

    if max_x2 > 1100 or max_y2 > 1100:
        def norm_box(bb: List[int]) -> List[int]:
            x0, y0, x1, y1 = bb
            return [
                max(0, min(1000, int(1000 * x0 / max(1, W)))),
                max(0, min(1000, int(1000 * y0 / max(1, H)))),
                max(0, min(1000, int(1000 * x1 / max(1, W)))),
                max(0, min(1000, int(1000 * y1 / max(1, H)))),
            ]
        boxes = [norm_box(bb) for bb in boxes]

    json_row_ids = rec.get("row_ids")
    if json_row_ids and len(json_row_ids) == len(words):
        row_ids = [int(r) for r in json_row_ids]
    else:
        row_ids = _derive_row_ids_from_boxes(boxes)

    json_zones = rec.get("zones")
    if json_zones and len(json_zones) == len(words):
        zones = json_zones
    else:
        zones = None

    return {
        "id":          rec.get("id", json_path.stem),
        "words":       [str(w) for w in words],
        "bboxes":      boxes,
        "row_ids":     row_ids,
        "meta":        rec.get("meta", {}),
        "zones":       zones,
        "image":       image,
        "image_path":  str(img_file),
        "__json_path": str(json_path),
        "image_size":  [W, H],
    }


# ============================================================
# Force TAX anchor labels
# ============================================================
def norm_tok(s: str) -> str:
    s = (s or "").strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch in {"%", "-"})

def force_tax_anchor_labels(
    words: List[str],
    labels: List[str],
) -> List[str]:
    out = labels[:]
    n = len(words)
    for i in range(n):
        w = norm_tok(words[i])
        if (w in TAX_ANCHORS) or any(
            k in w for k in [
                "pajak", "ppn", "gst", "pst", "hst",
                "qst",   "vat", "tax", "pb1", "pb-1",
            ]
        ):
            out[i] = "B-TAX"
            for j in range(i + 1, min(i + 6, n)):
                if NUM_LIKE_RE.match(str(words[j]).strip()):
                    out[j] = "I-TAX"
                    break
    return out


def predict_word_labels_and_conf_chunked(
    example: Dict[str, Any],
    *,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
    word_chunk: int = WORD_CHUNK,
    normalize_words: bool = True,
    min_chunk_words: int = 20,
    apply_postproc: bool = True,
) -> Tuple[List[str], List[float]]:

    image = example["image"].convert("RGB")
    words = example["words"]
    boxes = example["bboxes"]
    n_words = len(words)

    word_currency: List[Optional[str]] = [None] * n_words
    if normalize_words:
        words, word_currency = normalize_words_keep_currency(words)

    if len(words) != len(boxes):
        raise ValueError(
            f"Word/box mismatch after normalize: {len(words)} vs {len(boxes)}"
        )

    example["_word_currency"] = word_currency

    final_labels = ["O"] * n_words
    final_conf = [0.0] * n_words

    start = 0
    while start < n_words:
        end = min(start + word_chunk, n_words)

        while True:
            chunk_words = words[start:end]
            chunk_boxes = boxes[start:end]

            encoding = cord_processor(
                image,
                chunk_words,
                boxes=chunk_boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

            if "pixel_values" in encoding:
                pv = encoding["pixel_values"]
                if pv.ndim == 5 and pv.shape[1] == 1:
                    encoding["pixel_values"] = pv.squeeze(1)

            if hasattr(encoding, "word_ids"):
                word_ids = encoding.word_ids(batch_index=0)
            else:
                word_ids = encoding.encodings[0].word_ids

            last_wi = len(chunk_words) - 1
            covered_ids = {wi for wi in word_ids if wi is not None}
            covered = (last_wi in covered_ids)

            if covered or (end - start) <= min_chunk_words:
                break

            new_size = max(min_chunk_words, int((end - start) * 0.85))
            end = start + new_size

        encoding_t = {k: v.to(DEVICE) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = cord_model(**encoding_t)

        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1).tolist()
        pred_confs = probs.max(dim=-1).values.tolist()

        seen = set()
        for ti, wi in enumerate(word_ids):
            if wi is None or wi in seen:
                continue
            seen.add(wi)
            gi = start + int(wi)
            lab = CORD_ID2LABEL[int(pred_ids[ti])]
            conf = float(pred_confs[ti])
            if gi < n_words and conf >= final_conf[gi]:
                final_labels[gi] = lab
                final_conf[gi] = conf

        if end >= n_words:
            break
        next_start = end - stride
        if next_start <= start:
            next_start = end
        start = next_start

    if apply_postproc:
        raw_labels = final_labels[:]
        raw_conf = final_conf[:]

        final_labels, final_conf = postprocess_predictions(
            words=example["words"],
            labels=final_labels,
            confs=final_conf,
            bboxes=example["bboxes"],
            row_ids=example.get("row_ids"),
            zones=example.get("zones"),
        )

        example["_raw_labels"] = raw_labels
        example["_raw_conf"] = raw_conf

    return final_labels, final_conf


# ============================================================
# Validation helpers
# ============================================================
def extract_subtotal_from_spans(
    spans: List[Dict[str, Any]],
) -> Optional[float]:
    for span in spans:
        if span.get("type") == "SUMMARY_LINE":
            text = span.get("text", "").upper()
            if "SUBTOTAL" in text:
                for token in span.get("tokens", []):
                    try:
                        value = float(token)
                        if 10 < value < 10000:
                            return value
                    except ValueError:
                        continue
    return None

def format_validation_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("VALIDATION REPORT".center(80))
    lines.append("=" * 80)

    subtotal_result = report.get("subtotal_check", {})
    lines.append("\n[1] SUBTOTAL CONSISTENCY")
    lines.append(f"    Status: {subtotal_result.get('status', 'UNKNOWN')}")
    if subtotal_result.get("status") != "SKIPPED":
        lines.append(
            f"    Expected:      ${subtotal_result.get('expected_subtotal', 0):.2f}"
        )
        lines.append(
            f"    Reconstructed: ${subtotal_result.get('reconstructed_total', 0):.2f}"
        )
        lines.append(
            f"    Discrepancy:   "
            f"${subtotal_result.get('discrepancy', 0):+.2f} "
            f"({subtotal_result.get('discrepancy_pct', 0):+.2f}%)"
        )
        for item in subtotal_result.get("problematic_items", [])[:3]:
            lines.append(
                f"      - {item.get('item_name','UNKNOWN'):<40} "
                f"${item.get('price',0):>7.2f} "
                f"(Z-score: {item.get('z_score',0):>5.1f})"
            )

    outliers_result = report.get("price_outliers", {})
    lines.append(f"\n[2] PRICE OUTLIER DETECTION")
    lines.append(f"    Status: {outliers_result.get('status', 'UNKNOWN')}")
    for outlier in outliers_result.get("outliers", [])[:3]:
        lines.append(
            f"      - {outlier.get('item_name','UNKNOWN'):<40} "
            f"${outlier.get('price',0):>7.2f}"
        )

    lines.append(f"\n[3] RECOMMENDATIONS")
    for rec in report.get("recommendations", []):
        lines.append(f"    {rec}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)

def get_words(ex) -> List[str]:
    if hasattr(ex, "words"):
        return ex.words
    if isinstance(ex, dict):
        for key in ("words", "tokens", "ocr_words"):
            if key in ex:
                return ex[key]
    raise KeyError(
        "Cannot find words in example "
        "(expected .words or dict key 'words'/'tokens'/'ocr_words')."
    )


# ============================================================
# Safe formatting helpers
# ============================================================
def safe_format_price(value: Any) -> str:
    if value is None:
        return "Not detected"
    try:
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "Invalid"

def safe_format_conf(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"


# ============================================================
# Define receipt Type
# ============================================================
def classify_receipt_family(example: Dict[str, Any]) -> str:
    words = [str(w).upper() for w in example.get("words", [])]
    text = " ".join(words)
    row_ids = example.get("row_ids", [])
    meta = example.get("meta", {}) or {}
    n_words = len(words)
    n_rows = len(set(row_ids)) if row_ids else 0
    score = 0

    if meta.get("ocr_long_receipt"):
        score += 2
    if n_words > 250:
        score += 1
    if n_rows > 120:
        score += 1
    if any(u in text for u in [" KG", "/KG", " LB", "/LB", " EA "]):
        score += 1
    if any(s in text for s in ["PRODUCE", "GROCERY", "BAKERY", "MEATS", "DELI"]):
        score += 2
    if any(v in text for v in ["SUPERSTORE", "COSTCO", "WALMART", "REAL CANADIAN"]):
        score += 2

    if score >= 4:
        return "long_grocery"
    return "default"


# ============================================================
# Config
# ============================================================
def create_geo_config(
    min_confidence: float = 0.50,
    fuzzy_enabled: bool = True,
    debug: bool = False,
) -> "CordGeoConfig":
    return CordGeoConfig(
        min_label_confidence=min_confidence,
        fuzzy_distance_threshold=2,
        use_fuzzy_fallback=fuzzy_enabled,
        debug_print=debug,
    )


# ============================================================
# Geo extraction wrapper (Short)
# ============================================================
def extract_geo_items(
    example: Dict[str, Any],
    cfg: Optional["CordGeoConfig"] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = create_geo_config(debug=verbose)
    if verbose:
        print("\n[GEO EXTRACTION] Starting cord_plus_geo_extract_v4...")
    try:
        result = cord_plus_geo_extract_v4(example, cfg=cfg)
        if verbose:
            print(f"[GEO EXTRACTION] Complete")
            print(f"  - Subtotal: {safe_format_price(result.get('SUBTOTAL'))}")
            print(f"  - Tax:      {safe_format_price(result.get('TAX'))}")
            print(f"  - Total:    {safe_format_price(result.get('TOTAL'))}")
            print(f"  - Items:    {len(result.get('ITEMS', []))}")
        return result
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "id":                   example.get("id"),
            "image_path":           example.get("image_path"),
            "SUBTOTAL":             None,
            "SUBTOTAL_CONFIDENCE":  0.0,
            "TAX":                  None,
            "TAX_CONFIDENCE":       0.0,
            "TOTAL":                None,
            "TOTAL_CONFIDENCE":     0.0,
            "DISCOUNT":             None,
            "DISCOUNT_CONFIDENCE":  0.0,
            "ITEMS":                [],
            "debug":                {"error": str(e), "error_type": type(e).__name__},
        }


# ============================================================
# Geo extraction wrapper (Long)
# ============================================================
def extract_geo_items_long(
    example: Dict[str, Any],
    cfg: Optional["CordGeoConfig"] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = create_geo_config(debug=verbose)
    if verbose:
        print("\n[GEO EXTRACTION - LONG] Starting cord_plus_geo_extract_long...")
    try:
        result = cord_plus_geo_extract_long(example, cfg=cfg)
        if verbose:
            print(f"[GEO EXTRACTION - LONG] Complete")
            print(f"  - Subtotal: {safe_format_price(result.get('SUBTOTAL'))}")
            print(f"  - Tax:      {safe_format_price(result.get('TAX'))}")
            print(f"  - Total:    {safe_format_price(result.get('TOTAL'))}")
            print(f"  - Items:    {len(result.get('ITEMS', []))}")
        return result
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "id":                   example.get("id"),
            "image_path":           example.get("image_path"),
            "SUBTOTAL":             None,
            "SUBTOTAL_CONFIDENCE":  0.0,
            "TAX":                  None,
            "TAX_CONFIDENCE":       0.0,
            "TOTAL":                None,
            "TOTAL_CONFIDENCE":     0.0,
            "DISCOUNT":             None,
            "DISCOUNT_CONFIDENCE":  0.0,
            "ITEMS":                [],
            "debug": {
                "error": str(e),
                "error_type": type(e).__name__,
                "extractor": "cord_plus_geo_extract_long",
            },
        }


# ============================================================
# Confidence filtering
# ============================================================
def filter_items_by_confidence(
    items: List[Dict[str, Any]],
    min_name_conf: float = 0.70,
    min_price_conf: float = 0.70,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    high_conf, low_conf = [], []
    for item in items:
        nc = item.get("name_confidence",  0.0)
        pc = item.get("price_confidence", 0.0)
        if nc >= min_name_conf and pc >= min_price_conf:
            high_conf.append(item)
        else:
            item["_confidence_flag"]   = "LOW_CONFIDENCE"
            item["_confidence_reason"] = (
                f"Name: {safe_format_conf(nc)}, Price: {safe_format_conf(pc)}"
            )
            low_conf.append(item)
    return high_conf, low_conf


# ============================================================
# Merge geo with validation
# ============================================================
def merge_geo_with_validation(
    geo_result: Dict[str, Any],
    validation_items: List[Dict[str, Any]],
    validation_report: Dict[str, Any],
) -> Dict[str, Any]:
    merged_items = []
    geo_items = geo_result.get("ITEMS", [])

    for i, geo_item in enumerate(geo_items):
        merged_item = {
            "geo_name":          geo_item.get("name"),
            "geo_price":         geo_item.get("price"),
            "geo_discount":      geo_item.get("discount"),
            "geo_name_conf":     geo_item.get("name_confidence",  0.0),
            "geo_price_conf":    geo_item.get("price_confidence", 0.0),
            "geo_discount_conf": geo_item.get("discount_confidence", 0.0),
            "validation_passed": None,
            "validation_flags":  [],
        }
        if i < len(validation_items):
            val_item = validation_items[i]
            merged_item.update({
                "validation_passed": not val_item.get("_flagged", False),
                "validation_flags":  val_item.get("_flags", []),
                "validation_reason": val_item.get("flag_reason"),
            })
        merged_items.append(merged_item)

    return {
        "merged_items":       merged_items,
        "geo_subtotal":       geo_result.get("SUBTOTAL"),
        "geo_tax":            geo_result.get("TAX"),
        "geo_total":          geo_result.get("TOTAL"),
        "geo_subtotal_conf":  geo_result.get("SUBTOTAL_CONFIDENCE", 0.0),
        "geo_tax_conf":       geo_result.get("TAX_CONFIDENCE",      0.0),
        "geo_total_conf":     geo_result.get("TOTAL_CONFIDENCE",    0.0),
        "n_geo_items":        len(geo_items),
        "n_validation_items": len(validation_items),
    }


def build_prediction_log_struct(
    example: Dict[str, Any],
    word_labels: List[str],
    word_conf: List[float],
) -> Dict[str, Any]:
    words = get_words(example)
    raw_labels = example.get("_raw_labels", [])
    n = min(len(words), len(word_labels), len(word_conf))

    rows = []
    for i in range(n):
        raw = raw_labels[i] if i < len(raw_labels) else word_labels[i]
        fin = word_labels[i]
        rows.append({
            "idx":        i,
            "word":       str(words[i]) if words[i] is not None else "",
            "raw_label":  str(raw),
            "label":      str(fin),
            "changed":    raw != fin,
            "confidence": float(word_conf[i]) if i < len(word_conf) else 0.0,
        })

    return {"total_words": len(words), "rows": rows}



def normalize_ocr_to_text(ocr_input) -> str:
    if ocr_input is None:
        return ""
    if isinstance(ocr_input, str):
        return ocr_input
    if isinstance(ocr_input, dict):
        rows = ocr_input.get("rows", [])
        return "\n".join(r.get("text", "") for r in rows if r.get("text"))
    raise TypeError(f"Unsupported OCR type: {type(ocr_input)}")


# ============================================================
# process_single_receipt_with_validation
# ============================================================
def process_single_receipt_with_validation(
    json_path: Path,
    dataset_root: Path,
    cfg: "CordGeoConfig" = None,
    verbose: bool = True,
    use_geo_extraction: bool = True,
    use_sorie_extraction: bool = True,
    min_confidence_filter: float = 0.70,
) -> Dict[str, Any]:

    if cfg is None:
        cfg = create_geo_config(
            min_confidence=0.50, fuzzy_enabled=True, debug=verbose
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {json_path.name}")
        print(f"{'='*70}")

    try:
        # ── STEP 1: Load ─────────────────────────────────────────────────────
        if verbose:
            print("\n[STEP 1] Loading receipt data...")

        # FIX 2: pass dataset_root as-is; resolve_image_path now handles the
        # per-image subdir layout (dataset_root/<stem>/best_images/) internally.
        example = load_cordlike_example(json_path, dataset_root=dataset_root)
        receipt_family = classify_receipt_family(example)
        n_words = len(example.get("words", []))

        if verbose:
            print(f"  ✓ Loaded {n_words} words")
            print(f"  ✓ row_ids: {len(example.get('row_ids', []))} entries")
            print(f"  ✓ zones:   {'from JSON' if example.get('zones') else 'will derive from bbox'}")
            print(f"  ✓ Receipt family: {receipt_family}")

        # ── STEP 2: Inference + Post-processing ──────────────────────────────
        if verbose:
            print("\n[STEP 2] Predicting word labels (inference + post-processing)...")

        word_labels, word_conf = predict_word_labels_and_conf_chunked(
            example,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            word_chunk=WORD_CHUNK,
            normalize_words=True,
            apply_postproc=True,
        )
        if verbose:
            print(f"  ✓ Predicted {len(word_labels)} labels")

        if verbose:
            words = get_words(example)
            raw_labels = example.get("_raw_labels", [])
            print(f"\n{'Index':>5}  {'Token':<25} {'Raw':>16} {'→ Final':>16}  {'Conf':>6}  {'Changed'}")
            print("─" * 82)
            for i in range(min(len(words), len(word_labels), len(word_conf))):
                w = str(words[i])[:24]
                raw = raw_labels[i] if i < len(raw_labels) else word_labels[i]
                fin = word_labels[i]
                c = float(word_conf[i]) if i < len(word_conf) else 0.0
                chg = "✓ FIXED" if fin != raw else ""
                print(f"{i:>05d}  {w:<25} {raw:>16} → {fin:<16}  {c:>6.3f}  {chg}")

        prediction_log = build_prediction_log_struct(example, word_labels, word_conf)

        # ── STEP 3: Reconstruct items ─────────────────────────────────────────
        if verbose:
            print("\n[STEP 3] Reconstructing items from sequence...")

        ex2 = dict(example)
        ex2["labels"] = word_labels
        ex2["confs"] = word_conf

        item_line_spans = reconstruct_items_from_sequence(
            ex2["words"],
            ex2["bboxes"],
            word_labels,
            word_conf,
        )
        item_lines = [s for s in item_line_spans if s.get("type") == "ITEM_LINE"]
        summary_lines = [s for s in item_line_spans if s.get("type") == "SUMMARY_LINE"]

        if verbose:
            print(f"  ✓ Reconstructed {len(item_lines)} item lines")
            print(f"  ✓ Found {len(summary_lines)} summary lines")

        # ── STEP 4: Validation ────────────────────────────────────────────────
        if verbose:
            print("\n[STEP 4] Running validation checks...")
        extracted_subtotal = extract_subtotal_from_spans(summary_lines)
        validator = ItemValidationEngine(subtotal=extracted_subtotal, tax=None)
        validation_report = validator.validate_all(item_lines)
        if verbose:
            print("  ✓ Validation complete")

        # ── STEP 5: Flags ─────────────────────────────────────────────────────
        if verbose:
            print("\n[STEP 5] Adding validation flags...")
        enricher = ItemFlagEnricher()
        item_lines_with_flags = enricher.add_flags_from_validation(
            item_lines, validation_report
        )
        clean_items = enricher.get_clean(item_lines_with_flags)
        flagged_items = enricher.get_flagged(item_lines_with_flags)
        flag_summary = enricher.summary(item_lines_with_flags)

        if verbose:
            print(f"  ✓ Clean items:   {flag_summary['clean']}")
            print(f"  ✓ Flagged items: {flag_summary['flagged']}")

        # ── STEP 6: Geo extraction ────────────────────────────────────────────
        geo_result = None
        geo_merged = None
        high_conf_geo_items = []
        low_conf_geo_items = []

        if use_geo_extraction:
            if verbose:
                print("\n[STEP 6] Running geo extraction...")
            if receipt_family == "long_grocery":
                geo_result = extract_geo_items_long(ex2, cfg=cfg, verbose=verbose)
            else:
                geo_result = extract_geo_items(ex2, cfg=cfg, verbose=verbose)

            if geo_result and geo_result.get("ITEMS"):
                try:
                    high_conf_geo_items, low_conf_geo_items = filter_items_by_confidence(
                        geo_result.get("ITEMS", []),
                        min_name_conf=min_confidence_filter,
                        min_price_conf=min_confidence_filter,
                    )
                    if verbose:
                        print(f"    - High confidence: {len(high_conf_geo_items)}")
                        print(f"    - Low confidence:  {len(low_conf_geo_items)}")

                    if verbose:
                        print("\n[STEP 7] Merging geo with validation...")
                    geo_merged = merge_geo_with_validation(
                        geo_result, item_lines_with_flags, validation_report
                    )
                    if verbose:
                        print("  ✓ Merge complete")

                except Exception as e:
                    if verbose:
                        print(f"  ✗ Filtering/merging failed: {e}")

        # ── STEP 8: SROIE extraction ──────────────────────────────────────────
        if use_sorie_extraction:
            if verbose:
                print("\n[STEP 8] Running SROIE field extraction...")

            sroie_out = extract_sroie_fields_from_ocr_json(
                json_path,
                dataset_root=dataset_root,
                processor=sroie_processor,
                model=sroie_model,
                device=DEVICE,
                max_length=MAX_LENGTH,
                iou_threshold=0.90,
            )

            if geo_result is None:
                geo_result = {}

            geo_result["sroie_fields"] = {
                "vendor":   sroie_out.get("vendor",   ""),
                "address":  sroie_out.get("address",  ""),
                "phone":    sroie_out.get("phone",    ""),
                "date":     sroie_out.get("date",     ""),
                "subtotal": sroie_out.get("subtotal", ""),
                "tax":      sroie_out.get("tax",      ""),
                "total":    sroie_out.get("total",    ""),
            }
            geo_result["sroie_entities"] = (
                sroie_out.get("debug", {}) or {}
            ).get("entities", {})

            if verbose:
                print("  ✓ SROIE Fields:", geo_result["sroie_fields"])

        # ── Attach prediction log and OCR text to geo_result ─────────────────
        if geo_result is not None:
            geo_result["prediction_log"] = prediction_log

            raw_ocr = load_ocr_text(example.get("id"))
            ocr_text = normalize_ocr_to_text(raw_ocr)
            geo_result["ocr_text"] = ocr_text
            geo_result["ocr_struct"] = {
                "rows": [
                    {"text": line.strip()}
                    for line in ocr_text.splitlines()
                    if line.strip()
                ]
            }

        # ── Summary print ─────────────────────────────────────────────────────
        if verbose:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"Total items:   {len(item_lines_with_flags)}")
            print(f"Clean items:   {len(clean_items)}")
            print(f"Flagged items: {len(flagged_items)}")

            if geo_result:
                print("\nGeo Extraction Results:")
                print(f"  Subtotal: {safe_format_price(geo_result.get('SUBTOTAL'))}")
                print(f"  Tax:      {safe_format_price(geo_result.get('TAX'))}")
                print(f"  Total:    {safe_format_price(geo_result.get('TOTAL'))}")

                s = geo_result.get("sroie_fields", {})
                print("\nSROIE Extraction Results:")
                for field in ("subtotal", "tax", "total", "vendor", "phone", "address", "date"):
                    print(f"  {field.capitalize():<10}: {s.get(field, '')}")

                if geo_merged:
                    items = geo_merged.get("merged_items", [])
                    print(f"\n  Top items: {len(items)}")
                    for item in items:
                        print(
                            f"    - {item.get('geo_name','Unknown')}: "
                            f"{safe_format_price(item.get('geo_price'))} "
                            f"(Conf: {safe_format_conf(item.get('geo_name_conf'))}/"
                            f"{safe_format_conf(item.get('geo_price_conf'))})"
                        )

        return {
            "json_path":              str(json_path),
            "example":                ex2,
            "word_labels":            word_labels,
            "word_conf":              word_conf,
            "item_line_spans":        item_line_spans,
            "item_lines":             item_lines_with_flags,
            "item_lines_clean":       clean_items,
            "item_lines_flagged":     flagged_items,
            "summary_lines":          summary_lines,
            "extracted_subtotal":     extracted_subtotal,
            "validation_report":      validation_report,
            "flag_summary":           flag_summary,
            "geo_extraction_enabled": use_geo_extraction,
            "geo_result":             geo_result,
            "geo_merged":             geo_merged,
            "geo_items_high_conf":    high_conf_geo_items,
            "geo_items_low_conf":     low_conf_geo_items,
            "receipt_id":             json_path.stem,
            "status":                 "complete",
        }

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "json_path":          str(json_path),
            "receipt_id":         json_path.stem,
            "status":             "failed",
            "error":              str(e),
            "error_type":         type(e).__name__,
            "example":            None,
            "word_labels":        [],
            "word_conf":          [],
            "item_lines":         [],
            "item_lines_clean":   [],
            "item_lines_flagged": [],
            "geo_result":         None,
            "geo_merged":         None,
        }


# ============================================================
# BATCH RUNNER: to SQLite WITH VALIDATION
# ============================================================
def run_batch_to_sqlite(
    json_dir: Path,
    out_dir: Path,
    db_path: Path,
    *,
    dataset_root: Path,
    pattern: str = "*.json",
    max_files: Optional[int] = None,
    cfg: Optional["CordGeoConfig"] = None,
    write_out_json: bool = False,
    summary_report: bool = True,
    verbose: bool = True,
    min_confidence_filter: float = 0.70,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    init_sqlite_schema(db_path)

    json_paths = sorted(json_dir.glob(pattern))
    if max_files is not None:
        json_paths = json_paths[:max_files]

    results: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        for jp in json_paths:
            try:
                proc_result = process_single_receipt_with_validation(
                    jp,
                    dataset_root=dataset_root,
                    cfg=cfg,
                    verbose=verbose,
                    use_geo_extraction=True,
                    min_confidence_filter=min_confidence_filter,
                )

                if proc_result.get("status") != "complete":
                    raise RuntimeError(proc_result.get("error", "unknown error"))

                geo_out = proc_result.get("geo_result")
                validation_report = proc_result.get("validation_report") or {}

                if not geo_out:
                    raise RuntimeError("geo_result is None (geo extraction failed)")

                out_path = out_dir / f"{jp.stem}_geo_out.json"
                if write_out_json:
                    out_path.write_text(
                        json.dumps(geo_out, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                print("debug predictionlog_json rows:", len((geo_out.get("prediction_log") or {}).get("rows", [])))

                if not isinstance(geo_out.get("ocr_text"), str):
                    geo_out["ocr_text"] = normalize_ocr_to_text(geo_out.get("ocr_text"))

                if not isinstance(geo_out.get("ocr_struct"), dict):
                    ocr_text = geo_out.get("ocr_text", "")
                    geo_out["ocr_struct"] = {
                        "rows": [{"text": ln.strip()} for ln in ocr_text.splitlines() if ln.strip()]
                    }

                print("debug ocr rows:", len((geo_out.get("ocr_struct") or {}).get("rows", [])))

                # FIX 3: summary_report must be a dict (or None) — not a bool.
                # Build the summary dict from sroie_fields so it is actually saved.
                summary_report_dict: Optional[Dict[str, Any]] = None
                if summary_report:
                    sroie = geo_out.get("sroie_fields") or {}
                    summary_report_dict = {
                        "fields": {
                            "vendor":   sroie.get("vendor",   ""),
                            "phone":    sroie.get("phone",    ""),
                            "address":  sroie.get("address",  ""),
                            "date":     sroie.get("date",     ""),
                            "subtotal": sroie.get("subtotal", ""),
                            "tax":      sroie.get("tax",      ""),
                            "total":    sroie.get("total",    ""),
                        }
                    }

                with conn:
                    receipt_id = save_geo_fusion_to_sqlite(
                        conn=conn,
                        out=geo_out,
                        validation_report=validation_report,
                        summary_report=summary_report_dict,  # FIX 3: proper dict or None
                        receipt_name_fallback=jp.stem,
                        default_currency="CAD",
                        min_confidence_filter=min_confidence_filter,
                    )

                subtotal_check = validation_report.get("subtotal_check", {})
                results.append(
                    {
                        "json": jp.name,
                        "receipt_id": receipt_id,
                        "source_id": geo_out.get("id"),
                        "SUBTOTAL": geo_out.get("SUBTOTAL"),
                        "SUBTOTAL_STATUS": subtotal_check.get("status"),
                        "OUTLIERS_COUNT": len(
                            validation_report.get("price_outliers", {}).get("outliers", []) or []
                        ),
                        "ITEMS": len(geo_out.get("ITEMS") or geo_out.get("ITEMS_GEO") or []),
                        "out_file": str(out_path) if write_out_json else None,
                    }
                )

                n_ok += 1
                if verbose:
                    print(
                        f"✓ receipt_id={receipt_id} | json={jp.name} "
                        f"| SUBTOTAL_STATUS={subtotal_check.get('status')} "
                        f"| ITEMS={len(geo_out.get('ITEMS') or geo_out.get('ITEMS_GEO') or [])}"
                    )

            except Exception as e:
                n_fail += 1
                if verbose:
                    print(f"✗ {jp.name}: {type(e).__name__}: {e}")

    finally:
        conn.close()

    summary_path = out_dir / "summary_sqlite.json"
    summary = {
        "json_dir":  str(json_dir),
        "pattern":   pattern,
        "max_files": max_files,
        "db_path":   str(db_path),
        "n_total":   len(json_paths),
        "n_ok":      n_ok,
        "n_fail":    n_fail,
        "results":   results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if verbose:
        print("\n" + "=" * 80)
        print("DONE")
        print("Summary:", summary_path)
        print("DB:", db_path)
        print(f"OK: {n_ok} | FAIL: {n_fail}")
        print("=" * 80)

    return summary_path