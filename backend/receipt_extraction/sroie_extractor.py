from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import json
import re

import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


# ----------------------------
# Regex & Post-processing
# ----------------------------
DATE_RE = re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{2,4}\b")

PHONE_FULL_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{3}\)?[-\s]?)\d{3}[-\s]?\d{4}\b"
)

AMOUNT_LOOSE_RE = re.compile(r"(\d{1,4})[:\.](\d{2})")
AMOUNT_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2}")

ZIP_RE = re.compile(r"\b\d{5}\b")
STREET_NO_RE = re.compile(r"^\d{3,6}$")


# ----------------------------
# Core helpers
# ----------------------------
def resolve_image_path(
    img_path: str,
    *,
    dataset_root: Path,
    json_path: Path,
) -> Path:
    if not isinstance(img_path, str) or not img_path.strip():
        raise FileNotFoundError(f"Missing image_path for {json_path.name}")

    img_name = Path(img_path.strip()).name
    stem = json_path.stem   # e.g. "01_59530236c3d14c6a8fea0b21fec11e2a"

    # 1. Flat: dataset_root / img_path  (original behaviour, works for old layout)
    p = (dataset_root / img_path.strip()).resolve()
    if p.is_file():
        return p

    # 2. Per-image subdir produced by run_ocr_batch:
    #    dataset_root/<stem>/best_images/<filename>
    p = (dataset_root / stem / "best_images" / img_name).resolve()
    if p.is_file():
        return p

    # 3. Any best_images/ folder under dataset_root (handles nesting variations)
    for candidate in dataset_root.rglob(f"best_images/{img_name}"):
        if candidate.is_file():
            return candidate.resolve()

    # 4. Full recursive fallback — find the filename anywhere under dataset_root
    matches = list(dataset_root.rglob(img_name))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Cannot resolve image_path='{img_path}'\n"
        f"  Tried flat:       {(dataset_root / img_path.strip()).resolve()}\n"
        f"  Tried subdir:     {(dataset_root / stem / 'best_images' / img_name).resolve()}\n"
        f"  Tried rglob:      no match for '{img_name}' under {dataset_root}\n"
        f"  Dataset root:     {dataset_root}\n"
        f"  JSON file:        {json_path}"
    )



def clamp_0_1000(v: float) -> int:
    return int(max(0, min(1000, round(v))))


def norm_xyxy_to_1000(box: List[float], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [
        clamp_0_1000(1000 * x1 / w),
        clamp_0_1000(1000 * y1 / h),
        clamp_0_1000(1000 * x2 / w),
        clamp_0_1000(1000 * y2 / h),
    ]


def is_already_norm(boxes: List[List[float]]) -> str:
    mx = max(max(b) for b in boxes[: min(50, len(boxes))])
    if mx <= 1.5:
        return "float01"
    if mx <= 1100:
        return "int1000"
    return "xyxy"


def unique_preserve_order(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pick_first_or_join(lst):
    return " ".join(lst) if lst else ""


def merge_same_box_concat(
    words: List[str], boxes: List[List[int]]
) -> Tuple[List[str], List[List[int]]]:
    groups: Dict[Tuple[int, int, int, int], List[str]] = {}
    order: List[Tuple[int, int, int, int]] = []

    for w, b in zip(words, boxes):
        key = tuple(int(x) for x in b)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(w)

    out_words, out_boxes = [], []
    for key in order:
        toks = groups[key]
        toks = [t for t in toks if t not in {"1"}]
        text = " ".join(toks)
        text = re.sub(r"\bB\s*vd\b", "BLVD", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out_words.append(text)
            out_boxes.append(list(key))

    return out_words, out_boxes


def merge_entities(words: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """
    Robust BIO merge:
    - If an entity begins with I- without a preceding B-, treat it as B-.
    - Allows single-token entities.
    """
    entities = {
        "HEAD.VENDOR":   [],
        "HEAD.ADDRESS":  [],
        "HEAD.PHONE":    [],
        "HEAD.DATE":     [],
        "SUM.SUBTOTAL":  [],
        "SUM.TAX":       [],
        "SUM.TOTAL":     [],
    }

    current_type = None
    buffer: List[str] = []

    def flush():
        nonlocal current_type, buffer
        if current_type and buffer:
            entities[current_type].append(" ".join(buffer))
        current_type, buffer = None, []

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            flush()
            current_type = label[2:]
            buffer = [word]
        elif label.startswith("I-"):
            typ = label[2:]
            if current_type != typ or not buffer:
                flush()
                current_type = typ
                buffer = [word]
            else:
                buffer.append(word)
        else:
            flush()

    flush()
    return entities


# ----------------------------
# Post-processing cleaners
# ----------------------------
def normalize_ocr_text(s: str) -> str:
    if not s:
        return ""
    u = s.upper()
    u = u.replace("COTAL", "TOTAL")
    u = u.replace("SUBTOTA", "SUBTOTAL")
    return u


def extract_amount(text: str) -> str:
    if not text:
        return ""
    matches = AMOUNT_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else ""


def extract_amount_loose(text: str) -> str:
    if not text:
        return ""
    text = normalize_ocr_text(text)
    strict = extract_amount(text)
    if strict:
        return strict
    m = AMOUNT_LOOSE_RE.search(text)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return ""


def phone_near_keyword(words: List[str], keyword: str = "phone", window: int = 18) -> str:
    for i, w in enumerate(words):
        if keyword in w.lower():
            chunk = " ".join(words[i : i + window])
            m = PHONE_FULL_RE.search(chunk)
            if m:
                return m.group(0).strip()
    m = PHONE_FULL_RE.search(" ".join(words))
    return m.group(0).strip() if m else ""


def extract_address_fallback(words: List[str]) -> str:
    zip_idx = None
    for i, w in enumerate(words):
        if ZIP_RE.search(w):
            zip_idx = i
            break
    if zip_idx is None:
        return ""

    start_idx = None
    for i in range(zip_idx - 1, max(-1, zip_idx - 30), -1):
        if STREET_NO_RE.match(words[i]):
            start_idx = i
            break
    if start_idx is None:
        return ""

    chunk = words[start_idx : zip_idx + 1]
    junk = {"1", "d", "G", "M", "sc"}
    chunk = [t for t in chunk if t not in junk]

    text = " ".join(chunk)
    text = re.sub(r"\bB\s*vd\b", "BLVD", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def total_fallback(words_clean: List[str]) -> str:
    text = " ".join(words_clean).upper()
    m = re.search(r"(TOTAL\s+DUE|TOTAL)\s+([0-9]+(?:\.[0-9]{2}))", text)
    return m.group(2) if m else ""


# ----------------------------
# Main callable function
# ----------------------------
def extract_sroie_fields_from_ocr_json(
    json_path: Path,
    *,
    dataset_root: Path,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForTokenClassification,
    device: Union[str, torch.device] = "cuda",
    max_length: int = 512,
    iou_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Run LayoutLMv3 token-classification on OCR-json (words + bboxes + image_path)
    and return cleaned fields: vendor, address, phone, date, subtotal, tax, total.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words: List[str] = data["words"]
    boxes = data["bboxes"]

    # FIX: now uses the updated resolve_image_path above which handles
    # the per-image subdir layout from run_ocr_batch:
    #   dataset_root/<stem>/best_images/<filename>
    img_file = resolve_image_path(
        str(data["image_path"]),
        dataset_root=dataset_root,
        json_path=json_path,
    )

    image = Image.open(img_file).convert("RGB")
    w_img, h_img = image.size

    mode = is_already_norm(boxes)
    if mode == "float01":
        boxes_1000 = [[clamp_0_1000(v * 1000) for v in b] for b in boxes]
    elif mode == "int1000":
        boxes_1000 = [[clamp_0_1000(v) for v in b] for b in boxes]
    else:
        boxes_1000 = [norm_xyxy_to_1000(b, w_img, h_img) for b in boxes]

    words, boxes_1000 = merge_same_box_concat(words, boxes_1000)

    encoding = processor(
        image,
        words,
        boxes=boxes_1000,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)
    encoding = {k: v.to(dev) for k, v in encoding.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze(0).tolist()

    word_ids = processor(
        image,
        words,
        boxes=boxes_1000,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).word_ids()

    words_clean: List[str] = []
    labels_clean: List[str] = []
    seen_word_ids = set()

    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid in seen_word_ids:
            continue
        seen_word_ids.add(wid)
        words_clean.append(words[wid])
        labels_clean.append(model.config.id2label[predictions[tok_idx]])

    entities = merge_entities(words_clean, labels_clean)

    date_text = pick_first_or_join(entities["HEAD.DATE"]).strip()
    phone_text = pick_first_or_join(entities["HEAD.PHONE"])
    m = PHONE_FULL_RE.search(phone_text)
    phone = m.group(0) if m else ""

    subtotal = extract_amount_loose(" ".join(entities["SUM.SUBTOTAL"]))
    tax      = extract_amount_loose(" ".join(entities["SUM.TAX"]))
    total    = extract_amount_loose(" ".join(entities["SUM.TOTAL"]))

    final: Dict[str, Any] = {
        "vendor":   pick_first_or_join(entities["HEAD.VENDOR"]),
        "address":  pick_first_or_join(entities["HEAD.ADDRESS"]),
        "phone":    phone,
        "date":     date_text,
        "subtotal": subtotal,
        "tax":      tax,
        "total":    total,
        "debug": {
            "entities":      entities,
            "aligned_words": len(words_clean),
            "input_words":   len(words),
        },
    }

    for k in ["vendor", "address", "phone", "date", "subtotal", "tax", "total"]:
        final[k] = " ".join(unique_preserve_order(str(final[k]).split())).strip()

    if not final["total"]:
        final["total"] = total_fallback(words_clean)

    if not final["phone"]:
        final["phone"] = phone_near_keyword(words)

    if not final["address"]:
        final["address"] = extract_address_fallback(words)

    final["subtotal"] = extract_amount(final["subtotal"])
    final["tax"]      = extract_amount(final["tax"])
    final["total"]    = extract_amount(final["total"])

    return final


# ----------------------------
# Optional CLI usage:
# python sroie_extractor.py <model_dir> <json_path> <dataset_root>
# ----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python sroie_extractor.py <model_dir> <json_path> <dataset_root>")
        raise SystemExit(2)

    model_dir    = sys.argv[1]
    json_path    = Path(sys.argv[2])
    dataset_root = Path(sys.argv[3])

    proc = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
    mdl  = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)

    out = extract_sroie_fields_from_ocr_json(
        json_path,
        dataset_root=dataset_root,
        processor=proc,
        model=mdl,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))