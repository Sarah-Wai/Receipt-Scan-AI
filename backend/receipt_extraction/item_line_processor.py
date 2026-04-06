from typing import List, Dict, Any, Optional, Tuple
import re


def bbox_union(boxes: List[List[int]]) -> List[int]:
    """Union of bounding boxes to create encompassing rectangle."""
    return [
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    ]


def reconstruct_items_from_sequence(
    words: List[str],
    bboxes: List[List[int]],
    labels: List[str],
    confs: List[float],
    word_currency: Optional[List[Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Advanced reconstruction using NER-style state machine.
    
    Key insight: B-MENU.NM / I-MENU.NM define item boundaries.
    Between B-MENU.NM tokens, collect all associated price/weight/SKU info.
    
    This handles:
    - Items split across multiple lines
    - Scattered B-MENU.PRICE and I-MENU.PRICE tokens
    - Voided items (filter out)
    - Weight/quantity tokens (1.44KG)
    - Coupon references (TPD/...)
    """
    if word_currency is None:
        word_currency = [None] * len(words)

    # ----------------------------
    # Regex Patterns
    # ----------------------------
    TPD_RE = re.compile(r"\bTPD\s*/\s*\d+\b|\bTPD/\d+\b", re.IGNORECASE)
    VOID_RE = re.compile(r"\bVOID\b", re.IGNORECASE)
    SKU_RE = re.compile(r"^\d{6,10}$")
    PRICE_RE = re.compile(r"^\d{1,6}([.:]\d{2})([A-Z]{1,3})?$", re.IGNORECASE)
    WEIGHT_RE = re.compile(r"(\d+\.?\d*)\s*(KG|LB|OZ|G|ML|L)", re.IGNORECASE)

    # ----------------------------
    # Helpers
    # ----------------------------
    def is_discount_token(tok: str) -> bool:
        """Detect discount indicators: 4.CO-, 3.CO-GP, 23.99-GP, etc."""
        t = (tok or "").strip().upper()
        return (
            t.endswith("-") or "CO-" in t or "CO-GP" in t or "-GP" in t
        ) and re.search(r"\d", t)

    def is_coupon_token(tok: str) -> bool:
        """Detect TPD coupon references"""
        return "TPD" in (tok or "").upper()

    def extract_price_value(tok: str) -> Optional[float]:
        """Extract numeric price, ignoring suffixes"""
        t = (tok or "").strip().upper()
        if not t:
            return None
        
        # Remove discount indicators
        t = t.rstrip("-")
        
        # Normalize OCR artifacts
        t = t.replace(":", ".")
        
        # Extract numeric prefix
        match = re.match(r"^(\d{1,6}(?:\.\d{1,2})?)", t)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return None

    # ----------------------------
    # 1) Filter header/footer and summary lines
    # ----------------------------
    filtered_items = []
    for w, bb, lab, cf, cur in zip(words, bboxes, labels, confs, word_currency):
        w = (str(w) if w is not None else "").strip()
        if not w:
            continue

        # Skip header/footer
        header_keywords = ["costco", "wholesale", "member", "basket", "date", "time"]
        if w.lower() in header_keywords:
            continue

        # Skip summary section (marked by B-SUM.*, B-TAX, B-SUM.TOTAL)
        if lab.startswith("B-SUM.") or lab == "B-TAX":
            continue

        x1, y1, x2, y2 = [int(v) for v in bb]
        filtered_items.append({
            "word": w,
            "bbox": [x1, y1, x2, y2],
            "label": lab,
            "conf": float(cf),
            "currency": cur,
            "idx": len(filtered_items),
        })

    # ----------------------------
    # 2) State machine: reconstruct items
    # ----------------------------
    items = []
    current_item = None

    for i, item in enumerate(filtered_items):
        word = item["word"]
        label = item["label"]
        conf = item["conf"]
        bbox = item["bbox"]

        # B-MENU.NM: START a new item
        if label.startswith("B-MENU.NM"):
            # Save previous item if exists
            if current_item is not None and current_item["names"]:
                items.append(current_item)

            # Start new item
            current_item = {
                "names": [word],
                "name_bboxes": [bbox],
                "name_confs": [conf],
                "prices": [],
                "price_bboxes": [],
                "price_confs": [],
                "skus": [],
                "weights": [],
                "discounts": [],
                "coupon_refs": [],
                "void": False,
            }

        # I-MENU.NM: continue current item name
        elif label.startswith("I-MENU.NM"):
            if current_item is not None:
                current_item["names"].append(word)
                current_item["name_bboxes"].append(bbox)
                current_item["name_confs"].append(conf)

        # B-MENU.PRICE or I-MENU.PRICE: add price
        elif label.startswith("B-MENU.PRICE") or label.startswith("I-MENU.PRICE"):
            if current_item is not None:
                if is_discount_token(word):
                    current_item["discounts"].append(word)
                else:
                    current_item["prices"].append(word)
                    current_item["price_bboxes"].append(bbox)
                    current_item["price_confs"].append(conf)

        # Check for SKU (pure digits, 6-10 chars, before names)
        elif SKU_RE.match(word):
            if current_item is not None:
                current_item["skus"].append(word)
            else:
                # SKU before any B-MENU.NM? Create pending item
                current_item = {
                    "names": [],
                    "name_bboxes": [],
                    "name_confs": [],
                    "prices": [],
                    "price_bboxes": [],
                    "price_confs": [],
                    "skus": [word],
                    "weights": [],
                    "discounts": [],
                    "coupon_refs": [],
                    "void": False,
                }

        # Check for weight/quantity
        elif WEIGHT_RE.search(word):
            if current_item is not None:
                current_item["weights"].append(word)

        # Check for coupon reference
        elif is_coupon_token(word):
            if current_item is not None:
                current_item["coupon_refs"].append(word)

        # Check for VOID marker
        if VOID_RE.search(word):
            if current_item is not None:
                current_item["void"] = True

    # Save last item
    if current_item is not None and current_item["names"]:
        items.append(current_item)

    # ----------------------------
    # 3) Convert to output spans
    # ----------------------------
    spans = []

    for item in items:
        # Skip voided items
        if item["void"]:
            continue

        # Skip items without name OR price
        if not item["names"] or not item["prices"]:
            continue

        # Skip coupon reference lines (TPD/...)
        if item["coupon_refs"] and not item["prices"]:
            continue

        # Build item name
        item_name = " ".join(item["names"])

        # Get primary price (first valid one)
        primary_price = None
        primary_price_str = None
        if item["prices"]:
            primary_price_str = item["prices"][0]
            primary_price = extract_price_value(primary_price_str)

        # Collect all relevant bboxes
        all_bboxes = (
            item["name_bboxes"] +
            item["price_bboxes"]
        )

        if not all_bboxes:
            continue

        item_bbox = bbox_union(all_bboxes)
        all_confs = item["name_confs"] + item["price_confs"]
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

        # Build tokens list (for reference)
        all_tokens = (
            item["skus"] +
            item["names"] +
            item["prices"] +
            item["weights"]
        )

        spans.append({
            "type": "ITEM_LINE",
            "item_name": item_name,
            "price": primary_price,
            "price_str": primary_price_str,
            "skus": item["skus"],
            "weight": item["weights"][0] if item["weights"] else None,
            "discounts": item["discounts"],
            "tokens": all_tokens,
            "bbox": item_bbox,
            "y1": int(item_bbox[1]),
            "y2": int(item_bbox[3]),
            "conf_mean": float(avg_conf),
        })

    return spans


def validate_items(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate reconstructed items against expected structure.
    
    Returns metrics and potential issues.
    """
    valid_items = []
    invalid_items = []
    
    for span in spans:
        issues = []
        
        # Check name
        if not span.get("item_name") or not span["item_name"].strip():
            issues.append("Missing item name")
        
        # Check price
        if span.get("price") is None:
            issues.append("Missing price")
        
        # Check confidence
        if span.get("conf_mean", 0) < 0.60:
            issues.append(f"Low confidence: {span.get('conf_mean'):.2f}")
        
        if issues:
            invalid_items.append({"span": span, "issues": issues})
        else:
            valid_items.append(span)
    
    return {
        "total": len(spans),
        "valid": len(valid_items),
        "invalid": len(invalid_items),
        "valid_items": valid_items,
        "invalid_items": invalid_items,
    }


def format_items_for_display(spans: List[Dict[str, Any]]) -> str:
    """Format reconstructed items for human review."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("RECONSTRUCTED ITEMS".center(80))
    output.append("=" * 80)
    
    for i, span in enumerate(spans, 1):
        item_name = span.get("item_name", "UNKNOWN")
        price = span.get("price")
        weight = span.get("weight")
        price_str = span.get("price_str", "")
        conf = span.get("conf_mean", 0)
        
        line = f"{i:2d}. {item_name:<40s} ${price:>7.2f}" if price else f"{i:2d}. {item_name}"
        
        if weight:
            line += f" ({weight})"
        
        line += f" [conf:{conf:.3f}]"
        
        output.append(line)
    
    output.append("=" * 80)
    total_price = sum(s["price"] for s in spans if s.get("price"))
    output.append(f"TOTAL: ${total_price:.2f}".rjust(80))
    output.append("=" * 80)
    
    return "\n".join(output)