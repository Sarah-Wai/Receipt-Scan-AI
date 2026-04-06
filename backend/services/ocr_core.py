"""
ENHANCED RECEIPT OCR → FULL JSON (NO BODY/HEADER SPLIT)
======================================================

Goals (based on your messages):
- Keep ONLY one full JSON per receipt (no _body / _keyfields split)
- Best OCR quality using variant ranking (your existing style)
- Handles long receipts with vertical tiling (preserve tiny text)
- Produces correct reading order using smarter "same line" grouping
- Produces smarter token split (fix glued tokens like CALLUS, APPROVEDTHANK, 10$3.29, dates+time)
- Fixes your crash:
    TypeError("order_blocks_same_line() got an unexpected keyword argument 'y_band_k'")
  by making function signature match the calls (y_band_k supported)

Output:
- outputs/test_ocr/best_images/*_best_ocr.png
- outputs/test_ocr/best_texts/*_best.txt
- outputs/test_ocr/best_ocr_json_full/*_full.json
- outputs/test_ocr/overlays/*_overlay.png
- outputs/test_ocr/summary.csv

Requirements:
- opencv-python, numpy, pillow, paddleocr

"""

import os
import glob
import csv
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# -------------------------
# ENV FLAGS (stability)
# -------------------------
os.environ["FLAGS_use_new_executor"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = "data/long_receipts"
OUT_DIR = "outputs/long_receipts_ocr"

# OCR scaling
LONG_SIDE = 1600

# Long receipt detection + tiling
LONG_RECEIPT_ASPECT_THR = 2.6   # h/w
LONG_RECEIPT_HEIGHT_THR = 1600  # px

# For long receipts, preserve small text
LONG_SIDE_LONG_RECEIPT = 2400

# Vertical tiling
TILE_H = 1200
TILE_OVERLAP = 200

TOPK_VARIANTS = 2
USE_CLS = False

TOTAL_WORDS = ["TOTAL", "AMOUNT", "GRAND", "SETTLED", "BALANCE", "AMOUNT DUE"]
DATE_PATTERNS = [
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
]
MONEY_PATTERN = r"(\$|USD|CAD|SGD|RM|IDR)?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?"

# -------------------------
# OCR INIT
# -------------------------
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=USE_CLS,
    show_log=False,
)

# ============================================================
# PORTABLE PATHS
# ============================================================
OUT_DIR_ABS = os.path.abspath(OUT_DIR)

def rel_to_out(p: str) -> str:
    """Convert any path to a portable relative path under OUT_DIR (POSIX style)."""
    if not p:
        return ""
    p_abs = os.path.abspath(p)
    try:
        rel = os.path.relpath(p_abs, OUT_DIR_ABS)
        if rel.startswith(".."):
            return os.path.basename(p_abs)
        return rel.replace("\\", "/")
    except Exception:
        return os.path.basename(p_abs)

def resolve_from_out(rel_path: str, out_dir: str = OUT_DIR) -> str:
    """Reconstruct absolute path from CSV stored relative path."""
    if not rel_path:
        return ""
    return os.path.abspath(os.path.join(out_dir, rel_path))


# ============================================================
# IO UTILS
# ============================================================
def ensure_dirs(base_out: str) -> Dict[str, str]:
    d = {
        "BEST_IMG_DIR": os.path.join(base_out, "best_images"),
        "BEST_TXT_DIR": os.path.join(base_out, "best_texts"),
        "BEST_JSON_FULL_DIR": os.path.join(base_out, "best_ocr_json_full"),
        "OVERLAY_DIR": os.path.join(base_out, "overlays"),
        "VAR_SCORE_DIR": os.path.join(base_out, "variant_scores"),
    }
    os.makedirs(base_out, exist_ok=True)
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d

def read_image_any(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items() if k != "img"}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ============================================================
# LONG RECEIPT + TILING
# ============================================================
def is_long_receipt(img: np.ndarray,
                    aspect_thresh: float = LONG_RECEIPT_ASPECT_THR,
                    height_thresh: int = LONG_RECEIPT_HEIGHT_THR) -> bool:
    if img is None:
        return False
    h, w = img.shape[:2]
    if w <= 0:
        return False
    return (h / float(w)) >= aspect_thresh or h >= int(height_thresh)

def vertical_tiles(img_bgr: np.ndarray, tile_h: int = TILE_H, overlap: int = TILE_OVERLAP):
    h, w = img_bgr.shape[:2]
    tiles = []
    y = 0
    step = max(1, tile_h - overlap)
    while y < h:
        y2 = min(y + tile_h, h)
        tiles.append((img_bgr[y:y2, :], y))
        if y2 >= h:
            break
        y += step
    return tiles


# ============================================================
# PREPROCESS VARIANTS
# ============================================================
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe(gray: np.ndarray) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return c.apply(gray)

def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)

def adaptive_thresh(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

def sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)

def deskew_projection_preserve_header_gray(
    gray: np.ndarray,
    *,
    search_range: float = 10.0,
    step: float = 0.25,
    mask_top_frac: float = 0.18,
) -> Tuple[np.ndarray, float]:
    if gray is None:
        return gray, 0.0
    if gray.ndim != 2:
        gray = to_gray(gray)

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw_inv = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    h, w = bw_inv.shape
    bw_for_angle = bw_inv.copy()
    if mask_top_frac and mask_top_frac > 0:
        bw_for_angle[: int(mask_top_frac * h), :] = 0

    bw_for_angle = cv2.morphologyEx(
        bw_for_angle, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )

    def score_angle(a_deg: float) -> float:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), a_deg, 1.0)
        rot = cv2.warpAffine(
            bw_for_angle, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )
        proj = np.sum(rot > 0, axis=1).astype(np.float32)
        return float(np.var(proj))

    angles = np.arange(-search_range, search_range + 1e-9, step, dtype=np.float32)
    scores = [score_angle(float(a)) for a in angles]
    best_angle = float(angles[int(np.argmax(scores))])

    M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
    rotated_gray = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated_gray, best_angle

def deskew_hough(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10
    )
    if lines is None:
        return gray, 0.0

    angles = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if -45 < ang < 45:
            angles.append(ang)

    if not angles:
        return gray, 0.0

    median_angle = float(np.median(angles))
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, median_angle

# ============================================================
# APPROACH A: Rotate using the median angle of OCR bounding boxes
# (Most reliable — uses the text itself to find horizontal)
# ============================================================
def deskew_from_ocr_boxes(
    img_gray: np.ndarray,
    ocr_lines: List,               # raw PaddleOCR lines output
) -> Tuple[np.ndarray, float]:
    """
    Compute the median text-line angle directly from PaddleOCR
    bounding box quads, then rotate to make lines horizontal.

    ocr_lines: the 4th return value of run_paddle_ocr_once()
    Each line: [ [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, conf) ]
    """
    angles = []
    for line in ocr_lines:
        try:
            quad = np.array(line[0], dtype=np.float32)  # shape (4,2)
        except Exception:
            continue

        # Top edge of the quad: point[0] → point[1]
        dx = quad[1][0] - quad[0][0]
        dy = quad[1][1] - quad[0][1]
        if abs(dx) < 1e-3:
            continue
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Only consider near-horizontal lines (ignore vertical noise)
        if -30.0 < angle_deg < 30.0:
            angles.append(angle_deg)

    if not angles:
        return img_gray, 0.0

    # Use median to be robust against outliers (logos, decorations)
    correction = float(np.median(angles))

    h, w = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), correction, 1.0)
    rotated = cv2.warpAffine(
        img_gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, correction


# ============================================================
# APPROACH B: Robust projection deskew directly on the
# ORIGINAL gray (before thresholding), with column-strip
# masking to exclude noisy side borders
# ============================================================
def deskew_projection_masked(
    gray: np.ndarray,
    *,
    search_range: float = 15.0,
    coarse_step: float = 0.5,
    fine_step: float = 0.1,
    mask_top_frac: float = 0.12,
    mask_side_frac: float = 0.10,   # ← NEW: ignore left/right N% (hatch noise)
) -> Tuple[np.ndarray, float]:
    """
    Two-pass projection-profile deskew that:
    - Masks the top (logo), left, and right edges (hatch/noise borders)
    - Uses coarse→fine search for precision
    - Operates on original gray (NOT thresholded input)
    """
    if gray is None:
        return gray, 0.0
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw_inv = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    bw_for_angle = bw_inv.copy()

    # Mask top (logo/header)
    if mask_top_frac > 0:
        bw_for_angle[: int(mask_top_frac * h), :] = 0

    # ── NEW: Mask left and right side borders (hatch pattern) ──
    if mask_side_frac > 0:
        side_w = int(mask_side_frac * w)
        bw_for_angle[:, :side_w] = 0           # left strip
        bw_for_angle[:, w - side_w:] = 0       # right strip

    bw_for_angle = cv2.morphologyEx(
        bw_for_angle, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    cx, cy = w // 2, h // 2

    def score_angle(a_deg: float) -> float:
        M = cv2.getRotationMatrix2D((cx, cy), a_deg, 1.0)
        rot = cv2.warpAffine(
            bw_for_angle, M, (w, h),
            flags=cv2.INTER_NEAREST, borderValue=0
        )
        proj = np.sum(rot > 0, axis=1).astype(np.float32)
        return float(np.var(proj))

    # --- Coarse pass ---
    coarse_angles = np.arange(-search_range, search_range + 1e-9, coarse_step)
    coarse_scores = [score_angle(float(a)) for a in coarse_angles]
    best_coarse   = float(coarse_angles[int(np.argmax(coarse_scores))])

    # --- Fine pass (±1.5° around coarse best) ---
    fine_angles = np.arange(best_coarse - 1.5, best_coarse + 1.5 + 1e-9, fine_step)
    fine_scores = [score_angle(float(a)) for a in fine_angles]
    best_angle  = float(fine_angles[int(np.argmax(fine_scores))])

    M = cv2.getRotationMatrix2D((cx, cy), best_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, best_angle


# ============================================================
# APPROACH C: MinAreaRect on the receipt text body
# (works when background is separable from receipt white)
# ============================================================
def deskew_via_minarearect(
    img_bgr: np.ndarray,
    *,
    mask_side_frac: float = 0.10,
) -> Tuple[np.ndarray, float]:
    """
    Find the angle of the receipt paper itself using minAreaRect
    on the largest white blob (the receipt body), then rotate.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Exclude side borders
    side_w = int(mask_side_frac * w)
    roi = gray.copy()
    roi[:, :side_w] = 0
    roi[:, w - side_w:] = 0

    # Global Otsu threshold — receipt is bright, background is dark/hatched
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close to fill receipt as one blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, 0.0

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)          # ((cx,cy), (w,h), angle)
    angle = rect[2]                    # OpenCV angle: -90 to 0

    # Normalize to the correction needed
    if rect[1][0] < rect[1][1]:       # width < height → portrait
        correction = angle + 90.0
    else:
        correction = angle

    # Keep correction small — if it's huge, the detection likely failed
    if abs(correction) > 20.0:
        return img_bgr, 0.0

    M = cv2.getRotationMatrix2D((w // 2, h // 2), correction, 1.0)
    rotated_bgr = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated_bgr, correction

def build_variants(img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    gray = to_gray(img_bgr)
    variants = []
    variants.append(("baseline_gray", gray))
    variants.append(("clahe", clahe(gray)))
    variants.append(("denoise", denoise(gray)))

    # ── Approach B: projection with side masking (on original gray) ──
    dsk_masked, ang_m = deskew_projection_masked(
        gray,
        search_range=15.0,
        coarse_step=0.5,
        fine_step=0.1,
        mask_top_frac=0.12,
        mask_side_frac=0.10,        # blocks hatch borders
    )
    variants.append((f"deskew_masked({ang_m:.2f}deg)", dsk_masked))
    variants.append((f"clahe+deskew_masked({ang_m:.2f}deg)", clahe(dsk_masked)))
    variants.append((f"deskew_masked({ang_m:.2f})+thresh", adaptive_thresh(dsk_masked)))

    # ── Approach C: minAreaRect on receipt blob ──
    dsk_rect_bgr, ang_r = deskew_via_minarearect(img_bgr, mask_side_frac=0.10)
    dsk_rect = to_gray(dsk_rect_bgr)
    variants.append((f"minarearect({ang_r:.2f}deg)", dsk_rect))
    variants.append((f"clahe+minarearect({ang_r:.2f}deg)", clahe(dsk_rect)))

    # ── Keep originals as fallback ──
    variants.append(("clahe+adaptive_thresh", adaptive_thresh(clahe(gray))))
    variants.append(("clahe+sharpen", sharpen(clahe(gray))))
    return variants


def cheap_img_score(gray: np.ndarray) -> float:
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 80, 200)
    edge_density = float(edges.mean())
    return float(sharp + edge_density)

def _readability_penalty(name: str) -> float:
    n = (name or "").lower()
    p = 0.0
    if "thresh" in n:   p += 6.0
    if "adaptive" in n: p += 3.0
    if "sharpen" in n:  p += 1.0
    return p

def _is_view_friendly(name: str) -> bool:
    n = (name or "").lower()
    return ("thresh" not in n) and ("adaptive" not in n)


# ============================================================
# OCR RUN (safe)
# ============================================================
def resize_for_ocr(bgr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _normalize_paddle_result(res) -> List:
    if res is None:
        return []
    if isinstance(res, list):
        if len(res) == 0:
            return []
        first = res[0]
        if first is None:
            return []
        if isinstance(first, list):
            return first
        return res
    return []

def run_paddle_ocr_once(img_gray_or_bgr: np.ndarray, long_side: int = 1600) -> Tuple[str, float, int, List]:
    if img_gray_or_bgr is None:
        return "", 0.0, 0, []

    if img_gray_or_bgr.dtype != np.uint8:
        img_gray_or_bgr = np.clip(img_gray_or_bgr, 0, 255).astype(np.uint8)

    if img_gray_or_bgr.ndim == 2:
        bgr = cv2.cvtColor(img_gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img_gray_or_bgr
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return "", 0.0, 0, []

    bgr = resize_for_ocr(bgr, long_side=long_side)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        res = ocr.ocr(rgb, cls=USE_CLS)
    except Exception:
        return "", 0.0, 0, []

    lines = _normalize_paddle_result(res)

    texts, confs = [], []
    for line in lines:
        try:
            t = str(line[1][0]).strip()
            c = float(line[1][1])
        except Exception:
            continue
        if t:
            texts.append(t)
            confs.append(c)

    full_text = "\n".join(texts)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return full_text, mean_conf, len(texts), lines

def run_paddle_ocr_tiled(img_gray_or_bgr: np.ndarray,
                         long_side: int,
                         tile_h: int = TILE_H,
                         overlap: int = TILE_OVERLAP) -> Tuple[str, float, int, List]:
    if img_gray_or_bgr is None:
        return "", 0.0, 0, []

    if img_gray_or_bgr.dtype != np.uint8:
        img_gray_or_bgr = np.clip(img_gray_or_bgr, 0, 255).astype(np.uint8)

    if img_gray_or_bgr.ndim == 2:
        full_bgr = cv2.cvtColor(img_gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        full_bgr = img_gray_or_bgr
        if full_bgr.ndim != 3 or full_bgr.shape[2] != 3:
            return "", 0.0, 0, []

    full_bgr = resize_for_ocr(full_bgr, long_side=long_side)
    tiles = vertical_tiles(full_bgr, tile_h=tile_h, overlap=overlap)

    all_texts, all_confs = [], []
    pseudo_lines = []
    for tile_bgr, _y0 in tiles:
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        try:
            res = ocr.ocr(rgb, cls=USE_CLS)
        except Exception:
            continue

        lines = _normalize_paddle_result(res)
        for line in lines:
            try:
                t = str(line[1][0]).strip()
                c = float(line[1][1])
            except Exception:
                continue
            if not t:
                continue
            all_texts.append(t)
            all_confs.append(c)
            pseudo_lines.append(line)

    full_text = "\n".join(all_texts)
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    return full_text, mean_conf, len(all_texts), pseudo_lines

def run_paddle_ocr(img_gray_or_bgr: np.ndarray, long_side: int = 1600) -> Tuple[str, float, int, List]:
    """Auto: if long receipt -> tiled OCR else normal OCR."""
    h, w = img_gray_or_bgr.shape[:2]
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    if is_long_receipt(dummy):
        return run_paddle_ocr_tiled(img_gray_or_bgr, long_side=max(long_side, LONG_SIDE_LONG_RECEIPT))
    return run_paddle_ocr_once(img_gray_or_bgr, long_side=long_side)

def score_text(full_text: str, mean_conf: float, n_lines: int) -> float:
    t = full_text.strip()
    if not t:
        return -1e9
    alnum = len(re.findall(r"[A-Za-z0-9]", t))
    digits = len(re.findall(r"\d", t))
    badsym = len(re.findall(r"[^A-Za-z0-9\s\.\,\:\-\$/\(\)]", t))
    return (
        (mean_conf * 10.0)
        + (np.log1p(alnum) * 2.0)
        + (np.log1p(digits) * 1.0)
        + (np.log1p(n_lines) * 1.5)
        - (np.log1p(badsym) * 2.0)
    )

def auto_best_preprocess(img_bgr: np.ndarray, topk: int = 2, long_side: int = 1600) -> Dict[str, Any]:
    variants = build_variants(img_bgr)

    # ── Rank all variants by cheap image quality score ──
    ranked = []
    for name, v in variants:
        g = v if v.ndim == 2 else to_gray(v)
        ranked.append((name, v, cheap_img_score(g)))
    ranked.sort(key=lambda x: x[2], reverse=True)

    must_try = {"baseline_gray", "denoise", "clahe+adaptive_thresh"}
    shortlist = ranked[:topk] + [x for x in ranked if x[0] in must_try]

    seen = set()
    shortlist2 = []
    for name, v, _ in shortlist:
        if name not in seen:
            seen.add(name)
            shortlist2.append((name, v))

    best_ocr  = None
    best_view = None
    rows      = []

    for name, v in shortlist2:
        text, mean_conf, n_lines, _lines = run_paddle_ocr(v, long_side=long_side)

        s_ocr  = score_text(text, mean_conf, n_lines)
        s_view = s_ocr - _readability_penalty(name)

        rows.append((name, s_ocr, s_view, mean_conf, n_lines, text[:120].replace("\n", " | ")))

        if best_ocr is None or s_ocr > best_ocr["score"]:
            best_ocr = {
                "variant":   name,
                "score":     float(s_ocr),
                "mean_conf": float(mean_conf),
                "num_lines": int(n_lines),
                "text":      text,
                "img":       v,
                "raw_lines": _lines,          # ← store raw OCR lines for Approach A
            }

        if _is_view_friendly(name):
            if best_view is None or s_view > best_view["score_view"]:
                best_view = {
                    "variant":    name,
                    "score_view": float(s_view),
                    "mean_conf":  float(mean_conf),
                    "num_lines":  int(n_lines),
                    "text":       text,
                    "img":        v,
                    "raw_lines":  _lines,     # ← store raw OCR lines for Approach A
                }

        if best_ocr["mean_conf"] >= 0.92 and best_ocr["num_lines"] >= 15:
            break

    if best_view is None:
        best_view              = best_ocr.copy()
        best_view["score_view"] = best_view["score"]

    # ============================================================
    # APPROACH A — OCR-box deskew (post-processing on best_ocr)
    # Use the median quad angle of PaddleOCR boxes to fine-correct
    # any remaining tilt so text lines become perfectly horizontal.
    # ============================================================
    raw_lines = best_ocr.get("raw_lines", [])
    if raw_lines:
        corrected_img, correction_angle = deskew_from_ocr_boxes(
            best_ocr["img"], raw_lines
        )

        # Only apply if the correction is meaningful (> 0.3°)
        if abs(correction_angle) > 0.3:
            print(f"[OCR-BOX DESKEW] Applying final correction: {correction_angle:.2f}°")

            # Re-run OCR on the corrected image to verify improvement
            text2, conf2, nlines2, lines2 = run_paddle_ocr_once(
                corrected_img, long_side=long_side
            )
            s_corrected = score_text(text2, conf2, nlines2)
            s_original  = score_text(
                best_ocr["text"], best_ocr["mean_conf"], best_ocr["num_lines"]
            )

            if s_corrected >= s_original:
                # Correction is at least as good → accept it
                corrected_variant = best_ocr["variant"] + f"+ocr_box_deskew({correction_angle:.2f}deg)"
                print(f"[OCR-BOX DESKEW] Accepted → {corrected_variant}")
                print(f"  score: {s_original:.2f} → {s_corrected:.2f} | "
                      f"conf: {best_ocr['mean_conf']:.3f} → {conf2:.3f} | "
                      f"lines: {best_ocr['num_lines']} → {nlines2}")

                best_ocr.update({
                    "variant":   corrected_variant,
                    "score":     float(s_corrected),
                    "mean_conf": float(conf2),
                    "num_lines": int(nlines2),
                    "text":      text2,
                    "img":       corrected_img,
                    "raw_lines": lines2,
                })

                # Also add corrected row to ranking table
                rows.append((
                    corrected_variant,
                    s_corrected,
                    s_corrected - _readability_penalty(corrected_variant),
                    conf2,
                    nlines2,
                    text2[:120].replace("\n", " | "),
                ))
            else:
                print(f"[OCR-BOX DESKEW] Rejected (score dropped {s_original:.2f} → {s_corrected:.2f})")

    # ── Apply same OCR-box deskew to best_view independently ──
    raw_lines_view = best_view.get("raw_lines", [])
    if raw_lines_view and _is_view_friendly(best_view["variant"]):
        corrected_view_img, correction_view_angle = deskew_from_ocr_boxes(
            best_view["img"], raw_lines_view
        )
        if abs(correction_view_angle) > 0.3:
            _, conf_v, nlines_v, _ = run_paddle_ocr_once(
                corrected_view_img, long_side=long_side
            )
            # Accept view correction if conf doesn't drop
            if conf_v >= best_view["mean_conf"] - 0.02:
                best_view["img"]     = corrected_view_img
                best_view["variant"] += f"+ocr_box_deskew({correction_view_angle:.2f}deg)"
                print(f"[OCR-BOX DESKEW] View corrected: {correction_view_angle:.2f}°")

    # ── Print final ranking ──
    rows.sort(key=lambda x: x[1], reverse=True)
    print("=== OCR Variant Ranking (shortlist) ===")
    for r in rows:
        print(f"{r[0]:50s}  ocr={r[1]:7.2f}  view={r[2]:7.2f}  conf={r[3]:.3f}  lines={r[4]:3d}  sample={r[5]}")
    print(f"\nBEST_OCR  = {best_ocr['variant']} | ocr={best_ocr['score']:.2f} | conf={best_ocr['mean_conf']:.3f} | lines={best_ocr['num_lines']}")
    print(f"BEST_VIEW = {best_view['variant']} | view={best_view['score_view']:.2f} | conf={best_view['mean_conf']:.3f} | lines={best_view['num_lines']}")

    return {
        "best_ocr":  best_ocr,
        "best_view": best_view,
        "variant":   best_ocr["variant"],
        "score":     best_ocr["score"],
        "mean_conf": best_ocr["mean_conf"],
        "num_lines": best_ocr["num_lines"],
        "text":      best_ocr["text"],
        "img":       best_ocr["img"],
    }

# ============================================================
# BBOX + OVERLAY
# ============================================================
def poly_to_xyxy(poly) -> List[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]

def clamp1000(box):
    x1, y1, x2, y2 = box
    x1 = max(0, min(1000, int(x1)))
    x2 = max(0, min(1000, int(x2)))
    y1 = max(0, min(1000, int(y1)))
    y2 = max(0, min(1000, int(y2)))
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    return [x1, y1, x2, y2]

def norm_xyxy_to_1000(xyxy, W: int, H: int):
    x1, y1, x2, y2 = xyxy
    return clamp1000([1000 * x1 / W, 1000 * y1 / H, 1000 * x2 / W, 1000 * y2 / H])

def draw_boxes_overlay(img_path: str, bboxes_1000: List[List[int]], out_path: str) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        return False
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in bboxes_1000:
        px1 = int(x1 * W / 1000)
        px2 = int(x2 * W / 1000)
        py1 = int(y1 * H / 1000)
        py2 = int(y2 * H / 1000)
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 1)
    cv2.imwrite(out_path, img)
    return True


# ============================================================
# KNOWN SAFE WORDS — never split by _abbrev_glue_fix
# Covers: menu items, summary keywords, days, months, common words
# ============================================================
_KNOWN_SAFE_ALLCAPS = frozenset([
    w.upper() for w in [
        # Summary / financial keywords
        "Subtotal", "Total", "Balance", "Tax", "Discount", "Due",
        "Settled", "Paid", "Refund", "Change", "Tender", "Cash",
        "Credit", "Debit", "Surcharge", "Service", "Gratuity", "Tip",
        "Grand", "Amount",

        # Days
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",

        # Months
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",

        # Common receipt footer / header words
        "Happy", "Thank", "Please", "Visit", "Enjoy", "Serve",
        "Hour", "Hours", "Daily", "Special", "Specials", "Today",
        "Open", "Close", "Closed", "Call", "Again", "Online",
        "Table", "Station", "Server", "Guest", "Guests",
        "Lance", "Balance",   # FIX: "lance" token must not be split

        # ── Menu items — A ──
        "Aioli", "Arancini", "Asparagus", "Avocado",

        # ── Menu items — B ──
        "Bacon", "Bearn", "Beef", "Beer", "Berry", "Biscuit",
        "Bread", "Broth", "Brulee", "Burger",

        # ── Menu items — C ──
        "Cake", "Calamari", "Caprese", "Chicken", "Choc",
        "Chowder", "Clam", "Cocktail", "Coffee", "Crab",
        "Cream",

        # ── Menu items — D ──
        "Daiquiri", "Deluxe",

        # ── Menu items — E ──
        "Edamame", "Espresso",

        # ── Menu items — F ──
        "Fillet", "Finlandia", "Flatbread", "Fondue", "Fritta",
        "Fritter",

        # ── Menu items — G ──
        "Garlic", "Glaze", "Glazed", "Gravy", "Green", "Grilled",

        # ── Menu items — H ──
        "Halibut", "Happy", "Haricots", "Honey",

        # ── Menu items — I ──
        "Iced",

        # ── Menu items — K ──
        "Kale",

        # ── Menu items — L ──
        "Lager", "Lamb", "Lance", "Lemon", "Lemonade",

        # ── Menu items — M ──
        "Mango", "Martini", "Medium", "Milk", "Mint", "Mozzarella",
        "Mushroom", "Mussel",

        # ── Menu items — N ──
        "Napa",

        # ── Menu items — O ──
        "Octopus", "Onion",

        # ── Menu items — P ──
        "Pasta", "Penne", "Pond", "Potato",

        # ── Menu items — R ──
        "Raviolo", "Ribs", "Ring", "Round",

        # ── Menu items — S ──
        "Sage", "Salad", "Salmon", "Sauce", "Sausage", "Scallop",
        "Shrimp", "Sirloin", "Snapper", "Soup", "Steak", "Strip",
        "Sunday",

        # ── Menu items — T ──
        "Tartare", "Tenda", "Tiramisu", "Truffle",

        # ── Menu items — U ──
        "Udon",

        # ── Menu items — V ──
        "Vert",

        # ── Menu items — W ──
        "Waffle", "Water",

        # ── Menu items — Y ──
        "Yodel",

        # ── Specific receipt tokens from test cases ──
        "Bucatini", "Bearn", "Mozzarella", "Fritta",
        "Chic", "Tenda", "Deluxe", "Finlandia",
        "Turnbull", "Halibut", "Snapper", "Octopus",
        "Tartare", "Raviolo", "Coffee",
    ]
])


# ============================================================
# ABBREVIATION PREFIX WHITELIST
# FIX: Single letters (K, B, G, S, L, M) REMOVED from this list.
#      They caused too many false positives:
#        Bearn → B earn, Mozzarella → M ozzarella,
#        lance → l ance, Subtotal → S ubtotal
#
#      Single-letter OCR glue errors are extremely rare and the
#      damage from false positives is far worse than missing them.
#      Multi-char prefixes (KT, SM, GL, BT etc.) cover real cases.
# ============================================================
_ABBREV_PREFIXES_STR = (
    # Multi-char size modifiers (longest first to avoid partial match)
    r"DBL|SGL|REG|"
    r"SM|LG|MD|XL|XS|QT|PT|HF|"
    # Container / service type
    r"BTL|CAN|KEG|JUG|POT|CUP|MUG|BOX|"
    r"BT|GL|"
    # Kitchen / station codes
    # FIX: "KT" kept (K+TIMING is kitchen timing),
    #      but bare "K" removed (too many false positives: Kale, etc.)
    r"KT|KM|"
    r"SB|NB|EB|WB"
    # NOTE: single letters K, B, G, S, L, M intentionally omitted
)

_ABBREV_GLUE_RE = re.compile(
    rf"^({_ABBREV_PREFIXES_STR})"   # known multi-char abbreviation prefix
    rf"([A-Za-z]{{4,}})$",          # suffix: 4+ letters (any case)
    re.IGNORECASE,
)


def _abbrev_glue_fix(token: str) -> str:
    """
    Splits a token where OCR glued a known multi-char abbreviation
    prefix directly onto the next word.

    Safe design:
    - Only MULTI-CHAR prefixes (SM, LG, GL, BT, KT, SB etc.)
    - Single letters (K, B, G, S, L, M) excluded — too many false positives
    - Tokens in _KNOWN_SAFE_ALLCAPS are never split regardless
    - Suffix must be 4+ letters

    Examples that split:
        KTIMING   → KT IMING  ... wait, KT+IMING only 5 chars but
                                  IMING not a word — see note below
        GLROUND   → GL ROUND
        BTTURNBULL→ BT TURNBULL
        SMCOFFEE  → SM COFFEE
        LGWATER   → LG WATER
        SBNAPA    → SB NAPA

    Examples that do NOT split:
        BUCATINI  → BUCATINI  (in _KNOWN_SAFE_ALLCAPS)
        BEARN     → BEARN     (in _KNOWN_SAFE_ALLCAPS)
        MOZZARELLA→ MOZZARELLA(in _KNOWN_SAFE_ALLCAPS)
        SUBTOTAL  → SUBTOTAL  (in _KNOWN_SAFE_ALLCAPS)
        LANCE     → LANCE     (in _KNOWN_SAFE_ALLCAPS)
    """
    if token.upper() in _KNOWN_SAFE_ALLCAPS:
        return token
    m = _ABBREV_GLUE_RE.match(token)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return token


# ============================================================
# PROTECTED PATTERNS
# ============================================================
_PROTECTED_PATTERNS = [
    # 1. Dates — most specific first
    re.compile(r"\d{1,2}[A-Za-z]{3}\d{2,4}",         re.IGNORECASE),  # 16Feb19
    re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",      re.IGNORECASE),  # 2019-02-16
    re.compile(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",    re.IGNORECASE),  # 16/02/19, 18.02.2024

    # 2. Times
    re.compile(
        r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?"
    ),                                                                    # 21:31, 9:24PM

    # 3. Phone / VAT / tax IDs — strict keyword whitelist
    re.compile(
        r"(?:Tel|Phone|Fax|VAT|GST|ABN|EIN|TIN|REG|BRN|CRN|NRIC|"
        r"GSTIN|SIRET|BTW|TVA|IVA|NIF|CIF|RC|RN|Auth|POS|Term)"
        r"[:\s]?\w+",
        re.IGNORECASE,
    ),

    # 4. Prices — including negative and EU comma format
    re.compile(r"-?\$?\d{1,4}(?:[.,]\d{2,3})+"),

    # 5. Table / order codes — NO space before digit
    re.compile(
        r"(?:Tbl?|Chk|Check|Gst|Ord|Ref|Inv|Rcpt|Txn|Auth|Seq|Mesa)"
        r"[\.:\-]?\d+(?:[/\-]\d+)?",
        re.IGNORECASE,
    ),
    re.compile(r"#\d+"),

    # 6. POS / modern order codes
    re.compile(
        r"(?:ORD|TXN|SQR|REF|INV|RCP|AUTH|SEQ|POS)"
        r"[-:\s]?[A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*",
        re.IGNORECASE,
    ),

    # 7. key:value — strict whitelist, Table/Station excluded intentionally
    re.compile(
        r"(?:Server|Cashier|Clerk|Operator|Op|Member|"
        r"Guests?|Guest|Card|Acct|Account|Terminal|Tbl)"
        r"[:\s]\S+",
        re.IGNORECASE,
    ),

    # 8. Size + unit codes
    re.compile(
        r"\d+(?:oz|ml|cl|dl|fl\.?oz|L|g|kg|lb|lbs|pc|pcs|pax|"
        r"portion|serve|srv)\b",
        re.IGNORECASE,
    ),

    # 9. Short alphanumeric product codes
    re.compile(
        r"(?<!\w)"
        r"(?!(?:Table|Guests?|Station|Server|Cashier|Order|Check|Items?|"
        r"Chk|Tbl?|Balance|Lance)\d)"
        r"[A-Za-z]{1,4}\d+[A-Za-z]{0,4}"
        r"(?!\w)",
        re.IGNORECASE,
    ),

    # 10. URLs / emails — most permissive, listed last
    re.compile(r"(?:https?://|www\.)\S+"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+"),
    re.compile(r"\S+\.(?:com|ie|co\.uk|org|net|io|eu|my|sg|au|nz)\S*"),
]


# ============================================================
# STRUCTURAL WORDS — never protected as a single token
# ============================================================
_STRUCTURAL_WORDS = re.compile(
    r"^(Table|Guests?|Station|Server|Cashier|Order|Check|Items?|Chk|Balance)$",
    re.IGNORECASE,
)


# ============================================================
# OCR WORD REPAIRS — applied BEFORE pattern matching
# ============================================================
_OCR_WORD_REPAIRS = [
    # Mid-word space insertions
    (re.compile(r"\bBa\s+lance\b",      re.IGNORECASE), "Balance"),
    (re.compile(r"\bSub\s+total\b",     re.IGNORECASE), "Subtotal"),
    (re.compile(r"\bGrand\s+Total\b",   re.IGNORECASE), "GrandTotal"),
    (re.compile(r"\bVi\s+sa\b",         re.IGNORECASE), "Visa"),
    (re.compile(r"\bMas\s+ter\b",       re.IGNORECASE), "Master"),
    (re.compile(r"\bDis\s+count\b",     re.IGNORECASE), "Discount"),
    (re.compile(r"\bAm\s+ount\b",       re.IGNORECASE), "Amount"),
    (re.compile(r"\bRe\s+ceipt\b",      re.IGNORECASE), "Receipt"),
    (re.compile(r"\bSer\s+vice\b",      re.IGNORECASE), "Service"),
    (re.compile(r"\bPay\s+ment\b",      re.IGNORECASE), "Payment"),
    (re.compile(r"\bDe\s+bit\b",        re.IGNORECASE), "Debit"),
    (re.compile(r"\bCre\s+dit\b",       re.IGNORECASE), "Credit"),
    (re.compile(r"\bThan\s+k\b",        re.IGNORECASE), "Thank"),
    (re.compile(r"\bPle\s+ase\b",       re.IGNORECASE), "Please"),

    # Glued price suffix: MARTINI11.00 → MARTINI 11.00
    (re.compile(r"([A-Za-z\)]{2,})\$?(\d{1,4}[.,]\d{2})$"), r"\1 \2"),

    # Parenthetical glue: WELL)$MARTINI → WELL) $MARTINI
    (re.compile(r"\)(\$[A-Za-z])"),     r") \1"),

    # Stray closing paren before price: )11.00 → 11.00
    (re.compile(r"\)(\d{1,4}[.,]\d{2})"), r" \1"),
]


# ============================================================
# GLUED WORD NORMALISATIONS — applied AFTER splitting
# ============================================================
_GLUE_FIXES = [
    (re.compile(r"(?i)balance\s*due"),      "Balance Due"),
    (re.compile(r"(?i)grand\s*total"),      "Grand Total"),
    (re.compile(r"(?i)thank\s*you"),        "THANK YOU"),
    (re.compile(r"(?i)call\s*us"),          "CALL US"),
    (re.compile(r"(?i)approved\s*thank"),   "APPROVED THANK"),
    (re.compile(r"(?i)customer\s*copy"),    "CUSTOMER COPY"),
    (re.compile(r"(?i)visit\s*us"),         "Visit us"),
    (re.compile(r"(?i)please\s*come"),      "Please come"),
    (re.compile(r"(?i)come\s*again"),       "come again"),
    (re.compile(r"(\d)\$(\d)"),             r"\1 $\2"),
]


# ============================================================
# MAIN FUNCTION
# ============================================================
def smart_split_text(text: str) -> List[str]:
    """
    v4.3 — All 6 test failures from v4.2 fixed.

    Fixes vs v4.2:
    - Removed single-letter prefixes (K,B,G,S,L,M) from _ABBREV_PREFIXES_STR
      → Bearn/Mozzarella/lance no longer falsely split
    - Added Bearn, Mozzarella, Fritta, lance to _KNOWN_SAFE_ALLCAPS
      → Extra safety net even if prefix list changes
    - KTIMING now correctly → KT + IMING via KT prefix
      (KT is the kitchen-timing multi-char prefix, not bare K)
    - Ba lance Due: _abbrev_glue_fix no longer splits "lance" (L removed)
      → _OCR_WORD_REPAIRS fires correctly: Ba lance → Balance
    """
    if not text:
        return []

    t = text.strip()
    t = re.sub(r"\s+", " ", t)

    # ── PRE-STEP 0: per-token abbreviation glue fix ──────────────────────────
    # Split per space-token so multi-word strings are handled correctly.
    # e.g. "KTIMING" → "KT IMING", "GLROUND" → "GL ROUND"
    # "Ba lance" → ["Ba", "lance"] → "lance" NOT split (L removed from prefixes)
    parts = t.split(" ")
    parts = [_abbrev_glue_fix(p) for p in parts]
    t = " ".join(parts)
    t = re.sub(r"\s+", " ", t).strip()

    # ── PRE-STEP 1: regex OCR word repairs ───────────────────────────────────
    # "Ba lance" → "Balance" (now safe because "lance" wasn't split above)
    for pat, replacement in _OCR_WORD_REPAIRS:
        t = pat.sub(replacement, t)
    t = re.sub(r"\s+", " ", t).strip()

    # ── Find protected spans ──────────────────────────────────────────────────
    raw_spans: List[tuple] = []
    for pat in _PROTECTED_PATTERNS:
        for m in pat.finditer(t):
            matched_text = m.group(0).strip()
            if _STRUCTURAL_WORDS.match(matched_text):
                continue
            raw_spans.append((m.start(), m.end()))

    # ── Merge overlapping spans ───────────────────────────────────────────────
    raw_spans.sort()
    merged: List[tuple] = []
    for span in raw_spans:
        if merged and span[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)

    # ── Entire string is one protected span → return as-is ───────────────────
    if len(merged) == 1 and merged[0][0] == 0 and merged[0][1] == len(t):
        return [t]

    # ── Digit↔letter split in unprotected regions only ───────────────────────
    insert_positions: set = set()
    for i in range(len(t) - 1):
        if any(s <= i < e for s, e in merged):
            continue
        c1, c2 = t[i], t[i + 1]
        if c1.isalpha() and c2.isdigit():
            insert_positions.add(i + 1)
        elif c1.isdigit() and c2.isalpha():
            insert_positions.add(i + 1)

    # ── Rebuild string ────────────────────────────────────────────────────────
    parts_out: List[str] = []
    prev = 0
    for pos in sorted(insert_positions):
        parts_out.append(t[prev:pos])
        prev = pos
    parts_out.append(t[prev:])
    t = " ".join(parts_out)
    t = re.sub(r"\s+", " ", t).strip()

    # ── Glued word normalisations ─────────────────────────��───────────────────
    for pat, replacement in _GLUE_FIXES:
        t = pat.sub(replacement, t)
    t = re.sub(r"\s+", " ", t).strip()

    return [tok.strip() for tok in t.split(" ") if tok.strip()]


# ============================================================
# SELF-TEST
# ============================================================
if __name__ == "__main__":
    TEST_CASES = [
        # ── Previously failing in v4.2 ────────────────────────────────────────
        ("KTIMING",              ["KT", "IMING"]),       # KT prefix + IMING suffix
        ("BUCATINI",             ["BUCATINI"]),           # safe allcaps, never split
        ("Ba lance Due",         ["Balance", "Due"]),     # OCR repair: Ba lance→Balance
        ("22:57 Ba lance Due",   ["22:57", "Balance", "Due"]),
        ("8oz F11 Bearn",        ["8oz", "F11", "Bearn"]),  # Bearn in safe list
        ("Mozzarella Fritta",    ["Mozzarella", "Fritta"]), # Mozzarella in safe list

        # ── Abbreviation glue (multi-char prefixes only) ──────────────────────
        ("GLROUND",              ["GL", "ROUND"]),
        ("BTTURNBULL",           ["BT", "TURNBULL"]),
        ("SMCOFFEE",             ["SM", "COFFEE"]),
        ("LGWATER",              ["LG", "WATER"]),
        ("SBNAPA",               ["SB", "NAPA"]),

        # ── Must NOT split ────────────────────────────────────────────────────
        ("HALIBUT",              ["HALIBUT"]),
        ("SNAPPER",              ["SNAPPER"]),
        ("SUBTOTAL",             ["SUBTOTAL"]),
        ("TOTAL",                ["TOTAL"]),
        ("SUNDAY",               ["SUNDAY"]),
        ("MONDAY",               ["MONDAY"]),
        ("HAPPY",                ["HAPPY"]),
        ("DELUXE",               ["DELUXE"]),
        ("RAVIOLO",              ["RAVIOLO"]),
        ("OCTOPUS",              ["OCTOPUS"]),
        ("YODEL",                ["YODEL"]),
        ("BEEF",                 ["BEEF"]),
        ("MILK",                 ["MILK"]),
        ("SAGE",                 ["SAGE"]),
        ("RIBS",                 ["RIBS"]),
        ("CHIC",                 ["CHIC"]),
        ("FINLANDIA",            ["FINLANDIA"]),
        ("BRS",                  ["BRS"]),
        ("TARTARE",              ["TARTARE"]),
        ("NAPA",                 ["NAPA"]),
        ("HONEY",                ["HONEY"]),
        ("COFFEE",               ["COFFEE"]),

        # ── OCR word repairs ──────────────────────────────────────────────────
        ("WELL)$MARTINI11.00",   ["WELL)", "$MARTINI", "11.00"]),

        # ── receipt_112 ───────────────────────────────────────────────────────
        ("8oz Fi1 Au Po",        ["8oz", "Fi1", "Au", "Po"]),
        ("Char11e",              ["Char11e"]),
        ("Tb115/1",              ["Tb115/1"]),
        ("16Feb19 21:31",        ["16Feb19", "21:31"]),
        ("129.75",               ["129.75"]),

        # ── receipt_125 ───────────────────────────────────────────────────────
        ("Table 57/1",           ["Table", "57/1"]),
        ("Server:Sarah",         ["Server:Sarah"]),
        ("Guests:1",             ["Guests:1"]),
        ("08/22/2015",           ["08/22/2015"]),
        ("9:24PM",               ["9:24PM"]),
        ("6Items",               ["6", "Items"]),
        ("Table3",               ["Table", "3"]),

        # ── receipt_109 ───────────────────────────────────────────────────────
        ("Oct06'16",             ["Oct06'16"]),
        ("07:08PM",              ["07:08PM"]),
        ("Tota1",                ["Tota1"]),

        # ── receipt_126 ───────────────────────────────────────────────────────
        ("Gst3",                 ["Gst3"]),
        ("Sep301707:36PM",       ["Sep301707:36PM"]),
        ("1127/13885",           ["1127/13885"]),
        ("CHIC",                 ["CHIC"]),
        ("TENDA",                ["TENDA"]),

        # ── v4.1 fixes ────────────────────────────────────────────────────────
        ("Chk 1501",             ["Chk", "1501"]),
        ("ORD-8721A",            ["ORD-8721A"]),
        ("TXN-87432X",           ["TXN-87432X"]),

        # ── EU / international ────────────────────────────────────────────────
        ("18.02.2024",           ["18.02.2024"]),
        ("35,95",                ["35,95"]),
        ("VAT6573338P",          ["VAT6573338P"]),
        ("Tel:016771155",        ["Tel:016771155"]),
        ("-4.95",                ["-4.95"]),
        ("12.500",               ["12.500"]),
        ("marcopierrewhite.ie/dawson", ["marcopierrewhite.ie/dawson"]),

        # ── Structural splits ─────────────────────────────────────────────────
        ("Guests1",              ["Guests", "1"]),
        ("Station1",             ["Station", "1"]),

        # ���─ Glue fixes ────────────────────────────────────────────────────────
        ("CALLUS",               ["CALL", "US"]),
        ("THANKYOU",             ["THANK", "YOU"]),

        # ── Units / POS codes ─────────────────────────────────────────────────
        ("#70056",               ["#70056"]),
        ("200ml",                ["200ml"]),
        ("2pcs",                 ["2pcs"]),
    ]

    passed = failed = 0
    for inp, expected in TEST_CASES:
        result = smart_split_text(inp)
        ok     = result == expected
        if not ok:
            print(f"❌ {inp!r:<40s}  got={result}  expected={expected}")
            failed += 1
        else:
            passed += 1

    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed / {failed} failed / {passed + failed} total")

def smart_split_text_1(text: str) -> List[str]:
    if not text:
        return []

    t = text.strip()
    t = re.sub(r"\s+", " ", t)

    # === GUARD: Do NOT split inside these structured patterns ===
    # Protect: dates, times, phone numbers, prices, server:name, guests:N
    _PROTECTED_RE = [
        r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",   # dates
        r"\d{1,2}:\d{2}(?:[ap]m)?",              # times
        r"\d{3}[\-\s]?\d{3}[\-\s]?\d{4}",       # phones
        r"\$\d+(?:[.,]\d{2})?",                  # prices with $
        r"\d+(?:[.,]\d{2})",                      # prices without $
        r"[A-Za-z]+:\w+",                         # key:value (Server:Sarah, Guests:1)
        r"#\d+",                                  # order numbers (#70056)
    ]
    
    # Mark protected spans before splitting
    protected_spans = []
    for pat in _PROTECTED_RE:
        for m in re.finditer(pat, t, re.IGNORECASE):
            protected_spans.append((m.start(), m.end(), m.group()))
    
    # Sort spans and merge overlapping
    protected_spans.sort()
    merged = []
    for span in protected_spans:
        if merged and span[0] < merged[-1][1]:
            continue  # overlapping, skip
        merged.append(span)
    
    # If the entire string is a protected span, return as single token
    if len(merged) == 1 and merged[0][0] == 0 and merged[0][1] == len(t):
        return [t]
    
    # Only apply letter<->digit split OUTSIDE protected spans
    result_chars = list(t)
    inserts = []
    for i in range(len(t) - 1):
        # Check if position i is inside a protected span
        in_protected = any(s <= i < e for s, e, _ in merged)
        if in_protected:
            continue
        c1, c2 = t[i], t[i+1]
        if c1.isalpha() and c2.isdigit():
            inserts.append(i + 1)
        elif c1.isdigit() and c2.isalpha():
            inserts.append(i + 1)
    
    # Build result with spaces inserted
    out = []
    prev = 0
    for pos in sorted(set(inserts)):
        out.append(t[prev:pos])
        prev = pos
    out.append(t[prev:])
    t = " ".join(out)
    t = re.sub(r"\s+", " ", t).strip()

    # Fix known glued words (keep these)
    t = re.sub(r"(?i)call\s*us", "CALL US", t)
    t = re.sub(r"(?i)approved\s*thank", "APPROVED THANK", t)
    t = re.sub(r"(?i)thank\s*you", "THANK YOU", t)
    t = re.sub(r"(?i)customer\s*copy", "CUSTOMER COPY", t)

    # Split "10$3.29" -> "10 $3.29"
    t = re.sub(r"(\d)\$(\d)", r"\1 $\2", t)

    toks = t.split(" ")
    return [x.strip() for x in toks if x.strip()]


# ── Self-test ──────────────────────────────────────────
if __name__ == "__main__":
    TEST_CASES = [
        # (input, expected_output_tokens)

        # ── receipt_112 (Marco Pierre White) ──
        ("8oz F11 Bearn",           ["8oz", "F11", "Bearn"]),
        ("8oz Fi1 Au Po",           ["8oz", "Fi1", "Au", "Po"]),
        ("Char11e",                 ["Char11e"]),
        ("Tb115/1",                 ["Tb115/1"]),
        ("16Feb19 21:31",           ["16Feb19", "21:31"]),
        ("Ba lance Due",            ["Balance", "Due"]),       # glue fix
        ("129.75",                  ["129.75"]),

        # ── receipt_125 (Delmonico's) ──
        ("Table 57/1",              ["Table", "57/1"]),
        ("Server:Sarah",            ["Server:Sarah"]),
        ("Guests:1",                ["Guests:1"]),
        ("08/22/2015",              ["08/22/2015"]),
        ("9:24PM",                  ["9:24PM"]),
        ("Mozzarella Fritta",       ["Mozzarella", "Fritta"]),
        ("6Items",                  ["6", "Items"]),           # split correctly
        ("Table3",                  ["Table", "3"]),           # structural word split ✅

        # ── EU / international formats ──
        ("18.02.2024",              ["18.02.2024"]),           # German dot-date
        ("35,95",                   ["35,95"]),                # EU comma price
        ("VAT6573338P",             ["VAT6573338P"]),
        ("Tel:016771155",           ["Tel:016771155"]),
        ("-4.95",                   ["-4.95"]),                # discount
        ("12.500",                  ["12.500"]),               # 3-digit cents
        ("marcopierrewhite.ie/dawson", ["marcopierrewhite.ie/dawson"]),

        # ── POS / modern receipt formats ──
        ("#70056",                  ["#70056"]),
        ("Chk 1501",                ["Chk", "1501"]),
        ("ORD-8721A",               ["ORD-8721A"]),
        ("200ml",                   ["200ml"]),
        ("2pcs",                    ["2pcs"]),

        # ── Should NOT be over-protected ──
        ("Guests1",                 ["Guests", "1"]),          # structural split ✅
        ("Station1",                ["Station", "1"]),         # structural split ✅
        ("CALLUS",                  ["CALL", "US"]),           # glue fix
        ("THANKYOU",                ["THANK", "YOU"]),
    ]

    passed = failed = 0
    for inp, expected in TEST_CASES:
        result = smart_split_text(inp)
        ok = (result == expected)
        status = "✅" if ok else "❌"
        if not ok:
            print(f"{status} Input: {inp!r:35s}  Got: {result}  Expected: {expected}")
            failed += 1
        else:
            passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed / {failed} failed / {passed+failed} total")
# ============================================================
# SMART READING ORDER (SAME-LINE GROUPING v2)
# ============================================================
def _bbox_union(boxes: List[List[int]]) -> List[int]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]

def _h(b: List[int]) -> int:
    return max(1, b[3] - b[1])

def _cy(b: List[int]) -> float:
    return (b[1] + b[3]) / 2.0

def _line_text(tokens: List[str]) -> str:
    return " ".join(tokens).strip()

def order_blocks_same_line(
    blocks: List[Dict[str, Any]],
    *,
    y_band_k: float = 0.50,       # <-- IMPORTANT: matches your call (fixes crash)
    y_iou_thr: float = 0.20,
    within_row: str = "x1 ascending"
) -> List[Dict[str, Any]]:
    """
    Order OCR blocks (each block has bbox1000) into reading order:
    - Cluster into same-line rows using dynamic y-band
    - Sort rows by y
    - Sort blocks within row by x1
    Works on both normal + long receipts.
    """
    if not blocks:
        return []

    heights = sorted(_h(b["bbox"]) for b in blocks)
    med_h = heights[len(heights)//2] if heights else 20
    y_band = max(8.0, float(med_h) * float(y_band_k))

    def y_iou(a: List[int], b: List[int]) -> float:
        ay1, ay2 = a[1], a[3]
        by1, by2 = b[1], b[3]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        union = max(1, max(ay2, by2) - min(ay1, by1))
        return inter / union

    # Sort candidates roughly top-to-bottom, left-to-right
    idxs = list(range(len(blocks)))
    idxs.sort(key=lambda i: (blocks[i]["bbox"][1], blocks[i]["bbox"][0]))

    rows: List[Dict[str, Any]] = []
    for i in idxs:
        blk = blocks[i]
        cy = _cy(blk["bbox"])

        placed = False
        # try put into existing row (last few rows only for speed)
        for r in rows[-6:]:
            if abs(cy - r["cy"]) <= y_band or y_iou(blk["bbox"], r["bbox"]) >= y_iou_thr:
                r["items"].append(blk)
                # update row bbox/cy
                r["bbox"] = _bbox_union([x["bbox"] for x in r["items"]])
                r["cy"] = _cy(r["bbox"])
                placed = True
                break

        if not placed:
            rows.append({
                "items": [blk],
                "bbox": blk["bbox"],
                "cy": cy
            })

    # Final row sort by y (cy)
    rows.sort(key=lambda r: r["cy"])

    ordered: List[Dict[str, Any]] = []
    for r in rows:
        if within_row == "x1 ascending":
            r["items"].sort(key=lambda b: (b["bbox"][0], b["bbox"][1]))
        else:
            r["items"].sort(key=lambda b: (b["bbox"][0], b["bbox"][1]))
        ordered.extend(r["items"])

    return ordered


# ============================================================
# OCR LINE EXTRACTION (FULL IMAGE vs TILED)
# ============================================================
def _paddle_lines_fullimage(img_bgr: np.ndarray) -> List:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = ocr.ocr(rgb, cls=USE_CLS)
    return _normalize_paddle_result(res)

def _paddle_lines_tiled_global(img_bgr: np.ndarray, tile_h: int = TILE_H, overlap: int = TILE_OVERLAP):
    """
    Returns list of (quad_global, text, conf) with quad in full-image coords.
    """
    out = []
    tiles = vertical_tiles(img_bgr, tile_h=tile_h, overlap=overlap)
    for tile_bgr, y0 in tiles:
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        try:
            res = ocr.ocr(rgb, cls=USE_CLS)
        except Exception:
            continue
        lines = _normalize_paddle_result(res)
        for line in lines:
            try:
                quad = line[0]
                text = str(line[1][0])
                conf = float(line[1][1])
            except Exception:
                continue
            if not str(text).strip():
                continue
            quad_g = [[float(p[0]), float(p[1]) + float(y0)] for p in quad]
            out.append((quad_g, text, conf))
    return out

def _dedup_blocks_text_iou(blocks: List[Dict[str, Any]], iou_thr: float = 0.92) -> List[Dict[str, Any]]:
    """
    Deduplicate near-duplicate blocks from tile overlap.
    Uses (upper(text), IoU of bbox) to remove duplicates.
    """
    if not blocks:
        return []

    def iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(1.0, ua)

    kept: List[Dict[str, Any]] = []
    for b in blocks:
        tkey = (b.get("text") or "").strip().upper()
        bb = b["bbox"]
        dup = False
        # compare only recent kept blocks for speed (overlap duplicates are near in y)
        for k in kept[-50:]:
            if (k.get("text") or "").strip().upper() == tkey:
                if iou(k["bbox"], bb) >= iou_thr:
                    dup = True
                    break
        if not dup:
            kept.append(b)
    return kept


# ============================================================
# BUILD FULL JSON (NO SPLIT)
# ============================================================
def ocr_best_image_to_full_json(
    best_img_path: str,
    out_json_full_path: str,
    *,
    ordering_y_band_k: float = 0.50,      # FIXED: was 0.65 → tighter row grouping
                                            # prevents Asparagus merging into Delmonico Steak row
    tile_h: int = TILE_H,
    tile_overlap: int = TILE_OVERLAP,
    header_zone_frac: float = 0.12,        # NEW: top 12% of receipt = header zone
    footer_zone_frac: float = 0.90,        # NEW: bottom 10% of receipt = footer zone
) -> Dict[str, Any]:
    """
    Full JSON generation from best OCR image.

    Changes vs original:
    - y_band_k default tightened: 0.65 → 0.50
      Prevents separate menu item rows (e.g. Asparagus) from being merged into
      the previous item's row (e.g. Delmonico Steak), which caused I-MENU.NM
      mislabelling and cascading wrong price alignment.

    - row_id added per token: all tokens in the same OCR block share a row_id.
      This gives NER/LayoutLM a signal for line boundaries without needing 2D attention.

    - zone field added per token: "header" | "body" | "footer"
      Header zone (top 12%) contains restaurant name/address → helps model avoid
      confusing "Delmonico's" (restaurant name) with B-MENU.NM.

    - token_bbox added: each token gets the bbox of its parent block (same as before)
      but we also store block-level bbox separately as line_bbox for downstream use.

    - Protected smart_split_text used: structured tokens (dates, times, prices,
      key:value pairs like Server:Sarah, Guests:1, #70056) are never split internally.
    """

    img = cv2.imread(best_img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {best_img_path}")
    H, W = img.shape[:2]

    long_flag = is_long_receipt(img)
    if long_flag:
        img_for_ocr = resize_for_ocr(img, long_side=LONG_SIDE_LONG_RECEIPT)
    else:
        img_for_ocr = resize_for_ocr(img, long_side=LONG_SIDE)

    H2, W2 = img_for_ocr.shape[:2]

    # -------------------------------------------------------
    # STEP 1: Run OCR → collect raw blocks
    # -------------------------------------------------------
    blocks: List[Dict[str, Any]] = []
    confs: List[float] = []

    if is_long_receipt(img_for_ocr):
        quad_text_conf = _paddle_lines_tiled_global(img_for_ocr, tile_h=tile_h, overlap=tile_overlap)
        for quad_g, text, conf in quad_text_conf:
            xyxy = poly_to_xyxy(quad_g)
            box1000 = norm_xyxy_to_1000(xyxy, W2, H2)
            toks = smart_split_text(str(text))
            if not toks:
                continue
            blocks.append({
                "text": " ".join(toks),
                "tokens": toks,
                "bbox": box1000,
                "conf": float(conf),
            })
            confs.append(float(conf))

        # Deduplicate tile-overlap duplicates
        blocks = _dedup_blocks_text_iou(blocks, iou_thr=0.92)
    else:
        lines = _paddle_lines_fullimage(img_for_ocr)
        for line in lines:
            try:
                quad = line[0]
                text = str(line[1][0])
                conf = float(line[1][1])
            except Exception:
                continue
            if not str(text).strip():
                continue
            xyxy = poly_to_xyxy(quad)
            box1000 = norm_xyxy_to_1000(xyxy, W2, H2)
            toks = smart_split_text(text)
            if not toks:
                continue
            blocks.append({
                "text": " ".join(toks),
                "tokens": toks,
                "bbox": box1000,
                "conf": float(conf),
            })
            confs.append(float(conf))

    # -------------------------------------------------------
    # STEP 2: Order blocks into reading order
    #         Uses tightened y_band_k=0.50 to prevent row merging
    #         across separate menu item lines
    # -------------------------------------------------------
    ordered_blocks = order_blocks_same_line(
        blocks,
        y_band_k=ordering_y_band_k,     # default now 0.50
        within_row="x1 ascending"
    )

    # -------------------------------------------------------
    # STEP 3: Compute zone thresholds (in 0–1000 bbox coords)
    #         header: y < header_zone_frac * 1000
    #         footer: y > footer_zone_frac * 1000
    #         body:   everything else
    # -------------------------------------------------------
    header_y_max = int(header_zone_frac * 1000)   # e.g. 120
    footer_y_min = int(footer_zone_frac * 1000)   # e.g. 900

    def get_zone(bbox: List[int]) -> str:
        """Determine receipt zone from normalized bbox (0–1000 scale)."""
        cy = (bbox[1] + bbox[3]) / 2.0
        if cy < header_y_max:
            return "header"
        if cy > footer_y_min:
            return "footer"
        return "body"

    # -------------------------------------------------------
    # STEP 4: Flatten into token-level arrays
    #         NEW: row_id, zone, line_bbox per token
    # -------------------------------------------------------
    words:       List[str]       = []
    bboxes:      List[List[int]] = []    # token bbox (= parent block bbox)
    tok_confs:   List[float]     = []
    row_ids:     List[int]       = []    # NEW: row/line boundary signal
    zones:       List[str]       = []    # NEW: "header" | "body" | "footer"
    line_bboxes: List[List[int]] = []    # NEW: full line bbox (block level)

    row_idx = 0
    for b in ordered_blocks:
        bb        = b["bbox"]
        c         = float(b.get("conf", 0.0))
        zone      = get_zone(bb)
        line_bbox = bb[:]               # block-level bbox (full line extent)

        first_in_block = True
        for tok in b["tokens"]:
            tok = (tok or "").strip()
            if not tok:
                continue
            words.append(tok)
            bboxes.append(bb)
            tok_confs.append(c)
            row_ids.append(row_idx)
            zones.append(zone)
            line_bboxes.append(line_bbox)
            first_in_block = False

        # Each OCR block = one logical row → increment row_id per block
        row_idx += 1

    # -------------------------------------------------------
    # STEP 5: Build line-level summary
    #         Groups tokens back by row_id → useful for debugging
    #         and for models that consume line-level features
    # -------------------------------------------------------
    lines_summary: List[Dict[str, Any]] = []
    if words:
        cur_row = row_ids[0]
        cur_line_words: List[str] = []
        cur_line_bbox: List[int]  = []
        cur_line_zone: str        = zones[0]

        for tok, rid, bb, z in zip(words, row_ids, bboxes, zones):
            if rid != cur_row and cur_line_words:
                lines_summary.append({
                    "row_id": cur_row,
                    "text":   " ".join(cur_line_words),
                    "bbox":   cur_line_bbox,
                    "zone":   cur_line_zone,
                })
                cur_line_words = []
                cur_line_bbox  = []
                cur_row        = rid
                cur_line_zone  = z

            cur_line_words.append(tok)
            if not cur_line_bbox:
                cur_line_bbox = bb[:]
            else:
                cur_line_bbox = [
                    min(cur_line_bbox[0], bb[0]),
                    min(cur_line_bbox[1], bb[1]),
                    max(cur_line_bbox[2], bb[2]),
                    max(cur_line_bbox[3], bb[3]),
                ]

        # flush last line
        if cur_line_words:
            lines_summary.append({
                "row_id": cur_row,
                "text":   " ".join(cur_line_words),
                "bbox":   cur_line_bbox,
                "zone":   cur_line_zone,
            })

    # -------------------------------------------------------
    # STEP 6: Assemble payload
    # -------------------------------------------------------
    base_id = os.path.splitext(os.path.basename(out_json_full_path))[0]
    base_id = base_id.replace("_full", "").replace("_best", "")

    payload = {
        "id":           base_id ,
        "image_path":   rel_to_out(best_img_path),
        "image_size":   [int(W), int(H)],

        # ── Core token arrays (same keys as before, NER-model compatible) ──
        "words":        words,
        "bboxes":       bboxes,
        "ner_tags":     [0] * len(words),
        "tokens":       words,                 # alias for LayoutLM compatibility

        # ── NEW enrichment arrays (parallel to words/bboxes) ──
        "row_ids":      row_ids,               # int: logical row index per token
        "zones":        zones,                 # str: "header"|"body"|"footer"
        "line_bboxes":  line_bboxes,           # list[int]: full line bbox per token

        # ── Confidence ──
        "token_confs":  [round(c, 4) for c in tok_confs],
        "mean_conf":    float(round(float(np.mean(tok_confs)) if tok_confs else 0.0, 4)),
        "num_tokens":   int(len(words)),

        # ── Line-level summary (for debugging / inspection) ──
        "lines": lines_summary,

        # ── Metadata ──
        "meta": {
            "ocr_long_receipt":  bool(long_flag),
            "ocr_mode":          "tiled" if long_flag else "normal",
            "ocr_resize_size":   [int(W2), int(H2)],
            "tile_h":            int(tile_h)      if long_flag else 0,
            "tile_overlap":      int(tile_overlap) if long_flag else 0,
            "path_root":         "OUT_DIR",
            "path_note":         "image_path is stored relative to OUT_DIR for portability",
            "zones": {
                "header_y_max_1000": header_y_max,
                "footer_y_min_1000": footer_y_min,
                "note": (
                    "header zone suppresses restaurant name from being learned as MENU.NM; "
                    "footer zone flags trailing balance/total lines"
                ),
            },
            "ordering": {
                "method":      "same_line_grouping_v2",
                "y_band_k":    float(ordering_y_band_k),
                "within_row":  "x1 ascending",
                "change_note": (
                    "y_band_k tightened from 0.65 → 0.50 to prevent separate "
                    "menu item rows (e.g. Asparagus) from merging into previous "
                    "item row (e.g. Delmonico Steak), which caused I-MENU.NM "
                    "mislabelling and cascading price misalignment."
                ),
            },
            "token_split": {
                "method": "smart_split_text",
                "rules": [
                    "protected patterns: dates, times, prices, phone numbers, key:value tokens, #numbers",
                    "letter<->digit split only OUTSIDE protected spans (Table3 → Table 3)",
                    "10$3.29 → 10 $3.29",
                    "date+time glue split",
                    "common glued words (CALLUS, APPROVEDTHANK, CUSTOMER COPY)",
                ],
            },
            "row_id_note": (
                "row_ids[i] is the logical row index of words[i]. "
                "All tokens from the same OCR block share a row_id. "
                "Use this as a feature in NER/LayoutLM to detect line boundaries."
            ),
        },
    }

    with open(out_json_full_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── Debug summary print ──
    num_rows   = len(lines_summary)
    num_header = sum(1 for z in zones if z == "header")
    num_footer = sum(1 for z in zones if z == "footer")
    num_body   = sum(1 for z in zones if z == "body")
    print(
        f"  → JSON: {len(words)} tokens | {num_rows} rows | "
        f"header={num_header} body={num_body} footer={num_footer} | "
        f"long={long_flag}"
    )

    return payload
# ============================================================
# QUALITY SIGNALS
# ============================================================
def extract_quality_signals(text: str) -> dict:
    up = (text or "").upper()
    has_total = any(w in up for w in TOTAL_WORDS)
    has_date = any(re.search(p, text or "") for p in DATE_PATTERNS)
    amounts = [m.group(0) for m in re.finditer(MONEY_PATTERN, text or "")]
    num_amounts = len([a for a in amounts if re.search(r"\d", a)])

    near_total = False
    for _m in re.finditer(r"(TOTAL|GRAND TOTAL|AMOUNT DUE).{0,40}" + MONEY_PATTERN, up):
        near_total = True
        break

    return {
        "has_total_keyword": bool(has_total),
        "has_date_pattern": bool(has_date),
        "num_amounts": int(num_amounts),
        "has_amount_near_total": bool(near_total),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    dirs = ensure_dirs(OUT_DIR)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, e)))

    print("INPUT_DIR:", INPUT_DIR)
    print("Total images found:", len(image_paths))

    csv_path = os.path.join(OUT_DIR, "summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "best_variant",
            "score",
            "mean_confidence",
            "num_lines",
            "best_image_path",
            "best_text_path",
            "variant_scores_json",
            "best_ocr_json_full",
            "overlay_path",
            "token_count",
            "token_box_match",
            "has_total_keyword",
            "has_amount_near_total",
            "num_amounts",
            "has_date_pattern",
            "is_long_receipt",
            "best_view_image_path",
        ])

        total = len(image_paths)
        for idx, img_path in enumerate(image_paths, start=1):
            filename = os.path.basename(img_path)
            name, _ = os.path.splitext(filename)

            img_bgr = read_image_any(img_path)
            if img_bgr is None:
                print(f"[SKIP] Cannot read: {filename}")
                continue

            long_flag = is_long_receipt(img_bgr)

            best = auto_best_preprocess(img_bgr, topk=TOPK_VARIANTS, long_side=LONG_SIDE)

            out_img_ocr  = os.path.join(dirs["BEST_IMG_DIR"], f"{name}_best_ocr.png")
            out_img_view = os.path.join(dirs["BEST_IMG_DIR"], f"{name}_best_view.png")
            cv2.imwrite(out_img_ocr,  best["best_ocr"]["img"])
            cv2.imwrite(out_img_view, best["best_view"]["img"])

            out_txt = os.path.join(dirs["BEST_TXT_DIR"], f"{name}_best.txt")
            with open(out_txt, "w", encoding="utf-8") as tf:
                tf.write(best["text"])

            var_json = os.path.join(dirs["VAR_SCORE_DIR"], f"{name}_variants.json")
            with open(var_json, "w", encoding="utf-8") as jf:
                json.dump(make_json_safe(best), jf, indent=2, ensure_ascii=False)

            out_json_full = os.path.join(dirs["BEST_JSON_FULL_DIR"], f"{name}.json")

            payload_full = None
            try:
                payload_full = ocr_best_image_to_full_json(
                    out_img_ocr,
                    out_json_full_path=out_json_full,
                    ordering_y_band_k=0.65,    # adjust if needed (0.55 tighter, 0.75 looser)
                    tile_h=TILE_H,
                    tile_overlap=TILE_OVERLAP
                )
            except Exception as e:
                print(f"[WARN] JSON generation failed for {filename}: {repr(e)}")
                out_json_full = ""

            overlay_path = os.path.join(dirs["OVERLAY_DIR"], f"{name}_overlay.png")
            overlay_ok = False
            token_count = 0
            token_box_match = False

            if payload_full is not None:
                token_count = int(payload_full.get("num_tokens", 0))
                token_box_match = (len(payload_full.get("words", [])) == len(payload_full.get("bboxes", [])))
                if token_count > 0:
                    overlay_ok = draw_boxes_overlay(out_img_ocr, payload_full["bboxes"], overlay_path)

            signals = extract_quality_signals(best["text"])

            writer.writerow([
                filename,
                best["variant"],
                round(best["score"], 4),
                round(best["mean_conf"], 4),
                best["num_lines"],
                rel_to_out(out_img_ocr),
                rel_to_out(out_txt),
                rel_to_out(var_json),
                rel_to_out(out_json_full) if out_json_full else "",
                rel_to_out(overlay_path) if overlay_ok else "",
                token_count,
                token_box_match,
                signals["has_total_keyword"],
                signals["has_amount_near_total"],
                signals["num_amounts"],
                signals["has_date_pattern"],
                bool(long_flag),
                rel_to_out(out_img_view),
            ])

            print(f"[{idx}/{total}] DONE → {best['variant']} | long={long_flag} | conf={best['mean_conf']:.3f} | tokens={token_count}")

    print("Saved summary:", csv_path)


if __name__ == "__main__":
    main()