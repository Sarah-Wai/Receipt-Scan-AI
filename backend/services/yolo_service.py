from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


# =========================
# Config
# =========================


@dataclass
class YoloSegConfig:
    """
    Config for YOLO segmentation prediction.

    model_path:
        Can be either .pt or .onnx
    source:
        Can be a single image path or a directory
    run_dir:
        Base output folder for this receipt/process run
        Example:
            FinScanAI/backend/outputs/process_runs/receipt_001/yolo
    """

    model_path: str
    source: str
    run_dir: str

    predict_name: str = "predict"
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.70
    device: Union[int, str] = "cpu"

    save_txt: bool = True
    save: bool = True
    save_crop: bool = False
    save_masks: bool = False


@dataclass
class RectifyConfig:
    """Config for mask cleanup + contour → warp."""

    close_kernel: Tuple[int, int] = (9, 9)
    close_iters: int = 2
    approx_eps_frac: float = 0.02
    min_output_side: int = 2


@dataclass
class YoloRunResult:
    success: bool
    source_path: str
    model_path: str
    run_dir: str
    predict_dir: str
    labels_dir: str
    masks_dir: str
    rectified_dir: str
    rectified_files: List[str]
    error: Optional[str] = None


# =========================
# Helpers
# =========================


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


def list_images(images_dir: str | Path) -> List[Path]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []

    image_paths = [p for p in images_dir.iterdir() if is_image_file(p)]
    image_paths.sort()
    return image_paths


def resolve_source_images(source: str | Path) -> List[Path]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_file():
        if not is_image_file(source):
            raise ValueError(f"Source file is not a supported image: {source}")
        return [source]

    if source.is_dir():
        images = list_images(source)
        if not images:
            raise FileNotFoundError(f"No supported images found in directory: {source}")
        return images

    raise ValueError(f"Invalid source: {source}")


# =========================
# YOLO prediction
# =========================


def run_yolo_segmentation(cfg: YoloSegConfig) -> Path:
    """
    Run YOLOv8 segmentation prediction on a single image or directory.

    Returns
    -------
    Path
        predict directory path
    """
    from ultralytics import YOLO

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    source_path = Path(cfg.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    run_dir = ensure_dir(cfg.run_dir)

    model = YOLO(str(model_path))
    model.predict(
        task="segment",
        source=str(source_path),
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        save=cfg.save,
        save_txt=cfg.save_txt,
        save_crop=cfg.save_crop,
        project=str(run_dir),
        name=cfg.predict_name,
        exist_ok=True,
    )

    predict_dir = run_dir / cfg.predict_name
    return predict_dir


# =========================
# YOLO label → mask
# =========================
def yolo_seg_txt_to_mask(
    txt_path: Union[str, Path],
    image_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a YOLO label file into a binary mask for the given image.

    Supports:
    - segmentation format: class x1 y1 x2 y2 x3 y3 ...
    - detection format:    class xc yc w h
    """
    txt_path = Path(txt_path)
    image_path = Path(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing label file: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        nums = list(map(float, parts[1:]))

        # --------------------------------------------------
        # Case 1: detection label -> class xc yc w h
        # --------------------------------------------------
        if len(nums) == 4:
            xc, yc, bw, bh = nums
            x1 = int((xc - bw / 2) * W)
            y1 = int((yc - bh / 2) * H)
            x2 = int((xc + bw / 2) * W)
            y2 = int((yc + bh / 2) * H)

            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            if x2 > x1 and y2 > y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255
            continue

        # --------------------------------------------------
        # Case 2: segmentation polygon label
        # --------------------------------------------------
        if len(nums) >= 6 and len(nums) % 2 == 0:
            pts = np.array(nums, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            pts = np.clip(pts, [0, 0], [W - 1, H - 1]).astype(np.int32)

            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)

    return img, mask


# =========================
# Geometry helpers
# =========================


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_warp(
    image: np.ndarray,
    pts: np.ndarray,
    *,
    min_output_side: int = 2,
) -> np.ndarray:
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = max(min_output_side, int(round(max(wA, wB))))
    maxH = max(min_output_side, int(round(max(hA, hB))))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


def rectify_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: RectifyConfig = RectifyConfig(),
) -> Optional[np.ndarray]:
    """
    Rectify image using the largest contour in a binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.close_kernel)
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iters)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)

    peri = cv2.arcLength(c, True)
    eps = cfg.approx_eps_frac * peri
    approx = cv2.approxPolyDP(c, eps, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        return four_point_warp(image, pts, min_output_side=cfg.min_output_side)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return four_point_warp(image, box, min_output_side=cfg.min_output_side)


# =========================
# Batch rectify
# =========================
def batch_rectify_from_yolo_labels(
    source_images: List[Path],
    labels_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    rectify_cfg: RectifyConfig = RectifyConfig(),
    save_masks: bool = True,
) -> List[Path]:
    """
    Rectify input images using YOLO segmentation labels.

    Fallback:
        If contour rectification fails, crop by mask bounding box.

    Returns
    -------
    List[Path]
        rectified/cropped image file paths
    """
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)

    masks_dir = ensure_dir(out_dir / "masks")
    rectified_dir = ensure_dir(out_dir / "rectified")

    rectified_files: List[Path] = []

    ok, fail, skip = 0, 0, 0

    for img_path in source_images:
        base = img_path.stem
        lbl_path = labels_dir / f"{base}.txt"

        if not lbl_path.exists():
            print(f"[SKIP] Missing label: {lbl_path.name}")
            skip += 1
            continue

        try:
            img, mask = yolo_seg_txt_to_mask(lbl_path, img_path)

            if save_masks:
                cv2.imwrite(str(masks_dir / f"{base}_mask.png"), mask)

            warped = rectify_from_mask(img, mask, cfg=rectify_cfg)

            # --------------------------------------------------
            # Fallback: use mask bounding box crop if no contour
            # --------------------------------------------------
            if warped is None:
                ys, xs = np.where(mask > 0)

                if len(xs) == 0 or len(ys) == 0:
                    print(f"[FAIL] Empty mask: {base}")
                    fail += 1
                    continue

                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())

                # small padding
                pad = 8
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img.shape[1] - 1, x2 + pad)
                y2 = min(img.shape[0] - 1, y2 + pad)

                if x2 <= x1 or y2 <= y1:
                    print(f"[FAIL] Invalid fallback crop: {base}")
                    fail += 1
                    continue

                warped = img[y1 : y2 + 1, x1 : x2 + 1]
                print(f"[WARN] Contour failed, used bbox crop: {base}")

            out_file = rectified_dir / f"{base}.png"
            cv2.imwrite(str(out_file), warped)
            rectified_files.append(out_file)
            ok += 1

        except Exception as e:
            print(f"[ERROR] {base}: {e}")
            fail += 1

    print(f"\nDone. OK={ok} FAIL={fail} SKIP={skip}")
    print("Labels    :", labels_dir)
    print("Masks     :", masks_dir)
    print("Rectified :", rectified_dir)

    return rectified_files


# =========================
# Full pipeline helper
# =========================
def run_yolo_and_rectify(
    yolo_cfg: YoloSegConfig,
    rectify_cfg: RectifyConfig = RectifyConfig(),
) -> YoloRunResult:
    """
    Full step:
        1. run YOLO segmentation
        2. use YOLO labels to generate masks
        3. rectify images
        4. save rectified outputs for OCR

    Returns
    -------
    YoloRunResult
    """
    try:
        source_images = resolve_source_images(yolo_cfg.source)

        run_dir = ensure_dir(yolo_cfg.run_dir)
        predict_dir = run_yolo_segmentation(yolo_cfg)
        labels_dir = predict_dir / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"YOLO labels directory not found: {labels_dir}")

        rectified_files = batch_rectify_from_yolo_labels(
            source_images=source_images,
            labels_dir=labels_dir,
            out_dir=run_dir,
            rectify_cfg=rectify_cfg,
            save_masks=yolo_cfg.save_masks,
        )

        if not rectified_files:
            raise RuntimeError(
                "YOLO rectification produced no output files (no valid contour found)."
            )

        return YoloRunResult(
            success=True,
            source_path=str(Path(yolo_cfg.source).resolve()),
            model_path=str(Path(yolo_cfg.model_path).resolve()),
            run_dir=str(run_dir.resolve()),
            predict_dir=str(predict_dir.resolve()),
            labels_dir=str(labels_dir.resolve()),
            masks_dir=str((run_dir / "masks").resolve()),
            rectified_dir=str((run_dir / "rectified").resolve()),
            rectified_files=[str(p.resolve()) for p in rectified_files],
            error=None,
        )

    except Exception as e:
        run_dir = Path(yolo_cfg.run_dir)
        predict_dir = run_dir / yolo_cfg.predict_name
        return YoloRunResult(
            success=False,
            source_path=str(Path(yolo_cfg.source)),
            model_path=str(Path(yolo_cfg.model_path)),
            run_dir=str(run_dir),
            predict_dir=str(predict_dir),
            labels_dir=str(predict_dir / "labels"),
            masks_dir=str(run_dir / "masks"),
            rectified_dir=str(run_dir / "rectified"),
            rectified_files=[],
            error=f"{type(e).__name__}: {e}",
        )


# =========================
# Example usage (local)
# =========================
if __name__ == "__main__":
    # Example 1: single image
    # MODEL_PATH = "/Users/yourname/FinScanAI/models/yolo/best.onnx"
    # SOURCE = "/Users/yourname/FinScanAI/data/raw_data/my_receipts/receipt1.jpg"
    # RUN_DIR = "/Users/yourname/FinScanAI/backend/outputs/process_runs/receipt_001/yolo"

    # Example 2: folder
    MODEL_PATH = "/Users/yourname/FinScanAI/models/yolo/best.onnx"
    SOURCE = "/Users/yourname/FinScanAI/data/raw_data/my_receipts"
    RUN_DIR = (
        "/Users/yourname/FinScanAI/backend/outputs/process_runs/batch_run_001/yolo"
    )

    yolo_cfg = YoloSegConfig(
        model_path=MODEL_PATH,
        source=SOURCE,
        run_dir=RUN_DIR,
        predict_name="predict",
        imgsz=1024,
        conf=0.25,
        iou=0.70,
        device="cpu",  # use "cpu" for ONNX local, or 0 for GPU with .pt
        save_txt=True,
        save=True,
        save_crop=False,
        save_masks=False,
    )

    rectify_cfg = RectifyConfig(
        close_kernel=(9, 9),
        close_iters=2,
        approx_eps_frac=0.02,
        min_output_side=2,
    )

    result = run_yolo_and_rectify(yolo_cfg, rectify_cfg)

    print("\n=== YOLO RUN RESULT ===")
    print("success       :", result.success)
    print("source_path   :", result.source_path)
    print("model_path    :", result.model_path)
    print("run_dir       :", result.run_dir)
    print("predict_dir   :", result.predict_dir)
    print("labels_dir    :", result.labels_dir)
    print("rectified_dir :", result.rectified_dir)
    print("rectified     :", len(result.rectified_files))
    if result.error:
        print("error         :", result.error)
