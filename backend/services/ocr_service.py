# backend/services/ocr_service.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from services import ocr_core


@dataclass
class OCRServiceConfig:
    topk_variants: int = 2
    long_side: int = 1600
    tile_h: int = 1200
    tile_overlap: int = 200
    ordering_y_band_k: float = 0.65


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def _build_dirs(base_out: Path) -> Dict[str, Path]:
    d = {
        "BEST_IMG_DIR": base_out / "best_images",
        "BEST_TXT_DIR": base_out / "best_texts",
        "BEST_JSON_FULL_DIR": base_out / "best_ocr_json_full",
        "OVERLAY_DIR": base_out / "overlays",
        "VAR_SCORE_DIR": base_out / "variant_scores",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _safe_rel(base_out: Path, p: Path | str | None) -> str:
    if not p:
        return ""
    p = Path(p).resolve()
    try:
        return p.relative_to(base_out.resolve()).as_posix()
    except Exception:
        return p.name


def _write_summary_csv_header(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
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


def _append_summary_row(csv_path: Path, row: List[Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_ocr_for_image(
    image_path: str | Path,
    output_dir: str | Path,
    cfg: OCRServiceConfig | None = None,
) -> Dict[str, Any]:
    """
    Run your existing OCR logic on a single image.

    Output structure:
        output_dir/
            best_images/
            best_texts/
            best_ocr_json_full/
            overlays/
            variant_scores/
            summary.csv
    """
    cfg = cfg or OCRServiceConfig()

    image_path = Path(image_path)
    output_dir = ensure_dir(output_dir)
    dirs = _build_dirs(output_dir)
    csv_path = output_dir / "summary.csv"

    if not csv_path.exists():
        _write_summary_csv_header(csv_path)

    img_bgr = ocr_core.read_image_any(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    long_flag = ocr_core.is_long_receipt(img_bgr)

    best = ocr_core.auto_best_preprocess(
        img_bgr,
        topk=cfg.topk_variants,
        long_side=cfg.long_side,
    )

    name = image_path.stem

    out_img_ocr = dirs["BEST_IMG_DIR"] / f"{name}_best_ocr.png"
    out_img_view = dirs["BEST_IMG_DIR"] / f"{name}_best_view.png"
    cv2.imwrite(str(out_img_ocr), best["best_ocr"]["img"])
    cv2.imwrite(str(out_img_view), best["best_view"]["img"])

    out_txt = dirs["BEST_TXT_DIR"] / f"{name}_best.txt"
    out_txt.write_text(best["text"], encoding="utf-8")

    var_json = dirs["VAR_SCORE_DIR"] / f"{name}_variants.json"
    with var_json.open("w", encoding="utf-8") as jf:
        json.dump(ocr_core.make_json_safe(best), jf, indent=2, ensure_ascii=False)

    out_json_full = dirs["BEST_JSON_FULL_DIR"] / f"{name}.json"
    payload_full: Optional[Dict[str, Any]] = None

    try:
        payload_full = ocr_core.ocr_best_image_to_full_json(
            str(out_img_ocr),
            out_json_full_path=str(out_json_full),
            ordering_y_band_k=cfg.ordering_y_band_k,
            tile_h=cfg.tile_h,
            tile_overlap=cfg.tile_overlap,
        )
    except Exception as e:
        payload_full = None
        out_json_full = Path("")

    overlay_path = dirs["OVERLAY_DIR"] / f"{name}_overlay.png"
    overlay_ok = False
    token_count = 0
    token_box_match = False

    if payload_full is not None:
        token_count = int(payload_full.get("num_tokens", 0))
        token_box_match = (len(payload_full.get("words", [])) == len(payload_full.get("bboxes", [])))
        if token_count > 0:
            overlay_ok = ocr_core.draw_boxes_overlay(
                str(out_img_ocr),
                payload_full["bboxes"],
                str(overlay_path),
            )

    signals = ocr_core.extract_quality_signals(best["text"])

    _append_summary_row(
        csv_path,
        [
            image_path.name,
            best["variant"],
            round(best["score"], 4),
            round(best["mean_conf"], 4),
            best["num_lines"],
            _safe_rel(output_dir, out_img_ocr),
            _safe_rel(output_dir, out_txt),
            _safe_rel(output_dir, var_json),
            _safe_rel(output_dir, out_json_full) if str(out_json_full) else "",
            _safe_rel(output_dir, overlay_path) if overlay_ok else "",
            token_count,
            token_box_match,
            signals["has_total_keyword"],
            signals["has_amount_near_total"],
            signals["num_amounts"],
            signals["has_date_pattern"],
            bool(long_flag),
            _safe_rel(output_dir, out_img_view),
        ],
    )

    return {
        "success": True,
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "is_long_receipt": bool(long_flag),
        "best_variant": best["variant"],
        "score": float(best["score"]),
        "mean_confidence": float(best["mean_conf"]),
        "num_lines": int(best["num_lines"]),
        "best_text": best["text"],
        "best_image_path": str(out_img_ocr.resolve()),
        "best_view_image_path": str(out_img_view.resolve()),
        "best_text_path": str(out_txt.resolve()),
        "variant_scores_json": str(var_json.resolve()),
        "best_ocr_json_full": str(out_json_full.resolve()) if str(out_json_full) else "",
        "overlay_path": str(overlay_path.resolve()) if overlay_ok else "",
        "token_count": int(token_count),
        "token_box_match": bool(token_box_match),
        "quality_signals": signals,
    }


def run_ocr_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: OCRServiceConfig | None = None,
) -> Dict[str, Any]:
    """
    Run OCR for all images in a folder.
    This is the function your receipt pipeline should call.
    """
    cfg = cfg or OCRServiceConfig()

    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for img_path in image_paths:
        per_image_out = ensure_dir(output_dir / img_path.stem)
        try:
            result = run_ocr_for_image(
                image_path=img_path,
                output_dir=per_image_out,
                cfg=cfg,
            )
            results.append(result)
        except Exception as e:
            failed.append({
                "image_name": img_path.name,
                "image_path": str(img_path),
                "error": f"{type(e).__name__}: {e}",
            })

    return {
        "success": len(results) > 0,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_images": len(image_paths),
        "success_count": len(results),
        "failed_count": len(failed),
        "results": results,
        "failed": failed,
    }


def run_ocr_on_files(
    image_paths: List[str | Path],
    output_dir: str | Path,
    cfg: OCRServiceConfig | None = None,
) -> Dict[str, Any]:
    """
    Optional helper when pipeline already has a file list.
    """
    cfg = cfg or OCRServiceConfig()
    output_dir = ensure_dir(output_dir)

    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for p in image_paths:
        img_path = Path(p)
        per_image_out = ensure_dir(output_dir / img_path.stem)
        try:
            result = run_ocr_for_image(
                image_path=img_path,
                output_dir=per_image_out,
                cfg=cfg,
            )
            results.append(result)
        except Exception as e:
            failed.append({
                "image_name": img_path.name,
                "image_path": str(img_path),
                "error": f"{type(e).__name__}: {e}",
            })

    return {
        "success": len(results) > 0,
        "output_dir": str(output_dir.resolve()),
        "total_images": len(image_paths),
        "success_count": len(results),
        "failed_count": len(failed),
        "results": results,
        "failed": failed,
    }