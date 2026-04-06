# backend/services/extraction_service.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from services import extraction_core


@dataclass
class ExtractionServiceConfig:
    pattern: str = "*.json"
    max_files: Optional[int] = None
    write_out_json: bool = True
    summary_report: bool = True
    verbose: bool = True
    min_confidence_filter: float = 0.70

    # geo config
    min_label_confidence: float = 0.50
    fuzzy_enabled: bool = True
    debug: bool = False


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_json_files(folder: str | Path, pattern: str = "*.json") -> List[Path]:
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    return files


def build_geo_cfg(cfg: ExtractionServiceConfig):
    return extraction_core.create_geo_config(
        min_confidence=cfg.min_label_confidence,
        fuzzy_enabled=cfg.fuzzy_enabled,
        debug=cfg.debug,
    )


def run_extraction_for_single_json(
    json_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    cfg: ExtractionServiceConfig | None = None,
) -> Dict[str, Any]:
    """
    Process one OCR JSON through the CORD + SROIE pipeline.
    dataset_root is the top-level ocr dir; resolve_image_path inside
    extraction_core handles the per-image subdir layout automatically.
    """
    cfg = cfg or ExtractionServiceConfig()

    json_path = Path(json_path)
    dataset_root = Path(dataset_root)
    output_dir = ensure_dir(output_dir)

    geo_cfg = build_geo_cfg(cfg)

    result = extraction_core.process_single_receipt_with_validation(
        json_path=json_path,
        dataset_root=dataset_root,
        cfg=geo_cfg,
        verbose=cfg.verbose,
        use_geo_extraction=True,
        use_sorie_extraction=True,
        min_confidence_filter=cfg.min_confidence_filter,
    )

    if result.get("status") != "complete":
        raise RuntimeError(result.get("error", "Extraction failed"))

    geo_result = result.get("geo_result") or {}
    validation_report = result.get("validation_report") or {}

    out_json_path = output_dir / f"{json_path.stem}_geo_out.json"
    if cfg.write_out_json:
        out_json_path.write_text(
            json.dumps(geo_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "success":             True,
        "json_name":           json_path.name,
        "json_path":           str(json_path.resolve()),
        "receipt_id":          result.get("receipt_id", json_path.stem),
        "status":              result.get("status"),
        "geo_result":          geo_result,
        "geo_merged":          result.get("geo_merged"),
        "validation_report":   validation_report,
        "item_lines":          result.get("item_lines", []),
        "item_lines_clean":    result.get("item_lines_clean", []),
        "item_lines_flagged":  result.get("item_lines_flagged", []),
        "summary_lines":       result.get("summary_lines", []),
        "geo_items_high_conf": result.get("geo_items_high_conf", []),
        "geo_items_low_conf":  result.get("geo_items_low_conf", []),
        "output_json_path":    str(out_json_path.resolve()) if cfg.write_out_json else "",
    }


def run_extraction_batch(
    json_dir: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    cfg: ExtractionServiceConfig | None = None,
) -> Dict[str, Any]:
    """
    Run extraction for all OCR JSON files in json_dir.
    This is the main function receipt_pipeline calls.

    FIX 2: dataset_root is passed straight through to
    process_single_receipt_with_validation. resolve_image_path
    now probes dataset_root/<stem>/best_images/ internally, so
    callers no longer need to adjust the path themselves.
    """
    cfg = cfg or ExtractionServiceConfig()

    json_dir = Path(json_dir)
    dataset_root = Path(dataset_root)
    output_dir = ensure_dir(output_dir)

    json_paths = list_json_files(json_dir, cfg.pattern)
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")

    if cfg.max_files is not None:
        json_paths = json_paths[:cfg.max_files]

    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for jp in json_paths:
        try:
            # Each file gets its own output subdirectory to mirror the
            # per-image layout that run_ocr_batch uses on the input side.
            per_file_out = ensure_dir(output_dir / jp.stem)
            result = run_extraction_for_single_json(
                json_path=jp,
                dataset_root=dataset_root,
                output_dir=per_file_out,
                cfg=cfg,
            )
            results.append(result)
        except Exception as e:
            failed.append(
                {
                    "json_name": jp.name,
                    "json_path": str(jp),
                    "error":     f"{type(e).__name__}: {e}",
                }
            )

    summary = {
        "success":          len(results) > 0,
        "json_dir":         str(json_dir.resolve()),
        "dataset_root":     str(dataset_root.resolve()),
        "output_dir":       str(output_dir.resolve()),
        "total_json_files": len(json_paths),
        "success_count":    len(results),
        "failed_count":     len(failed),
        "results":          results,
        "failed":           failed,
    }

    summary_path = output_dir / "summary_extraction.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary["summary_path"] = str(summary_path.resolve())
    return summary


def run_extraction_batch_to_sqlite(
    json_dir: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    db_path: str | Path,
    cfg: ExtractionServiceConfig | None = None,
) -> Dict[str, Any]:
    """
    Optional wrapper that delegates to extraction_core.run_batch_to_sqlite.
    Use this when you want results written directly to SQLite instead of
    returning them as a dict.
    """
    cfg = cfg or ExtractionServiceConfig()

    json_dir = Path(json_dir)
    dataset_root = Path(dataset_root)
    output_dir = ensure_dir(output_dir)
    db_path = Path(db_path)

    geo_cfg = build_geo_cfg(cfg)

    summary_path = extraction_core.run_batch_to_sqlite(
        json_dir=json_dir,
        out_dir=output_dir,
        db_path=db_path,
        dataset_root=dataset_root,
        pattern=cfg.pattern,
        max_files=cfg.max_files,
        cfg=geo_cfg,
        write_out_json=cfg.write_out_json,
        summary_report=cfg.summary_report,
        verbose=cfg.verbose,
        min_confidence_filter=cfg.min_confidence_filter,
    )

    summary_data: Dict[str, Any] = {}
    sp = Path(summary_path)
    if sp.exists():
        summary_data = json.loads(sp.read_text(encoding="utf-8"))

    return {
        "success":      True,
        "json_dir":     str(json_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "output_dir":   str(output_dir.resolve()),
        "db_path":      str(db_path.resolve()),
        "summary_path": str(sp.resolve()),
        "summary":      summary_data,
    }