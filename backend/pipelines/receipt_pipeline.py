from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from config import DB_PATH, RUN_MODE
from db.sqlite_writer import init_sqlite_schema, save_receipt_payload
from services.yolo_service import (
    YoloSegConfig,
    RectifyConfig,
    run_yolo_and_rectify,
)
from services.ocr_service import run_ocr_batch, OCRServiceConfig
from services.extraction_service import (
    run_extraction_batch,
    ExtractionServiceConfig,
)
from llm.llm_gemini import LLMRouterConfig, run_auto_fix_for_receipt

logger = logging.getLogger(__name__)


# =========================================================
# Config
# =========================================================
@dataclass
class ReceiptPipelineConfig:
    base_dir: Path

    raw_upload_dir: Path
    processed_dir: Path

    yolo_model_path: Path
    yolo_device: str = "cpu"
    yolo_imgsz: int = 1024
    yolo_conf: float = 0.25
    yolo_iou: float = 0.70
    yolo_save_masks: bool = False

    close_kernel: tuple[int, int] = (9, 9)
    close_iters: int = 2
    approx_eps_frac: float = 0.02
    min_output_side: int = 2

    ocr_device: str = "cpu"
    cord_model_path: Optional[Path] = None
    sroie_model_path: Optional[Path] = None


@dataclass
class StepResult:
    success: bool
    name: str
    elapsed_sec: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    success: bool
    job_id: str
    raw_dir: str
    image_count: int
    steps: Dict[str, StepResult]
    final_output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "raw_dir": self.raw_dir,
            "image_count": self.image_count,
            "error": self.error,
            "steps": {
                k: {
                    "success": v.success,
                    "name": v.name,
                    "elapsed_sec": round(v.elapsed_sec, 4),
                    "data": v.data,
                    "error": v.error,
                }
                for k, v in self.steps.items()
            },
            "final_output": self.final_output,
        }


# =========================================================
# Helpers
# =========================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


# =========================================================
# Main pipeline
# =========================================================
class ReceiptPipeline:
    def __init__(self, config: ReceiptPipelineConfig):
        self.config = config

    def run(self, job_id: str) -> PipelineResult:
        raw_dir = self.config.raw_upload_dir / job_id
        out_dir = self.config.processed_dir / job_id

        yolo_dir = out_dir / "yolo"
        ocr_dir = out_dir / "ocr"
        extraction_dir = out_dir / "extraction"

        steps: Dict[str, StepResult] = {}

        try:
            if not raw_dir.exists():
                raise FileNotFoundError(f"Raw upload folder not found: {raw_dir}")

            image_paths = list_images(raw_dir)
            if not image_paths:
                raise FileNotFoundError(f"No images found in: {raw_dir}")

            if len(image_paths) > 10:
                raise ValueError(
                    f"Too many images in job {job_id}. Max allowed is 10, found {len(image_paths)}"
                )

            # ── Step 1: YOLO + Rectify ────────────────────────────────────────
            step = self._run_step(
                "yolo",
                self._run_yolo,
                {"job_id": job_id, "raw_dir": raw_dir, "yolo_dir": yolo_dir},
            )
            steps["yolo"] = step
            if not step.success:
                raise RuntimeError(step.error or "YOLO step failed")

            rectified_dir = Path(step.data["rectified_dir"])
            rectified_files = step.data["rectified_files"]

            # ── Step 2: OCR ───────────────────────────────────────────────────
            step = self._run_step(
                "ocr",
                self._run_ocr,
                {
                    "job_id": job_id,
                    "rectified_dir": rectified_dir,
                    "rectified_files": rectified_files,
                    "ocr_dir": ocr_dir,
                },
            )
            steps["ocr"] = step
            if not step.success:
                raise RuntimeError(step.error or "OCR step failed")

            ocr_json_dir = step.data["ocr_json_dir"]
            ocr_results = step.data.get("ocr_results", [])

            # ── Step 3: Extraction / Gemini LLM / Local fallback ─────────────
            step = self._run_step(
                "extraction",
                self._run_extraction,
                {
                    "job_id": job_id,
                    "ocr_dir": ocr_dir,
                    "extraction_dir": extraction_dir,
                    "ocr_json_dir": ocr_json_dir,
                    "ocr_results": ocr_results,
                },
            )
            steps["extraction"] = step
            if not step.success:
                raise RuntimeError(step.error or "Extraction step failed")

            # ── Step 4: Save to SQLite ────────────────────────────────────────
            step = self._run_step(
                "sqlite",
                self._save_to_sqlite,
                {
                    "job_id": job_id,
                    "ocr_results": steps["ocr"].data.get("ocr_results", []),
                    "extraction_results": steps["extraction"].data.get(
                        "extraction_results", {}
                    ),
                },
            )
            steps["sqlite"] = step
            if not step.success:
                raise RuntimeError(step.error or "SQLite save step failed")

            receipt_ids: List[int] = step.data.get("receipt_ids", [])
            saved_receipts: List[Dict[str, Any]] = step.data.get("saved_receipts", [])

            final_output = {
                "receipt_id": receipt_ids[0] if receipt_ids else None,
                "receipt_ids": receipt_ids,
                "saved_receipts": saved_receipts,
                "saved_count": step.data.get("saved_count", 0),
                "job_id": job_id,
                "raw_dir": str(raw_dir.resolve()),
                "processed_dir": str(out_dir.resolve()),
                "yolo_dir": str(yolo_dir.resolve()),
                "ocr_dir": str(ocr_dir.resolve()),
                "extraction_dir": str(extraction_dir.resolve()),
                "rectified_dir": str(rectified_dir.resolve()),
                "rectified_files": rectified_files,
                "ocr_results": steps["ocr"].data.get("ocr_results", []),
                "extraction_results": steps["extraction"].data.get(
                    "extraction_results", {}
                ),
            }

            return PipelineResult(
                success=True,
                job_id=job_id,
                raw_dir=str(raw_dir.resolve()),
                image_count=len(image_paths),
                steps=steps,
                final_output=final_output,
            )

        except Exception as e:
            logger.exception("Receipt pipeline failed")
            return PipelineResult(
                success=False,
                job_id=job_id,
                raw_dir=str(raw_dir.resolve()) if raw_dir.exists() else str(raw_dir),
                image_count=len(list_images(raw_dir)) if raw_dir.exists() else 0,
                steps=steps,
                error=f"{type(e).__name__}: {e}",
                final_output={},
            )

    def _run_step(self, name: str, fn, ctx: Dict[str, Any]) -> StepResult:
        start = perf_counter()
        try:
            data = fn(ctx)
            return StepResult(
                success=True,
                name=name,
                elapsed_sec=perf_counter() - start,
                data=data or {},
            )
        except Exception as e:
            logger.exception("Step failed: %s", name)
            return StepResult(
                success=False,
                name=name,
                elapsed_sec=perf_counter() - start,
                data={},
                error=f"{type(e).__name__}: {e}",
            )

    # =====================================================
    # Gemini LLM helpers
    # =====================================================
    def _should_use_llm(self, ocr_item: Dict[str, Any]) -> bool:
        run_mode = str(RUN_MODE).strip().upper()

        print(
            f"_should_use_llm -> RUN_MODE={run_mode}, "
            f"is_long_receipt={bool(ocr_item.get('is_long_receipt', False))}, "
            f"image={ocr_item.get('image_name', '')}"
        )

        return run_mode == "DEVELOPMENT"

    def _build_geo_result_from_llm(
        self,
        *,
        llm_parsed: Dict[str, Any],
        ocr_item: Dict[str, Any],
        extraction_source: str = "gemini_llm",
    ) -> Dict[str, Any]:
        image_stem = Path(ocr_item.get("image_name") or "").stem
        json_stem = Path(ocr_item.get("ocr_json_path") or "").stem

        base_id = image_stem or json_stem or "UNKNOWN"
        source_id = f"{ocr_item.get('job_id', 'JOB')}_{base_id}"
        receipt_name = base_id

        def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
            if value is None or value == "":
                return default
            try:
                return float(value)
            except Exception:
                return default

        items = llm_parsed.get("items") or []
        geo_items = []

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or item.get("item_name") or "").strip()
            price = _to_float(
                item.get("price") or item.get("line_price") or item.get("line_total"),
                0.0,
            )

            conf = item.get("confidence")
            if conf is None or conf == "":
                conf = 1.0
            else:
                conf = _to_float(conf, 1.0)

            geo_items.append(
                {
                    "line_no": idx,
                    "item_name": name,
                    "name": name,
                    "price": price,
                    "unit_price": price,
                    "line_price": price,
                    "line_total": price,
                    "confidence": conf,
                }
            )

        currency = str(llm_parsed.get("currency") or "CAD").strip() or "CAD"

        geo_result = {
            "source_id": source_id,
            "id": source_id,
            "receipt_name": receipt_name,
            "vendor": llm_parsed.get("vendor"),
            "phone": llm_parsed.get("phone"),
            "address": llm_parsed.get("address"),
            "receipt_date": llm_parsed.get("date"),
            "date": llm_parsed.get("date"),
            "subtotal": _to_float(llm_parsed.get("subtotal")),
            "tax": _to_float(llm_parsed.get("tax")),
            "total": _to_float(llm_parsed.get("total")),
            "currency": currency,
            "confidence": _to_float(llm_parsed.get("confidence"), 1.0),
            "items": geo_items,
            "receipt_items": geo_items,
            "ocr_txt_path": ocr_item.get("ocr_txt_path", ""),
            "ocr_json_path": ocr_item.get("ocr_json_path", ""),
            "source_image_path": ocr_item.get("image_path", ""),
            "image_path": ocr_item.get("image_path", ""),
            "sroie_fields": {
                "vendor": llm_parsed.get("vendor"),
                "phone": llm_parsed.get("phone"),
                "address": llm_parsed.get("address"),
                "date": llm_parsed.get("date"),
                "subtotal": _to_float(llm_parsed.get("subtotal")),
                "tax": _to_float(llm_parsed.get("tax")),
                "total": _to_float(llm_parsed.get("total")),
                "currency": currency,
            },
            "extraction_source": extraction_source,
        }

        return geo_result

    def _run_gemini_for_item(self, ocr_item: Dict[str, Any]) -> Dict[str, Any]:
        ocr_txt_path = str(ocr_item.get("ocr_txt_path") or "").strip()
        image_path = str(ocr_item.get("image_path") or "").strip()
        is_long = bool(ocr_item.get("is_long_receipt", False))

        force_route = "vision_image" if is_long else "ocr_text"

        ocr_text: Optional[str] = None
        final_image_path: Optional[str] = None
        ocr_confidence = ocr_item.get("mean_confidence")

        if is_long:
            if not image_path:
                return {
                    "success": False,
                    "error": "Long receipt route selected but image_path is missing",
                    "route_used": force_route,
                    "model_used": "gemini-2.5-flash-lite",
                    "raw_response_text": "",
                    "full_response": {},
                }

            final_image_path = image_path
            logger.info(
                "Gemini route = vision_image | image=%s | image_path=%s",
                ocr_item.get("image_name", ""),
                image_path,
            )
        else:
            if not ocr_txt_path:
                return {
                    "success": False,
                    "error": "Short receipt route selected but ocr_txt_path is missing",
                    "route_used": force_route,
                    "model_used": "gemini-2.5-flash-lite",
                    "raw_response_text": "",
                    "full_response": {},
                }

            try:
                ocr_text = Path(ocr_txt_path).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read OCR text file: {ocr_txt_path} | {e}",
                    "route_used": force_route,
                    "model_used": "gemini-2.5-flash-lite",
                    "raw_response_text": "",
                    "full_response": {},
                }

            if not ocr_text.strip():
                return {
                    "success": False,
                    "error": f"OCR text file is empty: {ocr_txt_path}",
                    "route_used": force_route,
                    "model_used": "gemini-2.5-flash-lite",
                    "raw_response_text": "",
                    "full_response": {},
                }

            logger.info(
                "Gemini route = ocr_text | image=%s | ocr_txt_path=%s",
                ocr_item.get("image_name", ""),
                ocr_txt_path,
            )

        cfg = LLMRouterConfig(
            free_tier_mode=True,
            allow_vision_fallback=False,
            temperature=0.0,
        )

        llm_result = run_auto_fix_for_receipt(
            ocr_text=ocr_text,
            image_path=final_image_path,
            ocr_confidence=ocr_confidence,
            config=cfg,
            force_route=force_route,
        )

        parsed_output = llm_result.get("parsed_output") or {}
        if not parsed_output:
            return {
                "success": False,
                "error": llm_result.get("error")
                or f"Gemini returned empty parsed_output | route={force_route}",
                "route_used": llm_result.get("route_used", force_route),
                "model_used": llm_result.get("model_used", "gemini-2.5-flash-lite"),
                "prompt_used": llm_result.get("prompt_used", ""),
                "raw_response_text": llm_result.get("raw_response_text", ""),
                "full_response": llm_result,
            }

        logger.info(
            "Gemini success | image=%s | route=%s | vendor=%s | total=%s | item_count=%s",
            ocr_item.get("image_name", ""),
            llm_result.get("route_used"),
            parsed_output.get("vendor"),
            parsed_output.get("total"),
            len(parsed_output.get("items") or []),
        )

        print("GEMINI PARSED OUTPUT:")
        print(json.dumps(parsed_output, indent=2, ensure_ascii=False))

        print("GEMINI VALIDATION:")
        print(json.dumps(llm_result.get("validation"), indent=2, ensure_ascii=False))

        geo_result = self._build_geo_result_from_llm(
            llm_parsed=parsed_output,
            ocr_item=ocr_item,
            extraction_source="gemini_llm",
        )

        return {
            "success": True,
            "geo_result": geo_result,
            "validation_report": llm_result.get("validation"),
            "summary_report": {
                "fields": geo_result.get("sroie_fields", {}),
                "items": geo_result.get("items", []),
            },
            "route_used": llm_result.get("route_used", force_route),
            "model_used": llm_result.get("model_used", "gemini-2.5-flash-lite"),
            "prompt_used": llm_result.get("prompt_used", ""),
            "raw_response_text": llm_result.get("raw_response_text", ""),
            "error": llm_result.get("error"),
            "full_response": llm_result,
        }

    def _run_local_extraction_for_item(
        self,
        *,
        item: Dict[str, Any],
        ocr_dir: Path,
        extraction_dir: Path,
    ) -> Dict[str, Any]:
        image_name = item.get("image_name", "")
        ocr_json_path = item.get("ocr_json_path")

        if not ocr_json_path:
            raise RuntimeError(f"Missing ocr_json_path for {image_name}")

        json_path = Path(ocr_json_path)
        if not json_path.exists():
            raise RuntimeError(f"OCR JSON file not found: {json_path}")

        per_file_json_dir = json_path.parent
        per_file_output_dir = ensure_dir(extraction_dir / json_path.stem)

        print(f"Calling run_extraction_batch for {image_name}")

        result = run_extraction_batch(
            json_dir=per_file_json_dir,
            dataset_root=ocr_dir,
            output_dir=per_file_output_dir,
            cfg=ExtractionServiceConfig(
                pattern=json_path.name,
                max_files=1,
                write_out_json=True,
                summary_report=True,
                verbose=True,
                min_confidence_filter=0.70,
                min_label_confidence=0.50,
                fuzzy_enabled=True,
                debug=False,
            ),
        )

        print(f"run_extraction_batch returned type={type(result)} for {image_name}")

        if not isinstance(result, dict):
            raise RuntimeError("Extraction returned invalid format")

        return result

    # ── Step 1: YOLO ─────────────────────────────────────────────────────────
    def _run_yolo(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raw_dir: Path = Path(ctx["raw_dir"])
        yolo_dir: Path = ensure_dir(ctx["yolo_dir"])

        if not raw_dir.exists():
            raise RuntimeError(f"Raw input directory not found: {raw_dir}")

        yolo_cfg = YoloSegConfig(
            model_path=str(self.config.yolo_model_path),
            source=str(raw_dir),
            run_dir=str(yolo_dir),
            predict_name="predict",
            imgsz=self.config.yolo_imgsz,
            conf=self.config.yolo_conf,
            iou=self.config.yolo_iou,
            device=self.config.yolo_device,
            save_txt=True,
            save=True,
            save_crop=False,
            save_masks=self.config.yolo_save_masks,
        )

        rectify_cfg = RectifyConfig(
            close_kernel=self.config.close_kernel,
            close_iters=self.config.close_iters,
            approx_eps_frac=self.config.approx_eps_frac,
            min_output_side=self.config.min_output_side,
        )

        result = run_yolo_and_rectify(yolo_cfg, rectify_cfg)

        if not result.success:
            raise RuntimeError(result.error or "run_yolo_and_rectify failed")

        rectified_dir = Path(result.rectified_dir)
        rectified_files = [str(Path(p).resolve()) for p in result.rectified_files]

        if not rectified_files:
            raise RuntimeError("YOLO step produced no rectified files.")

        return {
            "raw_dir": str(raw_dir.resolve()),
            "yolo_dir": str(yolo_dir.resolve()),
            "run_dir": str(Path(result.run_dir).resolve()),
            "predict_dir": str(Path(result.predict_dir).resolve()),
            "labels_dir": str(Path(result.labels_dir).resolve()),
            "masks_dir": str(Path(result.masks_dir).resolve()),
            "rectified_dir": str(rectified_dir.resolve()),
            "rectified_files": rectified_files,
        }

    # ── Step 2: OCR ──────────────────────────────────────────────────────────
    def _run_ocr(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        rectified_dir: Path = Path(ctx["rectified_dir"])
        rectified_files: list = ctx.get("rectified_files", [])
        ocr_dir: Path = ensure_dir(ctx["ocr_dir"])

        if not rectified_files:
            raise RuntimeError("No rectified images available for OCR.")

        ocr_batch = run_ocr_batch(
            input_dir=rectified_dir,
            output_dir=ocr_dir,
            cfg=OCRServiceConfig(
                topk_variants=2,
                long_side=1600,
                tile_h=1200,
                tile_overlap=200,
                ordering_y_band_k=0.65,
            ),
        )

        if not ocr_batch.get("success"):
            raise RuntimeError(
                f"OCR failed for all images. Failed: {ocr_batch.get('failed_count', 0)}"
            )

        per_image_results = ocr_batch.get("results", [])
        if not per_image_results:
            raise RuntimeError("OCR returned no successful results.")

        normalized_results = []

        for item in per_image_results:
            image_path = Path(item["image_path"])
            stem = image_path.stem
            per_image_dir = ocr_dir / stem

            src_json_str = item.get("best_ocr_json_full", "")
            src_json = (
                Path(src_json_str)
                if src_json_str
                else per_image_dir / "best_ocr_json_full" / f"{stem}.json"
            )

            src_txt_str = item.get("best_text_path", "")
            src_txt = (
                Path(src_txt_str)
                if src_txt_str
                else per_image_dir / "best_texts" / f"{stem}_best.txt"
            )

            if not src_json.exists():
                raise RuntimeError(
                    f"OCR JSON file does not exist: {src_json}\n"
                    f"Looked in: {per_image_dir / 'best_ocr_json_full'}\n"
                    f"Available keys from service: {list(item.keys())}"
                )

            ocr_long_receipt = False
            ocr_meta: Dict[str, Any] = {}
            try:
                with src_json.open("r", encoding="utf-8") as f:
                    ocr_json_obj = json.load(f)

                ocr_meta = ocr_json_obj.get("meta") or {}
                ocr_long_receipt = bool(ocr_meta.get("ocr_long_receipt", False))
            except Exception as e:
                logger.warning(
                    "Failed to read OCR meta from %s: %s",
                    src_json,
                    e,
                )
                ocr_long_receipt = False
                ocr_meta = {}

            logger.info(
                "OCR meta route check | image=%s | ocr_long_receipt=%s",
                image_path.name,
                ocr_long_receipt,
            )

            normalized_results.append(
                {
                    "job_id": ctx.get("job_id", ""),
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "ocr_json_path": str(src_json.resolve()),
                    "ocr_txt_path": str(src_txt.resolve()) if src_txt.exists() else "",
                    "best_variant": item.get("best_variant", ""),
                    "score": item.get("score", 0.0),
                    "mean_confidence": item.get("mean_confidence", 0.0),
                    "num_lines": item.get("num_lines", 0),
                    "token_count": item.get("token_count", 0),
                    "is_long_receipt": ocr_long_receipt,
                    "ocr_meta": ocr_meta,
                    "quality_signals": item.get("quality_signals", {}),
                }
            )

        primary_json_dir = (
            Path(normalized_results[0]["ocr_json_path"]).parent
            if normalized_results
            else ocr_dir
        )

        return {
            "ocr_dir": str(ocr_dir.resolve()),
            "ocr_input_dir": str(rectified_dir.resolve()),
            "ocr_json_dir": str(primary_json_dir.resolve()),
            "ocr_results": normalized_results,
            "ocr_failed": ocr_batch.get("failed", []),
            "ocr_total_images": ocr_batch.get("total_images", 0),
            "ocr_success_count": ocr_batch.get("success_count", 0),
            "ocr_failed_count": ocr_batch.get("failed_count", 0),
        }

    # ── Step 3: Extraction / Gemini LLM / Local fallback ────────────────────
    def _run_extraction(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ocr_dir: Path = Path(ctx["ocr_dir"])
        extraction_dir: Path = ensure_dir(ctx["extraction_dir"])
        ocr_results: List[Dict[str, Any]] = ctx.get("ocr_results", [])

        print(f"_run_extraction entered | ocr_results_count={len(ocr_results)}")

        if not ocr_results:
            raise RuntimeError("No OCR results available for extraction.")

        all_results: List[Dict[str, Any]] = []
        all_failed: List[Dict[str, Any]] = []

        for item in ocr_results:
            image_name = item.get("image_name", "")
            is_long = item.get("is_long_receipt", False)

            print(
                f"_run_extraction loop | image={image_name} | "
                f"is_long_receipt={is_long} | "
                f"ocr_json_path={item.get('ocr_json_path', '')} | "
                f"ocr_txt_path={item.get('ocr_txt_path', '')}"
            )

            used_local_fallback = False

            try:
                if self._should_use_llm(item):
                    route_name = "vision_image" if is_long else "ocr_text"
                    print(f"ROUTE => GEMINI_LLM ({route_name}) | image={image_name}")
                    logger.info(
                        "Extraction route = gemini_llm | image=%s | is_long_receipt=%s | route=%s | ocr_txt_path=%s | image_path=%s",
                        image_name,
                        is_long,
                        route_name,
                        item.get("ocr_txt_path", ""),
                        item.get("image_path", ""),
                    )

                    gemini_result = self._run_gemini_for_item(item)

                    print("GEMINI RESULT:")
                    print(json.dumps(gemini_result, indent=2, ensure_ascii=False))

                    if gemini_result.get("success"):
                        all_results.append(gemini_result)
                        continue

                    used_local_fallback = True
                    logger.warning(
                        "Gemini failed for %s | route=%s | error=%s | falling back to local extraction",
                        image_name,
                        gemini_result.get("route_used"),
                        gemini_result.get("error"),
                    )
                    print(
                        f"GEMINI FAILED => fallback to EXTRACTION_SERVICE | image={image_name} | "
                        f"error={gemini_result.get('error')}"
                    )

                print(f"ROUTE => EXTRACTION_SERVICE | image={image_name}")
                logger.info(
                    "Extraction route = extraction_service | image=%s | is_long_receipt=%s | ocr_json_path=%s",
                    image_name,
                    is_long,
                    item.get("ocr_json_path", ""),
                )

                result = self._run_local_extraction_for_item(
                    item=item,
                    ocr_dir=ocr_dir,
                    extraction_dir=extraction_dir,
                )

                if isinstance(result, dict):
                    local_results = result.get("results", []) or []
                    local_failed = result.get("failed", []) or []

                    all_results.extend(local_results)

                    if local_failed:
                        all_failed.extend(local_failed)
                    elif not local_results:
                        all_failed.append(
                            {
                                "image_name": image_name,
                                "error": "Local extraction returned no results",
                                "fallback_from_llm": used_local_fallback,
                            }
                        )
                else:
                    all_failed.append(
                        {
                            "image_name": image_name,
                            "error": "Extraction returned invalid format",
                            "fallback_from_llm": used_local_fallback,
                        }
                    )

            except Exception as e:
                print(
                    f"Extraction route failed for {image_name}: {type(e).__name__}: {e}"
                )
                logger.exception("Extraction route failed for %s", image_name)
                all_failed.append(
                    {
                        "image_name": image_name,
                        "error": f"{type(e).__name__}: {e}",
                        "fallback_from_llm": used_local_fallback,
                    }
                )

        print(
            f"_run_extraction done | success_count={len(all_results)} | failed_count={len(all_failed)}"
        )

        if not all_results:
            raise RuntimeError(
                f"No successful extraction results found. Failed count={len(all_failed)}"
            )

        extraction_results = {
            "success": len(all_results) > 0,
            "results": all_results,
            "failed": all_failed,
            "success_count": len(all_results),
            "failed_count": len(all_failed),
            "total_files": len(all_results) + len(all_failed),
        }

        return {
            "extraction_dir": str(extraction_dir.resolve()),
            "extraction_results": extraction_results,
        }

    # ── Step 4: Save to SQLite ───────────────────────────────────────────────
    def _save_to_sqlite(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        job_id = str(ctx["job_id"]).strip()
        extraction_results = ctx.get("extraction_results", {})
        ocr_results = ctx.get("ocr_results", [])

        print(f"_save_to_sqlite: job_id={job_id}")

        if not isinstance(extraction_results, dict):
            raise RuntimeError("Invalid extraction_results format")

        results = extraction_results.get("results", [])
        failed = extraction_results.get("failed", [])

        if not results:
            raise RuntimeError("No successful extraction results found")

        saved_receipts: List[Dict[str, Any]] = []

        ocr_map: Dict[str, Dict[str, Any]] = {}

        for idx, ocr in enumerate(ocr_results, start=1):
            if not isinstance(ocr, dict):
                continue

            image_name = str(ocr.get("image_name") or "").strip()
            image_path = str(ocr.get("image_path") or "").strip()
            ocr_json_path = str(ocr.get("ocr_json_path") or "").strip()
            ocr_txt_path = str(ocr.get("ocr_txt_path") or "").strip()

            candidates = set()

            for value in [image_name, image_path, ocr_json_path, ocr_txt_path]:
                if not value:
                    continue

                raw_value = value.strip().upper()
                if raw_value:
                    candidates.add(raw_value)

                stem = Path(value).stem.strip().upper()
                if stem:
                    candidates.add(stem)

                name = Path(value).name.strip().upper()
                if name:
                    candidates.add(name)

            candidates.add(f"{job_id}_{idx:03d}")

            meta = {
                "image_name": image_name,
                "image_path": image_path,
                "ocr_json_path": ocr_json_path,
                "ocr_txt_path": ocr_txt_path,
            }

            for key in candidates:
                if key and key not in ocr_map:
                    ocr_map[key] = meta

        init_sqlite_schema(DB_PATH)

        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

            for i, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    logger.warning("Skipping invalid extraction result at index %s", i)
                    continue

                geo_out = result.get("geo_result") or {}
                if not isinstance(geo_out, dict) or not geo_out:
                    logger.warning(
                        "Skipping result %s because geo_result is missing", i
                    )
                    continue

                raw_source_id = str(
                    geo_out.get("source_id")
                    or geo_out.get("id")
                    or geo_out.get("receipt_name")
                    or ""
                ).strip()

                source_id = (
                    raw_source_id.upper() if raw_source_id else f"{job_id}_{i:03d}"
                )

                lookup_keys = [
                    source_id,
                    str(geo_out.get("source_id") or "").strip().upper(),
                    str(geo_out.get("id") or "").strip().upper(),
                    str(geo_out.get("receipt_name") or "").strip().upper(),
                ]

                for value in [
                    geo_out.get("image_name"),
                    geo_out.get("image_path"),
                    geo_out.get("source_image_path"),
                    geo_out.get("ocr_json_path"),
                    geo_out.get("ocr_txt_path"),
                ]:
                    s = str(value or "").strip()
                    if s:
                        lookup_keys.append(s.upper())
                        lookup_keys.append(Path(s).stem.strip().upper())
                        lookup_keys.append(Path(s).name.strip().upper())

                ocr_meta = {}
                for key in lookup_keys:
                    if key and key in ocr_map:
                        ocr_meta = ocr_map[key]
                        break

                display_name = (
                    str(geo_out.get("receipt_name") or "").strip()
                    or str(geo_out.get("image_name") or "").strip()
                    or str(geo_out.get("ocr_json_path") or "").strip()
                )

                if display_name:
                    display_name = Path(display_name).stem.strip()

                for suffix in ("_best_view", "_best", "_view"):
                    if display_name.endswith(suffix):
                        display_name = display_name[: -len(suffix)]
                        break
                    else:
                        prefix = f"{job_id}_"
                    if source_id.startswith(prefix):
                        display_name = source_id[len(prefix) :]
                    else:
                        display_name = source_id

                geo_out["job_id"] = job_id
                geo_out["source_id"] = source_id
                geo_out["id"] = geo_out.get("id") or source_id
                geo_out["receipt_name"] = display_name

                if ocr_meta:
                    geo_out["source_image_path"] = ocr_meta.get("image_path", "")
                    geo_out["ocr_json_path"] = ocr_meta.get("ocr_json_path", "")
                    geo_out["ocr_txt_path"] = ocr_meta.get("ocr_txt_path", "")
                    geo_out["image_name"] = ocr_meta.get("image_name", "")

                validation_report = result.get("validation_report")
                summary_report = result.get("summary_report") or {
                    "fields": (
                        geo_out.get("sroie_fields") or geo_out.get("sorie_fields") or {}
                    ),
                    "items": geo_out.get("items") or geo_out.get("receipt_items") or [],
                }

                payload = {
                    "geo_out": geo_out,
                    "validation_report": validation_report,
                    "summary_report": summary_report,
                    "receipt_name_fallback": display_name,
                    "source_id_fallback": source_id,
                    "vendor_fallback": None,
                    "receipt_date_fallback": None,
                }

                receipt_id = save_receipt_payload(
                    conn=conn,
                    payload=payload,
                    default_currency="CAD",
                    min_confidence_filter=0.70,
                )

                saved_receipts.append(
                    {
                        "receipt_id": receipt_id,
                        "receipt_name": geo_out.get("receipt_name", source_id),
                        "source_id": source_id,
                        "job_id": job_id,
                        "ocr_txt_path": geo_out.get("ocr_txt_path", ""),
                        "ocr_json_path": geo_out.get("ocr_json_path", ""),
                        "source_image_path": geo_out.get("source_image_path", ""),
                    }
                )

            conn.commit()

        if not saved_receipts:
            raise RuntimeError("Extraction completed but nothing saved")

        return {
            "job_id": job_id,
            "saved_count": len(saved_receipts),
            "failed_count": len(failed),
            "receipt_ids": [r["receipt_id"] for r in saved_receipts],
            "saved_receipts": saved_receipts,
            "receipt_id": saved_receipts[0]["receipt_id"],
        }
