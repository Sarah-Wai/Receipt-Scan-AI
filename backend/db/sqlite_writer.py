# sqlite_writer.py
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from config import PROCESSED_UPLOAD_DIR

BEST_TEXT_DIR: Path | None = None  # set this from main


def set_best_text_dir(p: Path) -> None:
    global BEST_TEXT_DIR
    BEST_TEXT_DIR = p


# ============================================================
# OCR text loader
# ============================================================
def load_ocr_text(source_id) -> dict:
    processed_root = PROCESSED_UPLOAD_DIR
    s = "" if source_id is None else str(source_id).strip()

    if not s:
        return {"rows": []}

    source_key = s.upper()
    print(f"Using source key for OCR lookup: {source_key}")

    candidates = []
    raw_path = Path(s)

    # Full raw forms
    candidates.append(source_key)

    # File/path derived forms
    if s:
        candidates.append(raw_path.stem.strip().upper())
        candidates.append(raw_path.name.strip().upper())

    # De-duplicate while keeping order
    seen = set()
    ordered_candidates = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered_candidates.append(c)

    for key in ordered_candidates:
        patterns = [
            f"*/ocr/{key}/best_texts/{key}_best.txt",
            f"*/ocr/{key}/best_texts/*.txt",
            f"*/ocr/*/best_texts/{key}_best.txt",
            f"*/ocr/*/best_texts/*.txt",
        ]

        for pattern in patterns:
            matches = sorted(processed_root.glob(pattern))
            if not matches:
                continue

            chosen = None

            # Prefer exact file stem match
            for p in matches:
                stem_upper = p.stem.strip().upper()
                name_upper = p.name.strip().upper()
                parent_upper = p.parent.parent.name.strip().upper() if p.parent.parent else ""

                if stem_upper in {key, f"{key}_BEST"}:
                    chosen = p
                    break
                if name_upper == f"{key}_BEST.TXT":
                    chosen = p
                    break
                if parent_upper == key:
                    chosen = p
                    break

            p = chosen or matches[0]
            print(f"Looking for OCR text at: {p}")

            text = p.read_text(encoding="utf-8", errors="ignore")
            return {"rows": [{"text": line.strip()} for line in text.splitlines() if line.strip()]}

    print(f"OCR text file not found for {source_key} under: {processed_root}")
    return {"rows": []}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_optional_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _norm_name_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _price_key(p: Any) -> Optional[float]:
    try:
        return None if p is None else round(float(p), 2)
    except Exception:
        return None


def _to_json(obj: Any) -> str:
    return json.dumps(obj if obj is not None else {}, ensure_ascii=False, allow_nan=False, default=str)


def _conf100(v: Any) -> float:
    try:
        if v is None or v == "":
            return 0.0
        fv = float(v)
        if fv <= 1.0:
            return max(0.0, min(100.0, fv * 100.0))
        return max(0.0, min(100.0, fv))
    except Exception:
        return 0.0


def build_outlier_keyset(validation_report: Dict[str, Any]) -> set[tuple[str, float]]:
    outliers = validation_report.get("price_outliers", {}).get("outliers", []) or []
    keys: set[tuple[str, float]] = set()

    for o in outliers:
        name = _norm_name_key(o.get("item_name") or o.get("name") or "")
        pk = _price_key(o.get("price"))
        if name and pk is not None:
            keys.add((name, pk))

    return keys


# ============================================================
# SQLite schema
# ============================================================
def init_sqlite_schema(db_path: Path | str) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    ddl = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS receipts (
      receipt_id         INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id             TEXT NOT NULL,
      source_id          TEXT UNIQUE,
      receipt_name       TEXT NOT NULL,
      vendor             TEXT,
      receipt_date       TEXT,
      subtotal           REAL NOT NULL DEFAULT 0,
      tax                REAL NOT NULL DEFAULT 0,
      total              REAL NOT NULL DEFAULT 0,
      status             TEXT NOT NULL DEFAULT 'Pending'
                         CHECK (status IN ('Pending','Processed','Failed','Error')),
      confidence         REAL NOT NULL DEFAULT 0
                         CHECK (confidence >= 0 AND confidence <= 100),

      extraction_source  TEXT NOT NULL DEFAULT 'ocr'
                         CHECK (extraction_source IN ('ocr','ocr+llm','manual')),
      ai_review_status   TEXT NOT NULL DEFAULT 'not_requested'
                         CHECK (ai_review_status IN ('not_requested','pending','approved','rejected')),
      active_llm_run_id  INTEGER,

      raw_json           TEXT,
      ocr_json           TEXT,
      predictionlog_json TEXT,
      created_at         TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(receipt_date);
    CREATE INDEX IF NOT EXISTS idx_receipts_job_id ON receipts(job_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_receipts_source_id ON receipts(source_id);
    CREATE INDEX IF NOT EXISTS idx_receipts_ai_status ON receipts(ai_review_status);

    CREATE TABLE IF NOT EXISTS receipt_items (
      item_id        INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_id     INTEGER NOT NULL,
      line_no        INTEGER,
      item_name      TEXT NOT NULL,
      currency       TEXT NOT NULL DEFAULT 'CAD' CHECK (length(currency) = 3),
      unit_price     REAL NOT NULL DEFAULT 0,

      confidence     REAL NOT NULL DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 100),
      name_conf      REAL NOT NULL DEFAULT 0 CHECK (name_conf >= 0 AND name_conf <= 100),
      price_conf     REAL NOT NULL DEFAULT 0 CHECK (price_conf >= 0 AND price_conf <= 100),

      is_outlier     INTEGER NOT NULL DEFAULT 0 CHECK (is_outlier IN (0,1)),
      item_status    TEXT NOT NULL DEFAULT 'OK'
                     CHECK (item_status IN ('OK','OUTLIER','FLAGGED','LOW_CONF','ERROR')),
      status_reason  TEXT,

      FOREIGN KEY (receipt_id)
        REFERENCES receipts(receipt_id)
        ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_items_receipt ON receipt_items(receipt_id);

    CREATE TABLE IF NOT EXISTS receipt_validation (
      validation_id            INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_id               INTEGER NOT NULL UNIQUE,
      subtotal_status          TEXT,
      subtotal_discrepancy     REAL,
      subtotal_discrepancy_pct REAL,
      outliers_count           INTEGER DEFAULT 0,
      name_quality_issues      INTEGER DEFAULT 0,
      price_range_warnings     INTEGER DEFAULT 0,
      validation_json          TEXT,
      created_at               TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (receipt_id)
        REFERENCES receipts(receipt_id)
        ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_validation_receipt ON receipt_validation(receipt_id);

    CREATE TABLE IF NOT EXISTS receipt_summary (
      summary_id    INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_id    INTEGER NOT NULL UNIQUE,
      subtotal      REAL,
      tax           REAL,
      total         REAL,
      vendor        TEXT,
      phone         TEXT,
      address       TEXT,
      receipt_date  TEXT,
      summary_json  TEXT,
      created_at    TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (receipt_id)
        REFERENCES receipts(receipt_id)
        ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_summary_receipt ON receipt_summary(receipt_id);

    CREATE TABLE IF NOT EXISTS receipt_llm_runs (
      llm_run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_id         INTEGER NOT NULL,
      source_image_path  TEXT,
      raw_ocr_text       TEXT,

      llm_provider       TEXT,
      llm_model          TEXT,
      route_used         TEXT NOT NULL
                         CHECK (route_used IN ('ocr_text','vision_image')),
      prompt_version     TEXT,

      raw_llm_response   TEXT,
      parsed_json        TEXT,

      vendor             TEXT,
      phone              TEXT,
      address            TEXT,
      receipt_date       TEXT,
      subtotal           REAL,
      tax                REAL,
      total              REAL,
      currency           TEXT,

      confidence         REAL DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 100),
      validation_status  TEXT DEFAULT 'needs_review'
                         CHECK (validation_status IN ('unvalidated','valid','needs_review','failed')),
      approval_status    TEXT DEFAULT 'pending'
                         CHECK (approval_status IN ('pending','approved','rejected')),

      approved_by        TEXT,
      approved_at        TEXT,
      created_at         TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at         TEXT NOT NULL DEFAULT (datetime('now')),

      FOREIGN KEY (receipt_id)
        REFERENCES receipts(receipt_id)
        ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_llm_runs_receipt ON receipt_llm_runs(receipt_id);
    CREATE INDEX IF NOT EXISTS idx_llm_runs_approval ON receipt_llm_runs(approval_status);

    CREATE TABLE IF NOT EXISTS receipt_llm_items (
      llm_item_id      INTEGER PRIMARY KEY AUTOINCREMENT,
      llm_run_id       INTEGER NOT NULL,
      line_no          INTEGER,
      item_name        TEXT NOT NULL,
      qty              REAL,
      unit_price       REAL,
      line_price       REAL,
      currency         TEXT NOT NULL DEFAULT 'CAD' CHECK (length(currency) = 3),
      item_confidence  REAL DEFAULT 0 CHECK (item_confidence >= 0 AND item_confidence <= 100),
      created_at       TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at       TEXT NOT NULL DEFAULT (datetime('now')),

      FOREIGN KEY (llm_run_id)
        REFERENCES receipt_llm_runs(llm_run_id)
        ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_llm_items_run ON receipt_llm_items(llm_run_id);
    """

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(ddl)

        _ensure_column(conn, "receipts", "extraction_source", "TEXT NOT NULL DEFAULT 'ocr'")
        _ensure_column(conn, "receipts", "ai_review_status", "TEXT NOT NULL DEFAULT 'not_requested'")
        _ensure_column(conn, "receipts", "active_llm_run_id", "INTEGER")
        _ensure_column(conn, "receipts", "source_id", "TEXT")

        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_receipts_source_id ON receipts(source_id)")
        conn.commit()


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in cols}
    if column_name not in existing:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


# ============================================================
# Save OCR / geometry output into main tables
# ============================================================
def save_geo_fusion_to_sqlite(
    conn: sqlite3.Connection,
    out: Dict[str, Any],
    validation_report: Optional[Dict[str, Any]] = None,
    summary_report: Optional[Dict[str, Any]] = None,
    *,
    receipt_name_fallback: str = "Unknown Receipt",
    source_id_fallback: Optional[str] = None,
    default_currency: str = "CAD",
    vendor_fallback: Optional[str] = None,
    receipt_date_fallback: Optional[str] = None,
    min_confidence_filter: float = 0.70,
) -> int:
    conn.execute("PRAGMA foreign_keys = ON;")

    # job_id = batch-level id (same for all images in one upload batch)
    job_id_str = str(out.get("job_id") or "UNKNOWN_JOB").strip()

    # source_id = unique per image / receipt
    source_id_str = str(
        out.get("source_id")
        or out.get("id")
        or source_id_fallback
        or f"{job_id_str}-{uuid.uuid4().hex[:8]}"
    ).strip()

    receipt_name = (
        out.get("receipt_name")
        or out.get("id")
        or out.get("vendor")
        or source_id_str
        or receipt_name_fallback
    )

    print(
        f"save_geo_fusion_to_sqlite: job_id={job_id_str}, "
        f"source_id={source_id_str}, receipt={receipt_name}"
    )

    vendor = out.get("vendor") or vendor_fallback
    receipt_date = out.get("receipt_date") or receipt_date_fallback or datetime.now().isoformat(timespec="seconds")

    subtotal = _safe_float(out.get("SUBTOTAL", out.get("subtotal")), 0.0)
    tax = _safe_float(out.get("TAX", out.get("tax")), 0.0)
    total = _safe_float(out.get("TOTAL", out.get("total")), 0.0)
    confidence = _conf100(out.get("CONFIDENCE", out.get("confidence")))

    raw_json = _to_json(out)
    ocr_json = _to_json(out.get("ocr_struct") or {"rows": []})
    prediction_log_json = _to_json(out.get("prediction_log") or {})

    status = "Pending"

    conn.execute(
        """
        INSERT INTO receipts (
          job_id, source_id, receipt_name, vendor, receipt_date,
          subtotal, tax, total,
          status, confidence,
          extraction_source, ai_review_status, active_llm_run_id,
          raw_json, ocr_json, predictionlog_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(source_id) DO UPDATE SET
          job_id             = excluded.job_id,
          receipt_name       = excluded.receipt_name,
          vendor             = excluded.vendor,
          receipt_date       = excluded.receipt_date,
          subtotal           = excluded.subtotal,
          tax                = excluded.tax,
          total              = excluded.total,
          status             = excluded.status,
          confidence         = excluded.confidence,
          extraction_source  = excluded.extraction_source,
          ai_review_status   = excluded.ai_review_status,
          active_llm_run_id  = excluded.active_llm_run_id,
          raw_json           = excluded.raw_json,
          ocr_json           = excluded.ocr_json,
          predictionlog_json = excluded.predictionlog_json,
          updated_at         = datetime('now')
        """,
        (
            job_id_str,
            source_id_str,
            str(receipt_name),
            vendor,
            receipt_date,
            subtotal,
            tax,
            total,
            status,
            confidence,
            "ocr",
            "not_requested",
            None,
            raw_json,
            ocr_json,
            prediction_log_json,
        ),
    )

    row = conn.execute(
        "SELECT receipt_id FROM receipts WHERE source_id = ?",
        (source_id_str,),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to fetch receipt_id after upsert")

    receipt_id = int(row[0])

    if validation_report:
        subtotal_check = validation_report.get("subtotal_check", {}) or {}
        outliers = validation_report.get("price_outliers", {}).get("outliers", []) or []
        name_quality = validation_report.get("name_quality", {}) or {}
        price_ranges = validation_report.get("price_range", {}) or {}

        conn.execute(
            """
            INSERT OR REPLACE INTO receipt_validation (
              receipt_id,
              subtotal_status,
              subtotal_discrepancy,
              subtotal_discrepancy_pct,
              outliers_count,
              name_quality_issues,
              price_range_warnings,
              validation_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                subtotal_check.get("status"),
                _safe_optional_float(subtotal_check.get("discrepancy")),
                _safe_optional_float(subtotal_check.get("discrepancy_pct")),
                len(outliers),
                int(name_quality.get("num_issues", 0) or 0),
                len(price_ranges.get("suspicious_prices", []) or []),
                _to_json(validation_report),
            ),
        )

    if summary_report is not None:
        fields = (
            summary_report.get("fields")
            or out.get("sroie_fields")
            or out.get("sorie_fields")
            or {}
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO receipt_summary (
              receipt_id,
              subtotal,
              tax,
              total,
              vendor,
              phone,
              address,
              receipt_date,
              summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                _safe_optional_float(fields.get("subtotal", fields.get("SUBTOTAL"))),
                _safe_optional_float(fields.get("tax", fields.get("TAX"))),
                _safe_optional_float(fields.get("total", fields.get("TOTAL"))),
                fields.get("vendor") or fields.get("VENDOR"),
                fields.get("phone") or fields.get("PHONE"),
                fields.get("address") or fields.get("ADDRESS"),
                fields.get("date") or fields.get("DATE") or receipt_date,
                _to_json(summary_report),
            ),
        )

    conn.execute("DELETE FROM receipt_items WHERE receipt_id = ?", (receipt_id,))

    outlier_keys = build_outlier_keyset(validation_report or {})
    items: List[Dict[str, Any]] = (
        out.get("ITEMS")
        or out.get("ITEMS_GEO")
        or out.get("items")
        or out.get("receipt_items")
        or []
    )

    print(f"save_geo_fusion_to_sqlite: items_count={len(items)}")

    thr100 = min_confidence_filter * 100.0

    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue

        name = (it.get("name") or it.get("item_name") or "").strip() or "Unknown"

        price = _safe_float(
            it.get("price")
            or it.get("unit_price")
            or it.get("line_price")
            or it.get("line_total"),
            0.0,
        )

        currency = (it.get("currency") or out.get("currency") or default_currency or "CAD").strip().upper()[:3]
        if len(currency) != 3:
            currency = "CAD"

        overall_conf = _conf100(
            it.get("confidence", out.get("CONFIDENCE", out.get("confidence", 0.0)))
        )
        name_conf = _conf100(it.get("name_confidence", it.get("name_conf")))
        price_conf = _conf100(it.get("price_confidence", it.get("price_conf")))

        key = (_norm_name_key(name), round(float(price), 2))
        is_outlier = 1 if key in outlier_keys else 0

        status_reason = None
        item_status = "OK"

        if (0 < name_conf < thr100) or (0 < price_conf < thr100):
            item_status = "LOW_CONF"
            status_reason = (
                f"name_conf={name_conf:.1f}, price_conf={price_conf:.1f} (<{thr100:.1f})"
            )

        if is_outlier:
            item_status = "OUTLIER"
            status_reason = (status_reason + " | " if status_reason else "") + "validation_outlier"

        if it.get("_flagged") or it.get("flag") or it.get("item_status") == "FLAGGED":
            item_status = "FLAGGED"
            reason = (
                it.get("flag_reason")
                or it.get("_confidence_reason")
                or it.get("status_reason")
            )
            status_reason = (status_reason + " | " if status_reason else "") + (reason or "flagged")

        conn.execute(
            """
            INSERT INTO receipt_items (
              receipt_id, line_no, item_name, currency,
              unit_price,
              confidence, name_conf, price_conf,
              is_outlier, item_status, status_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                int(it.get("line_no") or i),
                name,
                currency,
                price,
                overall_conf,
                name_conf,
                price_conf,
                is_outlier,
                item_status,
                status_reason,
            ),
        )

    return receipt_id


def save_receipt_payload(
    conn: sqlite3.Connection,
    *,
    payload: Dict[str, Any],
    default_currency: str = "CAD",
    min_confidence_filter: float = 0.70,
) -> int:
    """
    Thin writer-layer wrapper so pipeline/service can pass one normalized payload
    without putting SQL logic in the pipeline.

    Expected payload keys:
      - geo_out
      - validation_report
      - summary_report
      - receipt_name_fallback
      - source_id_fallback
      - vendor_fallback
      - receipt_date_fallback
    """
    geo_out = payload.get("geo_out") or {}
    if not isinstance(geo_out, dict) or not geo_out:
        raise ValueError("payload.geo_out is required")

    validation_report = payload.get("validation_report")
    summary_report = payload.get("summary_report")

    return save_geo_fusion_to_sqlite(
        conn=conn,
        out=geo_out,
        validation_report=validation_report,
        summary_report=summary_report,
        receipt_name_fallback=payload.get("receipt_name_fallback", "Unknown Receipt"),
        source_id_fallback=payload.get("source_id_fallback"),
        default_currency=default_currency,
        vendor_fallback=payload.get("vendor_fallback"),
        receipt_date_fallback=payload.get("receipt_date_fallback"),
        min_confidence_filter=min_confidence_filter,
    )


# ============================================================
# Save LLM result into staging tables
# ============================================================
def save_llm_result_to_sqlite(
    conn: sqlite3.Connection,
    *,
    receipt_id: int,
    llm_result: Dict[str, Any],
    raw_llm_response: Optional[str] = None,
    route_used: str,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    source_image_path: Optional[str] = None,
    raw_ocr_text: Optional[str] = None,
    default_currency: str = "CAD",
    validation_status: str = "needs_review",
    set_active: bool = True,
) -> int:
    if route_used not in {"ocr_text", "vision_image"}:
        raise ValueError("route_used must be 'ocr_text' or 'vision_image'")

    vendor = llm_result.get("vendor")
    phone = llm_result.get("phone")
    address = llm_result.get("address")
    receipt_date = llm_result.get("date") or llm_result.get("receipt_date")

    subtotal = _safe_optional_float(llm_result.get("subtotal"))
    tax = _safe_optional_float(llm_result.get("tax"))
    total = _safe_optional_float(llm_result.get("total"))
    confidence = _conf100(llm_result.get("confidence"))

    currency = (llm_result.get("currency") or default_currency or "CAD").strip().upper()[:3]
    if len(currency) != 3:
        currency = "CAD"

    parsed_json = _to_json(llm_result)

    cur = conn.execute(
        """
        INSERT INTO receipt_llm_runs (
          receipt_id,
          source_image_path,
          raw_ocr_text,
          llm_provider,
          llm_model,
          route_used,
          prompt_version,
          raw_llm_response,
          parsed_json,
          vendor,
          phone,
          address,
          receipt_date,
          subtotal,
          tax,
          total,
          currency,
          confidence,
          validation_status,
          approval_status,
          updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', datetime('now'))
        """,
        (
            receipt_id,
            source_image_path,
            raw_ocr_text,
            llm_provider,
            llm_model,
            route_used,
            prompt_version,
            raw_llm_response,
            parsed_json,
            vendor,
            phone,
            address,
            receipt_date,
            subtotal,
            tax,
            total,
            currency,
            confidence,
            validation_status,
        ),
    )
    llm_run_id = int(cur.lastrowid)

    items = llm_result.get("items") or llm_result.get("receipt_items") or []
    for idx, item in enumerate(items, start=1):
        item_name = (item.get("name") or item.get("item_name") or "").strip() or "Unknown"
        qty = _safe_optional_float(item.get("qty"))
        unit_price = _safe_optional_float(item.get("unit_price"))
        line_price = _safe_optional_float(item.get("price") or item.get("line_price") or item.get("line_total"))

        item_currency = (item.get("currency") or currency or "CAD").strip().upper()[:3]
        if len(item_currency) != 3:
            item_currency = "CAD"

        item_confidence = _conf100(item.get("confidence"))

        conn.execute(
            """
            INSERT INTO receipt_llm_items (
              llm_run_id,
              line_no,
              item_name,
              qty,
              unit_price,
              line_price,
              currency,
              item_confidence,
              updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                llm_run_id,
                idx,
                item_name,
                qty,
                unit_price,
                line_price,
                item_currency,
                item_confidence,
            ),
        )

    if set_active:
        conn.execute(
            """
            UPDATE receipts
            SET ai_review_status = 'pending',
                active_llm_run_id = ?,
                updated_at = datetime('now')
            WHERE receipt_id = ?
            """,
            (llm_run_id, receipt_id),
        )

    return llm_run_id


# ============================================================
# Read LLM run(s) for UI
# ============================================================
def get_llm_run_with_items(
    conn: sqlite3.Connection,
    llm_run_id: int,
) -> Optional[Dict[str, Any]]:
    run_row = conn.execute(
        """
        SELECT
          llm_run_id, receipt_id, source_image_path, raw_ocr_text,
          llm_provider, llm_model, route_used, prompt_version,
          raw_llm_response, parsed_json,
          vendor, phone, address, receipt_date,
          subtotal, tax, total, currency,
          confidence, validation_status, approval_status,
          approved_by, approved_at, created_at, updated_at
        FROM receipt_llm_runs
        WHERE llm_run_id = ?
        """,
        (llm_run_id,),
    ).fetchone()

    if run_row is None:
        return None

    columns = [
        "llm_run_id", "receipt_id", "source_image_path", "raw_ocr_text",
        "llm_provider", "llm_model", "route_used", "prompt_version",
        "raw_llm_response", "parsed_json",
        "vendor", "phone", "address", "receipt_date",
        "subtotal", "tax", "total", "currency",
        "confidence", "validation_status", "approval_status",
        "approved_by", "approved_at", "created_at", "updated_at",
    ]
    run_data = dict(zip(columns, run_row))

    item_rows = conn.execute(
        """
        SELECT
          llm_item_id, llm_run_id, line_no, item_name,
          qty, unit_price, line_price, currency, item_confidence,
          created_at, updated_at
        FROM receipt_llm_items
        WHERE llm_run_id = ?
        ORDER BY line_no ASC, llm_item_id ASC
        """,
        (llm_run_id,),
    ).fetchall()

    run_data["items"] = [
        {
            "llm_item_id": r[0],
            "llm_run_id": r[1],
            "line_no": r[2],
            "item_name": r[3],
            "qty": r[4],
            "unit_price": r[5],
            "line_price": r[6],
            "currency": r[7],
            "item_confidence": r[8],
            "created_at": r[9],
            "updated_at": r[10],
        }
        for r in item_rows
    ]

    return run_data


def get_latest_llm_run_with_items(
    conn: sqlite3.Connection,
    receipt_id: int,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT llm_run_id
        FROM receipt_llm_runs
        WHERE receipt_id = ?
        ORDER BY llm_run_id DESC
        LIMIT 1
        """,
        (receipt_id,),
    ).fetchone()

    if row is None:
        return None

    return get_llm_run_with_items(conn, int(row[0]))


# ============================================================
# Approve / reject LLM result into main tables
# ============================================================
def approve_llm_run(
    conn: sqlite3.Connection,
    *,
    llm_run_id: int,
    approved_by: Optional[str] = None,
) -> int:
    conn.execute("PRAGMA foreign_keys = ON;")

    run_row = conn.execute(
        """
        SELECT
          receipt_id, vendor, receipt_date, subtotal, tax, total,
          currency, confidence, approval_status
        FROM receipt_llm_runs
        WHERE llm_run_id = ?
        """,
        (llm_run_id,),
    ).fetchone()

    if run_row is None:
        raise ValueError(f"LLM run not found: {llm_run_id}")

    receipt_id = int(run_row[0])
    vendor = run_row[1]
    receipt_date = run_row[2]
    subtotal = run_row[3]
    tax = run_row[4]
    total = run_row[5]
    currency = run_row[6] or "CAD"
    confidence = _conf100(run_row[7])
    approval_status = run_row[8]

    if approval_status == "approved":
        return receipt_id

    item_rows = conn.execute(
        """
        SELECT line_no, item_name, line_price, currency, item_confidence
        FROM receipt_llm_items
        WHERE llm_run_id = ?
        ORDER BY line_no ASC, llm_item_id ASC
        """,
        (llm_run_id,),
    ).fetchall()

    with conn:
        conn.execute(
            """
            UPDATE receipts
            SET vendor = COALESCE(?, vendor),
                receipt_date = COALESCE(?, receipt_date),
                subtotal = COALESCE(?, subtotal),
                tax = COALESCE(?, tax),
                total = COALESCE(?, total),
                confidence = ?,
                extraction_source = 'ocr+llm',
                ai_review_status = 'approved',
                active_llm_run_id = ?,
                updated_at = datetime('now')
            WHERE receipt_id = ?
            """,
            (
                vendor,
                receipt_date,
                subtotal,
                tax,
                total,
                confidence,
                llm_run_id,
                receipt_id,
            ),
        )

        conn.execute("DELETE FROM receipt_items WHERE receipt_id = ?", (receipt_id,))

        for row in item_rows:
            line_no, item_name, line_price, item_currency, item_confidence = row
            cur3 = (item_currency or currency or "CAD").strip().upper()[:3]
            if len(cur3) != 3:
                cur3 = "CAD"

            conn.execute(
                """
                INSERT INTO receipt_items (
                  receipt_id, line_no, item_name, currency,
                  unit_price,
                  confidence, name_conf, price_conf,
                  is_outlier, item_status, status_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'OK', ?)
                """,
                (
                    receipt_id,
                    line_no,
                    item_name,
                    cur3,
                    _safe_float(line_price, 0.0),
                    _conf100(item_confidence),
                    0.0,
                    _conf100(item_confidence),
                    f"approved_from_llm_run:{llm_run_id}",
                ),
            )

        conn.execute(
            """
            UPDATE receipt_llm_runs
            SET approval_status = 'approved',
                approved_by = ?,
                approved_at = datetime('now'),
                updated_at = datetime('now')
            WHERE llm_run_id = ?
            """,
            (approved_by, llm_run_id),
        )

    return receipt_id


def reject_llm_run(
    conn: sqlite3.Connection,
    *,
    llm_run_id: int,
    rejected_by: Optional[str] = None,
) -> int:
    row = conn.execute(
        "SELECT receipt_id FROM receipt_llm_runs WHERE llm_run_id = ?",
        (llm_run_id,),
    ).fetchone()

    if row is None:
        raise ValueError(f"LLM run not found: {llm_run_id}")

    receipt_id = int(row[0])

    with conn:
        conn.execute(
            """
            UPDATE receipt_llm_runs
            SET approval_status = 'rejected',
                approved_by = ?,
                approved_at = datetime('now'),
                updated_at = datetime('now')
            WHERE llm_run_id = ?
            """,
            (rejected_by, llm_run_id),
        )

        conn.execute(
            """
            UPDATE receipts
            SET ai_review_status = 'rejected',
                updated_at = datetime('now')
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        )

    return receipt_id


# ============================================================
# Optional helper: choose UI source
# ============================================================
def get_receipt_for_ui(conn: sqlite3.Connection, receipt_id: int) -> Dict[str, Any]:
    receipt_row = conn.execute(
        """
        SELECT
          receipt_id, job_id, source_id, receipt_name, vendor, receipt_date,
          subtotal, tax, total, status, confidence,
          extraction_source, ai_review_status, active_llm_run_id,
          raw_json, ocr_json, predictionlog_json,
          created_at, updated_at
        FROM receipts
        WHERE receipt_id = ?
        """,
        (receipt_id,),
    ).fetchone()

    if receipt_row is None:
        raise ValueError(f"Receipt not found: {receipt_id}")

    base = {
        "receipt_id": receipt_row[0],
        "job_id": receipt_row[1],
        "source_id": receipt_row[2],
        "receipt_name": receipt_row[3],
        "vendor": receipt_row[4],
        "receipt_date": receipt_row[5],
        "subtotal": receipt_row[6],
        "tax": receipt_row[7],
        "total": receipt_row[8],
        "status": receipt_row[9],
        "confidence": receipt_row[10],
        "extraction_source": receipt_row[11],
        "ai_review_status": receipt_row[12],
        "active_llm_run_id": receipt_row[13],
        "raw_json": receipt_row[14],
        "ocr_json": receipt_row[15],
        "predictionlog_json": receipt_row[16],
        "created_at": receipt_row[17],
        "updated_at": receipt_row[18],
    }

    active_llm_run_id = base["active_llm_run_id"]
    ai_review_status = (base["ai_review_status"] or "").strip().lower()

    if active_llm_run_id and ai_review_status == "pending":
        llm_data = get_llm_run_with_items(conn, int(active_llm_run_id))
        if llm_data:
            return {
                "source": "llm_staging",
                "receipt": base,
                "llm_run": llm_data,
            }

    item_rows = conn.execute(
        """
        SELECT item_id, line_no, item_name, currency, unit_price,
               confidence, name_conf, price_conf,
               is_outlier, item_status, status_reason
        FROM receipt_items
        WHERE receipt_id = ?
        ORDER BY line_no ASC, item_id ASC
        """,
        (receipt_id,),
    ).fetchall()

    items = [
        {
            "item_id": r[0],
            "line_no": r[1],
            "item_name": r[2],
            "currency": r[3],
            "unit_price": r[4],
            "confidence": r[5],
            "name_conf": r[6],
            "price_conf": r[7],
            "is_outlier": r[8],
            "item_status": r[9],
            "status_reason": r[10],
        }
        for r in item_rows
    ]

    return {
        "source": "main",
        "receipt": base,
        "items": items,
    }