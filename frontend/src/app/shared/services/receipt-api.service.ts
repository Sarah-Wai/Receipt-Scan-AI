import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { map, Observable } from "rxjs";
import {
  ReceiptRow,
  ReceiptStatus,
  ReceiptDetail,
  ReceiptDetailItem,
  ReceiptSummary,
  ReceiptValidation,
} from "../../models/receipt.model";

/* =========================
   Main receipt API types
   ========================= */

export interface ApiReceiptRow {
  receipt_id: number;
  receipt_name: string;
  vendor: string | null;
  receipt_date: string | null;
  subtotal: number | null;
  tax: number | null;
  total: number | null;
  status: ReceiptStatus;
  confidence: number | null;
  ocr_json: unknown | null;
  predictionlog_json: unknown | null;
  raw_json?: unknown | null;
  source_id: string | null;
  extraction_source: string | null;
  ai_review_status: "not_requested" | "pending" | "approved" | "rejected" | string | null;
  active_llm_run_id?: number | null;
  image_url?: string | null;
  data_source?: "main" | "llm_staging" | string;
}

export interface ApiReceiptItemRow {
  item_id: number | string;
  receipt_id: number;
  line_no?: number | null;
  item_name: string | null;
  currency: string | null;
  unit_price: number | null;
  confidence: number | null;
  name_conf?: number | null;
  price_conf?: number | null;
  is_outlier?: number | null;
  item_status?: string | null;
  status_reason?: string | null;
  qty?: number | null;
  line_price?: number | null;
}

export interface ApiReceiptValidationRow {
  validation_id: number;
  receipt_id: number;
  subtotal_status: string | null;
  subtotal_discrepancy: number | null;
  subtotal_discrepancy_pct: number | null;
  outliers_count: number | null;
  name_quality_issues: number | null;
  price_range_warnings: number | null;
  validation_json: unknown | null;
  created_at: string;
}

export interface ApiReceiptSummaryRow {
  summary_id: number | null;
  receipt_id: number;
  subtotal: number | null;
  tax: number | null;
  total: number | null;
  vendor: string | null;
  phone: string | null;
  address: string | null;
  receipt_date: string | null;
  summary_json?: unknown | null;
}

export interface ApiLlmRunRow {
  llm_run_id: number;
  receipt_id: number;
  vendor: string | null;
  phone: string | null;
  address: string | null;
  receipt_date: string | null;
  subtotal: number | null;
  tax: number | null;
  total: number | null;
  currency?: string | null;
  confidence?: number | null;
  parsed_json?: unknown | null;
  approval_status?: string | null;
  validation_status?: string | null;
  items?: any[];
}

export interface ApiReceiptDetailResponse {
  receipt: ApiReceiptRow;
  items: ApiReceiptItemRow[];
  validation?: ApiReceiptValidationRow | null;
  summary?: ApiReceiptSummaryRow | null;
  llm_run?: ApiLlmRunRow | null;
  data_source?: "main" | "llm_staging" | string;
  validation_skipped?: boolean;
}

export interface ApiLlmAutofixResponse {
  ok: boolean;
  receipt_id: number;
  llm_run_id: number;
  route_used: string;
  model_used: string;
  validation: {
    total_ok?: boolean;
    subtotal_ok?: boolean;
  } | null;
  llm_run: any | null;
}

/* =========================
   Approve payload
   ========================= */

export interface ApiApproveReceiptItem {
  line_no: number;
  item_name: string;
  currency: string;
  unit_price: number;
  confidence: number;
}

export interface ApiApproveReceiptPayload {
  vendor: string | null;
  receipt_date: string | null;
  subtotal: number | null;
  tax: number | null;
  total: number | null;

  summary_vendor: string | null;
  summary_phone: string | null;
  summary_address: string | null;
  summary_receipt_date: string | null;

  items: ApiApproveReceiptItem[];
}

/* =========================
   Upload / Process API
   ========================= */

export interface ApiUploadFileRow {
  filename: string;
  original_filename: string;
  image_path: string;
}

export interface ApiUploadResponse {
  ok: boolean;
  job_id: string;
  raw_dir: string;
  count: number;
  files: ApiUploadFileRow[];
}

export interface ApiProcessResponse {
  ok: boolean;
  receipt_id: number;
  job_id: string;
  result: {
    success: boolean;
    job_id: string;
    raw_dir: string;
    image_count: number;
    error?: string | null;
    steps?: Record<string, unknown>;
    final_output?: Record<string, unknown>;
  };
}

@Injectable({ providedIn: "root" })
export class ReceiptApiService {
  private baseUrl = "http://localhost:8000/api";

  constructor(private http: HttpClient) {}

  uploadReceipts(files: File[]): Observable<ApiUploadResponse> {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }
    return this.http.post<ApiUploadResponse>(`${this.baseUrl}/upload`, formData);
  }

  processReceipt(receiptId: number, jobId: string): Observable<ApiProcessResponse> {
    return this.http.post<ApiProcessResponse>(
      `${this.baseUrl}/receipts/${receiptId}/process`,
      { job_id: jobId },
    );
  }

  autoFixWithLlm(receiptId: number): Observable<ApiLlmAutofixResponse> {
    return this.http.post<ApiLlmAutofixResponse>(
      `${this.baseUrl}/receipts/${receiptId}/autofix-llm`,
      {},
    );
  }

  listReceipts(): Observable<ReceiptRow[]> {
    return this.http.get<ApiReceiptRow[]>(`${this.baseUrl}/receipts`).pipe(
      map((rows) =>
        rows.map((r) => ({
          id: r.receipt_id,
          receiptName: r.receipt_name ?? "",
          vendor: r.vendor ?? "",
          date: r.receipt_date ?? "",
          subtotal: Number(r.subtotal ?? 0),
          tax: Number(r.tax ?? 0),
          total: Number(r.total ?? 0),
          status: r.status,
          rawJson: this.parseJsonSafe(r.raw_json ?? null),
          ocrJson: this.parseJsonSafe(r.ocr_json ?? null),
          predictionLogJson: this.parseJsonSafe(r.predictionlog_json ?? null),
          confidence: Number(r.confidence ?? 0),
          hasLlmResult: !!r.active_llm_run_id,
          aiReviewStatus: r.ai_review_status ?? "not_requested",
          dataSource: r.data_source ?? "main",
        }) as any),
      ),
    );
  }

  getReceiptDetail(receiptId: number): Observable<ReceiptDetail> {
    return this.http
      .get<ApiReceiptDetailResponse>(
        `${this.baseUrl}/receipts/${receiptId}?include_validation=true`,
      )
      .pipe(
        map((res) => {
          const r = res.receipt;
          const v = res.validation ?? null;
          const s = res.summary ?? null;

          const items: ReceiptDetailItem[] = (res.items ?? []).map((it: ApiReceiptItemRow) => ({
            id: String(it.item_id ?? ""),
            name: it.item_name ?? "",
            currency: String(it.currency ?? "CAD").toUpperCase(),
            unitPrice: Number(it.unit_price ?? 0),
            confidence: Number(it.confidence ?? 0),
          }));

          const summary: ReceiptSummary | null = s
            ? {
                summaryId: Number(s.summary_id ?? 0),
                receiptId: Number(s.receipt_id ?? 0),
                subtotal: s.subtotal == null ? null : Number(s.subtotal),
                tax: s.tax == null ? null : Number(s.tax),
                total: s.total == null ? null : Number(s.total),
                vendor: s.vendor ?? null,
                phone: s.phone ?? null,
                address: s.address ?? null,
                receiptDate: s.receipt_date ?? null,
                source: (res.data_source === "llm_staging" ? "llm" : "summary") as any,
                summaryJson: this.parseJsonSafe(s.summary_json ?? null),
              } as any
            : null;

          const validation: ReceiptValidation | null = v
            ? {
                id: Number(v.validation_id ?? 0),
                receiptId: Number(v.receipt_id ?? 0),
                subtotalStatus: v.subtotal_status ?? "",
                subtotalDiscrepancy:
                  v.subtotal_discrepancy == null ? null : Number(v.subtotal_discrepancy),
                subtotalDiscrepancyPct:
                  v.subtotal_discrepancy_pct == null
                    ? null
                    : Number(v.subtotal_discrepancy_pct),
                outliersCount: Number(v.outliers_count ?? 0),
                nameQualityIssues: Number(v.name_quality_issues ?? 0),
                priceRangeWarnings: Number(v.price_range_warnings ?? 0),
                validationJson: this.parseJsonSafe(v.validation_json ?? null),
                createdAt: v.created_at ?? "",
              }
            : null;

          const detail: ReceiptDetail = {
            id: Number(r.receipt_id ?? 0),
            receiptName: r.receipt_name ?? "",
            vendor: r.vendor ?? "",
            date: r.receipt_date ?? "",
            subtotal: Number(r.subtotal ?? 0),
            tax: Number(r.tax ?? 0),
            total: Number(r.total ?? 0),
            status: (r.status ?? "") as ReceiptStatus,
            confidence: Number(r.confidence ?? 0),
            source_id: r.source_id ?? null,
            image_url: r.image_url ?? null,
            llmUsed: res.data_source === "llm_staging",
            items,
            ocrJson: this.parseJsonSafe(r.ocr_json ?? null),
            predictionLogJson: this.parseJsonSafe(r.predictionlog_json ?? null),
            rawJson: this.parseJsonSafe(r.raw_json ?? null),
            summary,
            validation,
            aiReviewStatus: r.ai_review_status ?? "not_requested",
            dataSource: res.data_source ?? "main",
            validationSkipped: !!res.validation_skipped,
            llmRun: res.llm_run ?? null,
          } as any;

          return detail;
        }),
      );
  }

  /* =========================
     Combined approve:
     save edits + mark Processed
     ========================= */
  approveReceipt(receiptId: number, payload: ApiApproveReceiptPayload) {
    return this.http.put<{
      ok: boolean;
      receipt: {
        receipt_id: number;
        status: string;
        ai_review_status?: string;
        active_llm_run_id?: number | null;
      };
    }>(`${this.baseUrl}/receipts/${receiptId}/approve`, payload);
  }

  /* =========================
     Simple status update
     For Failed / Error / etc.
     ========================= */
  updateReceiptStatus(receiptId: number, status: string) {
    return this.http.put<{
      ok: boolean;
      receipt: {
        receipt_id: number;
        status: string;
        ai_review_status?: string;
        active_llm_run_id?: number | null;
      };
    }>(`${this.baseUrl}/receipts/${receiptId}/status`, { status });
  }

  rejectLlm(receiptId: number) {
    return this.http.post<{
      ok: boolean;
      receipt_id: number;
      ai_review_status: string;
    }>(`${this.baseUrl}/receipts/${receiptId}/reject-llm`, {});
  }

  private parseJsonSafe(value: unknown): unknown {
    if (typeof value !== "string") return value;
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
}