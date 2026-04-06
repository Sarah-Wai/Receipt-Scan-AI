export type ReceiptStatus =
  | 'Approved'
  | 'Pending'
  | 'Rejected'
  | 'Processed'
  | 'Failed'
  | 'Error';


// ── List row (used in receipts table) ─────────────────────────────────────────
export interface ReceiptRow {
  id:           number;
  receiptName:  string;
  vendor:       string;
  date:         string;
  subtotal:     number;
  tax:          number;
  total:        number;
  status:       ReceiptStatus;
  confidence:   number;
}


// ── Detail item ───────────────────────────────────────────────────────────────
export interface ReceiptDetailItem {
  id:               string;
  name:             string;
  currency:         string;
  unitPrice:        number;
  confidence:       number;
  validationStatus?: string;
}


// ── Prediction log ────────────────────���───────────────────────────────────────
export interface PredictionLogRow {
  idx:        number;
  word:       string;
  raw_label:  string;
  label:      string;
  changed:    boolean;
  confidence: number;
}


// ── Summary (from receipt_summary table or LLM run) ───────────────────────────
export interface ReceiptSummary {
  summaryId:       number;
  receiptId:       number;
  subtotal:        number | null;
  tax:             number | null;
  total:           number | null;
  vendor:          string | null;
  phone:           string | null;
  address:         string | null;
  receiptDate:     string | null;
  source:          'llm' | 'summary';
  rawResponseJson?: unknown | null;
  summary_json?:   unknown | null;
}


// ── Validation (from receipt_validation table) ────────────────────────────────
export interface ReceiptValidation {
  id:                     number;
  receiptId:              number;
  subtotalStatus:         string | null;
  subtotalDiscrepancy:    number | null;
  subtotalDiscrepancyPct: number | null;
  outliersCount:          number | null;
  nameQualityIssues:      number | null;
  priceRangeWarnings:     number | null;
  validationJson:         unknown | null;
  createdAt:              string;
}


// ── Full detail (used in receipt-detail page) ─────────────────────────────────
export interface ReceiptDetail {
  id:           number;
  receiptName:  string;
  vendor:       string | null;
  date:         string | null;
  subtotal:     number | null;
  tax:          number | null;
  total:        number | null;
  status:       ReceiptStatus;
  confidence:   number | null;
  source_id:    string | null;
  image_url:    string | null;
  llmUsed:      boolean;

  items:             ReceiptDetailItem[];

  // FIX: typed as `any` — parseJsonSafe returns unknown, and these blobs are
  // read with runtime guards in the component (parsePredictionLog, etc.)
  // Strict structural types here gave no safety benefit and only caused errors.
  ocrJson:           any | null;
  predictionLogJson: any | null;
  rawJson:           any | null;

  summary:    ReceiptSummary    | null;
  validation: ReceiptValidation | null;
}


// ── LLM auto-fix response ─────────────────────────────────────────────────────
export interface ApiLlmAutofixResponse {
  ok:          boolean;
  receipt_id:  number;
  llm_run_id:  number;
  route_used:  string;
  model_used:  string;
  validation:  { total_ok?: boolean; subtotal_ok?: boolean } | null;
  llm_run:     any | null;
}