import { CommonModule } from "@angular/common";
import { Component, OnInit, OnDestroy } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { ActivatedRoute, Router } from "@angular/router";
import {
  ReceiptApiService,
  ApiLlmAutofixResponse,
  ApiApproveReceiptPayload,
} from "../../shared/services/receipt-api.service";
import { ReceiptStatus } from "../../models/receipt.model";
import { Subject, firstValueFrom, Observable } from "rxjs";
import { takeUntil } from "rxjs/operators";
import {
  buildSamplePredictionLines,
  parsePredictionLog,
  FormattedPredictionLine,
} from "../../utils/prediction-log";

type ItemValidationStatus =
  | "OK"
  | "OUTLIER"
  | "NAME_ISSUE"
  | "SUSPICIOUS_PRICE"
  | "CHECK";

type ReceiptItem = {
  id: string;
  name: string;
  currency: string;
  unitPrice: number;
  confidence: number;
  validationStatus: ItemValidationStatus;
};

@Component({
  selector: "app-receipt-detail",
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: "./receipt-detail.component.html",
  styleUrl: "./receipt-detail.component.css",
})
export class ReceiptDetailComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  leftW = 300;
  rightW = 320;

  private dragMode: "left" | "right" | null = null;
  private dragStartX = 0;
  private startLeftW = 0;
  private startRightW = 0;

  startDrag(e: PointerEvent, mode: "left" | "right") {
    this.dragMode = mode;
    this.dragStartX = e.clientX;
    this.startLeftW = this.leftW;
    this.startRightW = this.rightW;

    (e.target as HTMLElement).setPointerCapture(e.pointerId);

    window.addEventListener("pointermove", this.onDragMove);
    window.addEventListener("pointerup", this.onDragEnd);
  }

  private onDragMove = (e: PointerEvent) => {
    if (!this.dragMode) return;

    const dx = e.clientX - this.dragStartX;
    const LEFT_MIN = 220, LEFT_MAX = 520;
    const RIGHT_MIN = 240, RIGHT_MAX = 520;

    if (this.dragMode === "left") {
      this.leftW = Math.max(LEFT_MIN, Math.min(LEFT_MAX, this.startLeftW + dx));
    }
    if (this.dragMode === "right") {
      this.rightW = Math.max(RIGHT_MIN, Math.min(RIGHT_MAX, this.startRightW - dx));
    }
  };

  private onDragEnd = (_e: PointerEvent) => {
    this.dragMode = null;
    window.removeEventListener("pointermove", this.onDragMove);
    window.removeEventListener("pointerup", this.onDragEnd);
  };

  private receiptIds: number[] = [];
  private receiptIndex = -1;

  get canPrev() { return this.receiptIndex > 0; }
  get canNext() {
    return this.receiptIds.length > 0
      && this.receiptIndex >= 0
      && this.receiptIndex < this.receiptIds.length - 1;
  }

  zoom = 1;
  minZoom = 0.6;
  maxZoom = 2.5;
  zoomStep = 0.15;
  baseWidth = 400;

  zoomIn() { this.zoom = Math.min(this.zoom + this.zoomStep, this.maxZoom); }
  zoomOut() { this.zoom = Math.max(this.zoom - this.zoomStep, this.minZoom); }

  receiptId!: number;
  receiptNumericId = 0;
  vendor = "";
  receiptName = "";
  receiptDate = "";
  status: ReceiptStatus | "" = "";
  confidence = 0;

  currency: "CAD" | "USD" = "CAD";
  subtotal = 0;
  tax = 0;
  total = 0;

  activeTab: "items" | "ocr" | "validation" | "prediction" = "items";

  local_folder_path = "/assets/best_images/";
  receiptImageUrl = "/assets/receipt-demo.png";
  rawOcrText = `RAW OCR will show here...\n...`;
  predictionLines: FormattedPredictionLine[] = [];
  showPredictions = false;
  validationJson = "";

  summary_subtotal = 0;
  summary_tax = 0;
  summary_total = 0;
  summary_vendor: string | null = null;
  summary_phone: string | null = null;
  summary_address: string | null = null;
  summary_receipt_date: string | null = null;

  items: ReceiptItem[] = [];

  aiReviewStatus: "not_requested" | "pending" | "approved" | "rejected" | "" = "";
  dataSource: "main" | "llm_staging" | "" = "";
  validationSkipped = false;
  llmUsed = false;

  loading = true;
  errorMsg = "";
  approving = false;
  rejecting = false;
  exporting = false;

  autoFixing = false;
  autoFixStatus: "idle" | "success" | "error" = "idle";
  autoFixMsg = "";
  private autoFixMsgTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private receiptApi: ReceiptApiService,
  ) {}

  ngOnInit() {
    const st = (history.state ?? {}) as any;
    if (Array.isArray(st.ids) && st.ids.length > 0) {
      this.receiptIds = st.ids
        .map((x: any) => Number(x))
        .filter(Number.isFinite);
      this.receiptIndex = Number.isFinite(st.index) ? Number(st.index) : -1;
    }

    this.route.paramMap.pipe(takeUntil(this.destroy$)).subscribe((pm) => {
      const id = Number(pm.get("id"));

      if (!id || Number.isNaN(id)) {
        this.loading = false;
        this.errorMsg = "Invalid receipt id";
        return;
      }

      this.receiptId = id;
      this.receiptNumericId = id;

      if (this.receiptIds.length > 0) {
        const idx = this.receiptIds.indexOf(id);
        if (idx >= 0) this.receiptIndex = idx;
      }

      this.loadReceipt(id);
    });
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();

    if (this.autoFixMsgTimer) {
      clearTimeout(this.autoFixMsgTimer);
    }
  }

  get canRejectLlm(): boolean {
    return this.aiReviewStatus === "pending";
  }

  get showValidationTab(): boolean {
    return !this.validationSkipped;
  }

  get sourceBadge(): string {
    if (this.dataSource === "llm_staging") return "LLM Pending Review";
    if (this.aiReviewStatus === "approved") return "LLM Approved";
    if (this.aiReviewStatus === "rejected") return "LLM Rejected";
    return "OCR";
  }

  get isProcessed(): boolean {
    return this.status === "Processed";
  }

  private safeJsonParse(value: unknown): any {
    if (typeof value !== "string") return value;
    try { return JSON.parse(value); } catch { return null; }
  }

  private normName(s: unknown): string {
    return String(s ?? "").trim().toLowerCase().replace(/\s+/g, " ");
  }

  private buildValidationIndex(validationJson: any) {
    const outlierNames = new Set<string>();
    const nameIssueNames = new Set<string>();
    const suspiciousNames = new Set<string>();

    for (const o of validationJson?.price_outliers?.outliers ?? []) {
      outlierNames.add(this.normName(o?.item_name));
    }

    for (const x of validationJson?.name_quality?.issues ?? []) {
      nameIssueNames.add(this.normName(x?.item_name));
    }

    for (const s of validationJson?.price_range?.suspicious_prices ?? []) {
      suspiciousNames.add(this.normName(s?.item_name));
    }

    return { outlierNames, nameIssueNames, suspiciousNames };
  }

  private pickItemStatus(
    itemName: string,
    idx: {
      outlierNames: Set<string>;
      nameIssueNames: Set<string>;
      suspiciousNames: Set<string>;
    },
  ): ItemValidationStatus {
    const key = this.normName(itemName);
    if (idx.outlierNames.has(key)) return "OUTLIER";
    if (idx.suspiciousNames.has(key)) return "SUSPICIOUS_PRICE";
    if (idx.nameIssueNames.has(key)) return "NAME_ISSUE";
    return "OK";
  }

  private loadReceipt(id: number) {
    this.loading = true;
    this.errorMsg = "";

    this.receiptApi.getReceiptDetail(id).subscribe({
      next: (detail: any) => {
        this.subtotal = Number(detail.subtotal || 0);
        this.tax = Number(detail.tax || 0);
        this.total = Number(detail.total || 0);

        this.status = detail.status ?? this.status;
        this.vendor = detail.vendor ?? this.vendor;
        this.receiptName = detail.receiptName ?? this.receiptName;
        this.receiptDate = detail.date ?? this.receiptDate;

        const confRaw = Number(detail.confidence ?? this.confidence);
        this.confidence = confRaw <= 1 ? Math.round(confRaw * 100) : confRaw;

        this.aiReviewStatus = detail.aiReviewStatus ?? "not_requested";
        this.dataSource = detail.dataSource ?? "main";
        this.validationSkipped = !!detail.validationSkipped;
        this.llmUsed = !!detail.llmUsed;

        this.rawOcrText = detail.ocrJson
          ? JSON.stringify(detail.ocrJson["ocr_text"] ?? detail.ocrJson, null, 2)
          : "No OCR JSON data";

        const log = parsePredictionLog(detail.predictionLogJson);
        this.predictionLines = buildSamplePredictionLines(log);
        this.showPredictions = this.predictionLines.length > 0;

        this.receiptImageUrl = detail.image_url ?? "/assets/receipt-demo.png";
        console.log("Loaded receipt image URL:", 'http://localhost:8000' + this.receiptImageUrl);

        const vjson = detail?.validation?.validationJson ?? detail?.validationJson ?? null;
        const parsed = this.safeJsonParse(vjson);
        const idx = this.buildValidationIndex(parsed);

        const summary = detail.summary ?? null;
        if (summary) {
          this.summary_subtotal = Number(summary.subtotal ?? this.subtotal);
          this.summary_tax = Number(summary.tax ?? this.tax);
          this.summary_total = Number(summary.total ?? this.total);
          this.summary_vendor = summary.vendor ?? null;
          this.summary_phone = summary.phone ?? null;
          this.summary_address = summary.address ?? null;
          this.summary_receipt_date =
            summary.receiptDate ?? summary.receipt_date ?? null;
        } else {
          this.summary_subtotal = this.subtotal;
          this.summary_tax = this.tax;
          this.summary_total = this.total;
          this.summary_vendor = this.vendor || null;
          this.summary_phone = null;
          this.summary_address = null;
          this.summary_receipt_date = this.receiptDate || null;
        }

        const inItems = (detail.items ?? []) as any[];
        this.items = inItems.map((it: any) => {
          const name = String(it.name ?? it.item_name ?? "");
          const c = String(it.currency ?? "CAD").toUpperCase();
          const p = Number(it.unitPrice ?? it.unit_price ?? 0);
          const ic = Number(it.confidence ?? 0);
          const ic100 = ic <= 1 ? Math.round(ic * 100) : ic;

          return {
            id: String(it.id ?? it.item_id ?? ""),
            name,
            currency: c,
            unitPrice: p,
            confidence: Math.max(0, Math.min(100, ic100)),
            validationStatus: this.validationSkipped
              ? "OK"
              : this.pickItemStatus(name, idx),
          } as ReceiptItem;
        });

        if (this.validationSkipped && this.activeTab === "validation") {
          this.activeTab = "items";
        }

        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
        this.errorMsg = "Failed to load receipt detail";
      },
    });
  }

  async runAutoFix(): Promise<void> {
    if (this.autoFixing || this.isProcessed) return;

    this.autoFixing = true;
    this.autoFixStatus = "idle";
    this.autoFixMsg = "";

    if (this.autoFixMsgTimer) {
      clearTimeout(this.autoFixMsgTimer);
      this.autoFixMsgTimer = null;
    }

    try {
      const res = await firstValueFrom(
        this.receiptApi.autoFixWithLlm(this.receiptId) as Observable<ApiLlmAutofixResponse>,
      );

      this.autoFixStatus = "success";
      this.autoFixMsg = `AI fix applied via ${res?.route_used ?? "unknown"}.`;

      this.loadReceipt(this.receiptId);
    } catch (err) {
      console.error("Auto Fix failed:", err);
      this.autoFixStatus = "error";
      this.autoFixMsg = "AI fix failed. Check console for details.";
    } finally {
      this.autoFixing = false;

      this.autoFixMsgTimer = setTimeout(() => {
        this.autoFixMsg = "";
        this.autoFixStatus = "idle";
      }, 6000);
    }
  }

  rejectPendingLlm(): void {
    if (!this.canRejectLlm || this.rejecting || this.isProcessed) return;

    this.rejecting = true;

    this.receiptApi.rejectLlm(this.receiptId).subscribe({
      next: () => {
        this.rejecting = false;
        this.loadReceipt(this.receiptId);
      },
      error: (err) => {
        console.error(err);
        this.rejecting = false;
        alert("Failed to reject LLM result");
      },
    });
  }

  sanitiseItems() {
    this.items = this.items.map((it) => ({
      ...it,
      name: (it.name || "").trim(),
      unitPrice: Math.max(0, Number(it.unitPrice) || 0),
      confidence: Math.max(0, Math.min(100, Number(it.confidence) || 0)),
    }));
  }

  recalcTotal() {
    const s = Number(this.subtotal || 0);
    const t = Number(this.tax || 0);
    this.total = Math.round((s + t) * 100) / 100;
  }

  trackByItemId(_index: number, it: { id: string }) { return it.id; }
  trackByIndex(index: number): number { return index; }

  formatMoney(n: number, currency = "CAD") {
    try {
      return new Intl.NumberFormat("en-CA", { style: "currency", currency }).format(n);
    } catch {
      return `$${n.toFixed(2)}`;
    }
  }

  itemStatusLabel(s: ItemValidationStatus) {
    switch (s) {
      case "OUTLIER": return "Outlier";
      case "SUSPICIOUS_PRICE": return "Price Warn";
      case "NAME_ISSUE": return "Name Issue";
      case "CHECK": return "Check";
      default: return "OK";
    }
  }

  itemStatusClass(s: ItemValidationStatus) {
    switch (s) {
      case "OUTLIER":
        return "bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300";
      case "SUSPICIOUS_PRICE":
        return "bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300";
      case "NAME_ISSUE":
        return "bg-indigo-100 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300";
      case "CHECK":
        return "bg-yellow-100 text-yellow-700 dark:bg-yellow-500/10 dark:text-yellow-300";
      default:
        return "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300";
    }
  }

  confidenceClass(c: number) {
    if (c >= 90) return "bg-emerald-100 text-emerald-700";
    if (c >= 75) return "bg-blue-100 text-blue-700";
    return "bg-amber-100 text-amber-700";
  }

  labelClass(label: string): string {
    if (!label || label === "O") {
      return "bg-gray-200/50 text-gray-700 dark:bg-white/[0.08] dark:text-gray-300";
    }
    if (label.includes("PRICE")) {
      return "bg-green-200/50 text-green-700 dark:bg-green-500/20 dark:text-green-400";
    }
    if (label.includes("MENU")) {
      return "bg-blue-200/50 text-blue-700 dark:bg-blue-500/20 dark:text-blue-400";
    }
    if (label.includes("SUM")) {
      return "bg-orange-200/50 text-orange-700 dark:bg-orange-500/20 dark:text-orange-400";
    }
    if (label.includes("VENDOR") || label.includes("HEADER")) {
      return "bg-purple-200/50 text-purple-700 dark:bg-purple-500/20 dark:text-purple-400";
    }
    return "bg-gray-200/50 text-gray-700 dark:bg-white/[0.08] dark:text-gray-300";
  }

  formatPrice5(value: any): string {
    const num = Number(value);
    if (!isFinite(num)) return "";
    return num.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  }

  parsePrice5(raw: any): number {
    if (raw === null || raw === undefined) return 0;
    const num = Number(String(raw).replace(/\s+/g, ""));
    return isFinite(num) ? num : 0;
  }

  back() { this.router.navigate(["/receipts"]); }

  prevReceipt() {
    if (!this.canPrev) return;
    const nextIndex = this.receiptIndex - 1;
    this.receiptIndex = nextIndex;
    this.router.navigate(["/receipts", this.receiptIds[nextIndex]], {
      state: { ids: this.receiptIds, index: nextIndex },
    });
  }

  nextReceipt() {
    if (!this.canNext) return;
    const nextIndex = this.receiptIndex + 1;
    this.receiptIndex = nextIndex;
    this.router.navigate(["/receipts", this.receiptIds[nextIndex]], {
      state: { ids: this.receiptIds, index: nextIndex },
    });
  }

  approve() {
    if (this.approving || this.isProcessed) return;

    this.approving = true;

    const payload: ApiApproveReceiptPayload = {
      vendor: this.vendor || null,
      receipt_date: this.receiptDate || null,
      subtotal: Number(this.subtotal ?? 0),
      tax: Number(this.tax ?? 0),
      total: Number(this.total ?? 0),

      summary_vendor: this.summary_vendor,
      summary_phone: this.summary_phone,
      summary_address: this.summary_address,
      summary_receipt_date: this.summary_receipt_date,

      items: this.items.map((it, idx) => ({
        line_no: idx + 1,
        item_name: (it.name ?? "").trim(),
        currency: String(it.currency ?? "CAD").toUpperCase(),
        unit_price: Number(it.unitPrice ?? 0),
        confidence: Number(it.confidence ?? 0),
      })),
    };

    this.receiptApi.approveReceipt(this.receiptId, payload).subscribe({
      next: () => {
        this.approving = false;
        this.loadReceipt(this.receiptId);
      },
      error: (err) => {
        console.error(err);
        this.approving = false;
        alert("Failed to approve receipt");
      },
    });
  }

  WrongExtraction() {
    this.receiptApi.updateReceiptStatus(this.receiptId, "Failed").subscribe({
      next: (res) => {
        this.status = res.receipt.status as ReceiptStatus;
        this.loadReceipt(this.receiptId);
      },
      error: (err) => {
        console.error(err);
        alert("Failed to update status");
      },
    });
  }

  ErrorLayout() {
    this.receiptApi.updateReceiptStatus(this.receiptId, "Error").subscribe({
      next: (res) => {
        this.status = res.receipt.status as ReceiptStatus;
        this.loadReceipt(this.receiptId);
      },
      error: (err) => {
        console.error(err);
        alert("Failed to update status");
      },
    });
  }

  addRow() {
    if (this.isProcessed) return;

    this.items = [
      ...this.items,
      {
        id: String(Date.now()),
        name: "",
        currency: "CAD",
        unitPrice: 0,
        confidence: 0,
        validationStatus: "CHECK",
      },
    ];
  }

  deleteRow(rowId: string) {
    if (this.isProcessed) return;
    this.items = this.items.filter((it) => it.id !== rowId);
  }

  exportJson() {
    this.exporting = true;
    const payload = {
      receiptId: this.receiptId,
      receiptName: this.receiptName,
      vendor: this.vendor,
      date: this.receiptDate,
      status: this.status,
      confidence: this.confidence,
      aiReviewStatus: this.aiReviewStatus,
      dataSource: this.dataSource,
      items: this.items,
      summary: {
        subtotal: this.subtotal,
        tax: this.tax,
        total: this.total,
        vendor: this.summary_vendor,
        phone: this.summary_phone,
        address: this.summary_address,
        receiptDate: this.summary_receipt_date,
      },
      rawOcr: this.rawOcrText,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    this.downloadBlob(blob, `receipt_${this.receiptId}.json`);
    setTimeout(() => (this.exporting = false), 350);
  }

  exportCsv() {
    this.exporting = true;
    const header = ["Item Name", "Currency", "Unit Price", "Confidence", "Validation"];
    const rows = this.items.map((it) => [
      `"${(it.name || "").replace(/"/g, '""')}"`,
      it.currency,
      String(it.unitPrice ?? ""),
      String(it.confidence ?? ""),
      it.validationStatus,
    ]);
    const csv = [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    this.downloadBlob(blob, `receipt_${this.receiptId}.csv`);
    setTimeout(() => (this.exporting = false), 350);
  }

  private downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 800);
  }
}