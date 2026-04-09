import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  OnInit,
  OnDestroy,
  AfterViewInit,
  ViewChild,
  ElementRef,
} from '@angular/core';
import { Router } from '@angular/router';
import { Subject, forkJoin } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { Chart, registerables } from 'chart.js';
import { ReceiptApiService } from '../../../shared/services/receipt-api.service';
import {
  DashboardSummary,
  DashboardValidationSummary,
  DashboardMonthlyPoint,
  DashboardConfidenceDistribution,
  DashboardActivityItem,
} from '../../../shared/services/receipt-api.service';

Chart.register(...registerables);

// ── Types ─────────────────────────────────────────────────────────────────────

type DashboardReceiptRow = {
  receipt_id:        number;
  vendor:            string | null;
  receipt_date:      string;
  total:             number;
  ai_review_status:  string;
  extraction_source: string;
  confidence:        number;
  initials:          string;
  colorClass:        string;
};

type MetricCard = {
  label:     string;
  value:     string;
  badge:     string;
  badgeType: string;
};

type PipelineRow = {
  label: string;
  pct:   number;
  color: string;
};

type ValidationFlag = {
  label:    string;
  count:    number;
  severity: 'danger' | 'warning' | 'secondary';
};

// ── Constants ─────────────────────────────────────────────────────────────────

const VENDOR_COLORS = [
  'dot-blue', 'dot-teal', 'dot-amber',
  'dot-pink', 'dot-purple', 'dot-coral', 'dot-green',
];

const STATUS_LABELS: Record<string, string> = {
  approved:      'Approved',
  pending:       'Pending',
  not_requested: 'OCR only',
  rejected:      'Rejected',
};

// ── Component ─────────────────────────────────────────────────────────────────

@Component({
  selector: 'app-ecommerce',
  standalone: true,
  imports: [CommonModule, DatePipe, DecimalPipe],
  templateUrl: './ecommerce.component.html',
  styleUrl: './ecommerce.component.css',
})
export class EcommerceComponent implements OnInit, AfterViewInit, OnDestroy {

  @ViewChild('spendChart') spendChartRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('confChart')  confChartRef!:  ElementRef<HTMLCanvasElement>;

  private destroy$ = new Subject<void>();
  private spendChartInstance?: Chart;
  private confChartInstance?:  Chart;
  private chartsReady = false;
  private dataLoaded  = false;

  currentMonth = new Date().toLocaleString('default', { month: 'long', year: 'numeric' });

  pendingReviewCount = 0;
  loading  = true;
  errorMsg = '';

  metricCards: MetricCard[] = [];

  pipelineRows: PipelineRow[] = [
    { label: 'Processed',   pct: 0, color: '#378ADD' },
    { label: 'OCR only',    pct: 0, color: '#1D9E75' },
    { label: 'OCR + LLM',   pct: 0, color: '#7F77DD' },
    { label: 'AI approved', pct: 0, color: '#1D9E75' },
    { label: 'AI rejected', pct: 0, color: '#E24B4A' },
    { label: 'Failed',      pct: 0, color: '#EF9F27' },
  ];

  validationFlags: ValidationFlag[] = [
    { label: 'Price outliers',       count: 0, severity: 'danger'    },
    { label: 'Subtotal mismatch',    count: 0, severity: 'warning'   },
    { label: 'Name quality issues',  count: 0, severity: 'warning'   },
    { label: 'Price range warnings', count: 0, severity: 'warning'   },
    { label: 'Low confidence items', count: 0, severity: 'secondary' },
    { label: 'Missing vendor',       count: 0, severity: 'secondary' },
  ];

  recentReceipts: DashboardReceiptRow[] = [];
  activityFeed: DashboardActivityItem[] = [];

  private monthlyLabels: string[] = [];
  private monthlySpend:  number[] = [];
  private monthlyCount:  number[] = [];
  private confOcr = [0, 0, 0];
  private confLlm = [0, 0, 0];

  constructor(
    private router:     Router,
    private receiptApi: ReceiptApiService,
  ) {}

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  ngOnInit() {
    this.loadDashboard();
  }

  ngAfterViewInit() {
    this.initCharts();
    this.chartsReady = true;
    if (this.dataLoaded) this.updateCharts();
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
    this.spendChartInstance?.destroy();
    this.confChartInstance?.destroy();
  }

  // ── Data loading ───────────────────────────────────────────────────────────

  private loadDashboard() {
    this.loading  = true;
    this.errorMsg = '';

    forkJoin({
      summary:    this.receiptApi.getDashboardSummary(),
      receipts:   this.receiptApi.listReceipts(),
      validation: this.receiptApi.getDashboardValidationSummary(),
      monthly:    this.receiptApi.getDashboardMonthlyTrend(),
      confidence: this.receiptApi.getDashboardConfidenceDistribution(),
      activity:   this.receiptApi.getDashboardActivity(),
    })
    .pipe(takeUntil(this.destroy$))
    .subscribe({
      next: ({ summary, receipts, validation, monthly, confidence, activity }) => {
        this.applyMetrics(summary);
        this.applyPipeline(summary);
        this.recentReceipts = this.decorateReceipts(receipts.slice(0, 10));
        this.applyValidation(validation);
        this.applyMonthly(monthly);
        this.confOcr      = confidence.ocr;
        this.confLlm      = confidence.llm;
        this.activityFeed = activity;
        this.loading      = false;
        this.dataLoaded   = true;
        if (this.chartsReady) this.updateCharts();
      },
      error: (err) => {
        console.error(err);
        this.loading  = false;
        this.errorMsg = 'Failed to load dashboard';
      },
    });
  }

  // ── Apply helpers ──────────────────────────────────────────────────────────

  private applyMetrics(s: DashboardSummary) {
    this.pendingReviewCount = s.pending_review;

    this.metricCards = [
      {
        label:     'Total receipts',
        value:     s.total_receipts.toLocaleString(),
        badge:     `+${s.new_this_week} this week`,
        badgeType: 'success',
      },
      {
        label:     'Total spend (CAD)',
        value:     this.formatMoney(s.total_spend),
        badge:     `across ${s.vendor_count} vendors`,
        badgeType: 'info',
      },
      {
        label:     'Avg confidence',
        value:     Math.round(s.avg_confidence) + '%',
        badge:     'OCR + LLM',
        badgeType: 'success',
      },
      {
        label:     'Pending AI review',
        value:     String(s.pending_review),
        badge:     'needs action',
        badgeType: s.pending_review > 0 ? 'warn' : 'success',
      },
      {
        label:     'Flagged items',
        value:     String(s.flagged_items),
        badge:     'outliers & errors',
        badgeType: s.flagged_items > 0 ? 'danger' : 'success',
      },
    ];
  }

  private applyPipeline(s: DashboardSummary) {
    const total = s.total_receipts || 1;
    const pct   = (n: number) => Math.round((n / total) * 100);

    this.pipelineRows[0].pct = pct(total - s.failed_count);
    this.pipelineRows[1].pct = pct(s.ocr_only_count);
    this.pipelineRows[2].pct = pct(s.ocr_llm_count);
    this.pipelineRows[3].pct = pct(s.approved_count);
    this.pipelineRows[4].pct = pct(s.rejected_count);
    this.pipelineRows[5].pct = pct(s.failed_count);
  }

  private decorateReceipts(rows: any[]): DashboardReceiptRow[] {
    const colorMap = new Map<string, string>();
    let colorIdx = 0;

    return rows.map((r: any) => {
      const key = r.vendor ?? 'Unknown';
      if (!colorMap.has(key)) {
        colorMap.set(key, VENDOR_COLORS[colorIdx++ % VENDOR_COLORS.length]);
      }
      return {
        receipt_id:        r.id ?? r.receipt_id,
        vendor:            r.vendor ?? null,
        receipt_date:      r.date ?? r.receipt_date ?? '',
        total:             Number(r.total ?? 0),
        ai_review_status:  r.aiReviewStatus ?? r.ai_review_status ?? 'not_requested',
        extraction_source: r.extraction_source ?? 'ocr',
        confidence:        Number(r.confidence ?? 0),
        initials:          this.initials(r.vendor),
        colorClass:        colorMap.get(key) ?? 'dot-blue',
      };
    });
  }

  private applyValidation(v: DashboardValidationSummary) {
    this.validationFlags[0].count = v.price_outliers;
    this.validationFlags[1].count = v.subtotal_mismatch;
    this.validationFlags[2].count = v.name_quality_issues;
    this.validationFlags[3].count = v.price_range_warnings;
    this.validationFlags[4].count = v.low_confidence_items;
    this.validationFlags[5].count = v.missing_vendor;
  }

  private applyMonthly(rows: DashboardMonthlyPoint[]) {
    this.monthlyLabels = rows.map(r => r.month);
    this.monthlySpend  = rows.map(r => r.spend);
    this.monthlyCount  = rows.map(r => r.count);
  }

  // ── Charts ─────────────────────────────────────────────────────────────────

  private initCharts() {
    const gridColor = 'rgba(0,0,0,0.06)';
    const tickColor = 'rgba(0,0,0,0.45)';

    this.spendChartInstance = new Chart(this.spendChartRef.nativeElement, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Spend (CAD)',
            data: [],
            backgroundColor: '#378ADD',
            borderRadius: 4,
            yAxisID: 'y',
          },
          {
            label: 'Receipts',
            data: [],
            type: 'line' as any,
            borderColor: '#1D9E75',
            backgroundColor: 'transparent',
            pointBackgroundColor: '#1D9E75',
            pointRadius: 3,
            borderWidth: 2,
            tension: 0.3,
            yAxisID: 'y2',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { mode: 'index' },
        },
        scales: {
          x:  { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 } } },
          y:  {
                grid: { color: gridColor },
                ticks: {
                  color: tickColor,
                  font: { size: 11 },
                  callback: (v: number) => '$' + v.toLocaleString(),
                },
                position: 'left',
              },
          y2: { grid: { display: false }, ticks: { color: tickColor, font: { size: 11 } }, position: 'right' },
        },
      },
    });

    this.confChartInstance = new Chart(this.confChartRef.nativeElement, {
      type: 'bar',
      data: {
        labels: ['High (80–100%)', 'Medium (60–79%)', 'Low (<60%)'],
        datasets: [
          { label: 'OCR only',  data: [], backgroundColor: '#378ADD', borderRadius: 3 },
          { label: 'OCR + LLM', data: [], backgroundColor: '#7F77DD', borderRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 }, maxRotation: 0 } },
          y: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 } } },
        },
      },
    });
  }

  private updateCharts() {
    if (this.spendChartInstance) {
      this.spendChartInstance.data.labels           = this.monthlyLabels;
      this.spendChartInstance.data.datasets[0].data = this.monthlySpend;
      this.spendChartInstance.data.datasets[1].data = this.monthlyCount;
      this.spendChartInstance.update();
    }
    if (this.confChartInstance) {
      this.confChartInstance.data.datasets[0].data = this.confOcr;
      this.confChartInstance.data.datasets[1].data = this.confLlm;
      this.confChartInstance.update();
    }
  }

  // ── Template helpers ───────────────────────────────────────────────────────

  initials(vendor: string | null): string {
    if (!vendor) return '?';
    return vendor
      .split(/\s+/)
      .slice(0, 2)
      .map(w => w[0].toUpperCase())
      .join('');
  }

  // Replaces the pipe — called directly from template as statusLabel(r.ai_review_status)
  statusLabel(status: string): string {
    return STATUS_LABELS[status] ?? status;
  }

  statusPill(status: string): string {
    const map: Record<string, string> = {
      approved:      'pill-ok',
      pending:       'pill-pend',
      not_requested: 'pill-none',
      rejected:      'pill-rej',
    };
    return map[status] ?? 'pill-none';
  }

  formatMoney(n: number, currency = 'CAD'): string {
    try {
      return new Intl.NumberFormat('en-CA', { style: 'currency', currency }).format(n);
    } catch {
      return `$${n.toFixed(2)}`;
    }
  }

  // ── Navigation ─────────────────────────────────────────────────────────────

  uploadReceipt() { this.router.navigate(["/scan-receipt"]); }
  reviewPending()         { this.router.navigate(['/receipts'], { queryParams: { ai_review_status: 'pending' } }); }
  viewAll()               { this.router.navigate(['/receipts']); }
  resolveFlags()          { this.router.navigate(['/receipts'], { queryParams: { flagged: true } }); }
  openReceipt(id: number) { this.router.navigate(['/receipts', id]); }
}