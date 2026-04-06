import { CommonModule } from "@angular/common";
import { Component, EventEmitter, Input, Output } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { ButtonComponent } from "../ui/button/button.component";
import { TableDropdownComponent } from "../common/table-dropdown/table-dropdown.component";
import { BadgeComponent } from "../ui/badge/badge.component";
import { ReceiptRow, ReceiptStatus } from "../../../models/receipt.model";

@Component({
  selector: "app-receipt-table",
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ButtonComponent,
    TableDropdownComponent,
    BadgeComponent,
  ],
  templateUrl: "./receipt-table.component.html",
  styleUrl: "./receipt-table.component.css",
})
export class ReceiptTableComponent {
  // inputs
  @Input() title = "Receipts";
  @Input() rows: ReceiptRow[] = [];
  @Input() itemsPerPage = 10;
  @Input() newReceiptIds: number[] = [];
  private newSet = new Set<number>();

  // outputs (parent handles route/delete/backend)
  @Output() rowClick = new EventEmitter<number>();
  @Output() viewClick = new EventEmitter<{ id: number; ids: number[]; index: number }>();
  @Output() deleteClick = new EventEmitter<number>();

  // search + paging state (used by your HTML)
  searchText = "";
  currentPage = 1;

  // called by HTML: (ngModelChange)="onSearchChange()"
  onSearchChange() {
    this.currentPage = 1;
  }

  // ----- filtering (search bar) -----
  get filteredRows(): ReceiptRow[] {
    const q = (this.searchText || "").trim().toLowerCase();
    if (!q) return this.rows;

    return this.rows.filter((r) => {
      const hay = [
        r.id,
        r.vendor,
        r.date,
        String(r.total),
        r.status,
        String(r.confidence),
      ]
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
  }

  // used by HTML: totalPages
  get totalPages(): number {
    return Math.max(1, Math.ceil(this.filteredRows.length / this.itemsPerPage));
  }

  // used by HTML: currentItems
  get currentItems(): ReceiptRow[] {
    const start = (this.currentPage - 1) * this.itemsPerPage;
    return this.filteredRows.slice(start, start + this.itemsPerPage);
  }

  // used by HTML: goToPage(...)
  goToPage(page: number) {
    if (page < 1) return;
    if (page > this.totalPages) return;
    this.currentPage = page;
  }

  // row click (your HTML uses: (click)="onRow(item.id)")
  //onRow(id: number) {
    //this.rowClick.emit(id);
  //}

  // dropdown actions (your HTML uses: (click)="onView($event, item.id)")
 onView(e: MouseEvent, id: number, i: number) {
  e.stopPropagation();

  const ids = (this.filteredRows || []).map(r => Number(r.id)).filter(Number.isFinite);

  // global index across filtered list = pageOffset + i
  const start = (this.currentPage - 1) * this.itemsPerPage;
  const index = start + i;

  this.viewClick.emit({ id: Number(id), ids, index });
}


  // badge color mapping (your HTML uses: [color]="getBadgeColor(item.status)")
  getBadgeColor(
    status: ReceiptStatus,
  ): "success" | "warning" | "error" | "info" {
    if (status === "Approved") return "success";
    if (status === "Pending") return "warning";
    if (status === "Failed") return "error";
    return "info";
  }

  // total formatting (your HTML uses: {{ formatTotal(item.total) }})
  formatTotal(total: number): string {
    try {
      return new Intl.NumberFormat("en-CA", {
        style: "currency",
        currency: "CAD",
      }).format(total);
    } catch {
      return `$${Number(total || 0).toFixed(2)}`;
    }
  }

  ngOnChanges(): void {
    this.newSet = new Set(this.newReceiptIds ?? []);

    // Optional: if current page is now beyond total pages after filtering/data refresh, clamp it
    if (this.currentPage > this.totalPages) {
      this.currentPage = this.totalPages;
    }
    if (this.currentPage < 1) {
      this.currentPage = 1;
    }
  }

  isNew(id: number): boolean {
    return this.newSet.has(id);
  }
}
