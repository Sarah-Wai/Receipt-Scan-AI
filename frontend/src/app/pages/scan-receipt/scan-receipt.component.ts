import { Component, OnInit } from "@angular/core";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { Router } from "@angular/router";
import { firstValueFrom } from "rxjs";

import { ReceiptApiService } from "../../shared/services/receipt-api.service";

type ScanItem = {
  id: string;
  kind: "image" | "pdf";
  source: "upload" | "camera";
  name: string;
  file?: File;        // for upload items
  dataUrl?: string;   // for camera items (base64)
  previewUrl: string; // image: dataUrl or objectURL; pdf: empty
};

@Component({
  selector: "app-scan-receipt",
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: "./scan-receipt.component.html",
  styleUrl: "./scan-receipt.component.css",
})
export class ScanReceiptComponent implements OnInit {
  readonly MAX_ITEMS = 10;

  removeBg = false;
  deskew = false;

  items: ScanItem[] = [];
  selectedId?: string;

  limitMsg = "";
  statusMsg = "";

  skippedNames: string[] = [];

  isProcessing = false;
  progressLabel = "";
  private readonly NEW_KEY = "receipt_new_ids";
  newReceiptIds: number[] = [];

  constructor(
    private router: Router,
    private receiptApi: ReceiptApiService,
  ) {}

  ngOnInit(): void {
    const batch = sessionStorage.getItem("captured_receipts");
    if (batch) {
      try {
        const arr: string[] = JSON.parse(batch);
        arr.forEach((dataUrl, idx) =>
          this.addCameraDataUrl(dataUrl, `camera_${idx + 1}.jpg`),
        );
      } catch {
        // ignore
      }
      sessionStorage.removeItem("captured_receipts");
    }

    const single = sessionStorage.getItem("captured_receipt");
    if (single) {
      this.addCameraDataUrl(single, "camera_1.jpg");
      sessionStorage.removeItem("captured_receipt");
    }

    if (this.items[0]) this.select(this.items[0].id);
  }

  openCameraPage() {
    this.router.navigate(["/camera-scan"]);
  }

  onFiles(e: Event) {
    const input = e.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = "";

    this.skippedNames = [];

    const remaining = this.MAX_ITEMS - this.items.length;

    if (remaining <= 0) {
      this.skippedNames = files.map((f) => f.name);
      this.limitMsg = `Max ${this.MAX_ITEMS} items per batch.`;
      return;
    }

    const toAdd = files.slice(0, remaining);
    const skipped = files.slice(remaining);

    for (const f of toAdd) {
      if (f.type === "application/pdf") this.addPdfFile(f);
      else this.addImageFile(f);
    }

    this.skippedNames = skipped.map((f) => f.name);

    if (this.skippedNames.length > 0) {
      this.limitMsg = `Only added ${toAdd.length}. Max ${this.MAX_ITEMS} per batch.`;
    } else {
      this.limitMsg = "";
    }

    if (!this.selectedId && this.items[0]) this.select(this.items[0].id);
  }

  private addImageFile(file: File) {
    const id = this.makeId();
    const url = URL.createObjectURL(file);

    this.items.unshift({
      id,
      kind: "image",
      source: "upload",
      name: file.name,
      file,
      previewUrl: url,
    });
  }

  private addPdfFile(file: File) {
    const id = this.makeId();
    this.items.unshift({
      id,
      kind: "pdf",
      source: "upload",
      name: file.name,
      file,
      previewUrl: "",
    });
  }

  private addCameraDataUrl(dataUrl: string, name: string) {
    if (this.items.length >= this.MAX_ITEMS) {
      this.limitMsg = `Max ${this.MAX_ITEMS} items per batch.`;
      return;
    }

    const id = this.makeId();
    this.items.unshift({
      id,
      kind: "image",
      source: "camera",
      name,
      dataUrl,
      previewUrl: dataUrl,
    });
  }

  get selectedItem(): ScanItem | undefined {
    return this.items.find((i) => i.id === this.selectedId);
  }

  select(id: string) {
    this.selectedId = id;
    this.statusMsg = "";
  }

  remove(id: string) {
    const idx = this.items.findIndex((i) => i.id === id);
    if (idx < 0) return;

    const item = this.items[idx];

    if (item.source === "upload" && item.kind === "image" && item.previewUrl) {
      URL.revokeObjectURL(item.previewUrl);
    }

    this.items.splice(idx, 1);

    if (this.selectedId === id) {
      this.selectedId = this.items[0]?.id;
    }
  }

  clearAll() {
    for (const it of this.items) {
      if (it.source === "upload" && it.kind === "image" && it.previewUrl) {
        URL.revokeObjectURL(it.previewUrl);
      }
    }
    this.items = [];
    this.selectedId = undefined;
    this.limitMsg = "";
    this.statusMsg = "";
  }

  private saveNewIds() {
    localStorage.setItem(this.NEW_KEY, JSON.stringify(this.newReceiptIds));
  }

  async runExtraction() {
  if (this.items.length === 0 || this.isProcessing) return;

  this.isProcessing = true;
  this.progressLabel = "Preparing files…";
  this.statusMsg = "";
  this.newReceiptIds = [];

  try {
    const pdfItems = this.items.filter((x) => x.kind === "pdf");
    if (pdfItems.length > 0) {
      this.statusMsg =
        "PDF upload is not supported in the current backend yet. Please use images only.";
      return;
    }

    const filesToUpload = await this.buildUploadFilesFromItems(
      this.items.filter((x) => x.kind === "image"),
    );

    if (filesToUpload.length === 0) {
      this.statusMsg = "No valid image files found.";
      return;
    }

    this.progressLabel = "Uploading…";
    const uploadRes = await firstValueFrom(
      this.receiptApi.uploadReceipts(filesToUpload),
    );

    if (!uploadRes?.ok || !uploadRes.job_id) {
      throw new Error("Upload failed");
    }

    // FIX: placeholderId is only used as the URL parameter for the process
    // call (which the backend accepts but ignores for DB purposes).
    // It must NOT be used for anything after the process response arrives.
    const placeholderId = await this.createPlaceholderReceiptId(filesToUpload);

    this.progressLabel = "Processing receipt…";
    const processRes = await firstValueFrom(
      this.receiptApi.processReceipt(placeholderId, uploadRes.job_id),
    );

    if (!processRes?.ok || !processRes.result?.success) {
      throw new Error(processRes?.result?.error || "Process failed");
    }

    // FIX: use the real DB receipt_id returned by the backend, not the
    // placeholder timestamp. All subsequent calls must use this value.
    const receiptId = processRes.receipt_id;
    if (!receiptId) {
      throw new Error("Backend did not return a receipt_id after processing.");
    }

    this.newReceiptIds = [receiptId];
    this.saveNewIds();

    //this.progressLabel = "Applying auto-fix…";
    //try {
      //await firstValueFrom(this.receiptApi.autoFixWithLlm(receiptId));
    //} catch (llmErr) {
      //console.warn("LLM auto-fix failed:", llmErr);
    //}

    this.progressLabel = "Done";
    this.router.navigate(["/receipts", receiptId]);
  } catch (err) {
    console.error(err);
    this.statusMsg = "Processing failed. Please try again.";
  } finally {
    this.isProcessing = false;
    this.progressLabel = "";
  }
}

  private async buildUploadFilesFromItems(items: ScanItem[]): Promise<File[]> {
    const out: File[] = [];

    for (const item of items) {
      if (item.source === "upload" && item.file) {
        out.push(item.file);
        continue;
      }

      if (item.source === "camera" && item.dataUrl) {
        const file = this.dataUrlToFile(item.dataUrl, item.name || "camera.jpg");
        out.push(file);
      }
    }

    return out;
  }

  private dataUrlToFile(dataUrl: string, filename: string): File {
    const [meta, base64] = dataUrl.split(",");
    const mimeMatch = meta.match(/data:(.*?);base64/);
    const mime = mimeMatch?.[1] || "image/jpeg";

    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);

    for (let i = 0; i < len; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    return new File([bytes], filename, { type: mime });
  }

  private async createPlaceholderReceiptId(files: File[]): Promise<number> {
    // Replace this with your real "create receipt row" API if you already have one.
    // For now, generate a temporary local ID.
    return Date.now();
  }

  private makeId() {
    return (
      (crypto as any)?.randomUUID?.() ??
      `${Date.now()}_${Math.random().toString(16).slice(2)}`
    );
  }
}