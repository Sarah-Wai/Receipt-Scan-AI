import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';


declare const cv: any;

type ReceiptShot = {
  id: string;
  blob: Blob;
  url: string;
  source: 'camera' | 'upload';
  createdAt: number;
};

@Component({
  selector: 'app-camera-scan',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './camera-scan.component.html',
  styleUrl: './camera-scan.component.css',
})
export class CameraScanComponent implements AfterViewInit, OnDestroy {
  @ViewChild('video') videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;   // full-res capture
  @ViewChild('work') workRef!: ElementRef<HTMLCanvasElement>;       // low-res detection
  @ViewChild('overlay') overlayRef!: ElementRef<HTMLCanvasElement>; // overlay polygon

  // ============================
  // Batch / multi receipts
  // ============================
  readonly MAX_RECEIPTS = 10;
  receipts: ReceiptShot[] = [];

  selectedId?: string;
  selectedUrl?: string;
  selectedBlob?: Blob;

  limitMsg = '';
  uploadMsg = '';
  isUploading = false;

  // ============================
  // Camera / OpenCV
  // ============================
  private stream?: MediaStream;
  private usingFront = false;

  showGrid = true;

  canCapture = false;
  cameraHint = 'Loading OpenCV...';
  private cvReady = false;
  private detectTimer?: number;

  // ✅ FIX: separate raw corners (stability) vs smooth corners (overlay)
  private lastRawCorners: { x: number; y: number }[] | null = null;
  private lastSmoothCorners: { x: number; y: number }[] | null = null;

  private lastWorkCorners: { x: number; y: number }[] | null = null;
  private lastWorkW = 0;
  private lastWorkH = 0;

  // ============================
  // Output / Quality toggles
  // ============================
  saveAsPng = false;

  // Adobe-like enhancement toggles
  adobeLikeEnhance = true;     // white background + strong contrast (Adobe-like)
  adobeBW = true;              // black/white output (best for OCR)
  keepColorPreview = false;    // if true, keeps grayscale (less “scan” look)

  // Sharpen (IMPORTANT: we disable automatically for BW to avoid noise)
  sharpenAfterWarp = true;
  sharpenAmount = 0.12;        // 0.08 ~ 0.20

  // ============================
  // Long Receipt Mode (SMOOTH guided stitch)
  // ============================
  isLongMode = false;
  private longTimer?: number;

  private stitchCanvas?: HTMLCanvasElement;
  private stitchCtx?: CanvasRenderingContext2D | null;

  private longFrameCanvas?: HTMLCanvasElement;
  private longFrameCtx?: CanvasRenderingContext2D | null;

  private prevFrameCanvas?: HTMLCanvasElement;
  private prevFrameCtx?: CanvasRenderingContext2D | null;

  private noNewCount = 0;

  private readonly LONG_CAPTURE_MS = 320;
  private readonly LONG_FRAME_W = 720;
  private readonly LONG_MAX_STITCH_H = 14000;
  private readonly LONG_BLUR_MIN = 70;

  private readonly LONG_MIN_APPEND_H = 90;
  private readonly LONG_STOP_NO_NEW = 10;

  private readonly LONG_MATCH_SCALE = 0.40;
  private readonly LONG_CENTER_STRIP_W_FRAC = 0.55;
  private readonly LONG_STRIP_H_FRAC = 0.22;

  private readonly LONG_MIN_MATCH_SCORE = 0.55;
  private readonly LONG_SEAM_PAD = 8;

  // ✅ Stability score (unchanged concept, but tuned + fixed raw/smooth bug)
  private stableScore = 0;              // 0..100
  private readonly STABLE_ON = 65;      // green turns on
  private readonly STABLE_OFF = 45;     // hysteresis: stays green until drops below
  private readonly STABLE_GAIN = 18;    // add when good
  private readonly STABLE_DECAY = 10;   // subtract when shaky
  private readonly STABLE_MAX = 100;

  // Easier minimums (so green is reachable)
  private readonly MIN_AREA_FRAC = 0.09;      // was 0.12
  private readonly MIN_QUAD_QUALITY = 0.35;   // (0..1)


  constructor(private router: Router, private cdr: ChangeDetectorRef) { }

  // ============================
  // UI actions
  // ============================
  toggleGrid() { this.showGrid = !this.showGrid; }

  goBack() {
    this.router.navigate(['/scan-receipt']);
  }

  // ============================
  // Lifecycle
  // ============================
  async ngAfterViewInit() {
    await this.waitForOpenCV();
    await this.startCamera();
  }

  ngOnDestroy() {
    this.stopLongReceipt(false);
    this.stopCamera();
    this.receipts.forEach(r => URL.revokeObjectURL(r.url));
  }

  // ============================
  // OpenCV ready
  // ============================
  private async waitForOpenCV() {
    this.cameraHint = 'Loading OpenCV...';

    const maxWaitMs = 12000;
    const start = Date.now();

    while (!(window as any).cv) {
      if (Date.now() - start > maxWaitMs) {
        this.cameraHint = 'OpenCV not loaded. Check index.html script.';
        return;
      }
      await new Promise(r => setTimeout(r, 50));
    }

    const cvAny: any = (window as any).cv;

    await new Promise<void>((resolve) => {
      if (cvAny?.Mat) return resolve();
      cvAny['onRuntimeInitialized'] = () => resolve();
    });

    this.cvReady = true;
    this.cameraHint = 'Point camera at receipt';
  }

  // ============================
  // Camera start/stop
  // ============================
  async startCamera() {
    this.stopLongReceipt(false);
    this.stopCamera();

    const constraints: MediaStreamConstraints = {
      video: {
        facingMode: this.usingFront ? 'user' : { ideal: 'environment' },
        width: { ideal: 2560 },
        height: { ideal: 1440 },
        frameRate: { ideal: 30, max: 30 },
      } as any,
      audio: false,
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);

    try {
      const track = this.stream.getVideoTracks?.()[0];
      await track?.applyConstraints?.({
        advanced: [{ focusMode: 'continuous', exposureMode: 'continuous', whiteBalanceMode: 'continuous' }]
      } as any);
      console.log('camera settings:', track?.getSettings?.());
    } catch { }

    const video = this.videoRef.nativeElement;
    video.srcObject = this.stream;

    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    try { await video.play(); } catch { }
    await new Promise(r => setTimeout(r, 450)); // let focus settle

    // ✅ Reset stability state on start (important)
    this.stableScore = 0;
    this.lastRawCorners = null;
    this.lastSmoothCorners = null;

    this.resizeOverlayToVideoBox();
    this.startDetectLoop();
    window.addEventListener('resize', this.onResize);
  }

  stopCamera() {
    this.stopDetectLoop();
    window.removeEventListener('resize', this.onResize);

    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = undefined;
    }

    this.canCapture = false;

    // ✅ Reset stability state on stop
    this.stableScore = 0;
    this.lastRawCorners = null;
    this.lastSmoothCorners = null;

    this.lastWorkCorners = null;
    this.clearOverlay();
  }

  async switchCamera() {
    this.usingFront = !this.usingFront;
    await this.startCamera();
  }

  private onResize = () => {
    this.resizeOverlayToVideoBox();
  };

  private resizeOverlayToVideoBox() {
    const video = this.videoRef?.nativeElement;
    const overlay = this.overlayRef?.nativeElement;
    if (!video || !overlay) return;

    const rect = video.getBoundingClientRect();
    overlay.width = Math.max(1, Math.floor(rect.width));
    overlay.height = Math.max(1, Math.floor(rect.height));
  }



  // ============================
  // Live detection loop
  // ============================
  private startDetectLoop() {
    if (!this.cvReady) return;
    this.stopDetectLoop();
    this.detectTimer = window.setInterval(() => this.detectOnce(), 280);
  }

  private stopDetectLoop() {
    if (this.detectTimer) {
      clearInterval(this.detectTimer);
      this.detectTimer = undefined;
    }
  }

  private detectOnce() {
    if (!this.cvReady) return;
    if (this.isLongMode) return;

    const video = this.videoRef.nativeElement;
    if (!video.videoWidth || !video.videoHeight) return;

    const work = this.workRef.nativeElement;

    //const W = 420;
    //const H = Math.round((video.videoHeight / video.videoWidth) * W);
    //work.width = W;
    //work.height = H;

    //  FIX: Use wider work canvas so tall narrow receipts aren't squeezed
    const W = 640; // was 420 — wider = better edge detection on narrow receipts
    const H = Math.round((video.videoHeight / video.videoWidth) * W);
    work.width = W;
    work.height = H;

    const ctx = work.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, W, H);

    //  RAW corners for stability
    const raw = this.findReceiptCornersOnCanvas(work);
    const areaFrac = raw ? this.polyAreaFrac(raw) : 0;

    // cache RAW for full-res mapping
    if (raw) {
      this.lastWorkCorners = raw;
      this.lastWorkW = W;
      this.lastWorkH = H;
    }

    const ok = this.updateStability(raw, areaFrac);
    this.canCapture = ok;

    const progress = Math.round(this.stableScore);

    // ✅ Smooth corners ONLY for overlay drawing (stable visual)
    let smooth: { x: number; y: number }[] | null = null;
    if (raw) {
      smooth = this.smoothCorners(this.lastSmoothCorners, raw, 0.75);
      this.lastSmoothCorners = smooth;
    } else {
      this.lastSmoothCorners = null;
    }

    const MIN_AREA_FRAC_HINT = 0.09; // easier than 0.12
    if (!raw) {
      this.cameraHint = 'Point camera at receipt';
    } else if (areaFrac < MIN_AREA_FRAC_HINT) {
      this.cameraHint = 'Move closer — receipt too small';
    } else if (!ok) {
      this.cameraHint = `Hold steady… (${progress}%)`;
    } else if (this.receipts.length >= this.MAX_RECEIPTS) {
      this.cameraHint = `Limit reached (${this.MAX_RECEIPTS}). Process or clear.`;
    } else {
      this.cameraHint = 'Looks good — you can Capture.';
    }

    // draw smoothed overlay
    this.drawOverlayCorners(smooth, W, H);
  }


  private quadQuality(pts: { x: number; y: number }[]): number {
    const d = (a: any, b: any) => Math.hypot(a.x - b.x, a.y - b.y);

    const wTop = d(pts[0], pts[1]);
    const wBot = d(pts[3], pts[2]);
    const hL = d(pts[0], pts[3]);
    const hR = d(pts[1], pts[2]);

    const w = (wTop + wBot) / 2;
    const h = (hL + hR) / 2;
    if (w < 40 || h < 60) return 0;

    const wSim = 1 - Math.min(1, Math.abs(wTop - wBot) / Math.max(1, w));
    const hSim = 1 - Math.min(1, Math.abs(hL - hR) / Math.max(1, h));

    const areaFrac = this.polyAreaFrac(pts);
    const areaOk = Math.min(1, areaFrac / 0.20);

    return 0.45 * wSim + 0.45 * hSim + 0.10 * areaOk;
  }

  private updateStability(
    corners: { x: number; y: number }[] | null,
    areaFrac: number
  ): boolean {
    if (!corners || corners.length !== 4 || areaFrac < this.MIN_AREA_FRAC) {
      this.stableScore = Math.max(0, this.stableScore - this.STABLE_DECAY * 2);
      this.lastRawCorners = corners;
      return this.canCapture ? (this.stableScore >= this.STABLE_OFF) : false;
    }

    const q = this.quadQuality(corners);
    if (q < this.MIN_QUAD_QUALITY) {
      this.stableScore = Math.max(0, this.stableScore - this.STABLE_DECAY * 2);
      this.lastRawCorners = corners;
      return this.canCapture ? (this.stableScore >= this.STABLE_OFF) : false;
    }

    if (!this.lastRawCorners) {
      this.lastRawCorners = corners;
      this.stableScore = Math.max(0, this.stableScore - this.STABLE_DECAY);
      return false;
    }

    const avgMove = this.avgCornerMove(this.lastRawCorners, corners);
    this.lastRawCorners = corners;

    const moveOk = this.isMoveStable(avgMove);

    if (moveOk) {
      this.stableScore = Math.min(this.STABLE_MAX, this.stableScore + 22);
    } else {
      this.stableScore = Math.max(0, this.stableScore - 8);
    }

    if (this.canCapture) return this.stableScore >= this.STABLE_OFF;
    return this.stableScore >= this.STABLE_ON;
  }

  private isMoveStable(avgMove: number): boolean {
    const base = this.lastWorkW > 0 ? this.lastWorkW : 420;
    const thresh = Math.max(9, Math.min(20, base * 0.032));
    return avgMove < thresh;
  }

  private avgCornerMove(a: { x: number; y: number }[], b: { x: number; y: number }[]) {
    let sum = 0;
    for (let i = 0; i < 4; i++) sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    return sum / 4;
  }

  private polyAreaFrac(pts: { x: number; y: number }[]) {
    let a = 0;
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    a = Math.abs(a) / 2;

    const work = this.workRef.nativeElement;
    const denom = work.width * work.height;
    return denom > 0 ? a / denom : 0;
  }

  private smoothCorners(
    prev: { x: number; y: number }[] | null,
    next: { x: number; y: number }[],
    alpha = 0.75
  ) {
    if (!prev) return next;
    return next.map((p, i) => ({
      x: alpha * prev[i].x + (1 - alpha) * p.x,
      y: alpha * prev[i].y + (1 - alpha) * p.y,
    }));
  }

  private clearOverlay() {
    const overlay = this.overlayRef?.nativeElement;
    if (!overlay) return;
    const octx = overlay.getContext('2d');
    if (!octx) return;
    octx.clearRect(0, 0, overlay.width, overlay.height);
  }

  private drawOverlayCorners_old(
    corners: { x: number; y: number }[] | null,
    workW: number,
    workH: number
  ) {
    const overlay = this.overlayRef.nativeElement;
    const octx = overlay.getContext('2d');
    if (!octx) return;

    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!corners) return;

    const map = this.getContainMapping(workW, workH, overlay.width, overlay.height);
    const pts = corners.map(p => ({
      x: map.offsetX + p.x * map.scale,
      y: map.offsetY + p.y * map.scale,
    }));

    octx.lineWidth = this.canCapture ? 4 : 3;
    octx.strokeStyle = this.canCapture ? 'rgba(34,197,94,0.95)' : 'rgba(59,130,246,0.95)';

    octx.beginPath();
    octx.moveTo(pts[0].x, pts[0].y);
    octx.lineTo(pts[1].x, pts[1].y);
    octx.lineTo(pts[2].x, pts[2].y);
    octx.lineTo(pts[3].x, pts[3].y);
    octx.closePath();
    octx.stroke();

    for (const p of pts) {
      octx.fillStyle = octx.strokeStyle;
      octx.beginPath();
      octx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      octx.fill();
    }
  }

  private drawOverlayCorners(
    corners: { x: number; y: number }[] | null,
    workW: number,
    workH: number
  ) {
    const overlay = this.overlayRef.nativeElement;
    const octx = overlay.getContext('2d');
    if (!octx) return;

    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!corners) return;

    const map = this.getContainMapping(workW, workH, overlay.width, overlay.height);
    const pts = corners.map(p => ({
      x: map.offsetX + p.x * map.scale,
      y: map.offsetY + p.y * map.scale,
    }));

    const isGreen = this.canCapture;

    // ── Colors ────────────────────────────────────────────────
    const strokeColor = isGreen ? 'rgba(34, 220, 90, 1.0)' : 'rgba(59, 130, 246, 1.0)';
    const glowColor = isGreen ? 'rgba(34, 220, 90, 0.65)' : 'rgba(59, 130, 246, 0.55)';
    const fillColor = isGreen ? 'rgba(34, 220, 90, 0.12)' : 'rgba(59, 130, 246, 0.10)';
    const dotFill = isGreen ? 'rgba(34, 220, 90, 1.0)' : 'rgba(59, 130, 246, 1.0)';

    // ── Helper: draw the quad path ─────────────────────────────
    const drawPath = () => {
      octx.beginPath();
      octx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) octx.lineTo(pts[i].x, pts[i].y);
      octx.closePath();
    };

    // ── 1. Semi-transparent fill ───────────────────────────────
    octx.save();
    drawPath();
    octx.fillStyle = fillColor;
    octx.fill();
    octx.restore();

    // ── 2. Glow pass (wide, blurred shadow stroke) ─────────────
    octx.save();
    octx.shadowColor = glowColor;
    octx.shadowBlur = 18;
    octx.lineWidth = 5;
    octx.strokeStyle = strokeColor;
    drawPath();
    octx.stroke();
    octx.restore();

    // ── 3. Crisp solid stroke on top ───────────────────────────
    octx.save();
    octx.lineWidth = 3.5;
    octx.strokeStyle = strokeColor;
    octx.shadowBlur = 0;
    drawPath();
    octx.stroke();
    octx.restore();

    // ── 4. Corner dots (white ring + colored fill) ─────────────
    for (const p of pts) {
      // White outline ring
      octx.save();
      octx.beginPath();
      octx.arc(p.x, p.y, 9, 0, Math.PI * 2);
      octx.fillStyle = 'rgba(255,255,255,0.90)';
      octx.shadowColor = glowColor;
      octx.shadowBlur = 10;
      octx.fill();
      octx.restore();

      // Colored inner dot
      octx.save();
      octx.beginPath();
      octx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      octx.fillStyle = dotFill;
      octx.fill();
      octx.restore();
    }

    // ── 5. "READY" label when stable/green ────────────────────
    if (isGreen) {
      const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
      const cy = pts.reduce((s, p) => s + p.y, 0) / 4;

      octx.save();
      octx.font = 'bold 15px system-ui, sans-serif';
      octx.textAlign = 'center';
      octx.textBaseline = 'middle';

      // pill background
      const label = '✓ READY';
      const metrics = octx.measureText(label);
      const pw = metrics.width + 20;
      const ph = 26;
      const px = cx - pw / 2;
      const py = cy - ph / 2;

      octx.beginPath();
      octx.roundRect(px, py, pw, ph, 6);
      octx.fillStyle = 'rgba(34, 220, 90, 0.88)';
      octx.fill();

      octx.fillStyle = '#fff';
      octx.fillText(label, cx, cy);
      octx.restore();
    }
  }

  private getContainMapping(srcW: number, srcH: number, dstW: number, dstH: number) {
    const srcAR = srcW / srcH;
    const dstAR = dstW / dstH;

    let scale = 1;
    let drawW = dstW;
    let drawH = dstH;

    if (srcAR > dstAR) {
      scale = dstW / srcW;
      drawW = dstW;
      drawH = srcH * scale;
    } else {
      scale = dstH / srcH;
      drawH = dstH;
      drawW = srcW * scale;
    }

    const offsetX = (dstW - drawW) / 2;
    const offsetY = (dstH - drawH) / 2;
    return { scale, offsetX, offsetY };
  }

  // ============================
  // OpenCV: find corners
  // ============================
  private findReceiptCornersOnCanvas(canvasEl: HTMLCanvasElement): { x: number; y: number }[] | null {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return null;

    let src: any, gray: any, blur: any, edges: any, contours: any, hierarchy: any;
    try {
      src = cvAny.imread(canvasEl);

      gray = new cvAny.Mat();
      cvAny.cvtColor(src, gray, cvAny.COLOR_RGBA2GRAY);

      blur = new cvAny.Mat();
      cvAny.GaussianBlur(gray, blur, new cvAny.Size(5, 5), 0);

      edges = new cvAny.Mat();
      cvAny.Canny(blur, edges, 50, 150);

      const kernel = cvAny.getStructuringElement(cvAny.MORPH_RECT, new cvAny.Size(3, 3));
      cvAny.morphologyEx(edges, edges, cvAny.MORPH_CLOSE, kernel);
      kernel.delete();

      contours = new cvAny.MatVector();
      hierarchy = new cvAny.Mat();
      cvAny.findContours(edges, contours, hierarchy, cvAny.RETR_EXTERNAL, cvAny.CHAIN_APPROX_SIMPLE);

      let bestPts: { x: number; y: number }[] | null = null;
      let bestArea = 0;

      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        const area = cvAny.contourArea(cnt);
        if (area < bestArea) { cnt.delete(); continue; }

        const peri = cvAny.arcLength(cnt, true);
        const approx = new cvAny.Mat();
        cvAny.approxPolyDP(cnt, approx, 0.02 * peri, true);

        if (approx.rows === 4 && cvAny.isContourConvex(approx)) {
          bestArea = area;
          const pts: { x: number; y: number }[] = [];
          for (let r = 0; r < 4; r++) {
            const x = approx.intPtr(r, 0)[0];
            const y = approx.intPtr(r, 0)[1];
            pts.push({ x, y });
          }
          bestPts = this.orderCorners(pts);
        }

        approx.delete();
        cnt.delete();
      }

      return bestPts;
    } catch {
      return null;
    } finally {
      src?.delete?.();
      gray?.delete?.();
      blur?.delete?.();
      edges?.delete?.();
      contours?.delete?.();
      hierarchy?.delete?.();
    }
  }

  private orderCorners(pts: { x: number; y: number }[]) {
    const sum = pts.map(p => p.x + p.y);
    const diff = pts.map(p => p.x - p.y);

    const tl = pts[sum.indexOf(Math.min(...sum))];
    const br = pts[sum.indexOf(Math.max(...sum))];

    const tr = pts[diff.indexOf(Math.max(...diff))];
    const bl = pts[diff.indexOf(Math.min(...diff))];

    return [tl, tr, br, bl];
  }

  // ============================
  // Capture + warp (single shot)
  // ============================
  /**
 * Expands (or contracts) detected corners outward from the centroid by `px` pixels.
 * Positive px = expand (recover clipped border). Negative px = inset.
 * Clamps to canvas bounds so we never go outside the frame.
 */
  private expandCorners(
    pts: { x: number; y: number }[],
    canvasW: number,
    canvasH: number,
    px: number
  ): { x: number; y: number }[] {
    // Centroid
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;

    return pts.map(p => {
      const dx = p.x - cx;
      const dy = p.y - cy;
      const dist = Math.hypot(dx, dy) || 1;

      return {
        x: Math.max(0, Math.min(canvasW - 1, p.x + (dx / dist) * px)),
        y: Math.max(0, Math.min(canvasH - 1, p.y + (dy / dist) * px)),
      };
    });
  }
  captureWarped_old() {
    if (this.isLongMode) return;

    if (this.receipts.length >= this.MAX_RECEIPTS) {
      this.limitMsg = `Max ${this.MAX_RECEIPTS} receipts per batch. Process or clear first.`;
      return;
    }

    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;

    if (!video.videoWidth || !video.videoHeight) return;
    if (!this.cvReady) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    let corners = this.findReceiptCornersOnCanvas(canvas);

    if (!corners && this.lastWorkCorners && this.lastWorkW > 0 && this.lastWorkH > 0) {
      const scaleX = canvas.width / this.lastWorkW;
      const scaleY = canvas.height / this.lastWorkH;
      corners = this.lastWorkCorners.map(p => ({ x: p.x * scaleX, y: p.y * scaleY }));
    }

    let warpedCanvas: HTMLCanvasElement | null = null;
    if (corners) warpedCanvas = this.warpCanvasByCorners(canvas, corners);

    if (!warpedCanvas) {
      this.cameraHint = 'Corners not reliable; using full frame.';
      warpedCanvas = document.createElement('canvas');
      warpedCanvas.width = canvas.width;
      warpedCanvas.height = canvas.height;
      const wctx = warpedCanvas.getContext('2d');
      if (!wctx) return;
      wctx.drawImage(canvas, 0, 0);
    }

    this.postProcessAndAddSingle(warpedCanvas, 'camera');
  }

  /**
 * Nudges left-side corners (TL, BL) further left
 * and right-side corners (TR, BR) further right.
 * corners order: [TL, TR, BR, BL]
 * Clamps to canvas bounds.
 */
  private expandSideMargins(
    pts: { x: number; y: number }[],
    canvasW: number,
    px: number
  ): { x: number; y: number }[] {
    return pts.map((p, i) => {
      switch (i) {
        case 0: // TL → push left
          return { x: Math.max(0, p.x - px), y: p.y };
        case 1: // TR → push right
          return { x: Math.min(canvasW - 1, p.x + px), y: p.y };
        case 2: // BR → push right
          return { x: Math.min(canvasW - 1, p.x + px), y: p.y };
        case 3: // BL → push left
          return { x: Math.max(0, p.x - px), y: p.y };
        default:
          return p;
      }
    });
  }

  // Add these new state properties near your other UI state variables
  isProcessing = false;
  showCaptureFlash = false;
  captureWarped() {
    if (this.isLongMode) return;

    if (this.receipts.length >= this.MAX_RECEIPTS) {
      this.limitMsg = `Max ${this.MAX_RECEIPTS} receipts per batch. Process or clear first.`;
      return;
    }

    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;

    if (!video.videoWidth || !video.videoHeight) return;
    if (!this.cvReady) return;

    //  Step 1: Flash + freeze state immediately
    this.triggerCaptureFlash();
    this.isProcessing = true;
    this.cameraHint = 'Processing…';

    //  Step 2: Force Angular to paint the overlay NOW
    this.cdr.detectChanges();

    // Step 3: Grab the frame immediately (before setTimeout)
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) { this.isProcessing = false; return; }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    //  Step 4: Defer heavy OpenCV work by 2 frames (32ms) so overlay is visible
    setTimeout(() => {
      try {
        let corners = this.findReceiptCornersOnCanvas(canvas);

        if (!corners && this.lastWorkCorners && this.lastWorkW > 0 && this.lastWorkH > 0) {
          const scaleX = canvas.width / this.lastWorkW;
          const scaleY = canvas.height / this.lastWorkH;
          corners = this.lastWorkCorners.map(p => ({ x: p.x * scaleX, y: p.y * scaleY }));
        }

        if (corners) {
          const expandPx = Math.round(Math.min(canvas.width, canvas.height) * 0.012);
          corners = this.expandCorners(corners, canvas.width, canvas.height, expandPx);
          const extraSidePx = Math.round(canvas.width * 0.008);
          corners = this.expandSideMargins(corners, canvas.width, extraSidePx);
        }

        let warpedCanvas: HTMLCanvasElement | null = null;
        if (corners) warpedCanvas = this.warpCanvasByCorners(canvas, corners);

        if (!warpedCanvas) {
          this.cameraHint = 'Corners not reliable; using full frame.';
          warpedCanvas = document.createElement('canvas');
          warpedCanvas.width = canvas.width;
          warpedCanvas.height = canvas.height;
          const wctx = warpedCanvas.getContext('2d');
          if (!wctx) return;
          wctx.drawImage(canvas, 0, 0);
        }

        this.postProcessAndAddSingle(warpedCanvas, 'camera');
        this.cameraHint = 'Captured! You can take another.';

        setTimeout(() => {
          if (!this.isProcessing) {
            this.cameraHint = 'Looks good — you can Capture.';
            this.cdr.detectChanges();
          }
        }, 2000);

      } finally {
        // Always clear processing state even if something throws
        this.isProcessing = false;
        this.cdr.detectChanges();
      }
    }, 32); //  32ms = 2 paint frames — enough for overlay to appear
  }

  //  Shutter flash helper
  private triggerCaptureFlash() {
    this.showCaptureFlash = true;
    setTimeout(() => this.showCaptureFlash = false, 200);
  }

  private warpCanvasByCorners_old(canvasEl: HTMLCanvasElement, corners: { x: number; y: number }[]) {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return null;

    let src: any, dst: any, M: any, srcTri: any, dstTri: any;
    try {
      src = cvAny.imread(canvasEl);
      const [tl, tr, br, bl] = corners;

      const widthA = Math.hypot(br.x - bl.x, br.y - bl.y);
      const widthB = Math.hypot(tr.x - tl.x, tr.y - tl.y);
      const heightA = Math.hypot(tr.x - br.x, tr.y - br.y);
      const heightB = Math.hypot(tl.x - bl.x, tl.y - bl.y);

      const upscale = 1.8;
      let maxW = Math.min(3200, Math.max(1, Math.round(Math.max(widthA, widthB) * upscale)));
      let maxH = Math.min(9000, Math.max(1, Math.round(Math.max(heightA, heightB) * upscale)));

      const MAX_AR = 7.0;
      const ar = maxH / maxW;
      if (ar > MAX_AR) maxH = Math.round(maxW * MAX_AR);

      srcTri = cvAny.matFromArray(4, 1, cvAny.CV_32FC2, [
        tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y
      ]);
      dstTri = cvAny.matFromArray(4, 1, cvAny.CV_32FC2, [
        0, 0, maxW - 1, 0, maxW - 1, maxH - 1, 0, maxH - 1
      ]);

      M = cvAny.getPerspectiveTransform(srcTri, dstTri);
      dst = new cvAny.Mat();
      cvAny.warpPerspective(src, dst, M, new cvAny.Size(maxW, maxH));

      const out = document.createElement('canvas');
      out.width = maxW;
      out.height = maxH;
      cvAny.imshow(out, dst);
      return out;
    } catch (e) {
      console.warn('warpCanvasByCorners failed', e);
      return null;
    } finally {
      src?.delete?.(); dst?.delete?.(); M?.delete?.(); srcTri?.delete?.(); dstTri?.delete?.();
    }
  }

  private warpCanvasByCorners(canvasEl: HTMLCanvasElement, corners: { x: number; y: number }[]) {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return null;

    let src: any, dst: any, M: any, srcTri: any, dstTri: any;
    try {
      src = cvAny.imread(canvasEl);
      const [tl, tr, br, bl] = corners;

      const widthA = Math.hypot(br.x - bl.x, br.y - bl.y);
      const widthB = Math.hypot(tr.x - tl.x, tr.y - tl.y);
      const heightA = Math.hypot(tr.x - br.x, tr.y - br.y);
      const heightB = Math.hypot(tl.x - bl.x, tl.y - bl.y);

      // ✅ FIX: upscale 1.8→1.5 (less overshoot), height cap 9000→18000, MAX_AR 7→14
      const upscale = 1.5;
      let maxW = Math.min(3200, Math.max(1, Math.round(Math.max(widthA, widthB) * upscale)));
      let maxH = Math.min(18000, Math.max(1, Math.round(Math.max(heightA, heightB) * upscale)));

      const MAX_AR = 14.0; // thermal receipts can easily be 10:1
      const ar = maxH / maxW;
      if (ar > MAX_AR) maxH = Math.round(maxW * MAX_AR);

      srcTri = cvAny.matFromArray(4, 1, cvAny.CV_32FC2, [
        tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y
      ]);
      dstTri = cvAny.matFromArray(4, 1, cvAny.CV_32FC2, [
        0, 0, maxW - 1, 0, maxW - 1, maxH - 1, 0, maxH - 1
      ]);

      M = cvAny.getPerspectiveTransform(srcTri, dstTri);
      dst = new cvAny.Mat();
      cvAny.warpPerspective(src, dst, M, new cvAny.Size(maxW, maxH));

      const out = document.createElement('canvas');
      out.width = maxW;
      out.height = maxH;
      cvAny.imshow(out, dst);
      return out;
    } catch (e) {
      console.warn('warpCanvasByCorners failed', e);
      return null;
    } finally {
      src?.delete?.(); dst?.delete?.(); M?.delete?.(); srcTri?.delete?.(); dstTri?.delete?.();
    }
  }

  // ============================
  // Long Receipt Mode (SMOOTH)
  // ============================
  startLongReceipt() {
    if (!this.cvReady) return;
    if (this.isLongMode) return;

    if (this.receipts.length >= this.MAX_RECEIPTS) {
      this.limitMsg = `Max ${this.MAX_RECEIPTS} receipts per batch.`;
      return;
    }

    this.isLongMode = true;
    this.limitMsg = '';
    this.uploadMsg = '';
    this.noNewCount = 0;

    this.stitchCanvas = document.createElement('canvas');
    this.stitchCanvas.width = 0;
    this.stitchCanvas.height = 0;
    this.stitchCtx = this.stitchCanvas.getContext('2d');

    this.longFrameCanvas = document.createElement('canvas');
    this.longFrameCtx = this.longFrameCanvas.getContext('2d', { willReadFrequently: true });

    this.prevFrameCanvas = document.createElement('canvas');
    this.prevFrameCanvas.width = 0;
    this.prevFrameCanvas.height = 0;
    this.prevFrameCtx = this.prevFrameCanvas.getContext('2d', { willReadFrequently: true });

    this.cameraHint = 'Long mode: start at TOP, move DOWN slowly…';
    this.longTimer = window.setInterval(() => this.captureLongFrameFast(), this.LONG_CAPTURE_MS);
  }

  stopLongReceipt(save: boolean) {
    if (this.longTimer) {
      clearInterval(this.longTimer);
      this.longTimer = undefined;
    }

    const wasLong = this.isLongMode;
    this.isLongMode = false;

    const stitch = this.stitchCanvas;
    this.stitchCanvas = undefined;
    this.stitchCtx = null;
    this.longFrameCanvas = undefined;
    this.longFrameCtx = null;
    this.prevFrameCanvas = undefined;
    this.prevFrameCtx = null;

    if (!save || !wasLong) {
      this.cameraHint = 'Long mode canceled.';
      return;
    }

    if (!stitch || stitch.height < 200) {
      this.cameraHint = 'Long mode: not enough captured. Try again slower.';
      return;
    }

    this.postProcessAndAddSingle(stitch, 'camera');
    this.cameraHint = 'Saved long receipt as one image.';
  }

  private captureLongFrameFast() {
    const video = this.videoRef.nativeElement;
    if (!video.videoWidth || !video.videoHeight) return;

    if (!this.longFrameCanvas || !this.longFrameCtx || !this.prevFrameCanvas || !this.prevFrameCtx || !this.stitchCanvas || !this.stitchCtx) {
      return;
    }

    const W = this.LONG_FRAME_W;
    const H = Math.round((video.videoHeight / video.videoWidth) * W);

    this.longFrameCanvas.width = W;
    this.longFrameCanvas.height = H;

    this.longFrameCtx.drawImage(video, 0, 0, W, H);

    const blur = this.blurScore(this.longFrameCanvas);
    if (blur < this.LONG_BLUR_MIN) {
      this.cameraHint = 'Too blurry—hold steadier / more light…';
      return;
    }

    if (this.prevFrameCanvas.width === 0 || this.prevFrameCanvas.height === 0) {
      this.prevFrameCanvas.width = W;
      this.prevFrameCanvas.height = H;
      this.prevFrameCtx.drawImage(this.longFrameCanvas, 0, 0);

      this.ensureStitchCanvas(W, Math.min(H, this.LONG_MAX_STITCH_H));
      this.stitchCtx.drawImage(this.longFrameCanvas, 0, 0);

      this.cameraHint = 'Good. Now move DOWN slowly…';
      return;
    }

    const match = this.estimateOverlapCutYFast(this.prevFrameCanvas, this.longFrameCanvas);
    if (!match.ok) {
      this.noNewCount++;
      this.cameraHint = match.hint ?? 'Align center and move down slowly…';
      if (this.noNewCount >= this.LONG_STOP_NO_NEW) {
        this.cameraHint = 'No reliable new content—auto saving.';
        this.stopLongReceipt(true);
      }
      return;
    }

    let cutFromTop = match.cutY;
    cutFromTop = Math.max(0, Math.min(H - 1, cutFromTop - this.LONG_SEAM_PAD));

    const newH = H - cutFromTop;

    if (newH < this.LONG_MIN_APPEND_H) {
      this.noNewCount++;
      this.cameraHint = 'Move down a bit more…';
      if (this.noNewCount >= this.LONG_STOP_NO_NEW) {
        this.cameraHint = 'Not enough new content—auto saving.';
        this.stopLongReceipt(true);
      }
      return;
    }

    this.noNewCount = 0;

    const nextH = this.stitchCanvas.height + newH;
    if (nextH > this.LONG_MAX_STITCH_H) {
      this.cameraHint = 'Reached max length—saving.';
      this.stopLongReceipt(true);
      return;
    }

    this.growStitchCanvas(nextH);

    this.stitchCtx.drawImage(
      this.longFrameCanvas,
      0, cutFromTop, W, newH,
      0, this.stitchCanvas.height - newH, W, newH
    );

    this.prevFrameCtx.clearRect(0, 0, W, H);
    this.prevFrameCtx.drawImage(this.longFrameCanvas, 0, 0);

    this.cameraHint = `Capturing… (+${newH}px)`;
  }

  private blurScore(canvasEl: HTMLCanvasElement): number {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return 0;

    let src: any, gray: any, lap: any, mean: any, std: any;
    try {
      src = cvAny.imread(canvasEl);
      gray = new cvAny.Mat();
      cvAny.cvtColor(src, gray, cvAny.COLOR_RGBA2GRAY);

      lap = new cvAny.Mat();
      cvAny.Laplacian(gray, lap, cvAny.CV_64F);

      mean = new cvAny.Mat();
      std = new cvAny.Mat();
      cvAny.meanStdDev(lap, mean, std);

      const s = std.data64F[0];
      return s * s;
    } catch {
      return 0;
    } finally {
      src?.delete?.(); gray?.delete?.(); lap?.delete?.(); mean?.delete?.(); std?.delete?.();
    }
  }

  private estimateOverlapCutYFast(
    prevCanvas: HTMLCanvasElement,
    currCanvas: HTMLCanvasElement
  ): { ok: boolean; cutY: number; score: number; hint?: string } {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return { ok: false, cutY: 0, score: 0, hint: 'OpenCV not ready' };

    let prev: any, curr: any, pg: any, cg: any, res: any;
    try {
      const scale = this.LONG_MATCH_SCALE;

      prev = cvAny.imread(prevCanvas);
      curr = cvAny.imread(currCanvas);

      pg = new cvAny.Mat();
      cg = new cvAny.Mat();
      cvAny.cvtColor(prev, pg, cvAny.COLOR_RGBA2GRAY);
      cvAny.cvtColor(curr, cg, cvAny.COLOR_RGBA2GRAY);

      const pSmall = new cvAny.Mat();
      const cSmall = new cvAny.Mat();
      cvAny.resize(pg, pSmall, new cvAny.Size(0, 0), scale, scale, cvAny.INTER_AREA);
      cvAny.resize(cg, cSmall, new cvAny.Size(0, 0), scale, scale, cvAny.INTER_AREA);

      const W = pSmall.cols;
      const H = pSmall.rows;

      const stripW = Math.max(60, Math.floor(W * this.LONG_CENTER_STRIP_W_FRAC));
      const stripX = Math.max(0, Math.floor((W - stripW) / 2));

      const stripH = Math.max(70, Math.floor(H * this.LONG_STRIP_H_FRAC));
      const y0Prev = Math.max(0, H - stripH);

      const pRoi = pSmall.roi(new cvAny.Rect(stripX, y0Prev, stripW, stripH));
      const cRoi = cSmall.roi(new cvAny.Rect(stripX, 0, stripW, H));

      const resultCols = cRoi.cols - pRoi.cols + 1;
      const resultRows = cRoi.rows - pRoi.rows + 1;
      if (resultCols <= 0 || resultRows <= 0) {
        pRoi.delete(); cRoi.delete(); pSmall.delete(); cSmall.delete();
        return { ok: false, cutY: 0, score: 0, hint: 'Frame mismatch—try slower' };
      }

      res = new cvAny.Mat(resultRows, resultCols, cvAny.CV_32FC1);
      cvAny.matchTemplate(cRoi, pRoi, res, cvAny.TM_CCOEFF_NORMED);

      const mm = cvAny.minMaxLoc(res);
      const bestY = mm.maxLoc.y;
      const score = mm.maxVal;

      const cutSmall = bestY + stripH;
      const cutY = Math.round(cutSmall / scale);

      pRoi.delete(); cRoi.delete(); pSmall.delete(); cSmall.delete();

      if (score < this.LONG_MIN_MATCH_SCORE) {
        return { ok: false, cutY, score, hint: 'Can’t align—keep centered & move DOWN slower…' };
      }

      return { ok: true, cutY: Math.max(0, cutY), score };
    } catch {
      return { ok: false, cutY: 0, score: 0, hint: 'Matching failed—try again' };
    } finally {
      prev?.delete?.(); curr?.delete?.(); pg?.delete?.(); cg?.delete?.(); res?.delete?.();
    }
  }

  private ensureStitchCanvas(w: number, h: number) {
    if (!this.stitchCanvas || !this.stitchCtx) return;
    this.stitchCanvas.width = w;
    this.stitchCanvas.height = h;
  }

  private growStitchCanvas(newH: number) {
    if (!this.stitchCanvas) return;

    const old = this.stitchCanvas;
    const tmp = document.createElement('canvas');
    tmp.width = old.width;
    tmp.height = newH;

    const tctx = tmp.getContext('2d');
    if (!tctx) return;

    tctx.drawImage(old, 0, 0);

    this.stitchCanvas = tmp;
    this.stitchCtx = tctx;
  }

  // ============================
  // Post-process + add (SINGLE image)
  // ============================
  private postProcessAndAddSingle(
    baseCanvas: HTMLCanvasElement,
    source: 'camera' | 'upload'
  ) {
    const mime = this.saveAsPng ? 'image/png' : 'image/jpeg';
    const quality = this.saveAsPng ? undefined : 0.95;

    if (this.adobeLikeEnhance) {
      this.enhanceReceiptCanvasAdobeLike(
        baseCanvas,
        this.adobeBW,
        this.keepColorPreview
      );
    } else {
      this.enhanceReceiptCanvasMild(baseCanvas);
    }

    if (this.sharpenAfterWarp && !(this.adobeLikeEnhance && this.adobeBW)) {
      this.sharpenCanvas(baseCanvas, this.sharpenAmount);
    }

    baseCanvas.toBlob((blob: Blob | null) => {
      if (!blob) return;
      this.addReceipt(blob, source);
    }, mime as any, quality as any);
  }

  // ============================
  // Adobe-like enhancement (LOW-NOISE, stable)
  // ============================
  private enhanceReceiptCanvasAdobeLike(
    c: HTMLCanvasElement,
    bwOutput: boolean,
    keepColor: boolean
  ) {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return;

    let src: any, rgba: any, gray: any;
    let den: any, bg: any, norm: any, claheOut: any;
    let bw: any, out: any;
    let kBig: any, kSmall: any, kLine: any;
    let tmp1: any, tmp2: any;

    try {
      src = cvAny.imread(c);

      rgba = new cvAny.Mat();
      if (src.channels() === 4) src.copyTo(rgba);
      else cvAny.cvtColor(src, rgba, cvAny.COLOR_BGR2RGBA);

      gray = new cvAny.Mat();
      cvAny.cvtColor(rgba, gray, cvAny.COLOR_RGBA2GRAY);

      den = new cvAny.Mat();
      if (cvAny.fastNlMeansDenoising) {
        cvAny.fastNlMeansDenoising(gray, den, 12, 7, 21);
      } else {
        cvAny.bilateralFilter(gray, den, 7, 50, 50);
      }

      bg = new cvAny.Mat();
      kBig = cvAny.getStructuringElement(cvAny.MORPH_ELLIPSE, new cvAny.Size(41, 41));
      cvAny.morphologyEx(den, bg, cvAny.MORPH_OPEN, kBig);

      norm = new cvAny.Mat();
      cvAny.subtract(den, bg, norm);

      claheOut = new cvAny.Mat();
      const clahe = cvAny.createCLAHE(2.0, new cvAny.Size(8, 8));
      clahe.apply(norm, claheOut);
      clahe.delete();

      if (keepColor) {
        out = new cvAny.Mat();
        cvAny.cvtColor(claheOut, out, cvAny.COLOR_GRAY2RGBA);
        cvAny.imshow(c, out);
        return;
      }

      if (!bwOutput) {
        out = new cvAny.Mat();
        cvAny.cvtColor(claheOut, out, cvAny.COLOR_GRAY2RGBA);
        cvAny.imshow(c, out);
        return;
      }

      tmp1 = new cvAny.Mat();
      kLine = cvAny.getStructuringElement(cvAny.MORPH_RECT, new cvAny.Size(9, 9));
      cvAny.morphologyEx(claheOut, tmp1, cvAny.MORPH_BLACKHAT, kLine);

      tmp2 = new cvAny.Mat();
      cvAny.add(claheOut, tmp1, tmp2);

      cvAny.GaussianBlur(tmp2, tmp2, new cvAny.Size(3, 3), 0);

      bw = new cvAny.Mat();
      cvAny.threshold(tmp2, bw, 0, 255, cvAny.THRESH_BINARY | cvAny.THRESH_OTSU);

      const m = cvAny.mean(bw)[0];
      if (m < 127) cvAny.bitwise_not(bw, bw);

      kSmall = cvAny.getStructuringElement(cvAny.MORPH_RECT, new cvAny.Size(3, 3));
      cvAny.morphologyEx(bw, bw, cvAny.MORPH_OPEN, kSmall);
      cvAny.morphologyEx(bw, bw, cvAny.MORPH_CLOSE, kSmall);

      const kTiny = cvAny.getStructuringElement(cvAny.MORPH_RECT, new cvAny.Size(2, 2));
      cvAny.morphologyEx(bw, bw, cvAny.MORPH_OPEN, kTiny);
      kTiny.delete();

      out = new cvAny.Mat();
      cvAny.cvtColor(bw, out, cvAny.COLOR_GRAY2RGBA);
      cvAny.imshow(c, out);
    } catch (e) {
      console.warn('enhanceReceiptCanvasAdobeLike failed', e);
    } finally {
      src?.delete?.(); rgba?.delete?.(); gray?.delete?.();
      den?.delete?.(); bg?.delete?.(); norm?.delete?.(); claheOut?.delete?.();
      bw?.delete?.(); out?.delete?.();
      tmp1?.delete?.(); tmp2?.delete?.();
      kBig?.delete?.(); kSmall?.delete?.(); kLine?.delete?.();
    }
  }

  // Mild enhancement (fallback)
  private enhanceReceiptCanvasMild(c: HTMLCanvasElement) {
    const cvAny: any = (window as any).cv;
    if (!cvAny?.imread) return;

    let src: any, gray: any, denoised: any, claheDst: any;
    try {
      src = cvAny.imread(c);

      gray = new cvAny.Mat();
      cvAny.cvtColor(src, gray, cvAny.COLOR_RGBA2GRAY);

      denoised = new cvAny.Mat();
      cvAny.bilateralFilter(gray, denoised, 7, 50, 50);

      claheDst = new cvAny.Mat();
      const clahe = cvAny.createCLAHE(2.0, new cvAny.Size(8, 8));
      clahe.apply(denoised, claheDst);
      clahe.delete();

      const out = new cvAny.Mat();
      cvAny.cvtColor(claheDst, out, cvAny.COLOR_GRAY2RGBA);
      cvAny.imshow(c, out);
      out.delete();
    } catch (e) {
      console.warn('enhanceReceiptCanvasMild failed', e);
    } finally {
      src?.delete?.();
      gray?.delete?.();
      denoised?.delete?.();
      claheDst?.delete?.();
    }
  }

  private sharpenCanvas(c: HTMLCanvasElement, amount = 0.12) {
    const ctx = c.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const img = ctx.getImageData(0, 0, c.width, c.height);
    const data = img.data;
    const w = c.width;
    const h = c.height;

    const copy = new Uint8ClampedArray(data);
    const idx = (x: number, y: number) => (y * w + x) * 4;

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = idx(x, y);
        for (let ch = 0; ch < 3; ch++) {
          const center = copy[i + ch];
          const v =
            5 * center
            - copy[idx(x - 1, y) + ch] - copy[idx(x + 1, y) + ch]
            - copy[idx(x, y - 1) + ch] - copy[idx(x, y + 1) + ch];

          const out = center + amount * (v - center);
          data[i + ch] = Math.max(0, Math.min(255, out));
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // ============================
  // Upload multiple (max 10)
  // ============================
  onFiles(e: Event) {
    const input = e.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    const remaining = this.MAX_RECEIPTS - this.receipts.length;

    if (remaining <= 0) {
      this.limitMsg = `Max ${this.MAX_RECEIPTS} receipts per batch.`;
      input.value = '';
      return;
    }

    this.limitMsg = '';
    for (const f of files.slice(0, remaining)) {
      this.addReceipt(f, 'upload');
    }

    if (files.length > remaining) {
      this.limitMsg = `Only added ${remaining}. Max ${this.MAX_RECEIPTS} per batch.`;
    }

    input.value = '';
  }

  // ============================
  // Batch helpers
  // ============================
  private makeId() {
    return (crypto as any)?.randomUUID?.() ?? `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }

  private addReceipt(blob: Blob, source: 'camera' | 'upload') {
    if (this.receipts.length >= this.MAX_RECEIPTS) {
      this.limitMsg = `Max ${this.MAX_RECEIPTS} receipts per batch. Process or clear first.`;
      return false;
    }

    this.limitMsg = '';
    const id = this.makeId();
    const url = URL.createObjectURL(blob);

    const item: ReceiptShot = {
      id,
      blob,
      url,
      source,
      createdAt: Date.now(),
    };

    this.receipts.unshift(item);
    this.selectReceipt(id);
    return true;
  }

  selectReceipt(id: string) {
    const r = this.receipts.find(x => x.id === id);
    if (!r) return;
    this.selectedId = r.id;
    this.selectedUrl = r.url;
    this.selectedBlob = r.blob;
  }

  removeReceipt(id: string) {
    const idx = this.receipts.findIndex(r => r.id === id);
    if (idx < 0) return;

    URL.revokeObjectURL(this.receipts[idx].url);
    this.receipts.splice(idx, 1);

    if (this.selectedId === id) {
      const next = this.receipts[0];
      if (next) this.selectReceipt(next.id);
      else {
        this.selectedId = undefined;
        this.selectedUrl = undefined;
        this.selectedBlob = undefined;
      }
    }
  }

  clearAllReceipts() {
    this.receipts.forEach(r => URL.revokeObjectURL(r.url));
    this.receipts = [];
    this.selectedId = undefined;
    this.selectedUrl = undefined;
    this.selectedBlob = undefined;
    this.limitMsg = '';
    this.uploadMsg = '';
  }

  // ============================
  // Use selected + save debug
  // ============================
  useSelected() {
    if (!this.selectedBlob) return;

    const reader = new FileReader();
    reader.onload = () => {
      sessionStorage.setItem('captured_receipt', String(reader.result));
      this.router.navigate(['/scan-receipt']);
    };
    reader.readAsDataURL(this.selectedBlob);
  }

  async saveDebugSelected() {
    if (!this.selectedBlob) return;

    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const ext = this.selectedBlob.type.includes('png') ? 'png' : 'jpg';

    this.downloadBlob(this.selectedBlob, `debug_selected_${ts}.${ext}`);

    try {
      const dataUrl = await this.blobToDataURL(this.selectedBlob);
      sessionStorage.setItem('debug_receipt_selected', dataUrl);
      sessionStorage.setItem('debug_receipt_time', ts);
      this.uploadMsg = 'Saved debug (download + sessionStorage).';
    } catch {
      this.uploadMsg = 'Downloaded debug, but sessionStorage is too small.';
    }
  }

  // ============================
  // Backend upload (stub)
  // ============================
  async processAll() {
    this.uploadMsg = `Ready to send ${this.receipts.length} receipt(s) to backend.`;
  }

  // ============================
  // Utils
  // ============================
  private downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private blobToDataURL(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result));
      r.onerror = () => reject(new Error('FileReader failed'));
      r.readAsDataURL(blob);
    });
  }
}