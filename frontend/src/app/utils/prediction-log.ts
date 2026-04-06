// prediction-log.ts

export interface PredictionRow {
  idx: number;
  word: string;
  label: string;
  confidence: number;
}

export interface PredictionLog {
  total_words: number;
  rows: PredictionRow[];
}

export interface FormattedPredictionLine {
  idx: string;        // "000"
  word: string;       // padded for UI if you want
  label: string;      // padded for UI if you want
  confidence: string; // "0.973"
  line: string;       // full line: "000 | Coffee ... 0.973"
}

/**
 * Build lines like:
 * 000 | Coffee             B-MENU.NM        0.973
 *
 * Use this to show in HTML (no console).
 */
export function buildSamplePredictionLines(
  log: PredictionLog | null | undefined,
  opts?: {
    wordWidth?: number;   // default 18
    labelWidth?: number;  // default 16
    trimWordToWidth?: boolean; // default true
  },
): FormattedPredictionLine[] {
  if (!log?.rows?.length) return [];

const wordWidth = opts?.wordWidth ?? 18;
const labelWidth = opts?.labelWidth ?? 16;
const trimWordToWidth = opts?.trimWordToWidth ?? true;

return log.rows.map((row) => {
    const idx = padLeft(String(row.idx ?? 0), 3);

    const clean = cleanWord(row.word ?? "");
    const word = padRight(
        trimWordToWidth ? trimTo(clean, wordWidth) : clean,
        wordWidth,
    );

    const labelRaw = (row.label ?? "O").trim() || "O";
    const label = padRight(trimTo(labelRaw, labelWidth), labelWidth);

    const confidence = Number(row.confidence ?? 0).toFixed(3);

    const line = `${idx} | ${word} ${label} ${confidence}`;

    return { idx, word: word.trim(), label: label.trim(), confidence, line };
});
}

/**
 * If your API sometimes returns prediction_log_json as a string,
 * use this helper safely.
 */
export function parsePredictionLog(input: unknown): PredictionLog | null {
  try {
    if (!input) return null;
    if (typeof input === "string") return JSON.parse(input) as PredictionLog;
    return input as PredictionLog;
  } catch {
    return null;
  }
}

/* -----------------------------
   Helpers
--------------------------------*/

function padLeft(value: string, length: number): string {
  return value.padStart(length, "0");
}

function padRight(value: string, length: number): string {
  if (value.length >= length) return value; // keep full, UI can wrap/scroll
  return value + " ".repeat(length - value.length);
}

function trimTo(value: string, length: number): string {
  if (value.length <= length) return value;
  return value.slice(0, length);
}

function cleanWord(word: string): string {
  return String(word ?? "")
    .replace(/\s+/g, " ")
    .replace(/[\r\n]/g, "")
    .trim();
}
