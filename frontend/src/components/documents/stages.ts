import type { IngestionStatus } from "@/lib/api";

export const STAGE_LABEL: Record<IngestionStatus, string> = {
  // On the server, waiting for a worker; "QUEUED" is a file not sent yet.
  pending: "PENDING",
  extracting: "EXTRACTING",
  chunking: "CHUNKING",
  embedding: "EMBEDDING",
  persisting: "PERSISTING",
  complete: "INDEXED",
  failed: "FAILED",
};

// The steps a job goes through, in order: a running row says which it is on.
const STEPS: IngestionStatus[] = ["extracting", "chunking", "embedding", "persisting"];

/** "EMBEDDING (3/4)" for a step; the label alone for any other status. */
export function stageText(stage: IngestionStatus): string {
  const step = STEPS.indexOf(stage);
  return step === -1 ? STAGE_LABEL[stage] : `${STAGE_LABEL[stage]} (${step + 1}/${STEPS.length})`;
}

// What the backend takes (app/api/v1/documents.py, app/services/text_extractor.py).
const ACCEPTED_TYPES = ["pdf", "docx", "xlsx", "csv", "html", "htm", "md", "txt"];
export const ACCEPT_ATTR = ACCEPTED_TYPES.map((t) => `.${t}`).join(",");
export const ACCEPTED_LABEL = "PDF, DOCX, XLSX, CSV, HTML, MD, TXT";
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_NAME_LENGTH = 300;

const fmtMb = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

/** Why the backend would reject `file`, checked before it is sent; null if it would not. */
export function rejectReason(file: File): string | null {
  const ext = file.name.includes(".") ? file.name.split(".").pop()!.toLowerCase() : "";
  if (!ACCEPTED_TYPES.includes(ext)) {
    return `UNSUPPORTED TYPE${ext ? ` .${ext.toUpperCase()}` : ""}: ${ACCEPTED_LABEL} ONLY`;
  }
  if (file.size === 0) return "EMPTY FILE";
  if (file.size > MAX_FILE_BYTES) return `TOO LARGE: ${fmtMb(file.size)}, MAX 10 MB`;
  if (file.name.length > MAX_NAME_LENGTH) return `NAME LONGER THAN ${MAX_NAME_LENGTH} CHARACTERS`;
  return null;
}
