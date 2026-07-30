/**
 * Accepted upload formats — the single source of truth for the file picker,
 * the drag-and-drop filter and the overlay hint.
 *
 * Kept in sync by hand with SUPPORTED_SUFFIXES in backend/app/extract.py. Every
 * entry was verified end to end through the real extractor, not merely listed
 * as a docling InputFormat: docling names 30 formats, but "it appears in the
 * enum" and "it yields usable Markdown" are different claims.
 *
 * Audio and video are deliberately absent — they need speech-recognition models
 * this app does not ship, and the backend rejects them with a readable reason
 * rather than starting a model download mid-ingest.
 */
export const ACCEPTED_EXTENSIONS = [
  '.pdf',
  '.docx',
  '.pptx',
  '.xlsx',
  '.csv',
  '.html',
  '.htm',
  '.md',
  '.txt',
  '.png',
  '.jpg',
  '.jpeg',
  '.tiff',
  '.bmp',
  '.webp',
] as const

/** For an <input type="file"> accept attribute. */
export const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.join(',')

export function isAcceptedFile(name: string): boolean {
  const lower = name.toLowerCase()
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext))
}
