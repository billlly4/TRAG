export interface Citation {
  type: string
  cited_text: string
  document_title?: string | null
  document_index?: number
  start_char_index?: number
  end_char_index?: number
  start_page_number?: number
  end_page_number?: number
}

export interface TextBlock {
  type: 'text'
  text: string
  citations?: Citation[] | null
}

export type ContentBlock = TextBlock | { type: string; [key: string]: unknown }

export interface Message {
  id: string
  thread_id: string
  role: 'user' | 'assistant'
  /** Claude content blocks, not a string -- see backend/app/schemas.py */
  content: ContentBlock[]
  stop_reason?: string | null
  usage?: Record<string, number> | null
  created_at: string
}

export interface Thread {
  id: string
  title: string | null
  created_at: string
  updated_at: string
}

export interface DocumentMeta {
  id: string
  filename: string
  mime_type: string | null
  byte_size: number | null
  token_estimate: number | null
  created_at: string
  anthropic_file_id: string
}

/** Frames emitted by POST /api/chat over SSE. */
export type ChatFrame =
  | { type: 'delta'; text: string }
  | {
      type: 'done'
      message_id: string
      stop_reason: string | null
      truncated: boolean
      content: ContentBlock[]
      usage: Record<string, number> | null
    }
  | { type: 'error'; detail: string }

export function isTextBlock(block: ContentBlock): block is TextBlock {
  return block.type === 'text'
}

export function blocksToText(content: ContentBlock[]): string {
  return content.filter(isTextBlock).map((b) => b.text).join('')
}

export function blocksToCitations(content: ContentBlock[]): Citation[] {
  return content.filter(isTextBlock).flatMap((b) => b.citations ?? [])
}

/**
 * File ids already present in a thread's history.
 *
 * A document attached on an earlier turn is replayed on every subsequent call
 * -- that is what keeps the cached prefix stable. Attaching it again would put
 * the same text in context twice and invalidate the cache, so the UI uses this
 * to mark those documents as already in context rather than re-attachable.
 */
export function fileIdsInThread(messages: Message[]): Set<string> {
  const ids = new Set<string>()
  for (const message of messages) {
    for (const block of message.content) {
      if (block.type !== 'document') continue
      const source = (block as { source?: { file_id?: string } }).source
      if (source?.file_id) ids.add(source.file_id)
    }
  }
  return ids
}
