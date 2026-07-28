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

/** Ingestion pipeline states, pushed live over Supabase Realtime. */
export type DocumentStatus =
  | 'pending'
  | 'extracting'
  | 'chunking'
  | 'embedding'
  | 'ready'
  | 'failed'

export interface DocumentMeta {
  id: string
  filename: string
  mime_type: string | null
  byte_size: number | null
  created_at: string
  status: DocumentStatus
  error: string | null
  chunk_count: number | null
}

/** One retrieved passage, as attached to a tool_result block. */
export interface Source {
  document_id: string
  filename: string
  ordinal: number
  similarity: number
}

/** Frames emitted by POST /api/chat over SSE. */
export type ChatFrame =
  | { type: 'delta'; text: string }
  | {
      type: 'tool_use'
      id: string
      name: string
      input: { query?: string; top_k?: number }
    }
  | {
      type: 'tool_result'
      tool_use_id: string
      sources: Source[]
      is_error: boolean
    }
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

/** Search queries issued by an assistant turn (its tool_use blocks). */
export function toolQueries(content: ContentBlock[]): string[] {
  return content
    .filter((b) => b.type === 'tool_use')
    .map((b) => ((b as { input?: { query?: string } }).input?.query ?? '').trim())
    .filter(Boolean)
}

/** Retrieved sources carried by a message's tool_result blocks. */
export function toolSources(content: ContentBlock[]): Source[] {
  return content
    .filter((b) => b.type === 'tool_result')
    .flatMap((b) => ((b as { sources?: Source[] }).sources ?? []))
}

/** True for the synthetic user turns that only carry tool results. */
export function isToolResultMessage(m: Message): boolean {
  return m.content.length > 0 && m.content.every((b) => b.type === 'tool_result')
}
