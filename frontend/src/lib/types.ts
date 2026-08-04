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

/**
 * Quota consumption. Limits come from the server rather than being hardcoded
 * here: the real values live in the `quotas` table and can change without a
 * frontend deploy.
 */
export interface Usage {
  threads_used: number
  threads_limit: number
  /** Supabase Storage bucket only -- the original files. Chats and chunks are
   *  in Postgres and bounded by messages_per_thread_limit instead. */
  storage_used_bytes: number
  storage_limit_bytes: number
  messages_per_thread_limit: number
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
  /** LLM metadata extraction; runs before chunking because chunk embeddings
   *  are prefixed with the extracted title. */
  | 'analyzing'
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

  /**
   * Chunks were built under a different processing config than the current
   * one (or the row predates hashing). Computed by the backend, not stored --
   * a Realtime row merge never carries it, which is why `ready` transitions
   * refetch the list instead.
   */
  stale: boolean

  /**
   * Extracted by the LLM at ingest. All nullable: extraction can fail without
   * failing the ingest, and documents from before Module 4 have none until
   * they are re-processed.
   */
  title: string | null
  doc_type: string | null
  source_org: string | null
  published_year: number | null
  topics: string[] | null
  summary: string | null
}

/**
 * One retrieved passage, as attached to a tool_result block.
 *
 * These are REPLAYED FROM THE DATABASE, so this interface describes the shape
 * written *today* — messages stored by earlier versions are missing whatever
 * did not exist yet. Every field added after Module 2 is therefore optional,
 * and must be checked with `typeof`, not against `null`: `undefined !== null`
 * is true, and one `.toFixed()` on a missing field throws during render and
 * blanks the entire thread.
 */
export interface Source {
  document_id: string
  filename: string
  /** Heading path. Absent on messages stored before Module 4. */
  section?: string | null
  ordinal: number
  /** Vector cosine. 0 for a hit found only by the keyword channel. */
  similarity: number
  /**
   * Cross-encoder score — what the list is ORDERED by when present, which is
   * why it has to be displayed. Absent on messages stored before Module 6 and
   * whenever reranking is off.
   */
  rerank_score?: number | null
}

/**
 * A metadata query the agent ran, and how much it returned. The SQL is the
 * attribution: a bare number ("you have 12 documents") is unauditable, while
 * the query that produced it can be read and disagreed with.
 */
export interface SqlResult {
  sql: string
  row_count: number
  truncated?: boolean
  error?: string
}

/**
 * An exact count of documents containing a term, over the full-text index.
 *
 * Distinct from a search: `count` is a TOTAL across every document, where a
 * search result count only ever measures its own top-k. Shown with the term so
 * the claim can be checked, and labelled as literal-word matching — "4
 * documents mention forecasting" is not "4 documents are about forecasting".
 */
export interface CountResult {
  term: string
  /** Total documents containing the term. null when counting was unavailable. */
  count: number | null
  documents?: { filename: string; chunk_matches: number }[]
  is_error?: boolean
}

/** A page returned by the server-side web_search tool. */
export interface WebResult {
  url: string
  title: string
  /** Anthropic's freshness hint, e.g. "March 12, 2026". Often absent. */
  page_age?: string | null
}

/** Frames emitted by POST /api/chat over SSE. */
export type ChatFrame =
  | { type: 'delta'; text: string }
  | {
      type: 'tool_use'
      id: string
      name: string
      input: { query?: string; top_k?: number; sql?: string; term?: string }
    }
  | {
      type: 'tool_result'
      tool_use_id: string
      sources: Source[]
      /** Present only for query_document_metadata calls. */
      sql?: SqlResult | null
      /** Present only for count_documents_mentioning calls. */
      count?: CountResult | null
      is_error: boolean
    }
  | {
      /**
       * Web search runs on Anthropic's infrastructure, so its results arrive
       * with the assistant turn rather than through the tool loop — hence a
       * frame of its own rather than a `tool_result`.
       */
      type: 'web_results'
      tool_use_id: string | null
      results: WebResult[]
      /** Set when the search itself failed; results is then empty. */
      error: string | null
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

/** Metadata queries carried by a message's tool_result blocks. */
export function toolSqlResults(content: ContentBlock[]): SqlResult[] {
  return content
    .filter((b) => b.type === 'tool_result')
    .map((b) => (b as { sql?: SqlResult | null }).sql)
    .filter((s): s is SqlResult => !!s && typeof s.sql === 'string')
}

/** Exact content counts carried by a message's tool_result blocks. */
export function toolCounts(content: ContentBlock[]): CountResult[] {
  return content
    .filter((b) => b.type === 'tool_result')
    .map((b) => (b as { count?: CountResult | null }).count)
    .filter((c): c is CountResult => !!c && typeof c.term === 'string')
}

/**
 * Web pages carried by an assistant turn's web_search_tool_result blocks.
 *
 * Reads the persisted Anthropic block shape rather than our own, because these
 * blocks are stored verbatim — they have to round-trip unchanged for replay, so
 * there is nowhere to put a friendlier field. A failed search stores `content`
 * as an error OBJECT rather than an array, which is why the isArray check is
 * load-bearing: without it a failure renders as "no sources" instead of a
 * failure.
 */
export function webResults(content: ContentBlock[]): WebResult[] {
  return content
    .filter((b) => b.type === 'web_search_tool_result')
    .flatMap((b) => {
      const inner = (b as { content?: unknown }).content
      if (!Array.isArray(inner)) return []
      return inner
        .filter((r): r is { url: string; title?: string; page_age?: string } =>
          !!r && typeof (r as { url?: unknown }).url === 'string')
        .map((r) => ({
          url: r.url,
          title: r.title || r.url,
          page_age: r.page_age ?? null,
        }))
    })
}

/** True for the synthetic user turns that only carry tool results. */
export function isToolResultMessage(m: Message): boolean {
  return m.content.length > 0 && m.content.every((b) => b.type === 'tool_result')
}
