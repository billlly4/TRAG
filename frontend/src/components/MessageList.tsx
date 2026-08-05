import { useEffect, useRef } from 'react'

import type { LiveTool } from '../hooks/useChat'
import type { CountResult, Message, Source, SqlResult, WebResult } from '../lib/types'
import {
  blocksToCitations,
  blocksToText,
  isToolResultMessage,
  toolCounts,
  toolQueries,
  toolSources,
  toolSqlResults,
  webResults,
} from '../lib/types'
import { Banner } from './ui'

function Bubble({
  role,
  children,
}: {
  role: 'user' | 'assistant'
  children: React.ReactNode
}) {
  const mine = role === 'user'
  return (
    <div className={mine ? 'flex justify-end' : 'flex justify-start'}>
      <div
        className={
          mine
            ? 'max-w-[88%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm break-words whitespace-pre-wrap text-white sm:max-w-[75%]'
            : 'max-w-[88%] rounded-2xl rounded-bl-sm border border-zinc-200 bg-white px-4 py-2.5 text-sm break-words whitespace-pre-wrap text-zinc-800 sm:max-w-[75%] dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100'
        }
      >
        {children}
      </div>
    </div>
  )
}

/** "Searched documents: …" marker for an assistant turn's tool calls. */
function SearchMarker({ query, pending }: { query: string; pending?: boolean }) {
  return (
    <p className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
      {pending ? (
        <>
          <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500 align-middle" />
          Searching documents… <span className="italic">&ldquo;{query}&rdquo;</span>
        </>
      ) : (
        <>
          🔍 Searched documents: <span className="italic">&ldquo;{query}&rdquo;</span>
        </>
      )}
    </p>
  )
}

/** Collapsible list of retrieved passages (filename, ordinal, similarity). */
function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) {
    return (
      <p className="my-1 text-xs text-zinc-400">No relevant passages found.</p>
    )
  }
  return (
    <details className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
      <summary className="cursor-pointer select-none">
        {sources.length} source{sources.length === 1 ? '' : 's'}
      </summary>
      <ol className="mt-1 space-y-0.5 border-l-2 border-zinc-200 pl-3 dark:border-zinc-700">
        {sources.map((s, i) => (
          <li key={i}>
            <span className="font-medium">{s.filename}</span>
            {/* Section is null for documents with no Markdown headings, and
                for anything ingested before Module 4. */}
            {s.section && (
              <span className="text-zinc-400"> › {s.section}</span>
            )}
            {' · chunk '}
            {s.ordinal}
            {/* Every field below is checked with `typeof`, not against null.
                These objects are REPLAYED FROM THE DATABASE, so a message
                stored before a field existed simply has no such key --
                `undefined !== null` is true, and the render then throws on
                `.toFixed()`, unmounting the whole thread into a blank screen.
                The TypeScript types describe today's shape, not history's. */}
            {typeof s.similarity === 'number' && s.similarity > 0 &&
              ` · cos ${s.similarity.toFixed(3)}`}
            {/* Shown whenever reranking ran, because it is what the list is
                sorted by — without it the numbers appear out of order. */}
            {typeof s.rerank_score === 'number' && (
              <span className="text-zinc-400">
                {` · rerank ${s.rerank_score > 0 ? '+' : ''}${s.rerank_score.toFixed(2)}`}
              </span>
            )}
          </li>
        ))}
      </ol>
    </details>
  )
}

function SqlBlock({ result }: { result: SqlResult }) {
  return (
    <details className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
      <summary className="cursor-pointer select-none hover:text-zinc-700 dark:hover:text-zinc-200">
        🗄️ Queried document metadata —{' '}
        {result.error
          ? 'query failed'
          : `${result.row_count} row${result.row_count === 1 ? '' : 's'}`}
        {result.truncated && ' (truncated)'}
      </summary>
      <pre className="mt-1 overflow-x-auto rounded-lg bg-zinc-100 px-3 py-2 font-mono text-[11px] leading-relaxed text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
        {result.sql}
      </pre>
      {result.error && (
        <p className="mt-1 text-red-500">{result.error}</p>
      )}
    </details>
  )
}

function CountBlock({ result }: { result: CountResult }) {
  if (result.count === null) {
    return (
      <p className="my-1 text-xs text-red-500">
        Could not count documents mentioning &ldquo;{result.term}&rdquo;.
      </p>
    )
  }
  const docs = result.documents ?? []
  return (
    <details className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
      <summary className="cursor-pointer select-none hover:text-zinc-700 dark:hover:text-zinc-200">
        🔢 Counted <span className="italic">&ldquo;{result.term}&rdquo;</span> across all
        documents — {result.count} of them
      </summary>
      {docs.length > 0 && (
        <ol className="mt-1 space-y-0.5 border-l-2 border-emerald-300 pl-3 dark:border-emerald-800">
          {docs.map((d, i) => (
            <li key={i}>
              <span className="font-medium">{d.filename}</span>
              <span className="text-zinc-400">
                {' · '}
                {d.chunk_matches} passage{d.chunk_matches === 1 ? '' : 's'}
              </span>
            </li>
          ))}
        </ol>
      )}
      <p className="mt-1 text-zinc-400">
        Exact total over every document&rsquo;s full text. Matches the word
        literally — a document covering the subject in other words is not counted.
      </p>
    </details>
  )
}

function WebSourceList({ results }: { results: WebResult[] }) {
  if (results.length === 0) return null
  return (
    <details className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
      <summary className="cursor-pointer select-none hover:text-zinc-700 dark:hover:text-zinc-200">
        🌐 {results.length} web source{results.length === 1 ? '' : 's'} — not from
        your documents
      </summary>
      <ol className="mt-1 space-y-1 border-l-2 border-sky-300 pl-3 dark:border-sky-800">
        {results.map((r, i) => (
          <li key={i}>
            <a
              href={r.url}
              target="_blank"
              rel="noreferrer noopener"
              className="font-medium text-sky-700 hover:underline dark:text-sky-400"
            >
              {r.title}
            </a>
            <span className="ml-1 text-zinc-400">
              {(() => {
                try {
                  return new URL(r.url).hostname.replace(/^www\./, '')
                } catch {
                  return r.url
                }
              })()}
              {r.page_age && ` · ${r.page_age}`}
            </span>
          </li>
        ))}
      </ol>
    </details>
  )
}

function Citations({ message }: { message: Message }) {
  const cites = blocksToCitations(message.content)
  if (cites.length === 0) return null

  return (
    <ol className="mt-2 space-y-1 border-l-2 border-zinc-200 pl-3 text-xs text-zinc-500 dark:border-zinc-700">
      {cites.map((c, i) => {
        const where =
          c.start_page_number != null
            ? `p. ${c.start_page_number}`
            : c.start_char_index != null
              ? `chars ${c.start_char_index}–${c.end_char_index}`
              : ''
        return (
          <li key={i}>
            <span className="font-medium">
              {c.document_title || 'Source'}
              {where && ` · ${where}`}
            </span>
            {c.cited_text && (
              <span className="italic"> &ldquo;{c.cited_text.trim()}&rdquo;</span>
            )}
          </li>
        )
      })}
    </ol>
  )
}

export function MessageList({
  messages,
  streamText,
  streaming,
  liveTools,
  liveWeb,
  error,
  truncated,
}: {
  messages: Message[]
  streamText: string
  streaming: boolean
  liveTools: LiveTool[]
  liveWeb: WebResult[]
  error: string | null
  truncated: boolean
}) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streamText, liveTools.length])

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3 sm:p-6">
      {messages.length === 0 && !streaming && (
        <p className="pt-12 text-center text-sm text-zinc-400">
          Ask a question. Uploaded documents are searched automatically.
        </p>
      )}

      {messages.map((m) => {
        if (m.role === 'user' && isToolResultMessage(m)) {
          const sqls = toolSqlResults(m.content)
          const counts = toolCounts(m.content)
          const sources = toolSources(m.content)
          const showSources = sources.length > 0 || (sqls.length === 0 && counts.length === 0)
          return (
            <div key={m.id}>
              {sqls.map((s, i) => (
                <SqlBlock key={i} result={s} />
              ))}
              {counts.map((c, i) => (
                <CountBlock key={i} result={c} />
              ))}
              {showSources && <SourceList sources={sources} />}
            </div>
          )
        }

        const text = blocksToText(m.content)
        const queries = m.role === 'assistant' ? toolQueries(m.content) : []
        const web = m.role === 'assistant' ? webResults(m.content) : []
        return (
          <div key={m.id}>
            {queries.map((q, i) => (
              <SearchMarker key={i} query={q} />
            ))}
            {web.length > 0 && <WebSourceList results={web} />}
            {text && <Bubble role={m.role}>{text}</Bubble>}
            {m.role === 'assistant' && <Citations message={m} />}
          </div>
        )
      })}

      {/* Live tool activity for the in-flight turn */}
      {streaming &&
        liveTools.map((t, i) => {
          const pending = t.sources === null && t.sql === null && t.count === null
          if (t.kind === 'count') {
            return (
              <div key={i}>
                {pending ? (
                  <p className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
                    <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500 align-middle" />
                    Counting documents mentioning{' '}
                    <span className="italic">&ldquo;{t.query}&rdquo;</span>…
                  </p>
                ) : t.count ? (
                  <CountBlock result={t.count} />
                ) : null}
              </div>
            )
          }
          if (t.kind === 'sql') {
            return (
              <div key={i}>
                {pending ? (
                  <p className="my-1 text-xs text-zinc-500 dark:text-zinc-400">
                    <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500 align-middle" />
                    Querying document metadata…
                  </p>
                ) : t.sql ? (
                  <SqlBlock result={t.sql} />
                ) : null}
                {t.isError && !t.sql && (
                  <p className="my-1 text-xs text-red-500">Metadata query failed.</p>
                )}
              </div>
            )
          }
          return (
            <div key={i}>
              <SearchMarker query={t.query} pending={pending} />
              {t.sources !== null && !t.isError && <SourceList sources={t.sources} />}
              {t.isError && (
                <p className="my-1 text-xs text-red-500">Search failed.</p>
              )}
            </div>
          )
        })}

      {streaming && liveWeb.length > 0 && <WebSourceList results={liveWeb} />}

      {streaming && streamText && (
        <Bubble role="assistant">
          {streamText}
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-zinc-400 align-text-bottom" />
        </Bubble>
      )}

      {streaming && !streamText && liveTools.length === 0 && (
        <p className="text-sm text-zinc-400">Thinking…</p>
      )}

      {/* A max_tokens stop looks identical to a finished answer on the wire.
          Saying so is the only thing separating a truncated reply from a
          confident half-sentence. */}
      {truncated && (
        <Banner tone="warning">
          Response was cut off at the output limit. Ask for a shorter answer, or
          raise MAX_OUTPUT_TOKENS.
        </Banner>
      )}

      {error && <Banner tone="error">{error}</Banner>}

      <div ref={endRef} />
    </div>
  )
}
