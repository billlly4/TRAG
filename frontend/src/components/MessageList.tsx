import { useEffect, useRef } from 'react'

import type { LiveTool } from '../hooks/useChat'
import type { Message, Source } from '../lib/types'
import {
  blocksToCitations,
  blocksToText,
  isToolResultMessage,
  toolQueries,
  toolSources,
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
            ? 'max-w-[75%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm whitespace-pre-wrap text-white'
            : 'max-w-[75%] rounded-2xl rounded-bl-sm border border-zinc-200 bg-white px-4 py-2.5 text-sm whitespace-pre-wrap text-zinc-800 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100'
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

/** Citation spans only exist on Module 1 answers; kept so old threads render. */
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
  error,
  truncated,
}: {
  messages: Message[]
  streamText: string
  streaming: boolean
  liveTools: LiveTool[]
  error: string | null
  truncated: boolean
}) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streamText, liveTools.length])

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-6">
      {messages.length === 0 && !streaming && (
        <p className="pt-12 text-center text-sm text-zinc-400">
          Ask a question. Uploaded documents are searched automatically.
        </p>
      )}

      {messages.map((m) => {
        // Synthetic user turns that only carry tool results render as a
        // source list, not as something the user said.
        if (m.role === 'user' && isToolResultMessage(m)) {
          return <SourceList key={m.id} sources={toolSources(m.content)} />
        }

        const text = blocksToText(m.content)
        const queries = m.role === 'assistant' ? toolQueries(m.content) : []
        return (
          <div key={m.id}>
            {queries.map((q, i) => (
              <SearchMarker key={i} query={q} />
            ))}
            {text && <Bubble role={m.role}>{text}</Bubble>}
            {m.role === 'assistant' && <Citations message={m} />}
          </div>
        )
      })}

      {/* Live tool activity for the in-flight turn */}
      {streaming &&
        liveTools.map((t, i) => (
          <div key={i}>
            <SearchMarker query={t.query} pending={t.sources === null} />
            {t.sources !== null && !t.isError && <SourceList sources={t.sources} />}
            {t.isError && (
              <p className="my-1 text-xs text-red-500">Search failed.</p>
            )}
          </div>
        ))}

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
