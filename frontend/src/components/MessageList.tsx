import { useEffect, useRef } from 'react'

import type { Message } from '../lib/types'
import { blocksToCitations, blocksToText } from '../lib/types'
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
  error,
  truncated,
}: {
  messages: Message[]
  streamText: string
  streaming: boolean
  error: string | null
  truncated: boolean
}) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streamText])

  return (
    <div className="flex-1 space-y-4 overflow-y-auto p-6">
      {messages.length === 0 && !streaming && (
        <p className="pt-12 text-center text-sm text-zinc-400">
          Ask a question. Attach a document to ground the answer.
        </p>
      )}

      {messages.map((m) => (
        <div key={m.id}>
          <Bubble role={m.role}>{blocksToText(m.content)}</Bubble>
          {m.role === 'assistant' && <Citations message={m} />}
        </div>
      ))}

      {streaming && streamText && (
        <Bubble role="assistant">
          {streamText}
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-zinc-400 align-text-bottom" />
        </Bubble>
      )}

      {streaming && !streamText && (
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
