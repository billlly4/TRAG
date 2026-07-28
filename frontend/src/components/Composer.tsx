import { useState } from 'react'

import { Button } from './ui'

/**
 * No document picker here any more. Module 1 attached whole documents to a
 * message by hand; retrieval is now automatic -- the assistant searches
 * uploaded documents itself when the question calls for it.
 */
export function Composer({
  streaming,
  onSend,
  onStop,
}: {
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
}) {
  const [text, setText] = useState('')

  function submit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || streaming) return
    onSend(trimmed)
    setText('')
  }

  return (
    <form
      onSubmit={submit}
      className="border-t border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
    >
      <div className="flex items-end gap-2">
        <textarea
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) submit(e)
          }}
          placeholder="Ask something…  (Enter to send, Shift+Enter for a newline)"
          className="flex-1 resize-none rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
        />
        {streaming ? (
          <Button type="button" variant="ghost" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <Button type="submit" disabled={!text.trim()}>
            Send
          </Button>
        )}
      </div>
    </form>
  )
}
