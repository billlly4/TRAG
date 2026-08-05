import { useState } from 'react'

import { AuthForm } from './components/AuthForm'
import { Composer } from './components/Composer'
import { DropZone } from './components/DropZone'
import { MessageList } from './components/MessageList'
import { Sidebar } from './components/Sidebar'
import { cx } from './components/ui'
import { useAuth } from './hooks/useAuth'
import { useChat } from './hooks/useChat'

/** Tailwind's `md`. Kept as a constant so the JS default matches the CSS. */
const MD = 768

function Chat({ email, onSignOut }: { email?: string; onSignOut: () => void }) {
  const chat = useChat()

  // Open on a desktop, closed on a phone -- where the sidebar
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= MD)

  /** On a phone the sidebar overlays the chat, so acting closes it. */
  function actAndClose(fn: () => void) {
    return () => {
      fn()
      if (window.innerWidth < MD) setSidebarOpen(false)
    }
  }

  const activeTitle = chat.threads.find((t) => t.id === chat.activeId)?.title

  return (

    <div className="flex h-full overflow-hidden">
      {/* Backdrop, phones only. Desktop collapses to reclaim width and needs no
          dismissal target; on a phone, tapping beside a drawer to close it is
          the expected gesture. */}
      {sidebarOpen && (
        <button
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
        />
      )}

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        threads={chat.threads}
        activeId={chat.activeId}
        documents={chat.documents}
        notice={chat.notice}
        usage={chat.usage}
        email={email}
        onSelect={(id) => actAndClose(() => chat.setActiveId(id))()}
        onNew={actAndClose(() => void chat.newThread())}
        onDeleteThread={(id) => void chat.removeThread(id)}
        onUpload={chat.upload}
        onDeleteDocument={(id) => void chat.removeDocument(id)}
        onReprocess={(id) => void chat.reprocess(id)}
        onReprocessAll={() => void chat.reprocessAll()}
        onSignOut={onSignOut}
      />

      <main
        className={cx(
          'flex min-w-0 flex-1 flex-col bg-zinc-100 transition-[margin] duration-200 dark:bg-zinc-950',
          sidebarOpen ? 'md:ml-72' : 'ml-0',
        )}
      >
        <header className="flex items-center gap-2 border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
            aria-expanded={sidebarOpen}
            className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-200/70 dark:hover:bg-zinc-800"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M4 6h16M4 12h16M4 18h16"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <p className="truncate text-sm font-medium text-zinc-700 dark:text-zinc-200">
            {activeTitle || 'TRAG'}
          </p>
        </header>

        {/* Drop anywhere over the conversation, not just on a small target. */}
        <DropZone onFiles={(files) => void chat.upload(files)}>
          <MessageList
            messages={chat.messages}
            streamText={chat.streamText}
            streaming={chat.streaming}
            liveTools={chat.liveTools}
            liveWeb={chat.liveWeb}
            error={chat.error}
            truncated={chat.truncated}
          />
          <Composer
            streaming={chat.streaming}
            full={chat.chatFull}
            webSearch={chat.webSearch}
            onWebSearchChange={chat.setWebSearch}
            onSend={(text) => void chat.send(text)}
            onStop={chat.stop}
          />
        </DropZone>
      </main>
    </div>
  )
}

export default function App() {
  const { session, loading, signOut } = useAuth()

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-zinc-400">
        Loading…
      </div>
    )
  }

  if (!session) return <AuthForm />

  return (
    <Chat
      key={session.user.id}
      email={session.user.email}
      onSignOut={() => void signOut()}
    />
  )
}
