import { AuthForm } from './components/AuthForm'
import { Composer } from './components/Composer'
import { DropZone } from './components/DropZone'
import { MessageList } from './components/MessageList'
import { Sidebar } from './components/Sidebar'
import { useAuth } from './hooks/useAuth'
import { useChat } from './hooks/useChat'

function Chat({ email, onSignOut }: { email?: string; onSignOut: () => void }) {
  const chat = useChat()

  return (
    // overflow-hidden pins the app to the viewport: the ONLY scroll containers
    // are the message list and the sidebar list, never the page itself.
    <div className="flex h-full overflow-hidden">
      <Sidebar
        threads={chat.threads}
        activeId={chat.activeId}
        documents={chat.documents}
        notice={chat.notice}
        email={email}
        onSelect={chat.setActiveId}
        onNew={() => void chat.newThread()}
        onDeleteThread={(id) => void chat.removeThread(id)}
        onUpload={chat.upload}
        onDeleteDocument={(id) => void chat.removeDocument(id)}
        onReprocess={(id) => void chat.reprocess(id)}
        onReprocessAll={() => void chat.reprocessAll()}
        onSignOut={onSignOut}
      />

      <main className="flex min-w-0 flex-1 flex-col bg-zinc-100 dark:bg-zinc-950">
        {/* Drop anywhere over the conversation, not just on a small target. */}
        <DropZone onFiles={(files) => void chat.upload(files)}>
          <MessageList
            messages={chat.messages}
            streamText={chat.streamText}
            streaming={chat.streaming}
            liveTools={chat.liveTools}
            error={chat.error}
            truncated={chat.truncated}
          />
          <Composer
            streaming={chat.streaming}
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

  // Keyed on user id so switching accounts remounts the chat with clean state
  // rather than briefly showing the previous user's threads.
  return (
    <Chat
      key={session.user.id}
      email={session.user.email}
      onSignOut={() => void signOut()}
    />
  )
}
