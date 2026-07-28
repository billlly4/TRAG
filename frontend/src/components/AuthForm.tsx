import { useState } from 'react'

import { supabase } from '../lib/supabase'
import { Banner, Button, Input } from './ui'

export function AuthForm() {
  const [mode, setMode] = useState<'signin' | 'signup'>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)

    const { error } =
      mode === 'signup'
        ? await supabase.auth.signUp({ email, password })
        : await supabase.auth.signInWithPassword({ email, password })

    if (error) setError(error.message)
    setBusy(false)
  }

  return (
    <div className="flex h-full items-center justify-center p-6">
      <form
        onSubmit={submit}
        className="w-full max-w-sm space-y-4 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
      >
        <div>
          <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            {mode === 'signup' ? 'Create an account' : 'Sign in'}
          </h1>
          <p className="mt-1 text-sm text-zinc-500">TRAG &mdash; Module 1</p>
        </div>

        <Input
          type="email"
          required
          autoComplete="email"
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Input
          type="password"
          required
          minLength={6}
          autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {error && <Banner tone="error">{error}</Banner>}

        <Button type="submit" disabled={busy} className="w-full">
          {busy ? 'Working…' : mode === 'signup' ? 'Sign up' : 'Sign in'}
        </Button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === 'signup' ? 'signin' : 'signup')
            setError(null)
          }}
          className="w-full text-sm text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
        >
          {mode === 'signup'
            ? 'Already have an account? Sign in'
            : 'Need an account? Sign up'}
        </button>
      </form>
    </div>
  )
}
