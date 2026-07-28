# AGENTS.md

RAG app with chat (default) and document ingestion interfaces. Config via env vars, no admin UI.

Dev-time instructions only — the running app never reads this file.

## Stack
- Frontend: React + TypeScript + Vite + Tailwind + shadcn/ui
- Backend: Python + FastAPI
- Database: Supabase (Postgres, pgvector, Auth, Storage, Realtime)
- LLM: Anthropic Claude via the Messages API (`claude-haiku-4-5`, set by `ANTHROPIC_MODEL`)
- Observability: LangSmith via `wrap_anthropic`

## Rules
- Python runs in the root `.venv`, managed by `uv`. Add deps to `pyproject.toml`, then `uv sync` — never `pip install` directly
- No LangChain, no LangGraph — raw SDK calls only. The LangSmith wrapper is tracing, not a framework
- Use Pydantic for structured LLM outputs
- All tables need Row-Level Security — users only see their own data
- The backend forwards the caller's JWT so Postgres enforces RLS. Do not use the `service_role` key to work around a policy
- Stream chat responses via SSE
- Ingestion is manual file upload only — no connectors or automated pipelines
- No knowledge graph / GraphRAG — deliberately out of scope, see "Deferred: GraphRAG" in `agentic-rag-plan/PRD.md`

## Git
- **Never run git commands.** No `add`, `commit`, `push`, `branch`, `checkout`, or `reset` — the user handles all git manually
- Read-only inspection (`git status`, `git check-ignore`) is fine for verification
- When a new build artifact appears, add it to `.gitignore` — do not pre-populate with entries for files that do not exist yet

## Secrets
- `.env` is gitignored and holds real keys. `.env.example` is committed and holds placeholders only
- Never print secret values in output — variable names only
- Only `VITE_`-prefixed vars reach the browser. `ANTHROPIC_API_KEY` must never be one

## Claude API notes
- `max_tokens` bounds **output**, not input, and is not a cost control — you pay for tokens generated, not the ceiling. Chat uses 8192
- Always check `stop_reason == "max_tokens"` before treating a reply as complete, and surface truncation in the UI
- Prompt caching needs a **4096-token minimum prefix** on Haiku 4.5. Below that it silently does not cache — verify via `usage.cache_read_input_tokens`, never assume
- `cache_control` goes at the end of the stable prefix, with varying content after it
- No `thinking` parameter — Haiku 4.5 rejects `effort` and a chat app does not need it

## Planning
- Plans live in `agentic-rag-plan/.agent/plans/` as `{sequence}.{plan-name}.md`
- Each task needs at least one validation test
- Complexity marker at the top: ✅ Simple / ⚠️ Medium / 🔴 Complex (break 🔴 into checkpoints before executing)

## Progress
`agentic-rag-plan/PROGRESS.md` tracks module status. Update it as tasks complete.
