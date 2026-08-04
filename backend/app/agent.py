"""The agent, as a LangGraph graph.

Replaces the hand-rolled `for turn in range(max_tool_turns)` loop that used to
live in `routers/chat.py`. What that loop did by hand -- call the model, notice
`stop_reason == "tool_use"`, dispatch, append results, repeat -- `create_agent`
does as a compiled graph, which is what makes Module 8's sub-agents tractable.

**Retrieval is untouched.** `hybrid_search`, RRF fusion, the cross-encoder and
the abstention gate were each tuned against the golden set, and LangChain's
equivalents (`EnsembleRetriever`, `CrossEncoderReranker`) weight and threshold
differently. Wrapping the measured code in a `@tool` keeps the golden set a
valid regression test for this rewrite; swapping the internals would have made
it measure something else.

Two properties from the old loop that had to survive the port, because both are
guarantees rather than conveniences:

  * **Capability gating.** The web-search tool is absent from the tool list
    unless the user asked for it, so the model *cannot* reach the web. Tools are
    therefore built per request, not once at import.
  * **Per-channel attribution.** Each tool returns `(text, artifact)` via
    `response_format="content_and_artifact"`; the artifact rides on
    `ToolMessage.artifact` and becomes the `sources` / `sql` the UI renders. The
    model sees only the text.
"""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from supabase import Client

from .config import Settings
from .embeddings import EmbeddingError
from .llm import sanitize_for_api
from .retrieval import Filters, count_matching, hybrid_search, search
from .sql_tool import SCHEMA as SQL_SCHEMA
from .sql_tool import run_sql

log = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_documents"
SQL_TOOL_NAME = "query_document_metadata"
COUNT_TOOL_NAME = "count_documents_mentioning"


def _as_list(value: Any) -> list[str]:
    """Tolerate a bare string where the schema asks for a list."""
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value] if isinstance(value, list) else []


# How many filenames to name back to the model when a scope cannot be resolved.
# Enough to choose from, short enough not to crowd the turn.
_NAME_HINT_LIMIT = 15


def document_outline(db: Client, document_id: str) -> str:
    """The section headings of one document, in order, deduplicated.

    Headings are literal text lifted from the document, so this cannot invent
    anything -- which is why it is safe to show when the relevance gate has
    just refused to show passages. It answers a DIFFERENT question ("what is in
    here") with metadata, rather than lowering the bar on the question the gate
    declined.
    """
    try:
        res = (
            db.table("chunks")
            .select("ordinal,section")
            .eq("document_id", document_id)
            .order("ordinal")
            .execute()
        )
    except Exception:  # noqa: BLE001
        log.warning("outline unavailable for %s", document_id, exc_info=True)
        return ""

    seen: list[str] = []
    for row in res.data or []:
        section = (row.get("section") or "").strip()
        if section and section not in seen:
            seen.append(section)
    return " | ".join(seen[:40])


def resolve_document(db: Client, name: str) -> tuple[list[dict], str | None]:
    """Turn what the model typed into document rows. Returns (rows, error).

    Matching happens in Python over the user's own rows rather than as a
    PostgREST `ilike`. Filenames routinely contain `_` and `%`, which are LIKE
    metacharacters -- `Saliency_Driven_Report.docx` would match documents it
    should not, and PostgREST offers no reliable ESCAPE clause. The corpus is
    bounded by the storage quota and `corpus_summary` already selects every row
    on each request, so one more small select costs nothing and removes the
    quoting problem entirely.

    An unresolvable name comes back as an error STRING, not an exception: the
    model can read "did you mean one of these" and fix its next call, the same
    way `run_sql` hands back a bad column name.
    """
    q = (name or "").strip()
    if not q:
        return [], None

    try:
        res = (
            db.table("documents")
            .select("id,filename,title,chunk_count")
            .eq("status", "ready")
            .execute()
        )
    except Exception:  # noqa: BLE001
        log.warning("document resolution unavailable", exc_info=True)
        return [], "Could not look up documents. Retry the search without the document filter."

    rows = res.data or []
    if not rows:
        return [], "There are no processed documents to search within."

    def names(subset: list[dict], limit: int) -> str:
        return ", ".join(sorted(r["filename"] for r in subset)[:limit])

    # An id the model copied from a metadata query. Exact, so it wins outright.
    if hits := [r for r in rows if r["id"] == q]:
        return hits, None

    folded = q.casefold()

    def matches(predicate) -> list[dict]:
        return [
            r for r in rows
            if predicate(folded, (r.get("filename") or "").casefold())
            or predicate(folded, (r.get("title") or "").casefold())
        ]

    # Exact name before substring, so "chapter2" cannot be swallowed by a file
    # called "chapter2-and-chapter3". Several rows sharing a name is not
    # ambiguity -- it is the same document twice, so scope to all of them.
    if hits := matches(lambda needle, field: field == needle):
        return hits, None

    hits = matches(lambda needle, field: needle in field)
    if len(hits) == 1:
        return hits, None
    if hits:
        return [], (
            f"'{q}' matches {len(hits)} documents ({names(hits, 8)}). "
            f"Name one of them exactly, or search without the document filter."
        )

    return [], (
        f"No document matches '{q}'. Available: {names(rows, _NAME_HINT_LIMIT)}. "
        f"Retry with one of these, or search without the document filter."
    )


def build_tools(settings: Settings, db: Client, *, web_search: bool) -> list[Any]:
    """The tool set for one request.

    Bound per request for two reasons. The tools close over `db`, which is the
    caller's RLS-scoped client -- a module-level tool would have to reach for a
    connection and lose that. And web search is gated by CAPABILITY, not by
    instruction: when the user has not asked for it the tool is simply not in
    this list, so the model cannot use it. A system prompt saying "don't search
    the web" is a request; an undeclared tool is a guarantee, and it is what
    keeps an unanswerable document question landing on abstention.
    """

    @tool(SEARCH_TOOL_NAME, response_format="content_and_artifact")
    def search_documents(
        query: str,
        top_k: int | None = None,
        document: str | None = None,
        doc_type: list[str] | None = None,
        source_org: list[str] | None = None,
        topics: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> tuple[str, dict]:
        """Search the user's uploaded documents for passages relevant to a question.

        Call this whenever answering might depend on their documents -- do not
        answer from memory when the user refers to their files. The optional
        filters narrow the search before ranking; use them only when the question
        is explicitly about a kind of document, a source, or a period, and only
        with values listed in the corpus summary. Filtering on a value that is
        not in the corpus returns nothing.

        Args:
            query: A focused search query.
            top_k: Passages to return, 1-20. Default 5.
            document: Search inside ONE named document. Pass the filename or
                title when the user asks about a specific file ("what is in
                chapter7", "summarise report.pdf"); otherwise leave it out and
                search the whole corpus.
            doc_type: Restrict to these document types.
            source_org: Restrict to these publishing organisations.
            topics: Restrict to documents tagged with any of these topics.
            year_min: Earliest publication year.
            year_max: Latest publication year.
        """
        q = (query or "").strip()
        if not q:
            return "Error: 'query' is required.", {"sources": [], "is_error": True}

        scoped: list[dict] = []
        if document:
            scoped, resolve_error = resolve_document(db, document)
            if resolve_error:
                # Returned rather than searched-anyway. Silently widening a
                # scoped search back to the corpus is how a chapter 2 passage
                # ends up cited in an answer about chapter 7.
                return resolve_error, {"sources": [], "is_error": True}
        document_ids = [r["id"] for r in scoped]

        filters = Filters(
            doc_types=_as_list(doc_type),
            source_orgs=_as_list(source_org),
            topics=_as_list(topics),
            document_ids=document_ids,
            year_min=year_min if isinstance(year_min, int) else None,
            year_max=year_max if isinstance(year_max, int) else None,
        )
        k = top_k if isinstance(top_k, int) else None

        try:
            # Same call either way, so switching retrieval strategy stays
            # invisible to the model.
            hits = (
                hybrid_search(db, q, k, filters)
                if settings.retrieval_hybrid
                else search(db, q, k, filters)
            )
        except EmbeddingError as exc:
            return f"Search unavailable: {exc}", {"sources": [], "is_error": True}

        if not hits:
            # Two opposite failures used to share one message, and conflating
            # them is what made the model tell a user their file "may not have
            # been properly indexed":
            #
            #   * A METADATA filter can match no documents at all (doc_type
            #     "invoice" in a corpus with none). "Retry without the filters"
            #     is right.
            #   * A DOCUMENT scope always matches -- the document was resolved
            #     by name. So an empty result means the relevance gate rejected
            #     every passage, and "retry without the filters" is actively
            #     harmful: it pulls in other documents, which is the
            #     cross-document attribution bug 0012 exists to prevent.
            if len(scoped) == 1:
                doc = scoped[0]
                outline = document_outline(db, doc["id"])
                parts = [
                    f"'{doc['filename']}' is indexed ({doc.get('chunk_count') or 0} "
                    f"passages) and was searched, but no passage was relevant "
                    f"enough to this query. The document is fine -- the query is "
                    f"the problem. Ask about its subject matter in specific terms "
                    f"rather than asking what it contains."
                ]
                if outline:
                    # Headings, not content -- said plainly, because the model
                    # must not present a table of contents as though it had read
                    # the sections underneath it.
                    parts.append(
                        f"Its section headings, for orientation only (these are "
                        f"headings, NOT the text under them): {outline}"
                    )
                return "\n\n".join(parts), {"sources": []}

            # Naming the filters matters: told only "nothing found", the model
            # concludes the corpus cannot answer and stops. Told the filters
            # matched nothing, it retries without them.
            if filters.active():
                return (
                    f"No passages matched the filters ({filters.describe()}). "
                    f"The filters may not match this corpus — retry without them "
                    f"before concluding the documents have no answer.",
                    {"sources": []},
                )
            return (
                "No relevant passages found in the user's documents for this query.",
                {"sources": []},
            )

        parts, sources = [], []
        for i, h in enumerate(hits, 1):
            parts.append(
                f"[{i}] {h.label()} (chunk {h.ordinal}, similarity {h.similarity:.2f})\n"
                f"{h.content}"
            )
            sources.append(
                {
                    "document_id": h.document_id,
                    "filename": h.filename,
                    "section": h.section,
                    "ordinal": h.ordinal,
                    "similarity": round(h.similarity, 3),
                    # The score the list is actually ORDERED by. Cosine alone
                    # next to a reranked ranking shows numbers that contradict
                    # the order, and a keyword-only hit reads as 0.000 --
                    # looking irrelevant when it was matched exactly.
                    "rerank_score": (
                        round(h.rerank_score, 2) if h.rerank_score is not None else None
                    ),
                }
            )
        return "\n\n".join(parts), {"sources": sources}

    @tool(SQL_TOOL_NAME, response_format="content_and_artifact")
    def query_document_metadata(sql: str) -> tuple[str, dict]:
        """Run a read-only SQL SELECT over the user's document metadata.

        Use this for questions about the CORPUS ITSELF rather than its contents:
        counts, totals, groupings, date ranges, "how many", "which types",
        "what did I upload". Do NOT use it to find out what documents SAY --
        that is search_documents.

        Postgres dialect. One statement, no semicolon, SELECT or WITH only.
        Results are capped at 200 rows, so aggregate rather than listing
        everything. If a query errors, read the message and fix the SQL -- a
        wrong column name or a table other than documents_queryable is the usual
        cause.

        Args:
            sql: A single SELECT over documents_queryable, no trailing semicolon.
        """
        text, meta, is_error = run_sql(db, {"sql": sql})
        return text, {"sql": meta, "is_error": is_error}

    @tool(COUNT_TOOL_NAME, response_format="content_and_artifact")
    def count_documents_mentioning(term: str) -> tuple[str, dict]:
        """Count EXACTLY how many of the user's documents contain a word or phrase.

        Use this for "how many of my documents mention X", "which files talk
        about X", "do any of my files reference X". It reads the full text of
        every document, so the number is a TOTAL, not a sample -- unlike
        search_documents, whose count only ever measures its own top-k.

        It matches WORDS, not meaning: a document about the subject in different
        wording is not counted. Say so when reporting the number.

        It returns no passage text. To find out what the documents actually say,
        follow up with search_documents.

        Args:
            term: The word or phrase to count. Keep it short and literal.
        """
        t = (term or "").strip()
        if not t:
            return "Error: 'term' is required.", {"count": None, "is_error": True}

        try:
            rows = count_matching(db, t)
        except Exception as exc:  # noqa: BLE001
            # Returned, not raised: 0013 may not be applied yet, and that is
            # something the model should route around rather than a 500.
            log.warning("count failed for %r", t, exc_info=True)
            return (
                f"Counting is unavailable ({str(exc)[:200]}). Do not guess a "
                f"total -- search instead and say the number you found is a "
                f"lower bound.",
                {"count": None, "is_error": True},
            )

        meta = {
            "term": t,
            "count": len(rows),
            "documents": [
                {"filename": r["filename"], "chunk_matches": r["chunk_matches"]}
                for r in rows
            ],
        }
        if not rows:
            return (
                f"Exactly 0 documents contain the word '{t}'. Note this matches "
                f"words literally, so a document covering the subject in other "
                f"wording would not be counted.",
                meta,
            )

        listed = ", ".join(
            f"{r['filename']} ({r['chunk_matches']} passage"
            f"{'s' if r['chunk_matches'] != 1 else ''})"
            for r in rows[:25]
        )
        return (
            f"Exactly {len(rows)} document(s) contain '{t}': {listed}. "
            f"This is a complete count over the full text of every document, "
            f"not a sample -- report it as the total. It matches the word "
            f"literally, so say that a document discussing the subject in "
            f"different wording would not be included.",
            meta,
        )

    # The docstring is the tool description the model sees, and it has to carry
    # the exact column list -- without it the model invents names and every call
    # comes back as an undefined-column error.
    query_document_metadata.description += f"\n\n{SQL_SCHEMA}"

    tools: list[Any] = []
    if web_search and settings.web_search_enabled:
        # A raw Anthropic server-side tool dict, passed straight through by
        # ChatAnthropic. It executes on Anthropic's infrastructure, so unlike the
        # two below there is nothing here to run -- results arrive as
        # `web_search_tool_result` blocks on the assistant message.
        #
        # FIRST, not last, and that ordering is load-bearing. The prompt-caching
        # middleware places its tool breakpoint on tools[-1] and returns the list
        # UNCHANGED unless that entry is a BaseTool (langchain_anthropic
        # middleware/prompt_caching.py, _tag_tools). With this dict on the end,
        # tool caching silently did nothing on exactly the turns that cost most.
        # Order carries no meaning to the model; it only has to be stable.
        tools.append(
            {
                "type": settings.web_search_tool_version,
                "name": "web_search",
                "max_uses": settings.web_search_max_uses,
            }
        )
    tools.append(search_documents)
    if settings.count_tool_enabled:
        tools.append(count_documents_mentioning)
    if settings.sql_tool_enabled:
        tools.append(query_document_metadata)
    return tools


def build_agent(settings: Settings, db: Client, system_prompt: str, *, web_search: bool):
    """Compile the graph for one request."""
    model = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_output_tokens,
        streaming=True,
    )
    middleware: list[Any] = [
        # The old loop bounded itself with `for turn in range(max_tool_turns)`.
        # This is the same bound, declared instead of written: it stops a
        # model that keeps calling tools from looping forever.
        ModelCallLimitMiddleware(run_limit=settings.max_tool_turns, exit_behavior="end")
    ]

    if settings.prompt_cache_enabled:
        # The Messages API is stateless, so every turn re-sends the whole thread
        # at full input price -- and one user message can drive up to
        # max_tool_turns model calls, each re-sending a prefix that grows as
        # tool results are appended. This makes re-reads of that prefix cost 10%.
        #
        # The middleware rather than hand-placed cache_control blocks: Anthropic
        # allows only four breakpoints per request, and tool definitions travel
        # as one contiguous block, so exactly one trailing breakpoint caches the
        # whole tool set. Getting that budget wrong is easy and silent.
        middleware.append(
            AnthropicPromptCachingMiddleware(
                ttl=settings.prompt_cache_ttl,
                min_messages_to_cache=settings.prompt_cache_min_messages,
                # This app only ever passes ChatAnthropic, so an unsupported
                # model here would be a bug in build_agent, not a user setting.
                unsupported_model_behavior="raise",
            )
        )

    return create_agent(
        model=model,
        tools=build_tools(settings, db, web_search=web_search),
        system_prompt=system_prompt,
        middleware=middleware,
    )


# Tool names whose results carry document sources rather than SQL metadata.
# Used by the SSE layer to decide which frame shape to emit.
SOURCE_TOOLS = {SEARCH_TOOL_NAME}
SQL_TOOLS = {SQL_TOOL_NAME}


def to_lc_messages(rows: list[dict]) -> list[BaseMessage]:
    """Stored Anthropic blocks -> LangChain messages.

    Postgres stays the source of truth rather than a LangGraph checkpointer: the
    UI reads `messages` directly, threads have to survive a reload, and the
    quota triggers count rows. So conversion happens at the boundary, in both
    directions, and the graph is stateless between requests.

    `tool_use` blocks are dropped from AIMessage content and carried as
    `tool_calls` instead. Both would be sent, and ChatAnthropic renders
    `tool_calls` itself -- leaving them in content duplicates every tool call on
    the wire.
    """
    out: list[BaseMessage] = []
    for m in rows:
        # Module 1 turns carry `document` blocks referencing Anthropic's Files
        # API, which this app no longer uses. Dropping them keeps old threads
        # replayable; the text of those turns survives.
        blocks = [b for b in sanitize_for_api(m["content"]) if b.get("type") != "document"]
        if not blocks:
            continue

        if m["role"] == "assistant":
            tool_calls = [
                {"name": b["name"], "args": b.get("input") or {}, "id": b["id"]}
                for b in blocks
                if b.get("type") == "tool_use" and b.get("id")
            ]
            content = [b for b in blocks if b.get("type") != "tool_use"]
            out.append(AIMessage(content=content, tool_calls=tool_calls))
            continue

        # A user row that only carries tool_result blocks is the synthetic turn
        # the old loop wrote; it becomes one ToolMessage per result.
        results = [b for b in blocks if b.get("type") == "tool_result"]
        if results:
            for b in results:
                out.append(
                    ToolMessage(
                        content=b.get("content") or "",
                        tool_call_id=b.get("tool_use_id") or "",
                        status="error" if b.get("is_error") else "success",
                    )
                )
        else:
            out.append(HumanMessage(content=blocks))
    return out


def blocks_of(message: AIMessage) -> list[dict]:
    """AIMessage content as storable Anthropic blocks.

    ChatAnthropic already returns Anthropic-shaped blocks -- text, tool_use,
    server_tool_use, web_search_tool_result -- so this is mostly a normalisation
    for the case where content came back as a bare string.
    """
    if isinstance(message.content, list):
        return [b for b in message.content if isinstance(b, dict)]
    return [{"type": "text", "text": str(message.content)}] if message.content else []


def text_of(chunk: Any) -> str:
    """The visible text in a streamed chunk, ignoring tool and thinking blocks."""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )
