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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from supabase import Client

from .config import Settings
from .embeddings import EmbeddingError
from .llm import sanitize_for_api
from .retrieval import Filters, hybrid_search, search
from .sql_tool import SCHEMA as SQL_SCHEMA
from .sql_tool import run_sql

log = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_documents"
SQL_TOOL_NAME = "query_document_metadata"


def _as_list(value: Any) -> list[str]:
    """Tolerate a bare string where the schema asks for a list."""
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value] if isinstance(value, list) else []


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
            doc_type: Restrict to these document types.
            source_org: Restrict to these publishing organisations.
            topics: Restrict to documents tagged with any of these topics.
            year_min: Earliest publication year.
            year_max: Latest publication year.
        """
        q = (query or "").strip()
        if not q:
            return "Error: 'query' is required.", {"sources": [], "is_error": True}

        filters = Filters(
            doc_types=_as_list(doc_type),
            source_orgs=_as_list(source_org),
            topics=_as_list(topics),
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

    # The docstring is the tool description the model sees, and it has to carry
    # the exact column list -- without it the model invents names and every call
    # comes back as an undefined-column error.
    query_document_metadata.description += f"\n\n{SQL_SCHEMA}"

    tools: list[Any] = [search_documents]
    if settings.sql_tool_enabled:
        tools.append(query_document_metadata)
    if web_search and settings.web_search_enabled:
        # A raw Anthropic server-side tool dict, passed straight through by
        # ChatAnthropic. It executes on Anthropic's infrastructure, so unlike the
        # two above there is nothing here to run -- results arrive as
        # `web_search_tool_result` blocks on the assistant message.
        tools.append(
            {
                "type": settings.web_search_tool_version,
                "name": "web_search",
                "max_uses": settings.web_search_max_uses,
            }
        )
    return tools


def build_agent(settings: Settings, db: Client, system_prompt: str, *, web_search: bool):
    """Compile the graph for one request."""
    model = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_output_tokens,
        streaming=True,
    )
    return create_agent(
        model=model,
        tools=build_tools(settings, db, web_search=web_search),
        system_prompt=system_prompt,
        middleware=[
            # The old loop bounded itself with `for turn in range(max_tool_turns)`.
            # This is the same bound, declared instead of written: it stops a
            # model that keeps calling tools from looping forever.
            ModelCallLimitMiddleware(
                run_limit=settings.max_tool_turns, exit_behavior="end"
            )
        ],
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
