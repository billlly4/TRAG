"""Text-to-SQL over document metadata.

Vector search answers "what does my corpus say about X"; it cannot answer "how
many documents do I have from each source". The second is a `group by` over the
typed columns Module 4 added, and no amount of embedding similarity will produce
a count -- retrieval returns the top k passages, so counting what comes back
measures k, not the corpus.

The model writes SQL. It is executed by `query_documents` (0011), which drops to
a role granted SELECT on exactly one view, so the boundary is Postgres grants
rather than anything parsed here. Nothing in this module is load-bearing for
security; it exists to give the model a schema it can aim at and to turn errors
into something it can correct.
"""

import json
import logging
from typing import Any

from supabase import Client

log = logging.getLogger(__name__)

# Ships in the tool description. Without the exact column list the model invents
# names -- `upload_date`, `source`, `filetype` -- and every call comes back as a
# permissions or undefined-column error, which reads like the tool is broken.
SCHEMA = """\
View `documents_queryable` (the ONLY table you may query; already filtered to
this user, so never add a user_id condition):

  id              uuid         document id
  filename        text         original upload name, e.g. 'q3-report.pdf'
  mime_type       text         e.g. 'application/pdf', 'text/markdown'
  title           text         extracted title; NULL if extraction found none
  doc_type        text         extracted category; NULL if unknown
  source_org      text         publishing organisation; NULL if unknown
  authors         text[]       use unnest(authors) to count per author
  published_on    date         only set when a full date was printed
  published_year  int          set whenever the year is known -- prefer this
  language        text         ISO code, e.g. 'en'
  topics          text[]       use unnest(topics) to count per topic
  byte_size       bigint       size of the uploaded file
  chunk_count     int          passages the document was split into
  status          text         'ready' | 'pending' | 'failed' | ...
  created_at      timestamptz  when the user uploaded it

There is no updated_at column: you can answer when something was UPLOADED, not
when it was last changed."""

SQL_TOOL = {
    "name": "query_document_metadata",
    "description": (
        "Run a read-only SQL SELECT over the user's document metadata. Use this "
        "for questions about the CORPUS ITSELF rather than its contents: counts, "
        "totals, groupings, date ranges, 'how many', 'which types', 'what did I "
        "upload'. Do NOT use it to find out what documents SAY -- that is "
        "search_documents.\n\n"
        "Postgres dialect. One statement, no semicolon, SELECT or WITH only. "
        "Results are capped at 200 rows, so aggregate rather than listing "
        "everything.\n\n"
        f"{SCHEMA}\n\n"
        "If a query errors, read the message and fix the SQL -- a wrong column "
        "name or a table other than documents_queryable is the usual cause."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A single SELECT statement over documents_queryable, with "
                    "no trailing semicolon."
                ),
            }
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
}

# The model sees rows as JSON. Past a few dozen the useful signal is the
# aggregate anyway, and a wide result set crowds out the rest of the context.
MAX_ROWS = 200
MAX_CHARS = 20_000


def _error(message: str, sql: str) -> tuple[str, dict, bool]:
    return f"SQL error: {message}", {"sql": sql, "row_count": 0, "error": message}, True


def run_sql(db: Client, tool_input: dict[str, Any]) -> tuple[str, dict, bool]:
    """Execute one query_document_metadata call.

    Returns (text for the model, meta for the UI, is_error) -- the same shape as
    `_run_search`, so the chat loop dispatches on name and nothing else changes.

    Errors come back as tool results rather than exceptions on purpose: a
    mistyped column is something the model can fix on the next turn, and a
    500 would take the whole stream down instead.
    """
    sql = str(tool_input.get("sql") or "").strip().rstrip(";").strip()
    if not sql:
        return _error("'sql' is required.", sql)

    try:
        res = db.rpc("query_documents", {"sql": sql, "max_rows": MAX_ROWS}).execute()
    except Exception as exc:  # noqa: BLE001
        # Covers every rejection path: the function's own structural guards
        # (only_select_allowed, single_statement_only), a permission denial from
        # rag_readonly reaching for a table it cannot see, a syntax error, and
        # the 5s statement timeout. All of them are things the model should read
        # and retry, so none of them are special-cased.
        log.info("text-to-sql rejected: %s | sql=%r", exc, sql[:300])
        return _error(str(exc)[:400], sql)

    rows = res.data if isinstance(res.data, list) else []
    if not rows:
        return (
            "The query ran but matched no rows. If you filtered on a metadata "
            "value, it may not exist in this corpus -- try the query without "
            "that filter before concluding there is nothing to report.",
            {"sql": sql, "row_count": 0},
            False,
        )

    payload = json.dumps(rows, default=str)
    truncated = len(payload) > MAX_CHARS
    if truncated:
        # Cut whole rows, not the JSON string: a truncated blob is unparseable
        # and the model cannot tell a cut-off result from a malformed one.
        keep = rows[: max(1, len(rows) // 2)]
        while len(json.dumps(keep, default=str)) > MAX_CHARS and len(keep) > 1:
            keep = keep[: len(keep) // 2]
        payload = json.dumps(keep, default=str)

    text = f"{len(rows)} row(s):\n{payload}"
    if truncated:
        text += (
            f"\n\n(Showing part of {len(rows)} rows -- the result was too large. "
            f"Aggregate in SQL rather than listing rows.)"
        )

    return text, {"sql": sql, "row_count": len(rows), "truncated": truncated}, False
