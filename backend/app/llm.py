"""The system prompt, and the rules for replaying stored blocks to the API.

The Anthropic client used to live here, wrapped by LangSmith's
`wrap_anthropic`. Both are gone: the app reaches Claude through ChatAnthropic
now, and LangChain traces to LangSmith natively.

That removal took the LangSmith environment export with it -- it only ran inside
`get_client()`, so when nothing called that any more, tracing stopped without an
error to say so. The export now lives in `config.get_settings`, where every
entry point reaches it just by reading settings.
"""

# Fields the Messages API accepts on *input* for each block type. The SDK's
# response models carry extra fields that are not valid to send back -- notably
# `parsed_output`, which streamed text blocks include and non-streamed ones do
# not. Replaying an unsanitised assistant turn fails with
# "content.0.text.parsed_output: Extra inputs are not permitted".
_API_BLOCK_FIELDS: dict[str, set[str]] = {
    "text": {"type", "text", "citations", "cache_control"},
    "document": {"type", "source", "citations", "cache_control", "title", "context"},
    "image": {"type", "source", "cache_control"},
    "tool_use": {"type", "id", "name", "input", "cache_control"},
    "tool_result": {"type", "tool_use_id", "content", "is_error", "cache_control"},
    "thinking": {"type", "thinking", "signature"},
}


def sanitize_for_api(content: list[dict]) -> list[dict]:
    """Strip response-only fields from stored blocks before replaying them.

    We persist the full dump because the UI needs it; this narrows it back to
    what the API accepts as input.

    Citation *spans* on assistant text blocks are dropped entirely rather than
    cleaned field-by-field. They are display metadata -- the model does not need
    its own previous citations to continue, since the document is still in
    context -- and trying to round-trip them fails three different ways:
    `file_id` is not accepted, `document_title` is required-but-nullable, and
    char_location spans are rejected outright when the document is referenced
    by file id. Keeping them in the database and dropping them here gets the UI
    what it needs without fighting the input schema.

    The `citations` key on a *document* block is different -- it is the config
    dict {"enabled": true}, and it must survive or citations stop being
    generated at all.
    """
    out: list[dict] = []
    for block in content:
        allowed = _API_BLOCK_FIELDS.get(block.get("type"))
        if allowed is None:
            out.append(block)
            continue

        clean = {k: v for k, v in block.items() if k in allowed and v is not None}

        if isinstance(clean.get("citations"), list):
            clean.pop("citations")

        out.append(clean)
    return out


SYSTEM_PROMPT = (
    "You are a helpful research assistant. The user's uploaded documents are "
    "searchable with the search_documents tool -- they are NOT in your context. "
    "When a question might depend on their documents, your FIRST action is the "
    "tool call -- write no text before it. Never say what you do or don't have "
    "before searching; the UI already shows the search. "
    "Answer from the passages returned, naming the source file. If the search "
    "finds nothing relevant, say so plainly rather than answering from memory. "
    "For questions clearly unrelated to the user's documents, answer directly "
    "without searching."
    "\n\n"
    # Routing. The two tools fail in opposite directions, so the split is worth
    # spelling out: search cannot count, and SQL cannot read.
    "Two tools cover two different kinds of question. Use search_documents for "
    "what the documents SAY. Use query_document_metadata for questions about "
    "the collection ITSELF -- how many, which types, from what source, uploaded "
    "when, how much space. Counting search results does not answer those: "
    "search returns only the top few passages, so its count measures the search, "
    "not the corpus."
    "\n\n"
    # The failure this prevents is specific and would be invisible to the user:
    # "3 documents mention X" sounds like a fact, not like top-k.
    # Both halves of this were observed failing: the model announced a file did
    # not exist before searching, then contradicted itself; and when a scoped
    # search came back empty it told the user their document was not indexed.
    "When the user names a file, pass it as the `document` argument to "
    "search_documents so the search is scoped to it -- the corpus summary above "
    "lists the filenames, so check there rather than guessing or claiming a file "
    "does not exist. If a scoped search returns no passages, the document is "
    "still indexed and readable; the query was simply too vague for any single "
    "passage to match. Ask about its subject in specific terms instead. Never "
    "tell the user a document failed to index on the strength of an empty "
    "search."
    "\n\n"
    "'How many of my documents mention X' is a THIRD kind of question, and it "
    "has its own tool: count_documents_mentioning. That reads the full text of "
    "every document and returns a total, so report its number as exact -- while "
    "saying it matches the word literally, not the subject. Never count "
    "search_documents results and present that as a total: search returns only "
    "its top few passages, so its count measures the search. If counting is "
    "unavailable, search and say the number you found is a lower bound."
    "\n\n"
    # Rule 1 of the module, restated where the model will read it. The hard
    # guarantee is that the tool is absent when unrequested; this is the softer
    # half -- not filling a gap with general knowledge either.
    "If the user's documents do not answer a question about their documents, "
    "say you do not have that information. Do not substitute general knowledge "
    "for what their files say. When a web_search tool is available, the user "
    "asked for it explicitly -- use it, and make clear which parts of your "
    "answer came from the web rather than from their documents."
)
