"""Markdown-aware chunking.

Chunk size targets retrieval precision, not model limits: one vector averaging
many topics retrieves poorly. The embedding model's 8192-token window is 10x
the 800-token target, which is why a crude character-based token estimate is
safe here -- the ceiling is nowhere near.
"""

import re
from dataclasses import dataclass

from .config import get_settings

# ~4 chars/token for English text. Deliberately conservative; see module note.
CHARS_PER_TOKEN = 4

# Descending separators for recursive splitting: paragraph, line, sentence, word.
_SEPARATORS = ["\n\n", "\n", ". ", " "]

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Separates levels in a chunk's section path ("Chapter 3 > Inventory Models").
_SECTION_SEP = " › "


@dataclass
class Chunk:
    ordinal: int
    content: str
    token_count: int

    # Markdown heading path this chunk starts under. Free provenance: docling
    # already emits headings, so this costs no LLM call. It is what lets a
    # citation say which section a passage came from, and it is prepended to
    # the embedding input so a chunk is findable by its section's subject.
    section: str | None = None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _is_table(block: str) -> bool:
    lines = block.strip().splitlines()
    return len(lines) >= 2 and all(_TABLE_LINE.match(l) for l in lines)


def _split_blocks(text: str) -> list[tuple[str, str | None]]:
    """Split into (block, section path) pairs.

    Markdown tables stay whole. Headings do double duty: each one updates the
    running section path AND remains in the text, because a heading is content
    the reader (and the model) should still see in the chunk.
    """
    blocks: list[tuple[str, str | None]] = []
    stack: list[tuple[int, str]] = []  # (heading level, title)
    table: list[str] = []
    para: list[str] = []

    def section() -> str | None:
        return _SECTION_SEP.join(title for _, title in stack) or None

    def flush_para() -> None:
        joined = "\n".join(para).strip()
        if joined:
            blocks.append((joined, section()))
        para.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("\n".join(table), section()))
        table.clear()

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            # Flush first: the preceding text belongs to the OLD section.
            flush_table()
            flush_para()
            level = len(heading.group(1))
            # Pop siblings and deeper levels; an h2 replaces the previous h2
            # and everything nested under it, but keeps the h1 above it.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
            blocks.append((line.strip(), section()))
        elif _TABLE_LINE.match(line):
            flush_para()
            table.append(line)
        elif line.strip() == "":
            flush_table()
            flush_para()
        else:
            flush_table()
            para.append(line)
    flush_table()
    flush_para()
    return blocks


def _split_table(block: str, target_chars: int) -> list[str]:
    """Split an oversized table by rows, repeating the header on every piece.

    A table split mid-row leaves a chunk reading `| 4.2 | 891 |` --
    semantically meaningless and unretrievable. The header row is what makes
    each piece stand alone.
    """
    lines = block.splitlines()
    header: list[str] = []
    rows = lines
    if len(lines) >= 2 and _TABLE_DIVIDER.match(lines[1]):
        header, rows = lines[:2], lines[2:]

    header_len = sum(len(l) + 1 for l in header)
    pieces: list[str] = []
    group: list[str] = []
    group_len = header_len

    for row in rows:
        if group and group_len + len(row) + 1 > target_chars:
            pieces.append("\n".join(header + group))
            group, group_len = [], header_len
        group.append(row)
        group_len += len(row) + 1
    if group:
        pieces.append("\n".join(header + group))
    return pieces


def _split_recursive(text: str, target_chars: int, depth: int = 0) -> list[str]:
    """Split on descending separators; hard-cut when none remain.

    The hard cut matters: a 5,000-char run with no separators at all must
    still split rather than emitting one oversized chunk.
    """
    if len(text) <= target_chars:
        return [text]
    if depth >= len(_SEPARATORS):
        return [text[i : i + target_chars] for i in range(0, len(text), target_chars)]

    sep = _SEPARATORS[depth]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _split_recursive(text, target_chars, depth + 1)

    # Re-attach the separator so sentences keep their full stops.
    tail = sep.rstrip() if sep == ". " else ""
    out: list[str] = []
    for part in parts:
        piece = part + tail if tail else part
        out.extend(_split_recursive(piece, target_chars, depth + 1))
    return out


def chunk_text(text: str) -> list[Chunk]:
    settings = get_settings()
    target_chars = settings.chunk_target_tokens * CHARS_PER_TOKEN
    overlap_chars = settings.chunk_overlap_tokens * CHARS_PER_TOKEN

    # Pass 1: pieces that each fit the target, every one keeping the section
    # of the block it came from. Tables are atomic when they fit and split
    # header-repeated when they do not.
    pieces: list[tuple[str, str | None]] = []
    for block, sect in _split_blocks(text):
        if len(block) <= target_chars:
            pieces.append((block, sect))
        elif _is_table(block):
            pieces.extend((p, sect) for p in _split_table(block, target_chars))
        else:
            pieces.extend((p, sect) for p in _split_recursive(block, target_chars))

    # Pass 2: pack pieces into chunks, carrying overlap between neighbours.
    # Overlap is whole trailing pieces (never a sliced table or mid-word cut),
    # capped by the overlap budget.
    chunks: list[Chunk] = []
    current: list[tuple[str, str | None]] = []
    current_len = 0
    has_new = False  # False while `current` holds only carried overlap
    new_section: str | None = None  # section of the first non-carried piece

    def finalize() -> None:
        nonlocal current, current_len, has_new, new_section
        # Emit only when something new is pending -- a chunk that is nothing
        # but the previous chunk's overlap would be a pure duplicate.
        if has_new:
            content = "\n\n".join(t for t, _ in current).strip()
            if content:
                chunks.append(
                    Chunk(len(chunks), content, _estimate_tokens(content), new_section)
                )
        carry: list[tuple[str, str | None]] = []
        carry_len = 0
        for piece in reversed(current):
            if carry_len + len(piece[0]) > overlap_chars:
                break
            carry.insert(0, piece)
            carry_len += len(piece[0]) + 2
        current, current_len = carry, carry_len
        has_new = False
        new_section = None

    for piece in pieces:
        if current and current_len + len(piece[0]) + 2 > target_chars:
            finalize()
            # A near-target piece can overflow even with just the carry ahead
            # of it; the piece wins and the overlap is dropped.
            if current and current_len + len(piece[0]) + 2 > target_chars:
                current, current_len = [], 0
        current.append(piece)
        current_len += len(piece[0]) + 2
        # The chunk is labelled by where its new content starts, not by the
        # carried overlap -- otherwise every chunk inherits its predecessor's
        # section and the path is always one step behind.
        if not has_new:
            new_section = piece[1]
        has_new = True
    finalize()

    return chunks


# Bump when the layout below changes: it alters every stored vector, so
# record.py folds it into config_hash and the corpus reads stale.
EMBED_CONTEXT_VERSION = 1


def embed_text(chunk: Chunk, title: str) -> str:
    """The text actually sent to the embedding model for a chunk.

    Deliberately different from chunk.content, which stays the raw body. A
    chunk reading "margins fell 4% year over year" says nothing about whose
    margins; the answer is in the title and the heading above it, which the
    chunk itself never repeats. Prepending them makes the chunk retrievable by
    its own subject.

    Only the vector input is enriched. Feeding the same header back to Claude
    inside every retrieved passage would spend tokens restating what the
    source list already shows.
    """
    header = title if not chunk.section else f"{title}{_SECTION_SEP}{chunk.section}"
    return f"{header}\n\n{chunk.content}" if header else chunk.content
1