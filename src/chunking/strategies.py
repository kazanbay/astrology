"""Four chunking strategies used for retrieval A/B testing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from .document import (
    BlockSpan,
    BookDocument,
    TextBlock,
    flatten_document,
    iter_blocks,
    path_at,
)
from .tags import extract_meta_tags
from .tokens import Tokenizer, count_tokens, split_spans, trim_to_token_limit


FIXED_STRATEGY = "fixed_500"
SEMANTIC_STRATEGY = "semantic_paragraph"
STRUCTURAL_STRATEGY = "structural_xml"
PARENT_CHILD_STRATEGY = "parent_child"

STRATEGY_DESCRIPTIONS: dict[str, str] = {
    FIXED_STRATEGY: "Fixed-size 500 tokens with 50-token overlap, no document structure.",
    SEMANTIC_STRATEGY: "Paragraph-based chunks merged up to ~300-400 characters.",
    STRUCTURAL_STRATEGY: "One FB2 <section>/<title> per chunk, soft split over 1500 characters.",
    PARENT_CHILD_STRATEGY: "100-150 token children linked to 1000+ token parent sections.",
}
COLLECTION_NAMES: dict[str, str] = {
    FIXED_STRATEGY: "vronsky_fixed_500",
    SEMANTIC_STRATEGY: "vronsky_semantic_paragraph",
    STRUCTURAL_STRATEGY: "vronsky_structural_xml",
    PARENT_CHILD_STRATEGY: "vronsky_parent_child",
}
STRATEGIES: tuple[str, ...] = tuple(COLLECTION_NAMES)

FIXED_CHUNK_TOKENS = 500
FIXED_OVERLAP_TOKENS = 50
PARAGRAPH_MIN_CHARS = 300
PARAGRAPH_MAX_CHARS = 400
STRUCTURAL_MAX_CHARS = 1500
CHILD_CHUNK_TOKENS = 150
CHILD_OVERLAP_TOKENS = 20
PARENT_TARGET_TOKENS = 1500
PARENT_MAX_TOKENS = 2000

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Chunk:
    """A strategy-agnostic chunk that can be uploaded to Qdrant."""

    chunk_id: str
    strategy: str
    content: str
    title: str
    book_id: str
    source: str
    section_path: tuple[str, ...]
    chunk_index: int
    token_count: int
    char_count: int
    meta_tags: dict[str, list[object]] = field(default_factory=dict)
    parent_id: str | None = None
    parent_content: str | None = None
    parent_token_count: int | None = None
    parent_paths: tuple[tuple[str, ...], ...] = ()
    part_index: int | None = None
    part_count: int | None = None

    @property
    def section_title(self) -> str | None:
        return self.section_path[-1] if self.section_path else None

    @property
    def chapter_title(self) -> str | None:
        return self.section_path[0] if self.section_path else None


def collection_name(strategy: str) -> str:
    """Return the Qdrant collection configured for a strategy."""
    try:
        return COLLECTION_NAMES[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown chunking strategy: {strategy}") from exc


def _new_chunk(
    *,
    strategy: str,
    document: BookDocument,
    index: int,
    content: str,
    path: tuple[str, ...],
    tokenizer: Tokenizer,
    **extra: Any,
) -> Chunk:
    clean = content.strip()
    if not clean:
        raise ValueError("Chunk content must not be empty")
    token_count = count_tokens(clean, tokenizer)
    section_title = path[-1] if path else None
    return Chunk(
        chunk_id=str(uuid5(NAMESPACE_URL, f"astrology-chunk:{strategy}:{document.book_id}:{index}")),
        strategy=strategy,
        content=clean,
        title=document.display_title,
        book_id=document.book_id,
        source=document.source,
        section_path=path,
        chunk_index=index,
        token_count=token_count,
        char_count=len(clean),
        meta_tags=extract_meta_tags(section_title or "", clean),
        **extra,
    )


def build_chunks(document: BookDocument, strategy: str, tokenizer: Tokenizer) -> list[Chunk]:
    """Build all chunks of one strategy for one document."""
    builders = {
        FIXED_STRATEGY: _fixed_chunks,
        SEMANTIC_STRATEGY: _semantic_chunks,
        STRUCTURAL_STRATEGY: _structural_chunks,
        PARENT_CHILD_STRATEGY: _parent_child_chunks,
    }
    try:
        builder = builders[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown chunking strategy: {strategy}") from exc
    return builder(document, tokenizer)


def build_all_chunks(
    documents: Iterable[BookDocument],
    strategies: Iterable[str],
    tokenizer: Tokenizer,
) -> dict[str, list[Chunk]]:
    """Build chunks for every strategy, preserving document order."""
    selected = list(strategies)
    for strategy in selected:
        collection_name(strategy)
    result: dict[str, list[Chunk]] = {strategy: [] for strategy in selected}
    for document in documents:
        for strategy in selected:
            result[strategy].extend(build_chunks(document, strategy, tokenizer))
    return result


def _fixed_chunks(document: BookDocument, tokenizer: Tokenizer) -> list[Chunk]:
    text, spans = flatten_document(document)
    chunks: list[Chunk] = []
    for start, end in split_spans(
        text,
        FIXED_CHUNK_TOKENS,
        FIXED_OVERLAP_TOKENS,
        tokenizer,
        prefer_boundary=False,
    ):
        content = text[start:end].strip()
        if not content:
            continue
        content = trim_to_token_limit(content, FIXED_CHUNK_TOKENS, tokenizer)
        chunks.append(
            _new_chunk(
                strategy=FIXED_STRATEGY,
                document=document,
                index=len(chunks),
                content=content,
                path=path_at(spans, start),
                tokenizer=tokenizer,
            )
        )
    return chunks


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries, then hard-wrap."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for sentence in (part.strip() for part in SENTENCE_BOUNDARY.split(paragraph)):
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                pieces.append(sentence[start : start + max_chars])
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def _take_prefix(text: str, budget: int) -> tuple[str, str]:
    """Take at most ``budget`` characters, preferring a sentence or word boundary."""
    if budget <= 0:
        return "", text
    if len(text) <= budget:
        return text.strip(), ""

    window = text[: budget + 1]
    cut = max(window.rfind(mark) for mark in (". ", "! ", "? ", "… "))
    if cut >= max(1, budget // 2):
        cut += 1
    else:
        cut = window.rfind(" ")
        if cut < max(1, budget // 2):
            cut = budget
    head = text[:cut].strip()
    tail = text[cut:].strip()
    return head, tail


def _semantic_chunks(document: BookDocument, tokenizer: Tokenizer) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in iter_blocks(document):
        title_len = len(block.title) + 2 if block.title else 0
        body_max = max(120, PARAGRAPH_MAX_CHARS - title_len)
        body_min = max(120, PARAGRAPH_MIN_CHARS - title_len)
        pieces: list[str] = []
        for paragraph in block.paragraphs:
            pieces.extend(_split_paragraph(paragraph, body_max))

        buffer: list[str] = []
        section_start = len(chunks)

        def flush(merge_tail: bool = False) -> None:
            if not buffer:
                return
            body = "\n\n".join(buffer)
            if merge_tail and len(body) < 80 and len(chunks) > section_start:
                previous = chunks[-1]
                merged = f"{previous.content}\n\n{body}"
                if len(merged) <= 450:
                    chunks[-1] = replace(
                        previous,
                        content=merged,
                        token_count=count_tokens(merged, tokenizer),
                        char_count=len(merged),
                        meta_tags=extract_meta_tags(previous.section_title or "", merged),
                    )
                    buffer.clear()
                    return
            content = _with_section_title(block.title, body)
            chunks.append(
                _new_chunk(
                    strategy=SEMANTIC_STRATEGY,
                    document=document,
                    index=len(chunks),
                    content=content,
                    path=block.path,
                    tokenizer=tokenizer,
                )
            )
            buffer.clear()

        for piece in pieces:
            while piece:
                if not buffer:
                    buffer.append(piece)
                    break
                body = "\n\n".join(buffer)
                if len(body) + 2 + len(piece) <= body_max:
                    buffer.append(piece)
                    break

                space = body_max - len(body) - 2
                if len(body) < body_min and space >= 60:
                    head, tail = _take_prefix(piece, space)
                    if head:
                        buffer.append(head)
                        flush()
                        piece = tail
                        continue
                flush()
        flush(merge_tail=True)
    return chunks


def _with_section_title(title: str | None, body: str) -> str:
    return f"{title}\n\n{body}" if title else body


def _group_paragraphs(paragraphs: Iterable[str], max_chars: int) -> list[str]:
    """Greedily group paragraphs so each group stays within a character budget."""
    groups: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        for piece in _split_paragraph(paragraph, max_chars):
            candidate_len = len(piece) + (2 if current else 0)
            if current and current_len + candidate_len > max_chars:
                groups.append("\n\n".join(current))
                current = [piece]
                current_len = len(piece)
            else:
                current.append(piece)
                current_len += candidate_len
    if current:
        groups.append("\n\n".join(current))
    return groups


def _structural_chunks(document: BookDocument, tokenizer: Tokenizer) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in iter_blocks(document):
        header = block.title or (block.path[-1] if block.path else None)
        body = "\n\n".join(block.paragraphs)
        header_len = len(header) + 2 if header else 0
        body_limit = max(200, STRUCTURAL_MAX_CHARS - header_len)
        groups = _group_paragraphs(block.paragraphs, body_limit) if len(body) > body_limit else [body]
        for part_index, group in enumerate(groups):
            content = _with_section_title(header, group)
            chunks.append(
                _new_chunk(
                    strategy=STRUCTURAL_STRATEGY,
                    document=document,
                    index=len(chunks),
                    content=content,
                    path=block.path,
                    tokenizer=tokenizer,
                    part_index=part_index if len(groups) > 1 else None,
                    part_count=len(groups) if len(groups) > 1 else None,
                )
            )
    return chunks


def _parent_groups(blocks: list[TextBlock], tokenizer: Tokenizer) -> list[list[TextBlock]]:
    groups: list[list[TextBlock]] = []
    current: list[TextBlock] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = max(1, count_tokens(block.text, tokenizer))
        if current and current_tokens + block_tokens > PARENT_MAX_TOKENS:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
        if current_tokens >= PARENT_TARGET_TOKENS:
            groups.append(current)
            current = []
            current_tokens = 0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _paths_within(spans: list[BlockSpan], start: int, end: int) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    for span in spans:
        if span.end > start and span.start < end and span.path not in paths:
            paths.append(span.path)
    return tuple(paths)


def _parent_child_chunks(document: BookDocument, tokenizer: Tokenizer) -> list[Chunk]:
    blocks = list(iter_blocks(document))
    if not blocks:
        return []

    chunks: list[Chunk] = []
    parent_index = 0
    for group in _parent_groups(blocks, tokenizer):
        parent_text = "\n\n".join(block.text for block in group)
        parent_spans = _spans_for_blocks(group)
        for parent_start, parent_end in _split_parent(parent_text, tokenizer):
            parent_content = parent_text[parent_start:parent_end].strip()
            if not parent_content:
                continue
            parent_tokens = count_tokens(parent_content, tokenizer)
            parent_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"astrology-parent:{PARENT_CHILD_STRATEGY}:{document.book_id}:{parent_index}",
                )
            )
            parent_index += 1
            parent_slice = parent_text[parent_start:parent_end]
            for child_start, child_end in split_spans(
                parent_slice,
                CHILD_CHUNK_TOKENS,
                CHILD_OVERLAP_TOKENS,
                tokenizer,
                prefer_boundary=True,
            ):
                absolute_start = parent_start + child_start
                absolute_end = parent_start + child_end
                content = parent_text[absolute_start:absolute_end].strip()
                if not content:
                    continue
                content = trim_to_token_limit(content, CHILD_CHUNK_TOKENS, tokenizer)
                path = path_at(parent_spans, absolute_start)
                chunks.append(
                    _new_chunk(
                        strategy=PARENT_CHILD_STRATEGY,
                        document=document,
                        index=len(chunks),
                        content=content,
                        path=path,
                        tokenizer=tokenizer,
                        parent_id=parent_id,
                        parent_content=parent_content,
                        parent_token_count=parent_tokens,
                        parent_paths=_paths_within(parent_spans, parent_start, parent_end),
                    )
                )
    return chunks


def _spans_for_blocks(blocks: list[TextBlock]) -> list[BlockSpan]:
    spans: list[BlockSpan] = []
    position = 0
    for block in blocks:
        if spans:
            position += 2
        text = block.text
        spans.append(BlockSpan(start=position, end=position + len(text), path=block.path, title=block.title))
        position += len(text)
    return spans


def _split_parent(text: str, tokenizer: Tokenizer) -> list[tuple[int, int]]:
    """Keep parents in the 1000-2000 token band; split only oversized sections."""
    total = count_tokens(text, tokenizer)
    if total <= PARENT_MAX_TOKENS:
        return [(0, len(text))]
    parts = max(2, (total + PARENT_TARGET_TOKENS - 1) // PARENT_TARGET_TOKENS)
    target = (total + parts - 1) // parts
    return split_spans(text, target, 0, tokenizer, prefer_boundary=True)
