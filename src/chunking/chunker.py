"""Split cleaned Markdown into token-bounded, header-aware chunks."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
_HEADER = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def _sections(markdown: str) -> list[tuple[str, str, str]]:
    """Return section bodies and their nearest H2/H3 header context."""
    sections: list[tuple[str, str, str]] = []
    h2 = ""
    h3 = ""
    body: list[str] = []

    def flush() -> None:
        value = "\n".join(body).strip()
        if value:
            sections.append((h2, h3, value))
        body.clear()

    for line in markdown.splitlines():
        match = _HEADER.match(line)
        level = len(match.group(1)) if match else 0
        if level in (2, 3):
            flush()
            if level == 2:
                h2, h3 = match.group(2).strip(), ""
            else:
                h3 = match.group(2).strip()
            continue
        body.append(line)
    flush()
    return sections


def _encode_with_offsets(text: str, tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    return token_ids, [tuple(pair) for pair in offsets if pair[0] != pair[1]]


def _split_by_tokens(text: str, limit: int, overlap: int, tokenizer) -> list[str]:
    token_ids, offsets = _encode_with_offsets(text, tokenizer)
    if len(token_ids) <= limit:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    overlap = min(overlap, max(0, limit - 1))
    while start < len(offsets):
        end = min(start + limit, len(offsets))

        # Prefer a nearby paragraph or line boundary without exceeding the token cap.
        if end < len(offsets):
            minimum_end = start + max(1, int(limit * 0.7))
            boundary_end = end
            for token_index in range(end - 1, minimum_end - 1, -1):
                char_end = offsets[token_index][1]
                if char_end < len(text) and text[char_end : char_end + 2] == "\n\n":
                    boundary_end = token_index + 1
                    break
                if char_end < len(text) and text[char_end : char_end + 1] == "\n":
                    boundary_end = token_index + 1
                    break
            end = boundary_end

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_markdown(document_id: int, title: str, clean_markdown: str) -> list[dict[str, Any]]:
    """Create document chunks with H2/H3 context and BGE-M3 token counts."""
    if not isinstance(clean_markdown, str):
        raise TypeError("clean_markdown must be a string")
    tokenizer = _get_tokenizer()
    chunks: list[dict[str, Any]] = []
    for h2, h3, section_text in _sections(clean_markdown):
        section = " > ".join(part for part in (h2, h3) if part) or "Introduction"
        prefix = f"[Document: {title} | Section: {section}]\n\n"
        prefix_tokens = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        content_limit = CHUNK_SIZE - prefix_tokens - 2
        if content_limit < 1:
            raise ValueError("Document title and section headers exceed the chunk token limit")
        section_tokens = len(tokenizer(section_text, add_special_tokens=False)["input_ids"])
        if section_tokens + prefix_tokens <= CHUNK_SIZE:
            section_chunks = [section_text]
        else:
            section_chunks = _split_by_tokens(
                section_text,
                content_limit,
                min(CHUNK_OVERLAP, content_limit - 1),
                tokenizer,
            )

        for section_chunk in section_chunks:
            content = prefix + section_chunk
            token_count = len(tokenizer(content, add_special_tokens=False)["input_ids"])
            chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "content": content,
                    "token_count": token_count,
                    "metadata": {"h2": h2 or None, "h3": h3 or None},
                }
            )
    return chunks
