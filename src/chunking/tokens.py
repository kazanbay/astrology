"""Token counting and offset-based text splitting shared by chunking strategies."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Protocol


DEFAULT_MODEL_NAME = "BAAI/bge-m3"


class Tokenizer(Protocol):
    """Minimal Hugging Face tokenizer interface used by the chunkers."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_offsets_mapping: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


def default_model_name() -> str:
    """Return the embedding model used for token accounting."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str | None = None) -> Tokenizer:
    """Load and cache the Hugging Face tokenizer for the configured model."""
    import logging

    # Long books exceed the model's 8192-token limit; offsets are still valid.
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name or default_model_name(), use_fast=True)


def encode_with_offsets(text: str, tokenizer: Tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize text and keep only tokens that map to a real character span."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"] if pair[0] != pair[1]]
    return token_ids, offsets


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    """Count tokens without special tokens."""
    if not text or not text.strip():
        return 0
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return len(encoded["input_ids"])


def trim_to_token_limit(text: str, limit: int, tokenizer: Tokenizer) -> str:
    """Trim a chunk that grew by a boundary token after slicing the source text."""
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    if count_tokens(text, tokenizer) <= limit:
        return text

    target = max(1, limit - 1)
    while text and count_tokens(text, tokenizer) > limit:
        _token_ids, offsets = encode_with_offsets(text, tokenizer)
        if len(offsets) <= limit:
            break
        cut = offsets[min(len(offsets) - 1, target) - 1][1]
        if cut >= len(text):
            break
        text = text[:cut].strip()
    return text


def _prefer_boundary(text: str, offsets: list[tuple[int, int]], start: int, end: int) -> int:
    """Move a split point back to a nearby paragraph or line break when possible."""
    minimum_end = start + max(1, int((end - start) * 0.7))
    for token_index in range(end - 1, minimum_end - 1, -1):
        char_end = offsets[token_index][1]
        if text[char_end : char_end + 2] == "\n\n":
            return token_index + 1
        if text[char_end : char_end + 1] == "\n":
            return token_index + 1
    return end


def split_spans(
    text: str,
    limit: int,
    overlap: int,
    tokenizer: Tokenizer,
    *,
    prefer_boundary: bool = True,
) -> list[tuple[int, int]]:
    """Split text into character spans of at most ``limit`` tokens with ``overlap``."""
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    if not text or not text.strip():
        return []

    _token_ids, offsets = encode_with_offsets(text, tokenizer)
    if not offsets:
        return []
    if len(offsets) <= limit:
        return [(offsets[0][0], offsets[-1][1])]

    safe_overlap = max(0, min(overlap, limit - 1))
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(offsets):
        end = min(start + limit, len(offsets))
        if end < len(offsets) and prefer_boundary:
            end = _prefer_boundary(text, offsets, start, end)

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        if char_end > char_start:
            spans.append((char_start, char_end))
        if end >= len(offsets):
            break
        start = max(start + 1, end - safe_overlap)
    return spans


def split_text(
    text: str,
    limit: int,
    overlap: int,
    tokenizer: Tokenizer,
    *,
    prefer_boundary: bool = True,
) -> list[str]:
    """Return stripped chunk strings for the given token budget."""
    chunks: list[str] = []
    for start, end in split_spans(
        text,
        limit,
        overlap,
        tokenizer,
        prefer_boundary=prefer_boundary,
    ):
        value = text[start:end].strip()
        if value:
            chunks.append(value)
    return chunks
