"""Shaping RAG output into the JSON array stored in ``charts_cache``.

``get_structural_interpretations`` returns one match per chart element. Storing
that mapping verbatim would duplicate a whole section's text whenever one
chapter answers several elements (a node chapter, an aspect-pair chapter), so
the payload is a list of unique chunks, each carrying the elements it answered.
That is both the TZ shape ("массив найденных чанков") and a lossless one:

* the chunk text appears once;
* the element -> chunk link survives in ``elements``;
* ``element_chunk_index`` rebuilds the per-element view for the bot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Candidate fields copied into the cache. ``match_tier`` and ``rank_total``
#: keep the routing quality visible for later evaluation.
CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "title",
    "section_title",
    "section_path",
    "book_id",
    "chunk_index",
    "content",
    "match_tier",
    "rank_total",
    "is_transit",
)

ELEMENT_FIELDS: tuple[str, ...] = ("key", "kind", "display", "coverage", "rank")


def collect_unique_chunks(structural: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-element matches into a deduplicated chunk list.

    Order follows first appearance: the request order of the element queries
    and, inside one element, the router's ranking. The first chunk listed for
    an element is therefore its best candidate.
    """
    if not isinstance(structural, Mapping):
        raise ValueError("structural must be a mapping of element key -> match")

    chunks: dict[str, dict[str, Any]] = {}
    for element_key, match in structural.items():
        if not isinstance(match, Mapping):
            raise ValueError(f"structural[{element_key!r}] must be a mapping")
        candidates = match.get("candidates") or []
        if not isinstance(candidates, list):
            raise ValueError(f"structural[{element_key!r}]['candidates'] must be a list")

        for rank, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"structural[{element_key!r}]['candidates'][{rank}] must be a mapping")
            chunk_id = str(candidate.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(
                    f"structural[{element_key!r}]['candidates'][{rank}] has no chunk_id"
                )

            record = chunks.get(chunk_id)
            if record is None:
                record = {field: candidate.get(field) for field in CHUNK_FIELDS}
                record["chunk_id"] = chunk_id
                record["elements"] = []
                chunks[chunk_id] = record

            element = {
                "key": str(element_key),
                "kind": match.get("kind"),
                "display": match.get("element"),
                "coverage": match.get("coverage"),
                "rank": rank,
            }
            if element not in record["elements"]:
                record["elements"].append(element)

    return list(chunks.values())


def element_chunk_index(chunks: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map every element key to the first (best-ranked) chunk that answered it."""
    index: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise ValueError("chunks must be a list of mappings")
        for element in chunk.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            key = element.get("key")
            if key and key not in index:
                index[str(key)] = dict(chunk)
    return index


def element_stats(chunks: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count unique chunks and elements that received at least one section."""
    elements = {
        str(element.get("key"))
        for chunk in chunks
        for element in chunk.get("elements") or []
        if isinstance(element, Mapping) and element.get("key")
    }
    return {
        "chunks": len(chunks),
        "elements_with_chunks": len(elements),
    }
