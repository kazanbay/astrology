"""Multi-query knowledge-base retrieval for aspect-driven interpretations.

The semantic retriever answers one query at a time; interpretations need
several narrow queries (one per transit or synastry aspect) with
deduplication. This adapter keeps that logic in the RAG layer, so the service
layer only builds query strings and never talks to Qdrant directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

#: Chunks fetched per single query before deduplication.
DEFAULT_TOP_K = 2
#: Upper bound of unique chunks handed to the LLM.
DEFAULT_MAX_CHUNKS = 12


def get_relevant_chunks(
    queries: Iterable[str],
    *,
    retriever: Any | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[dict[str, Any]]:
    """Return deduplicated chunks for several aspect queries.

    Each query is searched separately: a narrow phrase like «Транзит Сатурна
    квадрат Солнца» matches the books better than one long sentence. A failure
    of a single query is logged and skipped; the rest still contribute.
    """
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if max_chunks < 1:
        raise ValueError("max_chunks must be greater than zero")
    cleaned = [str(query).strip() for query in queries if str(query).strip()]
    if not cleaned:
        return []

    owned = retriever is None
    client = retriever if retriever is not None else _default_retriever()
    collected: dict[str, dict[str, Any]] = {}
    try:
        for query in cleaned:
            try:
                found = client.search(query, top_k=top_k)
            except Exception:
                logger.warning("retrieval failed for one query; continuing", exc_info=True)
                continue
            for chunk in found or []:
                key = str(chunk.get("chunk_id") or chunk.get("id") or len(collected))
                collected.setdefault(key, chunk)
                if len(collected) >= max_chunks:
                    return list(collected.values())
        return list(collected.values())
    finally:
        if owned:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def _default_retriever() -> Any:
    """Build the production retriever lazily: the import pulls embeddings."""
    from .retriever import QdrantRetriever

    return QdrantRetriever()
