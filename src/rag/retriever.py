"""BGE-M3 semantic retrieval over an astrology Qdrant collection."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .embedding import load_bge_model


COLLECTION_NAME = "astrology_chunks"


class QdrantRetriever:
    """Encode queries with BGE-M3 and return matching chunk payloads."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        client: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if client is None and not qdrant_url:
            raise RuntimeError("QDRANT_URL is missing from the environment or .env")
        if client is None and not api_key:
            raise RuntimeError("QDRANT_API_KEY is missing from the environment or .env")

        self.model = model if model is not None else load_bge_model()
        self.collection_name = collection_name or COLLECTION_NAME
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=120)
        self.client = client

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k cosine-similar chunks in the RAG payload format."""
        results, _timings = self.search_with_timings(query, top_k=top_k)
        return results

    def search_with_timings(
        self,
        query: str,
        top_k: int = 5,
        *,
        include_payload: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Search and return separate embedding, Qdrant, and total durations."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        started = time.perf_counter()
        embedding_started = time.perf_counter()
        query_vector = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding_seconds = time.perf_counter() - embedding_started

        qdrant_started = time.perf_counter()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        qdrant_seconds = time.perf_counter() - qdrant_started

        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            point_id = point.id
            chunk_id = payload.get("chunk_id", point_id)
            document_id = payload.get("book_id", payload.get("document_id"))
            if document_id is not None:
                try:
                    document_id = int(document_id)
                except (TypeError, ValueError):
                    document_id = str(document_id)
            else:
                document_id = 0
            result = {
                "score": float(point.score),
                "document_id": document_id,
                "chunk_id": chunk_id,
                "title": str(payload.get("title") or ""),
                "content": str(payload.get("content") or ""),
                "metadata": payload.get("metadata") or {},
            }
            if include_payload:
                result["payload"] = payload
            results.append(result)

        timings = {
            "embedding_seconds": embedding_seconds,
            "qdrant_seconds": qdrant_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        return results, timings

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
