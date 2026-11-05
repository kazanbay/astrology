"""Qdrant collection management and batched embedding upload."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

try:  # support both `src.indexing` and top-level `indexing` imports
    from ..chunking.strategies import Chunk
except ImportError:  # pragma: no cover - depends on sys.path layout
    from chunking.strategies import Chunk


LOGGER = logging.getLogger(__name__)
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_SECONDS = 1.0

_SLEEP = Callable[[float], None]


@dataclass(frozen=True)
class UploadResult:
    """Counters returned after uploading one strategy collection."""

    uploaded: int
    skipped: int
    total: int
    seconds: float
    vector_size: int


def chunk_payload(chunk: Chunk, *, embedding_model: str) -> dict[str, Any]:
    """Build the flat, filterable payload schema shared by all four collections."""
    section_path = list(chunk.section_path)
    return {
        "chunk_id": chunk.chunk_id,
        "strategy": chunk.strategy,
        "book_id": chunk.book_id,
        "document_id": chunk.book_id,
        "source": chunk.source,
        "title": chunk.title,
        "content": chunk.content,
        "language": chunk.language,
        "planet": chunk.planet,
        "sign": chunk.sign,
        "house": chunk.house,
        "aspect": chunk.aspect,
        "school": chunk.school,
        "chunk_index": chunk.chunk_index,
        "section_path": section_path,
        "section_title": chunk.section_title,
        "chapter_title": chunk.chapter_title,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        "part_index": chunk.part_index,
        "part_count": chunk.part_count,
        "parent_id": chunk.parent_id,
        "parent_content": chunk.parent_content,
        "parent_token_count": chunk.parent_token_count,
        "parent_paths": [list(path) for path in chunk.parent_paths],
        "embedding_model": embedding_model,
    }


def ensure_payload_indexes(client: Any, name: str) -> None:
    """Create keyword/integer indexes used by agent-side payload filters."""
    from qdrant_client import models

    indexes: tuple[tuple[str, Any], ...] = (
        ("strategy", models.PayloadSchemaType.KEYWORD),
        ("book_id", models.PayloadSchemaType.KEYWORD),
        ("section_title", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
        ("language", models.PayloadSchemaType.KEYWORD),
        ("planet", models.PayloadSchemaType.KEYWORD),
        ("sign", models.PayloadSchemaType.KEYWORD),
        ("house", models.PayloadSchemaType.INTEGER),
        ("aspect", models.PayloadSchemaType.KEYWORD),
        ("school", models.PayloadSchemaType.KEYWORD),
    )
    for field_name, field_schema in indexes:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001 - an index is an optimization, not a blocker
            LOGGER.warning("Could not create payload index %s.%s: %s", name, field_name, exc)


def ensure_collection(
    client: Any,
    name: str,
    *,
    vector_size: int,
    recreate: bool = False,
) -> bool:
    """Create a cosine collection and its payload indexes; return True when created."""
    from qdrant_client import models

    exists = client.collection_exists(name)
    if recreate and exists:
        LOGGER.info("Deleting collection %s", name)
        client.delete_collection(name)
        exists = False

    if exists:
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        if size is None and isinstance(vectors, dict):
            size = next(iter(vectors.values())).size
        if size is not None and int(size) != vector_size:
            raise RuntimeError(
                f"Collection {name} stores {size}-dimensional vectors, "
                f"but the embedding model produces {vector_size}; rerun with --recreate"
            )
        return False

    LOGGER.info("Creating collection %s (%d dimensions, cosine)", name, vector_size)
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    ensure_payload_indexes(client, name)
    return True


def iter_batches(values: Sequence[Chunk], size: int) -> Iterable[Sequence[Chunk]]:
    """Yield consecutive batches without copying the whole sequence."""
    if size < 1:
        raise ValueError("batch size must be greater than zero")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def is_retryable_error(exc: BaseException) -> bool:
    """Treat transport, timeout and 5xx/429 failures as transient."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code >= 500 or status_code == 429
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    transient = {
        "ResponseHandlingException",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TransportError",
        "RemoteProtocolError",
        "TimeoutException",
    }
    return bool(names & transient)


def upsert_with_retry(
    client: Any,
    collection_name: str,
    points: list[Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BACKOFF_SECONDS,
    sleep: _SLEEP = time.sleep,
    rand: Callable[[], float] = random.random,
) -> None:
    """Upsert one batch, retrying transient network failures with jittered backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            return
        except Exception as exc:  # noqa: BLE001 - retry decision is made below
            if attempt >= max_attempts or not is_retryable_error(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) * (0.5 + rand())
            LOGGER.warning(
                "Qdrant upsert to %s failed (attempt %d/%d): %s; retrying in %.1fs",
                collection_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            sleep(delay)


def existing_ids(client: Any, collection_name: str, *, page_size: int = 1000) -> set[str]:
    """Scroll a collection and return its point ids for resumable uploads."""
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            break
    return ids


def _vectors_to_lists(vectors: Any, expected: int, vector_size: int) -> list[list[float]]:
    rows = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
    if len(rows) != expected:
        raise ValueError(f"Embedding model returned {len(rows)} vectors for {expected} chunks")
    result: list[list[float]] = []
    for row in rows:
        values = [float(value) for value in row]
        if len(values) != vector_size:
            raise ValueError(f"Expected {vector_size}-dimensional vectors, got {len(values)}")
        result.append(values)
    return result


def upload_chunks(
    client: Any,
    collection_name: str,
    chunks: Sequence[Chunk],
    model: Any,
    *,
    embedding_model: str,
    vector_size: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_batch_size: int = 16,
    skip_existing: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    progress: Callable[[int, int], None] | None = None,
) -> UploadResult:
    """Embed and upload chunks in fixed-size batches with retries."""
    from qdrant_client import models

    started = time.perf_counter()
    skipped = 0
    if skip_existing:
        present = existing_ids(client, collection_name)
        before = len(chunks)
        chunks = [chunk for chunk in chunks if chunk.chunk_id not in present]
        skipped = before - len(chunks)

    total = len(chunks)
    uploaded = 0
    for batch in iter_batches(chunks, batch_size):
        texts = [chunk.content for chunk in batch]
        vectors = model.encode(
            texts,
            batch_size=embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        rows = _vectors_to_lists(vectors, len(batch), vector_size)
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk_payload(chunk, embedding_model=embedding_model),
            )
            for chunk, vector in zip(batch, rows, strict=True)
        ]
        upsert_with_retry(
            client,
            collection_name,
            points,
            max_attempts=max_attempts,
        )
        uploaded += len(points)
        LOGGER.info("Uploaded %d/%d points to %s", uploaded, total, collection_name)
        if progress is not None:
            progress(uploaded, total)

    return UploadResult(
        uploaded=uploaded,
        skipped=skipped,
        total=total,
        seconds=time.perf_counter() - started,
        vector_size=vector_size,
    )
