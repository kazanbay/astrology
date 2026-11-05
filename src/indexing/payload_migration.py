"""Rebuild flat Qdrant payloads for existing collections without re-embedding."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Iterable, Mapping

try:  # support both `src.indexing` and top-level `indexing` imports
    from ..chunking.tags import DEFAULT_SCHOOL, extract_chunk_metadata
    from .qdrant_index import is_retryable_error
except ImportError:  # pragma: no cover - depends on sys.path layout
    from chunking.tags import DEFAULT_SCHOOL, extract_chunk_metadata
    from indexing.qdrant_index import is_retryable_error


LOGGER = logging.getLogger(__name__)
REMOVED_KEYS = {"meta_tags", "metadata", "hierarchy_path"}
METADATA_KEYS = ("language", "planet", "sign", "house", "aspect", "school")


def rebuild_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a flat payload with root-level English metadata fields."""
    section_path = payload.get("section_path")
    legacy_path = payload.get("hierarchy_path")
    if not isinstance(section_path, list):
        section_path = legacy_path if isinstance(legacy_path, list) else []

    section_title = payload.get("section_title")
    if not section_title and section_path:
        section_title = section_path[-1]

    school = str(payload.get("school") or DEFAULT_SCHOOL)
    metadata = extract_chunk_metadata(
        str(section_title or ""),
        str(payload.get("content") or ""),
        language=payload.get("language"),
        school=school,
    )

    rebuilt = {key: value for key, value in payload.items() if key not in REMOVED_KEYS}
    rebuilt["section_path"] = section_path
    rebuilt["section_title"] = section_title
    if not rebuilt.get("chapter_title") and section_path:
        rebuilt["chapter_title"] = section_path[0]
    rebuilt.update(
        language=metadata.language,
        planet=metadata.planet,
        sign=metadata.sign,
        house=metadata.house,
        aspect=metadata.aspect,
        school=metadata.school,
    )
    return rebuilt


def _distribution(payloads: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[Any]] = {key: Counter() for key in METADATA_KEYS}
    for payload in payloads:
        for key, counter in counters.items():
            value = payload.get(key)
            counter[value if value is not None else "null"] += 1
    return {
        key: {str(value): count for value, count in counter.most_common(12)}
        for key, counter in counters.items()
    }


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _update_with_retry(
    client: Any,
    collection_name: str,
    operations: list[Any],
    *,
    max_attempts: int,
    base_delay: float,
    sleep: Any = time.sleep,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            client.batch_update_points(
                collection_name=collection_name,
                update_operations=operations,
                wait=True,
            )
            return
        except Exception as exc:  # noqa: BLE001 - retry classification below
            if attempt >= max_attempts or not is_retryable_error(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "Payload update for %s failed (attempt %d/%d): %s; retrying in %.1fs",
                collection_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            sleep(delay)


def migrate_collection(
    client: Any,
    collection_name: str,
    *,
    batch_size: int = 100,
    page_size: int = 256,
    max_attempts: int = 5,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Scroll a collection and replace every point payload with the flat schema."""
    from qdrant_client import models

    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Collection {collection_name} does not exist")

    seen = 0
    updated = 0
    all_payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        payloads = [rebuild_payload(point.payload or {}) for point in points]
        all_payloads.extend(payloads)
        seen += len(points)
        LOGGER.info("Scanned %d points in %s", seen, collection_name)

        if not dry_run:
            for batch in _batches(list(zip(points, payloads, strict=True)), batch_size):
                operations = [
                    models.OverwritePayloadOperation(
                        overwrite_payload=models.SetPayload(payload=payload, points=[point.id])
                    )
                    for point, payload in batch
                ]
                _update_with_retry(
                    client,
                    collection_name,
                    operations,
                    max_attempts=max_attempts,
                    base_delay=0.5,
                )
                updated += len(operations)
                LOGGER.info("Updated %d/%d payloads in %s", updated, seen, collection_name)

        if limit is not None and seen >= limit:
            break
        if offset is None:
            break

    return {
        "collection": collection_name,
        "seen": seen,
        "updated": updated,
        "dry_run": dry_run,
        "distribution": _distribution(all_payloads),
    }
