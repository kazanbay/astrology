"""Embed PostgreSQL book chunks with BGE-M3 and upload them to Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag.embedding import get_embedding_backend  # noqa: E402
from src.rag.retriever import COLLECTION_NAME  # noqa: E402


VECTOR_SIZE = 1024
BATCH_SIZE = 8


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is missing from the environment or .env")
    if url.startswith("postgresql+asyncpg://"):
        url = "postgresql://" + url.removeprefix("postgresql+asyncpg://")
    return url


async def load_chunks() -> list[tuple[str, int | None, str, str, str, str, list[str], dict[str, Any]]]:
    """Read book chunk content and metadata from PostgreSQL."""
    import asyncpg

    connection = await asyncpg.connect(_database_url())
    try:
        rows = await connection.fetch(
            """SELECT c.id, c.book_id, b.title, b.volume_title,
                      c.chapter_title, c.section_title, c.hierarchy_path,
                      c.clean_text, c.meta_tags
               FROM chunks AS c
               LEFT JOIN books AS b ON b.id = c.book_id
               ORDER BY c.book_id, c.id"""
        )
        chunks = []
        for row in rows:
            book_title = row["title"] or "Без названия"
            volume_title = row["volume_title"]
            title = f"{book_title} — {volume_title}" if volume_title else book_title
            tags = row["meta_tags"] or {}
            if isinstance(tags, str):
                tags = json.loads(tags)
            chunks.append(
                (
                    str(row["id"]),
                    row["book_id"],
                    title,
                    row["clean_text"],
                    row["chapter_title"] or "",
                    row["section_title"] or "",
                    list(row["hierarchy_path"] or []),
                    tags,
                )
            )
        return chunks
    finally:
        await connection.close()


def _batches(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def index_chunks(chunks: list[tuple], *, recreate: bool = False) -> int:
    """Create/update the Qdrant collection and upload missing vectors."""
    from qdrant_client import QdrantClient, models

    qdrant_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is missing from the environment or .env")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY is missing from the environment or .env")
    if not chunks:
        raise RuntimeError("PostgreSQL chunks table is empty; load book chunks first")

    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=120)
    try:
        exists = client.collection_exists(COLLECTION_NAME)
        if recreate and exists:
            client.delete_collection(COLLECTION_NAME)
            exists = False
        if exists:
            collection = client.get_collection(COLLECTION_NAME)
            vector_config = collection.config.params.vectors
            if not isinstance(vector_config, models.VectorParams) or vector_config.size != VECTOR_SIZE:
                raise RuntimeError(
                    f"Collection {COLLECTION_NAME} has an incompatible vector configuration; "
                    "rerun with --recreate to replace it"
                )
            existing_ids: set[str] = set()
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                existing_ids.update(str(point.id) for point in points)
                if offset is None:
                    break
        else:
            existing_ids = set()
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        pending = [chunk for chunk in chunks if chunk[0] not in existing_ids]
        print(
            f"Chunks read: {len(chunks)}; already present: {len(existing_ids)}; "
            f"to upload: {len(pending)}",
            flush=True,
        )
        if not pending:
            return client.count(collection_name=COLLECTION_NAME, exact=True).count

        model = get_embedding_backend()
        print(f"Encoding batches of {BATCH_SIZE} with {model.model_name}", flush=True)
        uploaded = 0
        for batch in _batches(pending, BATCH_SIZE):
            vectors = model.encode(
                [chunk[3] for chunk in batch],
                batch_size=BATCH_SIZE,
            )
            points = []
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk_id, book_id, title, content, chapter, section, hierarchy, tags = chunk
                if len(vector) != VECTOR_SIZE:
                    raise ValueError(f"Expected {VECTOR_SIZE}-dimensional vectors, got {len(vector)}")
                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk_id,
                            "book_id": book_id,
                            "document_id": book_id,
                            "title": title,
                            "content": content,
                            "chapter_title": chapter,
                            "section_title": section,
                            "hierarchy_path": hierarchy,
                            "metadata": {
                                "hierarchy_path": hierarchy,
                                "chapter_title": chapter,
                                "section_title": section,
                                "meta_tags": tags,
                            },
                        },
                    )
                )
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            uploaded += len(points)
            print(f"Uploaded {uploaded}/{len(pending)} vectors", flush=True)

        total = client.count(collection_name=COLLECTION_NAME, exact=True).count
        print(f"Vectors uploaded: {uploaded}; vectors in {COLLECTION_NAME}: {total}")
        return total
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the collection")
    args = parser.parse_args()
    chunks = asyncio.run(load_chunks())
    index_chunks(chunks, recreate=args.recreate)


if __name__ == "__main__":
    main()
