from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

from chunking.strategies import PARENT_CHILD_STRATEGY, Chunk
from indexing.qdrant_index import (
    chunk_payload,
    ensure_collection,
    is_retryable_error,
    upload_chunks,
    upsert_with_retry,
)


class FakeModel:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return np.ones((len(texts), self.dimension), dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return self.dimension


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.deleted = []
        self.indexes = []
        self.batches = []
        self.fail_times = 0
        self.scroll_points = []

    def collection_exists(self, name):
        return name in self.collections

    def get_collection(self, name):
        size = self.collections[name]
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=size))))

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)

    def create_collection(self, name, vectors_config):
        self.collections[name] = vectors_config.size

    def create_payload_index(self, collection_name, field_name, field_schema, wait):
        self.indexes.append((collection_name, field_name))

    def upsert(self, collection_name, points, wait):
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionResetError("temporary network failure")
        self.batches.append((collection_name, len(points), points))

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        return self.scroll_points, None


def _chunk(index: int, *, parent: bool = False) -> Chunk:
    return Chunk(
        chunk_id=str(UUID(int=index + 1)),
        strategy=PARENT_CHILD_STRATEGY if parent else "fixed_500",
        content=f"Текст фрагмента номер {index}",
        title="Книга",
        book_id="book",
        source="book.fb2",
        section_path=("Том 1", "Глава"),
        chunk_index=index,
        token_count=3,
        char_count=24,
        parent_id="parent-1" if parent else None,
        parent_content="Большой родительский раздел" if parent else None,
        parent_token_count=1200 if parent else None,
        parent_paths=(("Том 1", "Глава"),) if parent else (),
    )


def test_chunk_payload_has_uniform_schema_and_parent_link():
    payload = chunk_payload(_chunk(0, parent=True), embedding_model="BAAI/bge-m3")

    assert payload["chunk_id"] == str(UUID(int=1))
    assert payload["strategy"] == PARENT_CHILD_STRATEGY
    assert payload["document_id"] == "book"
    assert payload["section_title"] == "Глава"
    assert payload["hierarchy_path"] == ["Том 1", "Глава"]
    assert payload["metadata"]["hierarchy_path"] == ["Том 1", "Глава"]
    assert payload["parent_id"] == "parent-1"
    assert payload["parent_content"] == "Большой родительский раздел"
    assert payload["parent_token_count"] == 1200
    assert payload["embedding_model"] == "BAAI/bge-m3"


def test_ensure_collection_creates_vectors_and_payload_indexes():
    client = FakeClient()
    created = ensure_collection(client, "test_collection", vector_size=1024)

    assert created is True
    assert client.collections["test_collection"] == 1024
    assert {field for _collection, field in client.indexes} == {
        "strategy",
        "book_id",
        "section_title",
        "chunk_index",
    }


def test_ensure_collection_rejects_dimension_mismatch():
    client = FakeClient()
    client.collections["test_collection"] = 768
    with pytest.raises(RuntimeError, match="768-dimensional"):
        ensure_collection(client, "test_collection", vector_size=1024)


def test_ensure_collection_recreates_on_request():
    client = FakeClient()
    client.collections["test_collection"] = 1024
    ensure_collection(client, "test_collection", vector_size=1024, recreate=True)
    assert client.deleted == ["test_collection"]
    assert client.collections["test_collection"] == 1024


def test_upload_chunks_uses_hundred_point_batches():
    client = FakeClient()
    model = FakeModel()
    chunks = [_chunk(index) for index in range(250)]

    result = upload_chunks(
        client,
        "test_collection",
        chunks,
        model,
        embedding_model="BAAI/bge-m3",
        vector_size=4,
        batch_size=100,
        embed_batch_size=8,
    )

    assert result.uploaded == 250
    assert result.total == 250
    assert [size for _collection, size, _points in client.batches] == [100, 100, 50]
    assert all(call["batch_size"] == 8 for call in model.calls)
    assert all(call["normalize_embeddings"] is True for call in model.calls)
    first_point = client.batches[0][2][0]
    assert first_point.payload["content"] == "Текст фрагмента номер 0"


def test_upload_chunks_skips_existing_ids():
    client = FakeClient()
    client.scroll_points = [SimpleNamespace(id=UUID(int=index + 1)) for index in range(50)]
    chunks = [_chunk(index) for index in range(250)]

    result = upload_chunks(
        client,
        "test_collection",
        chunks,
        FakeModel(),
        embedding_model="BAAI/bge-m3",
        vector_size=4,
        batch_size=100,
        skip_existing=True,
    )

    assert result.skipped == 50
    assert result.uploaded == 200
    assert [size for _collection, size, _points in client.batches] == [100, 100]


def test_upload_chunks_validates_vector_size():
    with pytest.raises(ValueError, match="Expected 8-dimensional"):
        upload_chunks(
            FakeClient(),
            "test_collection",
            [_chunk(0)],
            FakeModel(dimension=4),
            embedding_model="model",
            vector_size=8,
        )


def test_upsert_with_retry_retries_transient_errors():
    client = FakeClient()
    client.fail_times = 2
    delays = []

    upsert_with_retry(
        client,
        "test_collection",
        [object()],
        max_attempts=4,
        base_delay=0.5,
        sleep=delays.append,
        rand=lambda: 0.0,
    )

    assert delays == [0.25, 0.5]
    assert len(client.batches) == 1


def test_upsert_with_retry_does_not_retry_permanent_errors():
    class BadRequest(Exception):
        status_code = 400

    client = FakeClient()

    def broken_upsert(**_kwargs):
        raise BadRequest("bad payload")

    client.upsert = broken_upsert
    delays = []
    with pytest.raises(BadRequest):
        upsert_with_retry(client, "test_collection", [object()], sleep=delays.append)
    assert delays == []


def test_retryable_error_classification():
    assert is_retryable_error(ConnectionResetError())
    assert is_retryable_error(TimeoutError())

    transient = Exception("server unavailable")
    transient.status_code = 503
    assert is_retryable_error(transient)

    permanent = Exception("bad request")
    permanent.status_code = 400
    assert not is_retryable_error(permanent)
