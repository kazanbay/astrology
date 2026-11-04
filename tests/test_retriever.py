from types import SimpleNamespace

import numpy as np
import pytest

from rag.retriever import COLLECTION_NAME, QdrantRetriever


class FakeModel:
    def __init__(self):
        self.queries = []

    def encode(self, text, **kwargs):
        self.queries.append((text, kwargs))
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class FakeClient:
    def __init__(self):
        self.collection = None
        self.query = None
        self.limit = None
        self.closed = False

    def query_points(self, *, collection_name, query, limit, with_payload):
        self.collection = collection_name
        self.query = query
        self.limit = limit
        assert with_payload is True
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=17,
                    score=0.8765,
                    payload={
                        "document_id": 4,
                        "chunk_id": 17,
                        "title": "Guide",
                        "content": "[Document: Guide | Section: Law]\n\nFull chunk text",
                        "metadata": {"h2": "Law", "h3": None},
                    },
                )
            ]
        )

    def close(self):
        self.closed = True


def test_search_returns_formatted_results_and_queries_top_k():
    model = FakeModel()
    client = FakeClient()
    retriever = QdrantRetriever(model=model, client=client)

    results = retriever.search("  how to defer service?  ", top_k=3)

    assert client.collection == COLLECTION_NAME
    assert client.limit == 3
    assert client.query == pytest.approx([0.1, 0.2, 0.3])
    assert model.queries[0][0] == "how to defer service?"
    assert set(results[0]) == {
        "score",
        "document_id",
        "chunk_id",
        "title",
        "content",
        "metadata",
    }
    assert results[0] == {
        "score": 0.8765,
        "document_id": 4,
        "chunk_id": 17,
        "title": "Guide",
        "content": "[Document: Guide | Section: Law]\n\nFull chunk text",
        "metadata": {"h2": "Law", "h3": None},
    }


def test_search_with_timings_returns_nonnegative_durations():
    retriever = QdrantRetriever(model=FakeModel(), client=FakeClient())
    results, timings = retriever.search_with_timings("query", top_k=1)
    assert len(results) == 1
    assert set(timings) == {"embedding_seconds", "qdrant_seconds", "total_seconds"}
    assert all(duration >= 0 for duration in timings.values())


@pytest.mark.parametrize("query", ["", "   ", None])
def test_search_rejects_empty_query(query):
    retriever = QdrantRetriever(model=FakeModel(), client=FakeClient())
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search(query)


def test_search_rejects_invalid_top_k():
    retriever = QdrantRetriever(model=FakeModel(), client=FakeClient())
    with pytest.raises(ValueError, match="greater than zero"):
        retriever.search("query", top_k=0)


def test_retriever_uses_custom_collection_name():
    client = FakeClient()
    retriever = QdrantRetriever(
        model=FakeModel(),
        client=client,
        collection_name="vronsky_fixed_500",
    )
    retriever.search("Сатурн", top_k=1)
    assert client.collection == "vronsky_fixed_500"


def test_retriever_keeps_string_book_ids_and_can_return_payload():
    class StringBookClient(FakeClient):
        def query_points(self, *, collection_name, query, limit, with_payload):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="point-1",
                        score=0.5,
                        payload={
                            "book_id": "cbae3ff9-56ce-5fba-896b-9da5466ae7f6",
                            "chunk_id": "point-1",
                            "title": "Том 3",
                            "content": "Фрагмент",
                            "metadata": {"hierarchy_path": ["Том 3", "Глава"]},
                        },
                    )
                ]
            )

    client = StringBookClient()
    retriever = QdrantRetriever(model=FakeModel(), client=client)
    results, _timings = retriever.search_with_timings("запрос", include_payload=True)

    assert results[0]["document_id"] == "cbae3ff9-56ce-5fba-896b-9da5466ae7f6"
    assert results[0]["payload"]["chunk_id"] == "point-1"
