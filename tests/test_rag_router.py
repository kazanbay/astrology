"""Unit tests of the multi-query knowledge-base router."""

import pytest

from rag.router import get_relevant_chunks


class FakeRetriever:
    def __init__(self, results=None, fail_on=None):
        self.calls = []
        self.results = results or {}
        self.fail_on = fail_on
        self.closed = False

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("qdrant is down")
        return self.results.get(query, [])

    def close(self):
        self.closed = True


def chunk(chunk_id):
    return {"chunk_id": chunk_id, "content": "текст", "section_title": "раздел"}


def test_searches_each_query_and_deduplicates():
    retriever = FakeRetriever(
        {"a": [chunk("c1"), chunk("c2")], "b": [chunk("c2"), chunk("c3")]}
    )

    chunks = get_relevant_chunks(["a", "b"], retriever=retriever, top_k=3)

    assert [item["chunk_id"] for item in chunks] == ["c1", "c2", "c3"]
    assert retriever.calls == [("a", 3), ("b", 3)]
    assert retriever.closed is False  # an external retriever is not closed


def test_failed_query_is_skipped():
    retriever = FakeRetriever({"good": [chunk("c1")]}, fail_on="bad")

    chunks = get_relevant_chunks(["bad", "good"], retriever=retriever)

    assert [item["chunk_id"] for item in chunks] == ["c1"]


def test_cap_and_empty_queries():
    retriever = FakeRetriever({"a": [chunk("c1"), chunk("c2"), chunk("c3")]})

    assert len(get_relevant_chunks(["a"], retriever=retriever, max_chunks=2)) == 2
    assert get_relevant_chunks([], retriever=retriever) == []
    assert get_relevant_chunks(["  "], retriever=retriever) == []


def test_invalid_limits():
    with pytest.raises(ValueError):
        get_relevant_chunks(["a"], retriever=FakeRetriever(), top_k=0)
    with pytest.raises(ValueError):
        get_relevant_chunks(["a"], retriever=FakeRetriever(), max_chunks=0)
