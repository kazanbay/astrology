"""Tests for the embedding backends (no network, no PyTorch)."""

from types import SimpleNamespace

import pytest

from rag.embedding import (
    ApiEmbeddingBackend,
    EmbeddingError,
    LocalEmbeddingBackend,
    get_embedding_backend,
)


class FakeEmbeddings:
    def __init__(self, dimension=4, reverse=False, fail=None):
        self.dimension = dimension
        self.reverse = reverse
        self.fail = fail
        self.calls = []

    def create(self, *, model, input, encoding_format):
        self.calls.append(
            {"model": model, "input": list(input), "encoding_format": encoding_format}
        )
        if self.fail is not None:
            raise self.fail
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1)] * self.dimension)
            for index in range(len(input))
        ]
        if self.reverse:
            data.reverse()
        return SimpleNamespace(data=data)


def backend(embeddings, *, dimension=4, batch_size=32):
    return ApiEmbeddingBackend(
        model_name="BAAI/bge-m3",
        dimension=dimension,
        batch_size=batch_size,
        client=SimpleNamespace(embeddings=embeddings),
    )


def test_single_string_is_wrapped_and_returns_one_vector():
    embeddings = FakeEmbeddings()

    vectors = backend(embeddings).encode("натальная карта")

    assert vectors == [[1.0, 1.0, 1.0, 1.0]]
    assert embeddings.calls[0]["input"] == ["натальная карта"]
    assert embeddings.calls[0]["encoding_format"] == "float"


def test_vectors_follow_input_order_even_if_provider_sorts_them():
    embeddings = FakeEmbeddings(reverse=True)

    vectors = backend(embeddings).encode(["a", "b", "c"])

    assert vectors == [[1.0] * 4, [2.0] * 4, [3.0] * 4]


def test_batches_are_sent_separately():
    embeddings = FakeEmbeddings()

    vectors = backend(embeddings, batch_size=2).encode(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert [len(call["input"]) for call in embeddings.calls] == [2, 2, 1]


def test_dimension_mismatch_is_rejected():
    embeddings = FakeEmbeddings(dimension=3)

    with pytest.raises(EmbeddingError, match="1024"):
        backend(embeddings, dimension=1024).encode("текст")


def test_empty_input_makes_no_request():
    embeddings = FakeEmbeddings()

    assert backend(embeddings).encode([]) == []
    assert embeddings.calls == []


def test_provider_failure_is_wrapped():
    embeddings = FakeEmbeddings(fail=RuntimeError("402 Payment Required"))

    with pytest.raises(EmbeddingError, match="402"):
        backend(embeddings).encode("текст")


def test_blank_input_is_rejected():
    with pytest.raises(EmbeddingError, match="non-empty"):
        backend(FakeEmbeddings()).encode(["ok", "   "])


def test_missing_endpoint_key_is_reported_without_network():
    api = ApiEmbeddingBackend(base_url="https://example.test/v1", api_key="", client=None)

    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        api.encode("текст")


def test_openrouter_key_is_used_only_for_the_openrouter_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    openrouter = ApiEmbeddingBackend(base_url="https://openrouter.ai/api/v1", client=None)
    assert openrouter._api_key == "or-key"

    third_party = ApiEmbeddingBackend(base_url="https://api.example.test/v1", client=None)
    assert third_party._api_key == ""


def test_backend_factory_returns_local_without_importing_torch():
    local = get_embedding_backend("local")

    assert isinstance(local, LocalEmbeddingBackend)
    assert local.dimension == 1024


def test_backend_factory_rejects_unknown_name():
    with pytest.raises(EmbeddingError, match="unknown EMBEDDING_BACKEND"):
        get_embedding_backend("magic")
