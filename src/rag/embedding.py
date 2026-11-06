"""Embedding backends: cloud API (default) or the optional local BGE-M3.

The project's Qdrant collections were built with ``BAAI/bge-m3`` (1024
dimensions). Moving inference to the cloud keeps those collections usable only
if the API returns vectors of the *same* model and width, so the backend
validates every vector against :data:`EMBEDDING_DIMENSION` and never silently
switches models.

Two backends exist:

* ``api`` - any OpenAI-compatible ``/embeddings`` endpoint, configured through
  ``EMBEDDING_BASE_URL`` / ``EMBEDDING_API_KEY`` / ``EMBEDDING_MODEL``. The
  client is created lazily, so importing the package and starting the bot does
  not require the key to be present.
* ``local`` - the previous in-process SentenceTransformer path. It needs the
  ``local-embeddings`` extra (PyTorch and sentence-transformers are no longer
  part of the core runtime) and exists for offline re-indexing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

#: Width of every vector already stored in Qdrant. Do not change without
#: recreating the collections.
EMBEDDING_DIMENSION = 1024
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_API_BATCH_SIZE = 32
DEFAULT_API_TIMEOUT = 120.0


class EmbeddingError(RuntimeError):
    """Embeddings cannot be produced with the current configuration."""


class EmbeddingBackend(Protocol):
    """Minimal contract shared by the API and local backends."""

    model_name: str
    dimension: int

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]: ...


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resolve_credentials(
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str]:
    """Resolve endpoint and key, sharing the OpenRouter key only with OpenRouter.

    Falling back to ``OPENROUTER_API_KEY`` for a third-party host would leak the
    key to that provider, so the fallback applies only when the endpoint is
    OpenRouter itself (or not set at all).
    """
    url = (base_url or _env("EMBEDDING_BASE_URL")).rstrip("/")
    key = api_key if api_key is not None else _env("EMBEDDING_API_KEY")

    is_openrouter = not url or "openrouter.ai" in url
    if not url and _env("OPENROUTER_API_KEY"):
        url = "https://openrouter.ai/api/v1"
    if not key and is_openrouter:
        key = _env("OPENROUTER_API_KEY")
    return url, key


class ApiEmbeddingBackend:
    """OpenAI-compatible ``/embeddings`` client.

    ``client`` is injectable, which is how tests exercise request shaping
    without a network call.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimension: int = EMBEDDING_DIMENSION,
        batch_size: int = DEFAULT_API_BATCH_SIZE,
        timeout: float = DEFAULT_API_TIMEOUT,
        client: Any | None = None,
    ) -> None:
        self.model_name = (model_name or _env("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)).strip()
        self.dimension = int(dimension)
        if self.dimension < 1:
            raise EmbeddingError("embedding dimension must be positive")
        if batch_size < 1:
            raise EmbeddingError("batch_size must be greater than zero")
        self.batch_size = int(batch_size)
        self._base_url, self._api_key = _resolve_credentials(base_url, api_key)
        self._timeout = float(timeout)
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._base_url:
            raise EmbeddingError(
                "EMBEDDING_BASE_URL is not set. Point it at an OpenAI-compatible "
                "embeddings endpoint that serves BAAI/bge-m3 (1024 dimensions)."
            )
        if not self._api_key:
            raise EmbeddingError(
                "EMBEDDING_API_KEY is not set. Set it (or OPENROUTER_API_KEY for the "
                "OpenRouter endpoint) to an OpenAI-compatible embeddings provider."
            )
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        return self._client

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Return one 1024-dimensional vector per input text, in input order."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"embedding input #{index} must be a non-empty string")

        size = int(batch_size or self.batch_size)
        if size < 1:
            raise EmbeddingError("batch_size must be greater than zero")

        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(items), size):
            batch = items[start : start + size]
            try:
                response = client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float",
                )
            except Exception as exc:  # noqa: BLE001 - surface any provider failure as one error
                raise EmbeddingError(
                    f"embeddings request to {self._base_url or 'the API'} failed: {exc}"
                ) from exc

            data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            if len(data) != len(batch):
                raise EmbeddingError(
                    f"provider returned {len(data)} vectors for {len(batch)} inputs"
                )
            for item in data:
                vector = [float(value) for value in item.embedding]
                if len(vector) != self.dimension:
                    raise EmbeddingError(
                        f"expected {self.dimension}-dimensional vectors to match the "
                        f"existing Qdrant collections, got {len(vector)}; "
                        f"check EMBEDDING_MODEL"
                    )
                vectors.append(vector)
        return vectors


class LocalEmbeddingBackend:
    """In-process BGE-M3 inference; requires the ``local-embeddings`` extra."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        dimension: int = EMBEDDING_DIMENSION,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = (model_name or _env("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)).strip()
        self.dimension = int(dimension)
        self.device = (device or _env("EMBEDDING_DEVICE", "auto")).lower()
        if self.device not in {"auto", "cpu", "cuda"}:
            raise EmbeddingError("EMBEDDING_DEVICE must be one of: auto, cpu, cuda")
        self.batch_size = int(batch_size)
        self._model: Any | None = None
        self._device: str | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise EmbeddingError(
                "local embeddings need the optional extra: uv sync --extra local-embeddings"
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise EmbeddingError("EMBEDDING_DEVICE=cuda, but PyTorch cannot access a CUDA device")
        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        model_kwargs: dict[str, Any] = {"use_safetensors": False}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self._device = device
        self._model = SentenceTransformer(self.model_name, device=device, model_kwargs=model_kwargs)
        return self._model

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        model = self._load()
        vectors = model.encode(
            items,
            batch_size=int(batch_size or self.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        rows = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
        result = [[float(value) for value in row] for row in rows]
        for row in result:
            if len(row) != self.dimension:
                raise EmbeddingError(
                    f"expected {self.dimension}-dimensional vectors, got {len(row)}"
                )
        return result


def get_embedding_backend(backend: str | None = None) -> EmbeddingBackend:
    """Return the configured backend (``EMBEDDING_BACKEND``, API by default)."""
    name = (backend or _env("EMBEDDING_BACKEND", "api")).lower()
    if name == "api":
        return ApiEmbeddingBackend()
    if name == "local":
        return LocalEmbeddingBackend()
    raise EmbeddingError(f"unknown EMBEDDING_BACKEND {name!r}; expected 'api' or 'local'")
