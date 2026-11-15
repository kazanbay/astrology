"""Cache-aside LLM reading of a synastry pair (RAG + LLM).

``SynastryService`` computes the cross-aspect grid; this service turns it into
a book-grounded psychological portrait: narrow knowledge-base queries (aspects,
luminaries, Venus/Mars, the 7th house), the pair snapshot and the retrieved
fragments go to the LLM, and the text is cached per (pair of charts, prompt
version, model) in ``synastry_reports``. Compatibility is time-independent, so
the cache key carries no date.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from generator.synastry import SYNASTRY_PROMPT_VERSION, synthesize_synastry
from generator.synthesizer import DEFAULT_MODEL, DEFAULT_TEMPERATURE, normalize_temperature
from rag.router import get_relevant_chunks
from storage.store import ChartStore, SynastryReportRecord

from .presentation import synastry_queries

logger = logging.getLogger(__name__)

#: Chunks fetched per single aspect query (the router deduplicates).
SEARCH_TOP_K = 2
#: Upper bound of unique chunks sent to the model.
MAX_CHUNKS = 12

#: ``(synastry, chunks, model, temperature) -> text``
SynastrySynthesizer = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], str, float], Awaitable[str]
]


class SynastryReportService:
    """Generate and cache the LLM reading of a synastry pair."""

    def __init__(
        self,
        store: ChartStore,
        *,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        prompt_version: str = SYNASTRY_PROMPT_VERSION,
        synthesizer: SynastrySynthesizer | None = None,
        top_k: int = SEARCH_TOP_K,
        max_chunks: int = MAX_CHUNKS,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not prompt_version or not prompt_version.strip():
            raise ValueError("prompt_version must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if max_chunks < 1:
            raise ValueError("max_chunks must be greater than zero")
        self.store = store
        self.model_name = model_name.strip()
        self.temperature = normalize_temperature(temperature)
        self.prompt_version = prompt_version.strip()
        self.top_k = top_k
        self.max_chunks = max_chunks
        self._synthesize: SynastrySynthesizer = synthesizer or synthesize_synastry

    def report_id(self, chart_hash_a: str, chart_hash_b: str) -> str:
        """Deterministic cache key of one pair reading."""
        for value, label in ((chart_hash_a, "chart_hash_a"), (chart_hash_b, "chart_hash_b")):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{label} must be a 64-character SHA-256 hex string")
        payload = {
            "chart_hash_a": chart_hash_a,
            "chart_hash_b": chart_hash_b,
            "prompt_version": self.prompt_version,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get_or_generate(
        self,
        result: Mapping[str, Any],
        *,
        retriever: Any | None = None,
    ) -> dict[str, Any]:
        """Return the cached reading or build it with RAG + the LLM."""
        chart_hash_a = str(result.get("chart_hash_a") or "")
        chart_hash_b = str(result.get("chart_hash_b") or "")
        report_id = self.report_id(chart_hash_a, chart_hash_b)

        text = await self.store.get_synastry_report(report_id)
        if text is not None:
            return {"text": text, "cache_hit": True, "report_id": report_id, "chunks": 0}

        chunks: list[dict[str, Any]] = []
        if retriever is not None:
            queries = synastry_queries(result)
            try:
                chunks = await asyncio.to_thread(
                    get_relevant_chunks,
                    queries,
                    retriever=retriever,
                    top_k=self.top_k,
                    max_chunks=self.max_chunks,
                )
            except Exception:
                # The reading is still worth showing without fragments: the
                # prompt is instructed to stay general instead of inventing.
                logger.exception("synastry retrieval failed; interpreting without chunks")

        text = await self._synthesize(
            dict(result), list(chunks), self.model_name, self.temperature
        )
        try:
            await self.store.save_synastry_report(
                SynastryReportRecord(
                    report_id=report_id,
                    chart_hash_a=chart_hash_a,
                    chart_hash_b=chart_hash_b,
                    prompt_version=self.prompt_version,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    generated_text=text,
                )
            )
        except Exception:
            # The generated text is the expensive part: a cache write failure
            # must not hide it from the user.
            logger.exception("could not cache the synastry reading")
        return {
            "text": text,
            "cache_hit": False,
            "report_id": report_id,
            "chunks": len(chunks),
        }
