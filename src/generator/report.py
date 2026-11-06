"""Cache-aside generation of chart reports.

A report is identified by ``generation_id`` - SHA-256 over the chart hash, the
prompt version, the model and the temperature. The same chart requested twice
with the same settings is generated once; changing the prompt or the model
produces a new id and therefore a new report, without invalidating the old one.

Every request row keeps a link to the report it received
(``user_chart_requests.generation_id``), so history stays reproducible even
after the prompt evolves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

try:  # support both `src.generator` and top-level `generator` imports
    from ..storage.pipeline import ChartResult
    from ..storage.store import GenerationRecord
except ImportError:  # pragma: no cover - depends on sys.path layout
    from storage.pipeline import ChartResult
    from storage.store import GenerationRecord

from .prompts import PROMPT_VERSION
from .synthesizer import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    normalize_temperature,
    synthesize_chart_report,
)

GENERATION_ID_LENGTH = 64

#: ``(chart, chunks, model_name, temperature) -> report text``
Synthesizer = Callable[[dict[str, Any], list[dict[str, Any]], str, float], Awaitable[str]]


class GenerationStore(Protocol):
    """Persistence the report cache needs (implemented by ``PostgresChartStore``)."""

    async def get_generation(self, generation_id: str) -> str | None: ...

    async def save_generation(self, record: GenerationRecord) -> bool: ...

    async def attach_generation(self, request_id: int, generation_id: str) -> None: ...


@dataclass(frozen=True)
class ChartReport:
    """One report together with the identity that produced it."""

    generation_id: str
    text: str
    model_name: str
    prompt_version: str
    temperature: float
    cache_hit: bool


def generate_generation_id(
    chart_hash: str,
    prompt_version: str,
    model_name: str,
    temperature: float,
) -> str:
    """Return the deterministic cache key of one report."""
    if not isinstance(chart_hash, str) or len(chart_hash) != 64:
        raise ValueError("chart_hash must be a 64-character SHA-256 hex string")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompt_version must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    payload = {
        "chart_hash": chart_hash,
        "prompt_version": prompt_version.strip(),
        "model_name": model_name.strip(),
        "temperature": normalize_temperature(temperature),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ChartReportService:
    """Serve reports from ``llm_generations``, generating only on a miss."""

    def __init__(
        self,
        store: GenerationStore,
        *,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        prompt_version: str = PROMPT_VERSION,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not prompt_version or not prompt_version.strip():
            raise ValueError("prompt_version must be a non-empty string")
        self.store = store
        self.model_name = model_name.strip()
        self.temperature = normalize_temperature(temperature)
        self.prompt_version = prompt_version.strip()
        self._synthesize: Synthesizer = synthesizer or synthesize_chart_report

    def generation_id(self, chart_hash: str) -> str:
        """Return the report id for this chart under the service settings."""
        return generate_generation_id(
            chart_hash,
            self.prompt_version,
            self.model_name,
            self.temperature,
        )

    async def get_or_generate(self, result: ChartResult) -> ChartReport:
        """Return the cached report or generate, store and link a new one."""
        generation_id = self.generation_id(result.chart_hash)
        text = await self.store.get_generation(generation_id)
        cache_hit = text is not None

        if text is None:
            text = await self._synthesize(
                result.chart,
                result.chunks,
                self.model_name,
                self.temperature,
            )
            await self.store.save_generation(
                GenerationRecord(
                    generation_id=generation_id,
                    chart_hash=result.chart_hash,
                    prompt_version=self.prompt_version,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    generated_text=text,
                )
            )

        await self.store.attach_generation(result.request_id, generation_id)
        return ChartReport(
            generation_id=generation_id,
            text=text,
            model_name=self.model_name,
            prompt_version=self.prompt_version,
            temperature=self.temperature,
            cache_hit=cache_hit,
        )
