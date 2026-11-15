"""Cache-aside LLM interpretation of a computed forecast (RAG + LLM).

``ForecastService`` produces deterministic transit events; this service turns
them into a book-grounded interpretation: a knowledge-base query is built from
the strongest transits, the retrieved chunks and the event table go to the LLM,
and the text is cached per (chart, interval, local day, prompt version, model).
A repeated request on the same day therefore costs no LLM call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from generator.forecast import FORECAST_PROMPT_VERSION, synthesize_forecast
from generator.synthesizer import DEFAULT_MODEL, DEFAULT_TEMPERATURE, normalize_temperature
from storage.store import ChartStore, ForecastReportRecord

from .presentation import forecast_query

logger = logging.getLogger(__name__)

#: Knowledge-base chunks requested per forecast interpretation.
SEARCH_TOP_K = 6

#: ``(forecast, chunks, model, temperature) -> text``
ForecastSynthesizer = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], str, float], Awaitable[str]
]


class ForecastReportService:
    """Generate and cache the LLM interpretation of a transit forecast."""

    def __init__(
        self,
        store: ChartStore,
        *,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        prompt_version: str = FORECAST_PROMPT_VERSION,
        synthesizer: ForecastSynthesizer | None = None,
        top_k: int = SEARCH_TOP_K,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not prompt_version or not prompt_version.strip():
            raise ValueError("prompt_version must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        self.store = store
        self.model_name = model_name.strip()
        self.temperature = normalize_temperature(temperature)
        self.prompt_version = prompt_version.strip()
        self.top_k = top_k
        self._synthesize: ForecastSynthesizer = synthesizer or synthesize_forecast

    def report_id(self, chart_hash: str, interval: str, forecast_date: date) -> str:
        """Deterministic cache key of one interpretation."""
        if not isinstance(chart_hash, str) or len(chart_hash) != 64:
            raise ValueError("chart_hash must be a 64-character SHA-256 hex string")
        payload = {
            "chart_hash": chart_hash,
            "interval": str(interval).strip(),
            "forecast_date": forecast_date.isoformat(),
            "prompt_version": self.prompt_version,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def forecast_date(result: Mapping[str, Any], *, now: datetime | None = None) -> date:
        """Local date of the forecast start (day-granularity cache key)."""
        moment = now or datetime.now(timezone.utc)
        tz_name = (result.get("birth") or {}).get("timezone")
        target: Any = timezone.utc
        if tz_name:
            try:
                target = ZoneInfo(str(tz_name))
            except Exception:  # pragma: no cover - zones come from timezonefinder
                target = timezone.utc
        return moment.astimezone(target).date()

    async def get_or_generate(
        self,
        result: Mapping[str, Any],
        *,
        retriever: Any | None = None,
    ) -> dict[str, Any]:
        """Return the cached interpretation or build it with RAG + the LLM."""
        chart_hash = str(result.get("chart_hash") or "")
        if len(chart_hash) != 64:
            raise ValueError("forecast result has no chart_hash; ForecastService must set it")
        interval = str(result.get("interval") or "")
        day = self.forecast_date(result)
        report_id = self.report_id(chart_hash, interval, day)

        text = await self.store.get_forecast_report(report_id)
        if text is not None:
            return {
                "text": text,
                "cache_hit": True,
                "report_id": report_id,
                "forecast_date": day.isoformat(),
                "chunks": 0,
            }

        chunks: list[dict[str, Any]] = []
        if retriever is not None:
            query = forecast_query(result)
            try:
                chunks = await asyncio.to_thread(retriever.search, query, self.top_k)
            except Exception:
                # A missing collection or a network hiccup must not cost the
                # user the whole interpretation: the prompt is instructed to
                # stay general when no fragment is available.
                logger.exception("forecast retrieval failed; interpreting without chunks")

        text = await self._synthesize(
            dict(result), list(chunks), self.model_name, self.temperature
        )
        try:
            await self.store.save_forecast_report(
                ForecastReportRecord(
                    report_id=report_id,
                    chart_hash=chart_hash,
                    interval=interval,
                    forecast_date=day,
                    prompt_version=self.prompt_version,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    generated_text=text,
                )
            )
        except Exception:
            # The generated text is the expensive part: a cache write failure
            # must not hide it from the user.
            logger.exception("could not cache the forecast interpretation")
        return {
            "text": text,
            "cache_hit": False,
            "report_id": report_id,
            "forecast_date": day.isoformat(),
            "chunks": len(chunks),
        }
