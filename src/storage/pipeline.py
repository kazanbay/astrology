"""Cache-aside pipeline: user -> chart_hash -> charts_cache -> history.

The flow implements the TZ algorithm:

1. resolve (or create) the internal ``user_id`` by Telegram id / email;
2. compute ``chart_hash`` from the canonical birth inputs;
3. look the hash up in ``charts_cache``;
4. on a miss, run the astronomy calculator and the structural RAG lookup, then
   store the result (one row shared by every user with the same chart);
5. append the request to ``user_chart_requests``;
6. return the raw chart and chunks to the caller (bot, script, future API).

Heavy synchronous work (pyswisseph, the Qdrant round trip) goes through
``asyncio.to_thread`` so the bot stays responsive. Both dependencies are
injected, which keeps the pipeline testable without PostgreSQL, Qdrant or an
ephemeris on disk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .birth import BirthData
from .chunks import collect_unique_chunks, element_chunk_index
from .hashing import generate_chart_hash
from .store import CachedChart, ChartStore, RequestRecord, normalize_profile_name

DEFAULT_PROFILE_NAME = "Я"

#: ``BirthData`` -> ``NatalChart.to_dict()`` compatible mapping.
ChartBuilder = Callable[[BirthData], dict[str, Any]]

#: (chart mapping, birth data) -> deduplicated chunk list.
ChunkRetriever = Callable[[Mapping[str, Any], BirthData], list[dict[str, Any]]]


@dataclass(frozen=True)
class ChartResult:
    """What one cache-aside request produced."""

    user_id: int
    request_id: int
    chart_hash: str
    cache_hit: bool
    cache_written: bool
    profile_name: str
    chart: dict[str, Any]
    chunks: list[dict[str, Any]]
    input_params: dict[str, Any]

    def element_chunks(self) -> dict[str, dict[str, Any]]:
        """Map every chart element to the section that answered it."""
        return element_chunk_index(self.chunks)


def calculate_chart(birth: BirthData) -> dict[str, Any]:
    """Default chart builder: pyswisseph through the project calculator."""
    from astronomy.calculator import calculate_natal_chart

    return calculate_natal_chart(
        birth.dt_utc,
        birth.latitude,
        birth.longitude,
        birth.house_system,
    ).to_dict()


class ChartPipeline:
    """Cache-aside orchestration over a ``ChartStore``."""

    def __init__(
        self,
        store: ChartStore,
        *,
        chart_builder: ChartBuilder | None = None,
        chunk_retriever: ChunkRetriever | None = None,
        collection_name: str | None = None,
        per_element: int = 1,
    ) -> None:
        if per_element < 1:
            raise ValueError("per_element must be greater than zero")
        self.store = store
        self.per_element = per_element
        self.collection_name = collection_name
        self._chart_builder = chart_builder or calculate_chart
        self._chunk_retriever = chunk_retriever or self._structural_retriever()

    def _structural_retriever(self) -> ChunkRetriever:
        """Bind the structural router with the pipeline's collection settings."""

        def retrieve(chart: Mapping[str, Any], birth: BirthData) -> list[dict[str, Any]]:
            from rag.structural_router import get_structural_interpretations

            structural = get_structural_interpretations(
                chart,
                {"language": birth.language},
                per_element=self.per_element,
                collection_name=self.collection_name,
            )
            return collect_unique_chunks(structural)

        return retrieve

    async def load(
        self,
        birth: BirthData,
        *,
        telegram_id: int | str | None = None,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> ChartResult:
        """Return the chart for ``birth``, computing it only on a cache miss."""
        if not isinstance(birth, BirthData):
            raise ValueError("birth must be a BirthData instance")
        profile = normalize_profile_name(profile_name)

        user_id = await self.store.upsert_user(
            telegram_id=telegram_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
        )
        chart_hash = generate_chart_hash(
            birth.dt_utc,
            birth.latitude,
            birth.longitude,
            birth.house_system,
            birth.language,
        )

        cached = await self.store.get_chart(chart_hash)
        cache_hit = cached is not None
        cache_written = False
        if cached is None:
            chart = await asyncio.to_thread(self._chart_builder, birth)
            chunks = await asyncio.to_thread(self._chunk_retriever, dict(chart), birth)
            cached = CachedChart(
                chart_hash=chart_hash,
                input_params=birth.input_params(),
                calculated_chart=dict(chart),
                retrieved_chunks=list(chunks),
            )
            cache_written = await self.store.save_chart(cached)

        request_id = await self.store.record_request(
            user_id=user_id,
            chart_hash=chart_hash,
            profile_name=profile,
        )
        return ChartResult(
            user_id=user_id,
            request_id=request_id,
            chart_hash=chart_hash,
            cache_hit=cache_hit,
            cache_written=cache_written,
            profile_name=profile,
            chart=cached.calculated_chart,
            chunks=cached.retrieved_chunks,
            input_params=cached.input_params,
        )

    async def history(self, user_id: int, *, limit: int = 10) -> list[RequestRecord]:
        """Return the latest requests of one user (thin store passthrough)."""
        return await self.store.list_requests(user_id, limit=limit)

    async def resolve_user(
        self, *, telegram_id: int | str | None = None, email: str | None = None
    ) -> int | None:
        """Return the internal id of an existing user, or ``None``."""
        return await self.store.find_user(telegram_id=telegram_id, email=email)
