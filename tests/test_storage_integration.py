"""Integration check of the storage layer against a real PostgreSQL.

Skipped automatically when ``DATABASE_URL`` is not configured. The test writes
one user, one cached chart and one history row under a reserved test Telegram
id and removes exactly those rows afterwards.
"""

import asyncio
import hashlib
import uuid

import pytest

from storage.config import resolve_database_url
from storage.store import CachedChart, PostgresChartStore

try:
    DSN: str | None = resolve_database_url()
except RuntimeError:
    DSN = None

pytestmark = pytest.mark.skipif(not DSN, reason="DATABASE_URL is not configured")

TEST_TELEGRAM_ID = 999999999999
TEST_EMAIL = "storage-integration@example.invalid"
TEST_PROFILE = "Интеграционный тест"


async def _scenario() -> None:
    chart_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    user_id: int | None = None
    store = await PostgresChartStore.connect(max_size=2)
    try:
        user_id = await store.upsert_user(telegram_id=TEST_TELEGRAM_ID, first_name="Тест")
        assert await store.find_user(telegram_id=TEST_TELEGRAM_ID) == user_id
        assert await store.upsert_user(telegram_id=TEST_TELEGRAM_ID, first_name="Тест") == user_id

        cached = CachedChart(
            chart_hash=chart_hash,
            input_params={"city": "Москва", "dt_utc": "1990-08-15T08:00:00+00:00"},
            calculated_chart={"planets": {"Sun": {"sign": "Leo", "house": 10}}, "cusps": [0.0] * 12},
            retrieved_chunks=[{"chunk_id": "node-chapter", "content": "Текст", "elements": []}],
        )
        assert await store.save_chart(cached) is True
        assert await store.save_chart(cached) is False  # second writer loses the race

        loaded = await store.get_chart(chart_hash)
        assert loaded is not None
        assert loaded.calculated_chart == cached.calculated_chart
        assert loaded.retrieved_chunks == cached.retrieved_chunks
        assert loaded.input_params["city"] == "Москва"

        request_id = await store.record_request(
            user_id=user_id, chart_hash=chart_hash, profile_name=TEST_PROFILE
        )
        history = await store.list_requests(user_id, limit=5)
        assert history[0].request_id == request_id
        assert history[0].profile_name == TEST_PROFILE
        assert history[0].input_params["city"] == "Москва"

        assert await store.get_chart("0" * 64) is None
    finally:
        if user_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", user_id)
        await store.pool.execute("DELETE FROM charts_cache WHERE chart_hash = $1", chart_hash)
        await store.close()


def test_postgres_store_roundtrip():
    asyncio.run(_scenario())


async def _email_identity_scenario() -> None:
    user_id: int | None = None
    store = await PostgresChartStore.connect(max_size=2)
    try:
        user_id = await store.upsert_user(email=TEST_EMAIL, first_name="Веб")
        # Email is stored lowercased, so any spelling resolves to the same user.
        assert await store.find_user(email=TEST_EMAIL.upper()) == user_id
        assert await store.upsert_user(email=TEST_EMAIL, first_name="Веб") == user_id
    finally:
        if user_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", user_id)
        await store.close()


def test_postgres_store_resolves_user_by_email():
    asyncio.run(_email_identity_scenario())
