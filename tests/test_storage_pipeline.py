import asyncio
from datetime import datetime, timezone

import pytest

from storage.birth import BirthData
from storage.chunks import collect_unique_chunks
from storage.pipeline import ChartPipeline
from storage.store import RequestRecord

BIRTH = BirthData(
    dt_utc=datetime(1990, 8, 15, 8, 0, tzinfo=timezone.utc),
    latitude=55.7558,
    longitude=37.6173,
    city="Москва, Россия",
    timezone="Europe/Moscow",
    datetime_local="1990-08-15T12:00:00",
)

STRUCTURAL = {
    "NorthNode_in_Aquarius": {
        "kind": "planet_in_sign",
        "element": "NorthNode in Aquarius",
        "coverage": "single_element",
        "total_found": 1,
        "candidates": [
            {
                "chunk_id": "node-chapter",
                "title": "Классическая астрология",
                "section_title": "4.1.11. Восходящий Лунный Узел в знаке Водолея",
                "section_path": ["Том 4", "4.1.11"],
                "book_id": "book-4",
                "chunk_index": 11,
                "content": "Текст главы про узлы.",
                "match_tier": 2,
                "rank_total": 2,
                "is_transit": False,
            }
        ],
    },
    "SouthNode_in_Leo": {
        "kind": "planet_in_sign",
        "element": "SouthNode in Leo",
        "coverage": "single_element",
        "total_found": 1,
        "candidates": [
            {
                "chunk_id": "node-chapter",
                "title": "Классическая астрология",
                "section_title": "4.1.11. Восходящий Лунный Узел в знаке Водолея",
                "section_path": ["Том 4", "4.1.11"],
                "book_id": "book-4",
                "chunk_index": 11,
                "content": "Текст главы про узлы.",
                "match_tier": 2,
                "rank_total": 2,
                "is_transit": False,
            }
        ],
    },
}


class FakeStore:
    """In-memory ``ChartStore`` with call counters."""

    def __init__(self):
        self.users: dict[tuple[str, object], int] = {}
        self.display_names: dict[tuple[str, object], object] = {}
        self.charts = {}
        self.requests: list[RequestRecord] = []
        self.calls = {"upsert_user": 0, "get_chart": 0, "save_chart": 0, "record_request": 0}

    async def upsert_user(
        self, *, telegram_id=None, email=None, first_name=None, last_name=None, display_name=None
    ):
        if telegram_id is None and email is None:
            raise ValueError("a user needs at least one of telegram_id or email")
        self.calls["upsert_user"] += 1
        key = ("telegram", telegram_id) if telegram_id is not None else ("email", email)
        if key not in self.users:
            self.users[key] = len(self.users) + 1
        self.display_names[key] = display_name
        return self.users[key]

    async def find_user(self, *, telegram_id=None, email=None):
        key = ("telegram", telegram_id) if telegram_id is not None else ("email", email)
        return self.users.get(key)

    async def get_chart(self, chart_hash):
        self.calls["get_chart"] += 1
        return self.charts.get(chart_hash)

    async def save_chart(self, cached):
        self.calls["save_chart"] += 1
        if cached.chart_hash in self.charts:
            return False
        self.charts[cached.chart_hash] = cached
        return True

    async def record_request(self, *, user_id, chart_hash, profile_name):
        self.calls["record_request"] += 1
        record = RequestRecord(
            request_id=len(self.requests) + 1,
            chart_hash=chart_hash,
            profile_name=profile_name,
            requested_at=datetime(2026, 11, 6, 12, 0),
            input_params=dict(self.charts[chart_hash].input_params),
        )
        self.requests.append(record)
        return record.request_id

    async def list_requests(self, user_id, *, limit=10):
        return list(reversed(self.requests))[:limit]


class FakeBuilder:
    def __init__(self):
        self.calls = 0

    def __call__(self, birth):
        self.calls += 1
        return {
            "dt_utc": birth.dt_utc.isoformat(),
            "latitude": birth.latitude,
            "longitude": birth.longitude,
            "house_system": birth.house_system,
            "cusps": [0.0] * 12,
            "asc": {"longitude": 200.68, "sign": "Libra", "house": 1},
            "mc": {"longitude": 118.94, "sign": "Cancer", "house": 10},
            "planets": {
                "Sun": {
                    "longitude": 142.5,
                    "sign": "Leo",
                    "house": 10,
                    "speed": 0.96,
                    "is_retrograde": False,
                    "is_stationary": False,
                }
            },
            "aspects": [],
        }


class FakeRetriever:
    def __init__(self, structural=None):
        self.calls = 0
        self.structural = STRUCTURAL if structural is None else structural

    def __call__(self, chart, birth):
        self.calls += 1
        return collect_unique_chunks(self.structural)


def build_pipeline(**kwargs):
    store = FakeStore()
    builder = FakeBuilder()
    retriever = FakeRetriever()
    pipeline = ChartPipeline(
        store,
        chart_builder=builder,
        chunk_retriever=retriever,
        **kwargs,
    )
    return pipeline, store, builder, retriever


def test_miss_computes_chart_and_stores_it():
    pipeline, store, builder, retriever = build_pipeline()

    result = asyncio.run(pipeline.load(BIRTH, telegram_id=42, first_name="Анна"))

    assert result.cache_hit is False
    assert result.cache_written is True
    assert builder.calls == 1
    assert retriever.calls == 1
    assert store.calls["save_chart"] == 1
    assert store.calls["record_request"] == 1
    assert len(result.chart_hash) == 64
    assert result.user_id == 1
    assert result.request_id == 1
    assert result.profile_name == "Я"
    assert result.input_params["city"] == "Москва, Россия"
    assert [chunk["chunk_id"] for chunk in result.chunks] == ["node-chapter"]


def test_repeat_request_hits_cache_and_does_not_recompute():
    pipeline, store, builder, retriever = build_pipeline()

    first = asyncio.run(pipeline.load(BIRTH, telegram_id=42))
    second = asyncio.run(pipeline.load(BIRTH, telegram_id=42))

    assert second.cache_hit is True
    assert second.cache_written is False
    assert second.chart_hash == first.chart_hash
    assert builder.calls == 1
    assert retriever.calls == 1
    assert store.calls["get_chart"] == 2
    assert store.calls["upsert_user"] == 2
    assert len(store.requests) == 2
    assert [record.request_id for record in store.requests] == [1, 2]


def test_twins_share_one_chart_row():
    pipeline, store, builder, _retriever = build_pipeline()

    first = asyncio.run(pipeline.load(BIRTH, telegram_id=1))
    second = asyncio.run(pipeline.load(BIRTH, telegram_id=2))

    assert first.user_id != second.user_id
    assert first.chart_hash == second.chart_hash
    assert len(store.charts) == 1
    assert builder.calls == 1


def test_profiles_of_one_user_are_recorded_separately():
    pipeline, store, _builder, _retriever = build_pipeline()

    asyncio.run(pipeline.load(BIRTH, telegram_id=42, profile_name="Я"))
    result = asyncio.run(pipeline.load(BIRTH, telegram_id=42, profile_name="Мама"))

    assert result.cache_hit is True
    assert [record.profile_name for record in store.requests] == ["Я", "Мама"]
    assert len({record.chart_hash for record in store.requests}) == 1


def test_language_is_part_of_the_cache_key():
    pipeline, store, builder, _retriever = build_pipeline()

    russian = asyncio.run(pipeline.load(BIRTH, telegram_id=42))
    english_birth = BirthData(
        dt_utc=BIRTH.dt_utc,
        latitude=BIRTH.latitude,
        longitude=BIRTH.longitude,
        language="en",
    )
    english = asyncio.run(pipeline.load(english_birth, telegram_id=42))

    assert english.chart_hash != russian.chart_hash
    assert english.cache_hit is False
    assert builder.calls == 2
    assert len(store.charts) == 2


def test_default_house_system_and_language_flow_into_hash():
    pipeline, _store, _builder, _retriever = build_pipeline()

    result = asyncio.run(pipeline.load(BIRTH, telegram_id=42))

    assert result.input_params["house_system"] == "P"
    assert result.input_params["language"] == "ru"


def test_display_name_is_passed_to_the_store():
    pipeline, store, _builder, _retriever = build_pipeline()

    asyncio.run(pipeline.load(BIRTH, telegram_id=42, display_name="Анна"))

    assert store.display_names[("telegram", 42)] == "Анна"


def test_invalid_profile_name_fails_before_touching_the_database():
    pipeline, store, builder, _retriever = build_pipeline()

    with pytest.raises(ValueError):
        asyncio.run(pipeline.load(BIRTH, telegram_id=42, profile_name="   "))

    assert store.calls["upsert_user"] == 0
    assert builder.calls == 0


def test_request_without_identity_is_rejected():
    pipeline, store, builder, _retriever = build_pipeline()

    with pytest.raises(ValueError):
        asyncio.run(pipeline.load(BIRTH))

    assert builder.calls == 0
    assert store.calls["get_chart"] == 0


def test_history_and_resolve_user_delegate_to_store():
    pipeline, store, _builder, _retriever = build_pipeline()

    asyncio.run(pipeline.load(BIRTH, telegram_id=42))
    user_id = asyncio.run(pipeline.resolve_user(telegram_id=42))
    history = asyncio.run(pipeline.history(user_id, limit=5))

    assert user_id == 1
    assert len(history) == 1
    assert asyncio.run(pipeline.resolve_user(telegram_id=999)) is None


def test_per_element_must_be_positive():
    with pytest.raises(ValueError):
        ChartPipeline(FakeStore(), per_element=0)
