"""Unit tests of the cached LLM interpretation of forecasts."""

import asyncio
from datetime import date, datetime, timezone

import pytest

from services.forecast_report import ForecastReportService
from storage.store import ForecastReportRecord

CHART_HASH = "a" * 64


def make_forecast(**overrides):
    payload = {
        "tool": "forecast",
        "chart_hash": CHART_HASH,
        "interval": "week",
        "person": {"kind": "owner", "name": "Анна", "relation": None},
        "birth": {"city": "Москва", "timezone": "Europe/Moscow", "time_unknown": False},
        "range": {"days": 7},
        "natal": {"planets": {"Sun": {"sign": "Leo", "house": 10}}},
        "now_transits": [],
        "events": [
            {"transit": "Saturn", "natal": "Sun", "aspect": "square", "orb": 0.8},
            {"transit": "Moon", "natal": "Pluto", "aspect": "square", "orb": 0.3},
        ],
        "event_count": 2,
    }
    payload.update(overrides)
    return payload


class FakeStore:
    def __init__(self, cached=None):
        self.cached = cached
        self.saved: list[ForecastReportRecord] = []

    async def get_forecast_report(self, report_id):
        return self.cached

    async def save_forecast_report(self, record):
        self.saved.append(record)
        self.cached = record.generated_text
        return True


class FakeRetriever:
    def __init__(self, chunks=None):
        self.calls = []
        self.chunks = (
            chunks
            if chunks is not None
            else [{"chunk_id": "c1", "section_title": "Транзиты", "content": "текст"}]
        )

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks


class FakeSynth:
    def __init__(self, text="Интерпретация периода"):
        self.text = text
        self.calls = []

    async def __call__(self, forecast, chunks, model_name, temperature):
        self.calls.append((forecast, chunks, model_name, temperature))
        return self.text


def build_service(store, synth=None):
    return ForecastReportService(
        store, synthesizer=synth or FakeSynth(), model_name="model-a", temperature=0.7
    )


def test_cache_hit_skips_rag_and_llm():
    store = FakeStore(cached="Старый прогноз")
    retriever = FakeRetriever()
    synth = FakeSynth()
    service = build_service(store, synth)

    result = asyncio.run(service.get_or_generate(make_forecast(), retriever=retriever))

    assert result["text"] == "Старый прогноз"
    assert result["cache_hit"] is True
    assert retriever.calls == []
    assert synth.calls == []
    assert store.saved == []


def test_miss_retrieves_chunks_and_saves_the_report():
    store = FakeStore()
    retriever = FakeRetriever()
    synth = FakeSynth()
    service = build_service(store, synth)

    result = asyncio.run(service.get_or_generate(make_forecast(), retriever=retriever))

    assert result["cache_hit"] is False
    assert result["chunks"] == 1
    query, top_k = retriever.calls[0]
    assert query == "Транзит Сатурн квадрат Солнце"
    assert top_k == service.top_k
    assert retriever.calls[-1][0].startswith("Классическая астрология Вронского")
    assert synth.calls[0][1][0]["chunk_id"] == "c1"

    record = store.saved[0]
    assert len(record.report_id) == 64
    assert record.chart_hash == CHART_HASH
    assert record.interval == "week"
    assert record.prompt_version == service.prompt_version
    assert record.model_name == "model-a"
    assert record.generated_text == "Интерпретация периода"


def test_second_call_hits_the_cache():
    store = FakeStore()
    retriever = FakeRetriever()
    synth = FakeSynth()
    service = build_service(store, synth)

    asyncio.run(service.get_or_generate(make_forecast(), retriever=retriever))
    first_calls = len(retriever.calls)
    second = asyncio.run(service.get_or_generate(make_forecast(), retriever=retriever))

    assert second["cache_hit"] is True
    assert len(synth.calls) == 1
    assert len(retriever.calls) == first_calls


def test_without_retriever_the_llm_still_gets_the_forecast():
    store = FakeStore()
    synth = FakeSynth()
    service = build_service(store, synth)

    result = asyncio.run(service.get_or_generate(make_forecast()))

    assert result["chunks"] == 0
    assert synth.calls[0][1] == []


def test_forecast_date_uses_the_birth_timezone():
    result = make_forecast(birth={"timezone": "Europe/Moscow", "time_unknown": False})
    moment = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)

    assert ForecastReportService.forecast_date(result, now=moment) == date(2026, 1, 2)

    utc_result = make_forecast(birth={"timezone": None, "time_unknown": False})
    assert ForecastReportService.forecast_date(utc_result, now=moment) == date(2026, 1, 1)


def test_report_id_changes_with_interval_and_date():
    service = build_service(FakeStore())

    base = service.report_id(CHART_HASH, "week", date(2026, 1, 1))
    assert len(base) == 64
    assert base != service.report_id(CHART_HASH, "day", date(2026, 1, 1))
    assert base != service.report_id(CHART_HASH, "week", date(2026, 1, 2))
    assert base != service.report_id("b" * 64, "week", date(2026, 1, 1))


def test_missing_chart_hash_is_rejected():
    service = build_service(FakeStore())

    with pytest.raises(ValueError):
        asyncio.run(service.get_or_generate(make_forecast(chart_hash="")))


def test_synthesis_error_propagates():
    class BrokenSynth:
        async def __call__(self, *args, **kwargs):
            raise RuntimeError("LLM is down")

    service = build_service(FakeStore(), BrokenSynth())

    with pytest.raises(RuntimeError, match="LLM is down"):
        asyncio.run(service.get_or_generate(make_forecast()))


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        ForecastReportService(FakeStore(), model_name="")
    with pytest.raises(ValueError):
        ForecastReportService(FakeStore(), prompt_version=" ")
    with pytest.raises(ValueError):
        ForecastReportService(FakeStore(), synthesizer=FakeSynth(), top_k=0)
