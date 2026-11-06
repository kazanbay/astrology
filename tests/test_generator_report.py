"""Cache-aside report tests with an in-memory store and fake synthesizer."""

import asyncio

import pytest

from generator.prompts import PROMPT_VERSION
from generator.report import ChartReportService, generate_generation_id
from storage.pipeline import ChartResult

CHART_HASH = "a" * 64


def chart_result(**overrides):
    payload = {
        "user_id": 1,
        "request_id": 7,
        "chart_hash": CHART_HASH,
        "cache_hit": False,
        "cache_written": True,
        "profile_name": "Я",
        "chart": {"planets": {"Sun": {"sign": "Leo", "house": 10}}},
        "chunks": [{"chunk_id": "c1", "section_title": "Солнце во Льве", "content": "Текст"}],
        "input_params": {},
    }
    payload.update(overrides)
    return ChartResult(**payload)


class FakeStore:
    def __init__(self, cached_text=None):
        self.cached_text = cached_text
        self.saved = []
        self.attached = []
        self.requested_ids = []

    async def get_generation(self, generation_id):
        self.requested_ids.append(generation_id)
        return self.cached_text

    async def save_generation(self, record):
        self.saved.append(record)
        return True

    async def attach_generation(self, request_id, generation_id):
        self.attached.append((request_id, generation_id))


class FakeSynthesizer:
    def __init__(self, text="Готовый разбор"):
        self.text = text
        self.calls = []

    async def __call__(self, chart, chunks, model_name, temperature):
        self.calls.append((chart, chunks, model_name, temperature))
        return self.text


def test_generation_id_is_deterministic_and_sensitive():
    base = generate_generation_id(CHART_HASH, "v1.0", "model-a", 0.7)

    assert len(base) == 64
    assert base == generate_generation_id(CHART_HASH, "v1.0", "model-a", 0.7)
    assert base != generate_generation_id("b" * 64, "v1.0", "model-a", 0.7)
    assert base != generate_generation_id(CHART_HASH, "v1.1", "model-a", 0.7)
    assert base != generate_generation_id(CHART_HASH, "v1.0", "model-b", 0.7)
    assert base != generate_generation_id(CHART_HASH, "v1.0", "model-a", 0.9)


def test_generation_id_rounds_temperature():
    assert generate_generation_id(CHART_HASH, "v1.0", "m", 0.7000001) == generate_generation_id(
        CHART_HASH, "v1.0", "m", 0.7
    )


@pytest.mark.parametrize(
    "chart_hash, prompt_version, model_name",
    [("short", "v1", "m"), (CHART_HASH, " ", "m"), (CHART_HASH, "v1", "")],
)
def test_generation_id_rejects_invalid_identity(chart_hash, prompt_version, model_name):
    with pytest.raises(ValueError):
        generate_generation_id(chart_hash, prompt_version, model_name, 0.7)


def test_cache_hit_skips_synthesis_and_links_request():
    store = FakeStore(cached_text="Старый разбор")
    synthesizer = FakeSynthesizer()
    service = ChartReportService(store, synthesizer=synthesizer, model_name="model-a", temperature=0.7)
    result = chart_result()

    report = asyncio.run(service.get_or_generate(result))

    assert report.text == "Старый разбор"
    assert report.cache_hit is True
    assert synthesizer.calls == []
    assert store.saved == []
    assert store.attached == [(7, report.generation_id)]
    assert store.requested_ids == [
        generate_generation_id(CHART_HASH, PROMPT_VERSION, "model-a", 0.7)
    ]


def test_miss_generates_saves_and_links():
    store = FakeStore()
    synthesizer = FakeSynthesizer()
    service = ChartReportService(store, synthesizer=synthesizer, model_name="model-a", temperature=0.7)
    result = chart_result()

    report = asyncio.run(service.get_or_generate(result))

    assert report.text == "Готовый разбор"
    assert report.cache_hit is False
    assert synthesizer.calls == [(result.chart, result.chunks, "model-a", 0.7)]
    assert len(store.saved) == 1
    record = store.saved[0]
    assert record.generation_id == report.generation_id
    assert record.chart_hash == CHART_HASH
    assert record.prompt_version == PROMPT_VERSION
    assert record.model_name == "model-a"
    assert record.temperature == 0.7
    assert record.generated_text == "Готовый разбор"
    assert store.attached == [(7, report.generation_id)]


def test_prompt_version_and_temperature_change_the_key():
    store = FakeStore()
    service = ChartReportService(store, synthesizer=FakeSynthesizer(), model_name="model-a", temperature=0.7)
    other = ChartReportService(
        store,
        synthesizer=FakeSynthesizer(),
        model_name="model-a",
        temperature=0.9,
        prompt_version="v2.0",
    )

    first = asyncio.run(service.get_or_generate(chart_result()))
    second = asyncio.run(other.get_or_generate(chart_result()))

    assert first.generation_id != second.generation_id


def test_invalid_service_settings_are_rejected():
    with pytest.raises(ValueError):
        ChartReportService(FakeStore(), model_name="")
    with pytest.raises(ValueError):
        ChartReportService(FakeStore(), prompt_version=" ")


def test_temperature_is_normalized_for_the_cache_key():
    store = FakeStore()
    service = ChartReportService(store, synthesizer=FakeSynthesizer(), temperature=0.7000001)

    report = asyncio.run(service.get_or_generate(chart_result()))

    assert report.temperature == 0.7
    assert report.generation_id == generate_generation_id(
        CHART_HASH, PROMPT_VERSION, service.model_name, 0.7
    )
