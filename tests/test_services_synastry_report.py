"""Unit tests of the cached LLM reading of a synastry pair."""

import asyncio

import pytest

from services.synastry_report import SynastryReportService
from storage.store import SynastryReportRecord

HASH_A = "a" * 64
HASH_B = "b" * 64


def make_synastry(**overrides):
    payload = {
        "tool": "synastry",
        "chart_hash_a": HASH_A,
        "chart_hash_b": HASH_B,
        "person_a": {"kind": "owner", "positions": {"Sun": {"sign": "Leo"}}},
        "person_b": {"kind": "contact", "positions": {"Moon": {"sign": "Libra"}}},
        "events": [
            {"body_a": "Sun", "body_b": "Moon", "aspect": "sextile", "orb": 0.4},
            {"body_a": "Venus", "body_b": "Mars", "aspect": "square", "orb": 1.2},
        ],
        "event_count": 2,
    }
    payload.update(overrides)
    return payload


class FakeStore:
    def __init__(self, cached=None):
        self.cached = cached
        self.saved: list[SynastryReportRecord] = []

    async def get_synastry_report(self, report_id):
        return self.cached

    async def save_synastry_report(self, record):
        self.saved.append(record)
        self.cached = record.generated_text
        return True


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return [
            {
                "chunk_id": f"chunk-{len(self.calls)}",
                "section_title": "Синастрия",
                "content": "текст",
            }
        ]


class FakeSynth:
    def __init__(self, text="Портрет пары"):
        self.text = text
        self.calls = []

    async def __call__(self, result, chunks, model_name, temperature):
        self.calls.append((result, chunks, model_name, temperature))
        return self.text


def build_service(store, synth=None):
    return SynastryReportService(
        store, synthesizer=synth or FakeSynth(), model_name="model-a", temperature=0.7
    )


def test_cache_hit_skips_rag_and_llm():
    store = FakeStore(cached="Старый разбор")
    retriever = FakeRetriever()
    synth = FakeSynth()
    service = build_service(store, synth)

    result = asyncio.run(service.get_or_generate(make_synastry(), retriever=retriever))

    assert result["text"] == "Старый разбор"
    assert result["cache_hit"] is True
    assert retriever.calls == [] and synth.calls == []
    assert store.saved == []


def test_miss_queries_books_and_caches_the_reading():
    store = FakeStore()
    retriever = FakeRetriever()
    synth = FakeSynth()
    service = build_service(store, synth)

    result = asyncio.run(service.get_or_generate(make_synastry(), retriever=retriever))

    assert result["cache_hit"] is False
    assert result["chunks"] == len(retriever.calls)
    queries = " ".join(query for query, _top_k in retriever.calls)
    assert "Синастрия: Солнце секстиль Луна" in queries
    assert "Солнца и Луны" in queries
    assert "Венера и Марс" in queries
    assert "Седьмой дом" in queries
    assert retriever.calls[0][1] == service.top_k

    record = store.saved[0]
    assert record.chart_hash_a == HASH_A and record.chart_hash_b == HASH_B
    assert record.prompt_version == service.prompt_version
    assert record.generated_text == "Портрет пары"


def test_second_call_hits_the_cache():
    store = FakeStore()
    retriever = FakeRetriever()
    synth = FakeSynth()
    service = build_service(store, synth)

    asyncio.run(service.get_or_generate(make_synastry(), retriever=retriever))
    calls_after_first = len(retriever.calls)
    second = asyncio.run(service.get_or_generate(make_synastry(), retriever=retriever))

    assert second["cache_hit"] is True
    assert len(synth.calls) == 1
    assert len(retriever.calls) == calls_after_first


def test_report_id_is_pair_specific():
    service = build_service(FakeStore())

    base = service.report_id(HASH_A, HASH_B)
    assert len(base) == 64
    assert base != service.report_id(HASH_B, HASH_A)
    assert base != service.report_id(HASH_A, "c" * 64)


def test_missing_hashes_are_rejected():
    service = build_service(FakeStore())

    with pytest.raises(ValueError):
        asyncio.run(service.get_or_generate(make_synastry(chart_hash_a="")))


def test_synthesis_error_propagates():
    class BrokenSynth:
        async def __call__(self, *args, **kwargs):
            raise RuntimeError("LLM is down")

    service = build_service(FakeStore(), BrokenSynth())

    with pytest.raises(RuntimeError, match="LLM is down"):
        asyncio.run(service.get_or_generate(make_synastry()))


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        SynastryReportService(FakeStore(), model_name="")
    with pytest.raises(ValueError):
        SynastryReportService(FakeStore(), prompt_version=" ")
    with pytest.raises(ValueError):
        SynastryReportService(FakeStore(), synthesizer=FakeSynth(), top_k=0)
    with pytest.raises(ValueError):
        SynastryReportService(FakeStore(), synthesizer=FakeSynth(), max_chunks=0)
