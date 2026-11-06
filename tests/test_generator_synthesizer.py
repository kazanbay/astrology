"""Synthesis tests with a fake OpenAI-compatible client (no network)."""

import asyncio
from types import SimpleNamespace

import pytest

from generator.prompts import ASTROLOGER_SYSTEM_PROMPT
from generator.synthesizer import (
    SynthesisError,
    build_user_content,
    synthesize_chart_report,
)

CHART = {
    "dt_utc": "1990-08-15T08:00:00+00:00",
    "planets": {
        "Sun": {"longitude": 142.2, "sign": "Leo", "house": 10},
        "Moon": {"longitude": 74.4, "sign": "Gemini", "house": 8},
    },
    "asc": {"longitude": 200.68, "sign": "Libra", "house": 1},
    "aspects": [{"body_a": "Sun", "body_b": "Moon", "aspect": "sextile", "orb": 3.1}],
}

CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "section_title": "2.2.5. Солнце в знаке Льва",
        "content": "Солнце во Льве находится на пике своего величия.",
        "elements": [{"key": "Sun_in_Leo", "rank": 0}],
    },
    {
        "chunk_id": "chunk-2",
        "section_title": "3.3.3. Луна в знаке Близнецов",
        "content": "Луна в Близнецах делает эмоции подвижными.",
        "elements": [{"key": "Moon_in_Gemini", "rank": 0}],
    },
]


class FakeCompletions:
    def __init__(self, content="Разбор карты", error=None):
        self.content = content
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content="Разбор карты", error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_sends_system_prompt_chart_and_chunks():
    client = FakeClient()

    text = asyncio.run(
        synthesize_chart_report(CHART, CHUNKS, model_name="google/gemini-2.5-pro", client=client)
    )

    assert text == "Разбор карты"
    call = client.completions.calls[0]
    assert call["model"] == "google/gemini-2.5-pro"
    assert call["temperature"] == 0.7
    system, user = call["messages"]
    assert system == {"role": "system", "content": ASTROLOGER_SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert "Leo" in user["content"] and "142.2" in user["content"]
    assert "Солнце в знаке Льва" in user["content"]
    assert "Sun_in_Leo" in user["content"]


def test_chunks_are_truncated_to_the_per_chunk_budget():
    long_chunks = [
        {
            "chunk_id": "long",
            "section_title": "Длинный раздел",
            "content": "слово " * 500,
            "elements": [],
        }
    ]

    content = build_user_content(CHART, long_chunks, max_chunk_chars=200, max_total_chars=5000)

    assert "…" in content
    assert content.count("слово") < 500


def test_total_budget_stops_adding_chunks():
    many = [
        {
            "chunk_id": f"chunk-{index}",
            "section_title": f"Раздел {index}",
            "content": "текст раздела " * 50,
            "elements": [],
        }
        for index in range(50)
    ]

    content = build_user_content(CHART, many, max_chunk_chars=500, max_total_chars=3000)

    assert len(content) < 4200
    assert "Раздел 0" in content
    assert "Раздел 49" not in content


def test_empty_response_is_an_error():
    client = FakeClient(content="")

    with pytest.raises(SynthesisError, match="empty"):
        asyncio.run(synthesize_chart_report(CHART, CHUNKS, client=client))


def test_provider_error_is_wrapped():
    client = FakeClient(error=RuntimeError("401 Unauthorized"))

    with pytest.raises(SynthesisError, match="401"):
        asyncio.run(synthesize_chart_report(CHART, CHUNKS, client=client))


def test_multi_part_content_is_joined():
    client = FakeClient()
    client.completions.content = [
        SimpleNamespace(text="Часть 1"),
        {"type": "text", "text": "Часть 2"},
    ]

    text = asyncio.run(synthesize_chart_report(CHART, CHUNKS, client=client))

    assert text == "Часть 1\nЧасть 2"


@pytest.mark.parametrize("temperature", [-0.1, 2.1, "hot"])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(SynthesisError):
        asyncio.run(synthesize_chart_report(CHART, CHUNKS, temperature=temperature))


def test_empty_chart_is_rejected():
    with pytest.raises(SynthesisError, match="non-empty mapping"):
        build_user_content({}, CHUNKS)
