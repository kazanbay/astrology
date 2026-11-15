"""Forecast prompt/synthesis tests with a fake OpenAI-compatible client."""

import asyncio
from types import SimpleNamespace

import pytest

from generator.forecast import (
    FORECAST_PROMPT_VERSION,
    FORECAST_SYSTEM_PROMPT,
    build_forecast_messages,
    serialize_forecast,
    synthesize_forecast,
)
from generator.synthesizer import SynthesisError

FORECAST = {
    "tool": "forecast",
    "chart_hash": "a" * 64,
    "interval": "week",
    "person": {"kind": "owner", "name": "Анна", "relation": None},
    "birth": {"city": "Москва", "timezone": "Europe/Moscow", "time_unknown": False},
    "range": {
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-08T00:00:00+00:00",
        "days": 7,
    },
    "natal": {
        "planets": {"Sun": {"sign": "Leo", "house": 10}},
        "angles": {"Ascendant": {"sign": "Libra"}},
    },
    "now_transits": [{"name": "Saturn", "sign": "Aries", "house": 4}],
    "events": [
        {
            "transit": "Saturn",
            "natal": "Sun",
            "aspect": "square",
            "orb": 0.8,
            "hits": 3,
            "first_sample": "2026-01-01T00:00:00+00:00",
            "last_sample": "2026-01-05T00:00:00+00:00",
        }
    ],
    "event_count": 1,
}

CHUNKS = [
    {
        "chunk_id": "c1",
        "section_title": "Транзиты Сатурна",
        "content": "Сатурн требует зрелости и ответственности.",
    }
]


class FakeCompletions:
    def __init__(self, content="Прогноз периода", error=None):
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
    def __init__(self, content="Прогноз периода", error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_messages_carry_forecast_and_chunks():
    client = FakeClient()

    text = asyncio.run(
        synthesize_forecast(FORECAST, CHUNKS, model_name="google/gemini-2.5-pro", client=client)
    )

    assert text == "Прогноз периода"
    call = client.completions.calls[0]
    assert call["model"] == "google/gemini-2.5-pro"
    system, user = call["messages"]
    assert system["content"] == FORECAST_SYSTEM_PROMPT
    assert "week" in user["content"]
    assert "Saturn" in user["content"] and "square" in user["content"]
    assert "Транзиты Сатурна" in user["content"]
    assert "Сатурн требует зрелости" in user["content"]


def test_local_prompt_is_used_without_langfuse():
    messages = build_forecast_messages('{"interval": "day"}', "фрагменты")

    assert messages[0] == {"role": "system", "content": FORECAST_SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert '{"interval": "day"}' in messages[1]["content"]


def test_serialize_drops_coordinates():
    text = serialize_forecast({**FORECAST, "birth": {**FORECAST["birth"], "lat": 55.7, "lon": 37.6}})

    assert "55.7" not in text
    assert "Москва" in text


def test_empty_forecast_is_rejected():
    with pytest.raises(SynthesisError):
        serialize_forecast({})


def test_empty_response_is_an_error():
    with pytest.raises(SynthesisError, match="empty"):
        asyncio.run(synthesize_forecast(FORECAST, CHUNKS, client=FakeClient(content="")))


def test_provider_error_is_wrapped():
    client = FakeClient(error=RuntimeError("429 Too Many Requests"))

    with pytest.raises(SynthesisError, match="429"):
        asyncio.run(synthesize_forecast(FORECAST, CHUNKS, client=client))


def test_prompt_version_is_stable():
    assert FORECAST_PROMPT_VERSION == "f1.0"
