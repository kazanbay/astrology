"""Synastry prompt/synthesis tests with a fake OpenAI-compatible client."""

import asyncio
from types import SimpleNamespace

import pytest

from generator.synastry import (
    SYNASTRY_PROMPT_VERSION,
    SYNASTRY_SYSTEM_PROMPT,
    build_synastry_messages,
    serialize_synastry,
    synthesize_synastry,
)
from generator.synthesizer import SynthesisError

SYNASTRY = {
    "tool": "synastry",
    "chart_hash_a": "a" * 64,
    "chart_hash_b": "b" * 64,
    "person_a": {
        "kind": "owner",
        "name": "Алексей",
        "positions": {"Sun": {"sign": "Leo", "longitude": 142.5}},
    },
    "person_b": {
        "kind": "contact",
        "name": "Анна",
        "relation": "супруга",
        "positions": {"Moon": {"sign": "Libra", "longitude": 190.0}},
    },
    "events": [{"body_a": "Sun", "body_b": "Moon", "aspect": "sextile", "orb": 0.4}],
}

CHUNKS = [
    {
        "chunk_id": "c1",
        "section_title": "Синастрия светил",
        "content": "Солнце партнёра в аспекте с Луной другого создаёт притяжение.",
    }
]


class FakeCompletions:
    def __init__(self, content="Разбор пары", error=None):
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
    def __init__(self, content="Разбор пары", error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_messages_carry_aspects_and_chunks_without_names():
    client = FakeClient()

    text = asyncio.run(
        synthesize_synastry(SYNASTRY, CHUNKS, model_name="google/gemini-2.5-pro", client=client)
    )

    assert text == "Разбор пары"
    call = client.completions.calls[0]
    system, user = call["messages"]
    assert system["content"] == SYNASTRY_SYSTEM_PROMPT
    assert "Sun" in user["content"] and "sextile" in user["content"]
    assert "Синастрия светил" in user["content"]
    # Personal names never reach the model.
    assert "Алексей" not in user["content"]
    assert "Анна" not in user["content"]


def test_local_prompt_is_used_without_langfuse():
    messages = build_synastry_messages('{"events": []}', "фрагменты")

    assert messages[0] == {"role": "system", "content": SYNASTRY_SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert '{"events": []}' in messages[1]["content"]


def test_serialize_rejects_empty_result():
    with pytest.raises(SynthesisError):
        serialize_synastry({})


def test_serialize_keeps_only_placements():
    text = serialize_synastry(SYNASTRY)

    assert "positions" in text
    assert "Алексей" not in text and "Анна" not in text
    assert "супруга" not in text


def test_empty_response_is_an_error():
    with pytest.raises(SynthesisError, match="empty"):
        asyncio.run(synthesize_synastry(SYNASTRY, CHUNKS, client=FakeClient(content="")))


def test_provider_error_is_wrapped():
    client = FakeClient(error=RuntimeError("429 Too Many Requests"))

    with pytest.raises(SynthesisError, match="429"):
        asyncio.run(synthesize_synastry(SYNASTRY, CHUNKS, client=client))


def test_prompt_version_is_stable():
    assert SYNASTRY_PROMPT_VERSION == "s1.0"
