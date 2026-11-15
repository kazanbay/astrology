"""LLM reading of a synastry pair: cross aspects + book chunks.

The model receives the two compact chart snapshots (planets in signs/houses,
angles), the cross-aspect grid and retrieved book fragments. No names, birth
dates or coordinates are sent: the prompt speaks about «вы» and «партнёр», so
the reading stays useful without personal data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langfuse import observe

try:  # support both `src.generator` and top-level `generator` imports
    from ..llm_prompts import SYNASTRY_PROMPT_NAME, resolve_chat
except ImportError:  # pragma: no cover - depends on sys.path layout
    from llm_prompts import SYNASTRY_PROMPT_NAME, resolve_chat

from .forecast import _chunks_block
from .synthesizer import (
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_TOTAL_CHARS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    SynthesisError,
    _build_client,
    _message_text,
    normalize_temperature,
)

#: Version of the synastry prompts; participates in the report cache key.
SYNASTRY_PROMPT_VERSION = "s1.0"

#: Variables the Langfuse chat template (``astrology_synastry``) may reference.
SYNASTRY_PROMPT_VARIABLES = ("synastry_json", "chunks")

SYNASTRY_SYSTEM_PROMPT = """
Ты — профессиональный астролог-исследователь классической школы (в духе С.А. Вронского).
Ты пишешь разбор совместимости пары по синастрии: тепло, психологично, без фатализма и без выдуманных «авторских методик».

### ЖЁСТКИЕ ПРАВИЛА (АНТИГАЛЛЮЦИНАЦИЯ)
1. Используй только переданные данные: межкартные аспекты (events), положения планет обеих карт (person_a / person_b) и фрагменты книг (chunks).
2. Не придумывай аспекты, планеты и события, которых нет в данных.
3. Опирайся на смыслы из фрагментов книг; если для темы фрагмента нет — говори о ней общими словами, не выдумывая цитат.
4. Говори о тенденциях, а не о приговоре: «звёзды склоняют, но не принуждают».

### ОСОБОЕ ВНИМАНИЕ (классическая синастрия)
• Взаимодействие светил: Солнце и Луна одного партнёра с планетами другого.
• Аспекты Венеры и Марса — притяжение, страсть, конфликтность.
• Связи с седьмым домом — партнёрство и брак.

### ФОРМАТ ДЛЯ TELEGRAM
1. Никаких заголовков Markdown (`###`): вместо них эмодзи и **жирный** текст.
2. Списки — только маркеры `•` или `◦`; нумерованные списки допустимы.
3. Никаких сносок «согласно источнику [1]».

### СТРУКТУРА ОТВЕТА
1. **Портрет пары** — 3–5 предложений: как две карты сочетаются в целом.
2. **Что сближает** — 3–5 пунктов по гармоничным аспектам и светилам.
3. **Где напряжение** — 3–5 пунктов по напряжённым аспектам (Венера, Марс, седьмой дом) и как это проживать.
4. **Практические советы** — 3–4 пункта.

Без медицинских, финансовых и юридических гарантий.
"""

SYNASTRY_USER_TEMPLATE = """Данные синастрии (synastry):
{synastry_json}

Фрагменты базы знаний (retrieved_chunks):
{chunks}

Составь разбор совместимости пары по структуре из системного промпта.
Опирайся только на переданные аспекты и фрагменты; если для какого-то аспекта трактовки нет — не додумывай её.
Оформляй ответ по правилам для Telegram из системного промпта."""


def build_synastry_user_prompt(synastry_json: str, chunks: str) -> str:
    """Render the local user message from the already serialized inputs."""
    return SYNASTRY_USER_TEMPLATE.format(synastry_json=synastry_json, chunks=chunks)


def _sanitized_person(person: Any) -> dict[str, Any]:
    """Placements only: names and other personal data are dropped."""
    if not isinstance(person, Mapping):
        return {}
    return {"positions": person.get("positions")}


def serialize_synastry(
    result: Mapping[str, Any],
    *,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Compact JSON of the pair: placements and aspects, no personal data."""
    if not isinstance(result, Mapping) or not result:
        raise SynthesisError("synastry result must be a non-empty mapping")
    if max_total_chars < 1000:
        raise SynthesisError("max_total_chars must be at least 1000")

    payload: dict[str, Any] = {
        # Names and cities never leave for the model: only placements and aspects.
        "person_a": _sanitized_person(result.get("person_a")),
        "person_b": _sanitized_person(result.get("person_b")),
        "events": result.get("events"),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_total_chars:
        return text
    payload["events"] = list(payload.get("events") or [])[:15]
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) > max_total_chars:
        text = text[: max_total_chars - 1].rstrip() + "…"
    return text


def build_synastry_messages(synastry_json: str, chunks_text: str) -> list[dict[str, str]]:
    """Langfuse chat template or the local fallback prompts."""
    resolution = resolve_chat(
        SYNASTRY_PROMPT_NAME,
        allowed_variables=SYNASTRY_PROMPT_VARIABLES,
        local_version=SYNASTRY_PROMPT_VERSION,
    )
    if resolution.source == "langfuse":
        try:
            compiled = resolution.compile(synastry_json=synastry_json, chunks=chunks_text)
        except Exception as exc:  # noqa: BLE001 - template defects must fail loudly
            raise SynthesisError(
                f"Langfuse prompt {resolution.name!r} could not be compiled: {exc}"
            ) from exc
        if not compiled:
            raise SynthesisError(f"Langfuse prompt {resolution.name!r} compiled to no messages")
        return compiled
    return [
        {"role": "system", "content": SYNASTRY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_synastry_user_prompt(synastry_json=synastry_json, chunks=chunks_text),
        },
    ]


@observe(name="synthesize_synastry")
async def synthesize_synastry(
    result: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    *,
    client: Any | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Generate the pair reading for one computed synastry."""
    if not model_name or not str(model_name).strip():
        raise SynthesisError("model_name must be a non-empty string")
    model = str(model_name).strip()
    value = normalize_temperature(temperature)

    synastry_json = serialize_synastry(result, max_total_chars=max_total_chars)
    chunks_text = _chunks_block(
        chunks,
        max_chunk_chars=max_chunk_chars,
        max_total_chars=max_total_chars,
    )
    messages = build_synastry_messages(synastry_json, chunks_text)

    api = client if client is not None else _build_client()
    try:
        response = await api.chat.completions.create(
            model=model,
            messages=messages,
            temperature=value,
        )
    except Exception as exc:  # noqa: BLE001 - one error type for every provider failure
        raise SynthesisError(f"LLM request failed for {model}: {exc}") from exc

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise SynthesisError(f"LLM returned no choices for {model}")
    text = _message_text(choices[0].message)
    if not text:
        raise SynthesisError(f"LLM returned an empty report for {model}")
    return text
