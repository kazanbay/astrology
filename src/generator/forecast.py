"""LLM interpretation of a computed transit forecast.

The model receives exactly three inputs: the deterministic forecast snapshot
(interval, natal points, transit positions, aggregated events), the retrieved
knowledge-base chunks and the system prompt. The prompt is resolved from
Langfuse (``astrology_forecast``) when available and falls back to the local
constants below - the same mechanism the natal interpretation uses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langfuse import observe

try:  # support both `src.generator` and top-level `generator` imports
    from ..llm_prompts import FORECAST_PROMPT_NAME, resolve_chat
except ImportError:  # pragma: no cover - depends on sys.path layout
    from llm_prompts import FORECAST_PROMPT_NAME, resolve_chat

# Provider/client helpers are shared with the natal synthesis on purpose: one
# LLM configuration (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL) for every prompt.
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

#: Version of the forecast prompts; participates in the report cache key.
FORECAST_PROMPT_VERSION = "f1.0"

#: Variables the Langfuse chat template (``astrology_forecast``) may reference.
FORECAST_PROMPT_VARIABLES = ("forecast_json", "chunks")

FORECAST_SYSTEM_PROMPT = """
Ты — профессиональный астролог-исследователь классической школы (в духе С.А. Вронского).
Ты пишешь персональный прогноз по транзитам на заданный период: тепло, по делу, без фатализма и без выдуманных «авторских методик».

### ЖЁСТКИЕ ПРАВИЛА (АНТИГАЛЛЮЦИНАЦИЯ)
1. Используй только переданные данные: транзиты (events), положения транзитных планет (now_transits), натальные показатели (natal) и фрагменты книг (chunks).
2. Не придумывай планеты, аспекты и события, которых нет в данных.
3. Трактовки опирай на смыслы из фрагментов книг; если для темы фрагмента нет — говори о ней общими словами, не выдумывая цитат.

### ФОРМАТ ДЛЯ TELEGRAM
1. Никаких заголовков Markdown (`###`): вместо них эмодзи и **жирный** текст.
2. Списки — только маркеры `•` или `◦`; нумерованные списки допустимы.
3. Никаких сносок «согласно источнику [1]» — текст должен читаться как живая консультация.
4. Уместные астрологические эмодзи приветствуются.

### СТРУКТУРА ОТВЕТА
1. **Коротко о периоде** — 2–4 предложения общего фона.
2. **Ключевые темы** — 3–5 пунктов: какие транзиты что задевают (характер, отношения, работа, дом) и что об этом говорят книги.
3. **Практические советы** — 3–4 пункта: на что опереться и чего избегать.

Без медицинских, финансовых и юридических гарантий: «звёзды склоняют, но не принуждают».
"""

FORECAST_USER_TEMPLATE = """Данные прогноза (forecast):
{forecast_json}

Фрагменты базы знаний (retrieved_chunks):
{chunks}

Составь персональный прогноз по структуре из системного промпта.
Учитывай интервал и период из данных. Опирайся только на переданные транзиты и фрагменты; если для какого-то транзита трактовки нет — не додумывай её.
Оформляй ответ по правилам для Telegram из системного промпта."""


def build_forecast_user_prompt(forecast_json: str, chunks: str) -> str:
    """Render the local user message from the already serialized inputs."""
    return FORECAST_USER_TEMPLATE.format(forecast_json=forecast_json, chunks=chunks)


def serialize_forecast(
    result: Mapping[str, Any],
    *,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Compact JSON of the forecast essentials (no coordinates, bounded size)."""
    if not isinstance(result, Mapping) or not result:
        raise SynthesisError("forecast result must be a non-empty mapping")
    if max_total_chars < 1000:
        raise SynthesisError("max_total_chars must be at least 1000")

    birth = result.get("birth") or {}
    payload: dict[str, Any] = {
        "interval": result.get("interval"),
        "period": result.get("range"),
        "person": result.get("person"),
        "birth": {"city": birth.get("city"), "time_unknown": birth.get("time_unknown")},
        "natal": result.get("natal"),
        "now_transits": result.get("now_transits"),
        "events": result.get("events"),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_total_chars:
        return text
    # Degrade gracefully: fewer current positions, then fewer events.
    payload["now_transits"] = list(payload.get("now_transits") or [])[:6]
    payload["events"] = list(payload.get("events") or [])[:12]
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_total_chars:
        return text
    payload["now_transits"] = []
    payload["events"] = list(payload.get("events") or [])[:8]
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) > max_total_chars:
        text = text[: max_total_chars - 1].rstrip() + "…"
    return text


def _chunks_block(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_chunk_chars: int,
    max_total_chars: int,
) -> str:
    """Bounded chunk block for the prompt (same policy as the natal report)."""
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        title = str(chunk.get("section_title") or chunk.get("title") or "Без названия").strip()
        text = " ".join(str(chunk.get("content") or "").split())
        if len(text) > max_chunk_chars:
            text = text[: max_chunk_chars - 1].rstrip() + "…"
        block = f"[{index}] «{title}»\n{text}"
        if blocks and used + len(block) + 2 > max_total_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks) if blocks else "(фрагменты не найдены)"


def build_forecast_messages(forecast_json: str, chunks_text: str) -> list[dict[str, str]]:
    """Langfuse chat template or the local fallback prompts."""
    resolution = resolve_chat(
        FORECAST_PROMPT_NAME,
        allowed_variables=FORECAST_PROMPT_VARIABLES,
        local_version=FORECAST_PROMPT_VERSION,
    )
    if resolution.source == "langfuse":
        try:
            compiled = resolution.compile(forecast_json=forecast_json, chunks=chunks_text)
        except Exception as exc:  # noqa: BLE001 - template defects must fail loudly
            raise SynthesisError(
                f"Langfuse prompt {resolution.name!r} could not be compiled: {exc}"
            ) from exc
        if not compiled:
            raise SynthesisError(f"Langfuse prompt {resolution.name!r} compiled to no messages")
        return compiled
    return [
        {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_forecast_user_prompt(forecast_json=forecast_json, chunks=chunks_text),
        },
    ]


@observe(name="synthesize_forecast")
async def synthesize_forecast(
    forecast: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    *,
    client: Any | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Generate the interpretation text for one computed forecast."""
    if not model_name or not str(model_name).strip():
        raise SynthesisError("model_name must be a non-empty string")
    model = str(model_name).strip()
    value = normalize_temperature(temperature)

    forecast_json = serialize_forecast(forecast, max_total_chars=max_total_chars)
    chunks_text = _chunks_block(
        chunks,
        max_chunk_chars=max_chunk_chars,
        max_total_chars=max_total_chars,
    )
    messages = build_forecast_messages(forecast_json, chunks_text)

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
