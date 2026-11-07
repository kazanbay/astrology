"""Synthesis of a natal-chart report through an OpenAI-compatible LLM API.

The module is provider-agnostic: OpenRouter is the default endpoint, but any
OpenAI-compatible ``/chat/completions`` works via ``LLM_BASE_URL``. The model
receives exactly two inputs - the chart snapshot and the retrieved chunks - and
the system prompt forbids inventing anything outside them.

Context size is bounded: sections are stored whole in the cache (several
kilobytes each), while a report needs the meaning, not the full chapter. Every
chunk is cut to ``max_chunk_chars`` and the whole block to ``max_total_chars``,
so a request stays inside the context window of smaller models too.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import observe

from .prompts import ASTROLOGER_SYSTEM_PROMPT, build_user_prompt

# Load before reading the defaults below: the bot imports this module early.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-pro").strip() or "google/gemini-2.5-pro"
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip()
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

#: One knowledge-base section rarely needs more than this to carry its meaning.
DEFAULT_MAX_CHUNK_CHARS = 1200
#: Whole chunk block budget, roughly 8-10k tokens for Russian text.
DEFAULT_MAX_TOTAL_CHARS = 24000


class SynthesisError(RuntimeError):
    """The report could not be produced (configuration, transport or content)."""


def normalize_temperature(value: float) -> float:
    """Validate a temperature and round it for the deterministic cache key."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SynthesisError(f"temperature must be a number, got {value!r}") from exc
    if not 0.0 <= number <= 2.0:
        raise SynthesisError(f"temperature must be within [0, 2], got {number}")
    return round(number, 2)


def _chunk_block(index: int, chunk: Mapping[str, Any], *, max_chars: int) -> str:
    title = str(chunk.get("section_title") or chunk.get("title") or "Без названия").strip()
    elements = [
        str(element.get("key"))
        for element in chunk.get("elements") or []
        if isinstance(element, Mapping) and element.get("key")
    ]
    label = f"[{index}] «{title}»"
    if elements:
        label += f" (элементы: {', '.join(elements)})"
    text = " ".join(str(chunk.get("content") or "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return f"{label}\n{text}"


def build_user_content(
    chart_data: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Serialize the chart and the chunks into one bounded user message."""
    if not isinstance(chart_data, Mapping) or not chart_data:
        raise SynthesisError("chart_data must be a non-empty mapping")
    if max_chunk_chars < 100:
        raise SynthesisError("max_chunk_chars must be at least 100")
    if max_total_chars < 1000:
        raise SynthesisError("max_total_chars must be at least 1000")

    chart_json = json.dumps(chart_data, ensure_ascii=False, indent=2, sort_keys=True)
    if len(chart_json) > max_total_chars // 2:
        raise SynthesisError(
            "chart_data is unexpectedly large; check that the snapshot is the chart, not the cache row"
        )

    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = _chunk_block(index, chunk, max_chars=max_chunk_chars)
        # Keep at least one block, then stop once the budget is exhausted.
        if blocks and used + len(block) + 2 > max_total_chars:
            break
        if not blocks and len(block) > max_total_chars:
            block = block[: max_total_chars - 1].rstrip() + "…"
        blocks.append(block)
        used += len(block) + 2

    chunks_text = "\n\n".join(blocks) if blocks else "(фрагменты не найдены)"
    return build_user_prompt(chart_json=chart_json, chunks=chunks_text)


def _build_client() -> Any:
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SynthesisError(
            "LLM_API_KEY (or OPENROUTER_API_KEY) is not set; the report cannot be generated"
        )
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=DEFAULT_BASE_URL or None,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT,
    )


def _message_text(message: Any) -> str:
    """Extract text from a chat message, including multi-part content."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


@observe(name="synthesize_chart_report") 
async def synthesize_chart_report(
    chart_data: dict[str, Any],
    chunks: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    *,
    client: Any | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Generate the report text for one chart and its retrieved sections.

    ``client`` is injectable for tests; otherwise an ``AsyncOpenAI`` client is
    built from ``LLM_BASE_URL`` / ``LLM_API_KEY`` (falling back to
    ``OPENROUTER_API_KEY``).
    """
    if not model_name or not str(model_name).strip():
        raise SynthesisError("model_name must be a non-empty string")
    model = str(model_name).strip()
    value = normalize_temperature(temperature)

    content = build_user_content(
        chart_data,
        chunks,
        max_chunk_chars=max_chunk_chars,
        max_total_chars=max_total_chars,
    )

    api = client if client is not None else _build_client()
    try:
        response = await api.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ASTROLOGER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
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