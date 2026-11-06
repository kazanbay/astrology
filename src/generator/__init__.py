"""LLM synthesis of natal-chart reports.

Public surface:

* ``PROMPT_VERSION`` / ``ASTROLOGER_SYSTEM_PROMPT`` - the versioned prompt;
* ``synthesize_chart_report`` - one call to the chat model;
* ``generate_generation_id`` / ``ChartReportService`` - deterministic report
  cache on top of ``llm_generations``.

The package imports the OpenAI SDK lazily, so the bot can start - and serve the
chart summary - even when the LLM key is missing or expired.
"""

from __future__ import annotations

from .prompts import ASTROLOGER_SYSTEM_PROMPT, PROMPT_VERSION
from .report import ChartReport, ChartReportService, generate_generation_id
from .synthesizer import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    SynthesisError,
    build_user_content,
    normalize_temperature,
    synthesize_chart_report,
)

__all__ = [
    "ASTROLOGER_SYSTEM_PROMPT",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "PROMPT_VERSION",
    "ChartReport",
    "ChartReportService",
    "SynthesisError",
    "build_user_content",
    "generate_generation_id",
    "normalize_temperature",
    "synthesize_chart_report",
]
