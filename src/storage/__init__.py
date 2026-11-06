"""Persistent layer of the natal chart pipeline.

Public surface:

* ``BirthData`` / ``BirthInputError`` - validated birth input and its errors;
* ``generate_chart_hash`` - deterministic cache key;
* ``ChartPipeline`` - cache-aside orchestration;
* ``PostgresChartStore`` / ``ChartStore`` - persistence and its contract;
* ``collect_unique_chunks`` / ``element_chunk_index`` - RAG payload shaping;
* ``format_chart_reply`` / ``format_history`` - Russian rendering for bot/CLI.

Heavy optional dependencies (pyswisseph, Qdrant) are imported inside the
functions that need them, so importing this package stays cheap.
"""

from __future__ import annotations

from .birth import (
    BIRTH_COMMAND_USAGE,
    BirthData,
    BirthInputError,
    canonical_utc_string,
    detect_timezone,
    geocode_city,
    local_to_utc,
    normalize_utc,
    parse_birth_command,
    parse_local_datetime,
)
from .chunks import collect_unique_chunks, element_chunk_index
from .hashing import CHART_HASH_LENGTH, generate_chart_hash
from .pipeline import ChartPipeline, ChartResult, calculate_chart
from .presentation import format_chart_reply, format_history, format_section_previews, split_message
from .store import (
    CachedChart,
    ChartStore,
    GenerationRecord,
    PostgresChartStore,
    RequestRecord,
    normalize_profile_name,
)

__all__ = [
    "BIRTH_COMMAND_USAGE",
    "BirthData",
    "BirthInputError",
    "CHART_HASH_LENGTH",
    "CachedChart",
    "ChartPipeline",
    "ChartResult",
    "ChartStore",
    "GenerationRecord",
    "PostgresChartStore",
    "RequestRecord",
    "calculate_chart",
    "canonical_utc_string",
    "collect_unique_chunks",
    "detect_timezone",
    "element_chunk_index",
    "format_chart_reply",
    "format_history",
    "format_section_previews",
    "generate_chart_hash",
    "geocode_city",
    "local_to_utc",
    "normalize_profile_name",
    "normalize_utc",
    "parse_birth_command",
    "parse_local_datetime",
    "split_message",
]
