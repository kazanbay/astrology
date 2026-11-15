"""Langfuse-backed prompt management with local fallbacks.

Prompts are edited in the Langfuse UI (versioned, labeled, rollback-friendly)
and fetched by name and label. Design goals:

* The bot must keep working without Langfuse: missing credentials, network
  failures, a missing prompt or a template that violates the code contract
  all fall back to the local prompt constants, so behavior degrades to the
  pre-Langfuse state instead of failing.
* A fetched chat prompt must use exactly the documented ``{{variables}}``;
  anything else is rejected at resolve time and the local prompt is used
  instead. This keeps the report cache key (prompt version) consistent with
  the prompt that is actually sent to the model.
* Resolution is memoized for the process lifetime: one fetch per prompt, the
  same text and version everywhere (synthesizer and report cache key). New
  versions published in Langfuse are picked up after a restart or an explicit
  :func:`reset_cache`.

Environment knobs:

* ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_BASE_URL`` -
  Langfuse connection (already used by tracing).
* ``LANGFUSE_PROMPTS_ENABLED`` - explicit kill switch (default: enabled).
* ``LANGFUSE_PROMPT_LABEL`` - which label to resolve (default ``production``).
* ``LANGFUSE_PROMPT_NAME_INTERPRETATION`` / ``LANGFUSE_PROMPT_NAME_QA`` -
  rename the prompts without touching the code.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Default prompt names in the Langfuse UI (overridable via environment).
INTERPRETATION_PROMPT_NAME = "astrology_interpretation"
QA_PROMPT_NAME = "astrology_qa"
FORECAST_PROMPT_NAME = "astrology_forecast"
SYNASTRY_PROMPT_NAME = "astrology_synastry"

_PROMPT_NAME_OVERRIDES = {
    INTERPRETATION_PROMPT_NAME: "LANGFUSE_PROMPT_NAME_INTERPRETATION",
    QA_PROMPT_NAME: "LANGFUSE_PROMPT_NAME_QA",
    FORECAST_PROMPT_NAME: "LANGFUSE_PROMPT_NAME_FORECAST",
    SYNASTRY_PROMPT_NAME: "LANGFUSE_PROMPT_NAME_SYNASTRY",
}

_ENABLED_ENV_VAR = "LANGFUSE_PROMPTS_ENABLED"
_LABEL_ENV_VAR = "LANGFUSE_PROMPT_LABEL"
_DEFAULT_LABEL = "production"

#: Fail fast so a Langfuse outage never delays the first user request much.
_FETCH_TIMEOUT_SECONDS = 3.0
_FETCH_MAX_RETRIES = 1

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class ResolvedPrompt:
    """A text prompt (single string) resolved from Langfuse or the fallback."""

    name: str
    source: str  # "langfuse" | "local"
    version: str
    text: str | None = None  # None for the local fallback: caller owns the text


@dataclass(frozen=True)
class ResolvedChatPrompt:
    """A chat prompt template with ``{{variables}}`` from Langfuse or fallback.

    ``compile`` substitutes the variables through the Langfuse client and
    returns OpenAI-ready ``{"role", "content"}`` messages. For the local
    fallback (``client is None``) it returns ``None`` and the caller builds
    the messages from the local constants.
    """

    name: str
    source: str  # "langfuse" | "local"
    version: str
    client: Any = field(default=None, compare=False, repr=False)

    def compile(self, **variables: Any) -> list[dict[str, str]] | None:
        """Substitute the variables; ``None`` means "use the local prompt"."""
        if self.client is None:
            return None
        messages: list[dict[str, str]] = []
        for raw in self.client.compile(**variables):
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Langfuse prompt {self.name!r}: unsupported message entry {raw!r}; "
                    "only role/content messages are supported"
                )
            role = str(raw.get("role") or "").strip()
            content = raw.get("content")
            if not role or not isinstance(content, str) or not content.strip():
                raise ValueError(f"Langfuse prompt {self.name!r}: invalid chat message {raw!r}")
            messages.append({"role": role, "content": content})
        return messages


_text_cache: dict[str, ResolvedPrompt] = {}
_chat_cache: dict[str, ResolvedChatPrompt] = {}
_lock = threading.Lock()


def reset_cache() -> None:
    """Drop memoized resolutions (tests and manual reload)."""
    with _lock:
        _text_cache.clear()
        _chat_cache.clear()


def prompt_name(default_name: str) -> str:
    """Effective prompt name in Langfuse, honoring the env overrides."""
    override = _PROMPT_NAME_OVERRIDES.get(default_name)
    if override:
        value = os.getenv(override, "").strip()
        if value:
            return value
    return default_name


def _enabled() -> bool:
    flag = os.getenv(_ENABLED_ENV_VAR, "true").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def prompt_label() -> str:
    """Label the code resolves prompts under (default ``production``)."""
    return os.getenv(_LABEL_ENV_VAR, "").strip() or _DEFAULT_LABEL


def _fetch_client() -> Any:
    # Imported lazily so environments without langfuse installed keep working.
    from langfuse import get_client

    return get_client()


def _validate_chat_template(
    name: str,
    prompt: Any,
    allowed_variables: tuple[str, ...],
) -> str | None:
    """Return an error message when the template violates the code contract."""
    raw_messages = getattr(prompt, "prompt", None)
    if not isinstance(raw_messages, list) or not raw_messages:
        return "prompt has no messages"
    variables: set[str] = set()
    for raw in raw_messages:
        if not isinstance(raw, Mapping) or raw.get("type") != "message":
            return "only role/content messages are supported (no placeholders)"
        role = str(raw.get("role") or "").strip()
        content = raw.get("content")
        if not role:
            return "a message has an empty role"
        if not isinstance(content, str) or not content.strip():
            return "a message has empty content"
        variables.update(_PLACEHOLDER_RE.findall(content))
    unexpected = sorted(variables - set(allowed_variables))
    if unexpected:
        return (
            f"unexpected variables {unexpected}; the template must use exactly "
            f"{sorted(allowed_variables)}"
        )
    missing = sorted(set(allowed_variables) - variables)
    if missing:
        return f"variables {missing} are not used in the template"
    return None


def _fetch_text(name: str) -> ResolvedPrompt | None:
    try:
        prompt = _fetch_client().get_prompt(
            name,
            label=prompt_label(),
            type="text",
            max_retries=_FETCH_MAX_RETRIES,
            fetch_timeout_seconds=_FETCH_TIMEOUT_SECONDS,
        )
        text = getattr(prompt, "prompt", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("prompt text is empty")
        version = int(getattr(prompt, "version", 0))
    except Exception as exc:  # noqa: BLE001 - any failure means "use the local prompt"
        logger.warning("Langfuse text prompt %r is unavailable, using the local prompt (%s)", name, exc)
        return None
    return ResolvedPrompt(name=name, source="langfuse", version=f"langfuse:{name}:v{version}", text=text)


def _fetch_chat(name: str, allowed_variables: tuple[str, ...]) -> ResolvedChatPrompt | None:
    try:
        prompt = _fetch_client().get_prompt(
            name,
            label=prompt_label(),
            type="chat",
            max_retries=_FETCH_MAX_RETRIES,
            fetch_timeout_seconds=_FETCH_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "use the local prompt"
        logger.warning("Langfuse chat prompt %r is unavailable, using the local prompt (%s)", name, exc)
        return None
    error = _validate_chat_template(name, prompt, allowed_variables)
    if error is not None:
        logger.warning("Langfuse chat prompt %r rejected, using the local prompt: %s", name, error)
        return None
    return ResolvedChatPrompt(
        name=name,
        source="langfuse",
        version=f"langfuse:{name}:v{int(getattr(prompt, 'version', 0))}",
        client=prompt,
    )


def resolve(name: str = QA_PROMPT_NAME) -> ResolvedPrompt:
    """Resolve a text prompt by name (memoized for the process lifetime)."""
    name = prompt_name(name)
    with _lock:
        cached = _text_cache.get(name)
    if cached is not None:
        return cached
    resolved = _fetch_text(name) if _enabled() else None
    if resolved is None:
        resolved = ResolvedPrompt(name=name, source="local", version="local", text=None)
    with _lock:
        _text_cache[name] = resolved
    return resolved


def resolve_chat(
    name: str,
    *,
    allowed_variables: tuple[str, ...],
    local_version: str,
) -> ResolvedChatPrompt:
    """Resolve a chat prompt template by name (memoized for the process)."""
    name = prompt_name(name)
    with _lock:
        cached = _chat_cache.get(name)
    if cached is not None:
        return cached
    resolved = _fetch_chat(name, allowed_variables) if _enabled() else None
    if resolved is None:
        resolved = ResolvedChatPrompt(name=name, source="local", version=local_version, client=None)
    with _lock:
        _chat_cache[name] = resolved
    return resolved
