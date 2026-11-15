"""Log redaction: personal data never reaches stdout or log files.

Only anonymized identifiers (``chart_hash``, ``user_id``) belong in logs. This
filter masks emails, Telegram-like numeric ids and coordinates in every log
record, so a stray message cannot leak a name-adjacent detail by accident.
"""

from __future__ import annotations

import logging
import re

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
    (re.compile(r"(?<!\d)\d{8,13}(?!\d)"), "<id>"),
    (re.compile(r"(?<![\d.])-?\d{1,3}\.\d{3,6}(?![\d.])"), "<coord>"),
)


class PrivacyFilter(logging.Filter):
    """Mask personal data inside log messages before they are emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed record, pass through
            return True
        redacted = message
        for pattern, replacement in _REDACTIONS:
            redacted = pattern.sub(replacement, redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True
