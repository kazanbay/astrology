"""Feedback formatting and parsing shared by the bot and the admin group.

The notice posted to the admin group is compact HTML with a machine-readable
line ``user_id: 123``. The reply handler resolves the ticket by the replied
message id first (exact) and falls back to parsing this line, so a changed
message format never loses the reply.
"""

from __future__ import annotations

import html
import re

USER_ID_PATTERN = re.compile(r"user_id:\s*(\d+)")


def format_ticket_notice(
    *,
    ticket_id: int,
    username: str | None,
    user_id: int,
    message: str,
    created_at: str,
) -> str:
    """Render one ticket for the admin group (HTML)."""
    who = f"@{username}" if username else "без username"
    return (
        f"💬 <b>Обращение #{ticket_id}</b>\n"
        f"👤 {html.escape(who)} · user_id: {user_id}\n"
        f"🕒 {html.escape(created_at)}\n\n"
        f"{html.escape(message)}"
    )


def parse_user_id(text: str | None) -> int | None:
    """Extract ``user_id`` from a forwarded notice (regex fallback)."""
    if not text:
        return None
    match = USER_ID_PATTERN.search(text)
    return int(match.group(1)) if match else None


def format_admin_reply(text: str) -> str:
    """Render the support answer delivered to the user."""
    return f"📩 Ответ от поддержки:\n\n{str(text).strip()}"
