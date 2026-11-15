"""In-process scheduler for the admin digests.

The project has no task queue, so the schedule lives in an asyncio task next to
polling: asleep until the nearest firing point, then it sends the due digest.
Times are Moscow: daily at 21:00, weekly on Sunday at 20:00. A restart simply
computes the next occurrence; a missed window is skipped by design.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from storage.store import ChartStore

from .analytics import format_daily_digest, format_weekly_digest

logger = logging.getLogger(__name__)

MSK = ZoneInfo("Europe/Moscow")
DAILY_AT = time(21, 0)
WEEKLY_AT = time(20, 0)
SUNDAY = 6
#: A firing is considered due when the wake-up lands this close after it.
FIRE_WINDOW_SECONDS = 120.0
#: Do not spin on a clock that jumps backwards.
MIN_SLEEP_SECONDS = 1.0


def next_daily(now: datetime) -> datetime:
    """Nearest 21:00 MSK strictly after ``now``."""
    local = now.astimezone(MSK)
    candidate = local.replace(
        hour=DAILY_AT.hour, minute=DAILY_AT.minute, second=0, microsecond=0
    )
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate


def next_weekly(now: datetime) -> datetime:
    """Nearest Sunday 20:00 MSK strictly after ``now``."""
    local = now.astimezone(MSK)
    candidate = local.replace(
        hour=WEEKLY_AT.hour, minute=WEEKLY_AT.minute, second=0, microsecond=0
    )
    candidate += timedelta(days=(SUNDAY - local.weekday()) % 7)
    if candidate <= local:
        candidate += timedelta(days=7)
    return candidate


async def run_digest_scheduler(
    bot: Any,
    store: ChartStore,
    chat_id: int,
    *,
    now_fn: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Sleep until the nearest digest time and send what is due."""
    current_time = now_fn or (lambda: datetime.now(timezone.utc))
    wait = sleep or asyncio.sleep
    while True:
        now = current_time()
        daily = next_daily(now)
        weekly = next_weekly(now)
        target = min(daily, weekly)
        await wait(max(MIN_SLEEP_SECONDS, (target - now).total_seconds()))

        now = current_time()
        try:
            if daily <= now <= daily + timedelta(seconds=FIRE_WINDOW_SECONDS):
                metrics = await store.collect_analytics(since=now - timedelta(days=1))
                await bot.send_message(
                    chat_id,
                    format_daily_digest(metrics, now=now),
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            if weekly <= now <= weekly + timedelta(seconds=FIRE_WINDOW_SECONDS):
                metrics = await store.collect_analytics(since=now - timedelta(days=7))
                await bot.send_message(
                    chat_id,
                    format_weekly_digest(metrics, now=now),
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
        except Exception:
            logger.exception("digest delivery failed")
