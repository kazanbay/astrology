"""Product metrics digests for the admin group.

Pure formatting on top of ``PostgresChartStore.collect_analytics``: no personal
data, only counters, so a digest is safe to keep in chat history. Times are
rendered in Moscow time, the schedule the team lives by.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")


def _stickiness(dau: int, mau: int) -> str:
    if mau <= 0:
        return "—"
    return f"{round(100 * dau / mau)}%"


def _top_features(metrics: dict[str, int]) -> str:
    rows = [
        ("📅 Прогнозы", metrics.get("forecasts", 0)),
        ("🔮 Карты", metrics.get("charts", 0)),
        ("💞 Синастрия", metrics.get("synastry", 0)),
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{label} ({count})" for label, count in rows)


def format_daily_digest(metrics: dict[str, int], *, now: datetime) -> str:
    """Daily digest: the last 24 hours."""
    moment = now.astimezone(MSK)
    return "\n".join(
        [
            f"📊 <b>Дайджест дня</b> · {moment.strftime('%d.%m.%Y')}",
            "",
            f"👥 Пользователи: всего {metrics['total_users']} (+{metrics['new_users']} за сутки)",
            f"🔥 Активность: DAU {metrics['dau']} · WAU {metrics['wau']} · MAU {metrics['mau']}",
            (
                f"👑 Премиум: активных {metrics['active_premium']} · "
                f"покупок {metrics['new_payments']} · ⭐ {metrics['revenue_stars']}"
            ),
            (
                f"⚙️ Фичи: карты {metrics['charts']} · прогнозы {metrics['forecasts']} · "
                f"синастрия {metrics['synastry']}"
            ),
            f"💬 Поддержка: новых {metrics['feedback_new']} · в работе {metrics['feedback_open']}",
        ]
    )


def format_weekly_digest(metrics: dict[str, int], *, now: datetime) -> str:
    """Weekly digest: the last 7 days."""
    moment = now.astimezone(MSK)
    start = (moment - timedelta(days=6)).strftime("%d.%m")
    return "\n".join(
        [
            f"📊 <b>Итоги недели</b> · {start}–{moment.strftime('%d.%m.%Y')}",
            "",
            f"👥 База: всего {metrics['total_users']} (+{metrics['new_users']} за неделю)",
            (
                f"🔥 WAU {metrics['wau']} · MAU {metrics['mau']} · "
                f"Stickiness {_stickiness(metrics['dau'], metrics['mau'])}"
            ),
            (
                f"👑 Премиум: активных {metrics['active_premium']} · "
                f"покупок {metrics['new_payments']} · выручка ⭐ {metrics['revenue_stars']}"
            ),
            f"🏆 Топ функций: {_top_features(metrics)}",
            f"💬 Поддержка: новых {metrics['feedback_new']} · в работе {metrics['feedback_open']}",
        ]
    )


def format_stats_now(metrics: dict[str, int], *, now: datetime) -> str:
    """Manual /stats snapshot: the weekly view, marked as on-demand."""
    return format_weekly_digest(metrics, now=now).replace(
        "Итоги недели", "Сводка на сейчас", 1
    )
