"""Smoke: admin directory, block flag, stats and ticket list on the real DB."""

import asyncio

from services.admin import format_user_card, format_user_list, format_ticket_list
from storage.store import PostgresChartStore

SMOKE_ID = 777000799


async def main() -> None:
    store = await PostgresChartStore.connect(max_size=2)
    uid = None
    try:
        stale = await store.pool.fetchval(
            "SELECT user_id FROM users WHERE telegram_id = $1", SMOKE_ID
        )
        if stale:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", stale)
        uid = await store.upsert_user(
            telegram_id=SMOKE_ID, first_name="Админ", username="smoke_admin"
        )
        await store.touch_user(SMOKE_ID, "smoke_admin")
        await store.record_event(SMOKE_ID, "chart")
        await store.record_event(SMOKE_ID, "forecast")

        found = await store.list_users(query="smoke_admin", limit=3)
        print("search by username:", [(u.telegram_id, u.username) for u in found])
        print(format_user_list(found, query="smoke_admin"))

        profile = await store.get_user_profile(telegram_id=SMOKE_ID)
        stats = await store.admin_user_stats(profile.user_id)
        events = await store.list_user_events(SMOKE_ID, limit=5)
        print("stats:", stats)
        print(format_user_card(profile, stats, events=events))

        await store.set_user_blocked(profile.user_id, True)
        print("blocked:", await store.is_user_blocked(SMOKE_ID))
        await store.set_user_blocked(profile.user_id, False)
        print("unblocked:", await store.is_user_blocked(SMOKE_ID))

        tickets = await store.list_feedback(status="new", limit=3)
        print("open tickets:", [(t.id, t.username) for t in tickets])
        print(format_ticket_list(tickets)[:200])
    finally:
        if uid is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", uid)
        await store.close()
        print("cleanup OK")


asyncio.run(main())
