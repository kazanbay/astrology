"""Grant premium manually, with an audit trail.

Usage (from the project root, or inside the production container):

    python scripts/grant_premium.py --telegram-id 123456789 --months 1 \
        --reason "Подарок за подробный баг-репорт"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from storage.store import PostgresChartStore  # noqa: E402


async def grant(args: argparse.Namespace) -> int:
    store = await PostgresChartStore.connect(max_size=2)
    try:
        user_id = await store.find_user(telegram_id=args.telegram_id)
        if user_id is None:
            print(f"user with telegram_id {args.telegram_id} not found")
            return 1
        until = await store.grant_premium(
            user_id,
            args.months,
            source="manual",
            reason=args.reason,
            reference=args.reference,
        )
        print(f"granted {args.months} month(s) to user_id={user_id}; premium until {until}")
        return 0
    finally:
        await store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Grant premium manually")
    parser.add_argument("--telegram-id", type=int, required=True)
    parser.add_argument("--months", type=int, required=True)
    parser.add_argument("--reason", default="Ручное начисление")
    parser.add_argument("--reference", default=None, help="ticket id, chat link, etc.")
    return asyncio.run(grant(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
