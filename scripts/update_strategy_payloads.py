"""Migrate existing strategy collections to the flat, filterable payload schema."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.chunking.strategies import (  # noqa: E402
    COLLECTION_NAMES,
    STRATEGIES,
    collection_name,
)
from src.indexing.payload_migration import migrate_collection  # noqa: E402
from src.indexing.qdrant_index import ensure_payload_indexes  # noqa: E402

LOGGER = logging.getLogger("update_strategy_payloads")


def _parse_collections(value: str) -> list[str]:
    aliases: dict[str, str] = {}
    for strategy, collection in COLLECTION_NAMES.items():
        aliases[strategy.lower()] = strategy
        aliases[collection.lower()] = strategy
    selected: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item not in aliases:
            raise argparse.ArgumentTypeError(
                f"Unknown collection {raw!r}; choose from: {', '.join(STRATEGIES)}"
            )
        strategy = aliases[item]
        if strategy not in selected:
            selected.append(strategy)
    if not selected:
        raise argparse.ArgumentTypeError("At least one collection must be selected")
    return selected


def _client() -> Any:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=120, check_compatibility=False)


def _print_distribution(result: dict[str, Any]) -> None:
    print(f"\n{result['collection']}: seen={result['seen']} updated={result['updated']}")
    for field, values in result["distribution"].items():
        rendered = ", ".join(f"{key}={count}" for key, count in list(values.items())[:6])
        print(f"  {field}: {rendered}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        type=_parse_collections,
        default=list(STRATEGIES),
        help=f"Comma-separated strategies or collections (default: {', '.join(STRATEGIES)})",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Debug: stop after N points per collection")
    parser.add_argument("--dry-run", action="store_true", help="Only report the new metadata distribution")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.page_size < 1 or args.max_attempts < 1:
        parser.error("--batch-size, --page-size and --max-attempts must be positive")

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    client = _client()
    try:
        for strategy in args.collections:
            name = collection_name(strategy)
            try:
                if not args.dry_run:
                    ensure_payload_indexes(client, name)
                result = migrate_collection(
                    client,
                    name,
                    batch_size=args.batch_size,
                    page_size=args.page_size,
                    max_attempts=args.max_attempts,
                    dry_run=args.dry_run,
                    limit=args.limit,
                )
                _print_distribution(result)
            except Exception:  # noqa: BLE001 - one collection must not stop the others
                LOGGER.exception("Collection %s failed", name)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
