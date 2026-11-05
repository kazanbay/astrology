"""Print a natal chart as JSON from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astronomy.calculator import AstronomyError, calculate_natal_chart  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate a natal chart for a UTC instant")
    parser.add_argument("dt", help="UTC instant with offset, e.g. 1990-08-15T12:00:00+00:00")
    parser.add_argument("lat", type=float, help="geographical latitude, -90..90")
    parser.add_argument("lon", type=float, help="geographical longitude (east positive), -180..180")
    parser.add_argument("--house-system", default="P", help="Swiss Ephemeris house code (default: P)")
    parser.add_argument("--node-type", default="mean", choices=("mean", "true"))
    args = parser.parse_args()

    moment = datetime.fromisoformat(args.dt)
    if moment.tzinfo is None or moment.utcoffset() is None:
        parser.error("dt must carry a UTC offset, e.g. 1990-08-15T12:00:00+00:00")

    try:
        chart = calculate_natal_chart(
            moment,
            args.lat,
            args.lon,
            args.house_system,
            node_type=args.node_type,
        )
    except (AstronomyError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(chart.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
