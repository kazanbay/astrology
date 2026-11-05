"""Deterministic chart identity: ``chart_hash`` is the cache key of one chart.

The hash covers the inputs that fully determine the pair
"ephemeris snapshot + retrieved knowledge-base chunks":

* the instant in UTC;
* the coordinates rounded to four decimals (absorbs micro-offsets of
  geocoders, which would otherwise split one city into several cache rows);
* the house system;
* the chunk language.

Everything else in ``charts_cache`` is derived data. Two users born at the same
instant in the same place share one row, which is exactly the point of the
cache. Changing the payload below changes every hash, so a change of its
semantics requires re-filling ``charts_cache``.

The calculator's ``node_type`` and aspect ``orbs`` are fixed defaults of the
pipeline and are not part of the key; a caller that starts varying them must
mix them into the payload, otherwise two different charts collide.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .birth import BirthData, canonical_utc_string

#: Number of decimals kept for coordinates in the hash payload.
HASH_COORDINATE_PRECISION = 4

#: SHA-256 in hex, the width of ``charts_cache.chart_hash``.
CHART_HASH_LENGTH = 64


def generate_chart_hash(
    dt_utc: datetime | str,
    lat: float,
    lon: float,
    house_system: str,
    language: str = "ru",
) -> str:
    """Return the SHA-256 cache key of a chart request.

    ``BirthData`` validates and normalizes every field first, so a hash built
    from a string instant and one built from the equivalent datetime are
    identical.
    """
    birth = BirthData(
        dt_utc=dt_utc,  # type: ignore[arg-type]  # datetime | str is accepted by BirthData
        latitude=lat,
        longitude=lon,
        house_system=house_system,
        language=language,
    )
    payload: dict[str, Any] = {
        "dt_utc": canonical_utc_string(birth.dt_utc),
        "lat": _rounded(birth.latitude),
        "lon": _rounded(birth.longitude),
        "house_system": birth.house_system,
        "language": birth.language,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rounded(value: float) -> float:
    """Round for the payload and collapse ``-0.0`` into ``0.0``."""
    rounded = round(value, HASH_COORDINATE_PRECISION)
    return 0.0 if rounded == 0 else rounded
