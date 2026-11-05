"""Natal chart core: ephemeris positions, house cusps and major aspects.

The module keeps the two coordinate systems of a natal chart apart:

* Level 1 - astronomy: apparent geocentric ecliptic longitudes and daily speeds
  come from the Swiss Ephemeris bindings (the ``swisseph`` module, distributed on
  PyPI as ``pysweph``; ``pyswisseph`` exposes the same API). House cusps are the
  product of the Earth's rotation for the given latitude, longitude and instant.
* Level 2 - projection: longitudes are split into equal 30-degree zodiac signs
  and compared with the cusp ring to assign houses. This layer is pure
  arithmetic and knows nothing about the ephemeris.

Time is accepted in UTC only. Local time, DST and time zone resolution belong to
the backend layer above this module: the calculator turns the received instant
into a Julian Day and never consults the local clock or ``zoneinfo``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

import swisseph as swe


class AstronomyError(RuntimeError):
    """Raised when Swiss Ephemeris cannot compute a chart for the given input."""


ZODIAC_SIGNS: tuple[str, ...] = (
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
)
ZODIAC_SIGN_SIZE = 360.0 / len(ZODIAC_SIGNS)
HOUSE_COUNT = 12

#: Geocentric bodies reported for every chart, in display order.
BODIES: tuple[tuple[str, int], ...] = (
    ("Sun", swe.SUN),
    ("Moon", swe.MOON),
    ("Mercury", swe.MERCURY),
    ("Venus", swe.VENUS),
    ("Mars", swe.MARS),
    ("Jupiter", swe.JUPITER),
    ("Saturn", swe.SATURN),
    ("Uranus", swe.URANUS),
    ("Neptune", swe.NEPTUNE),
    ("Pluto", swe.PLUTO),
)

#: Lunar node flavours; the node itself is the North Node, the South Node is its
#: opposition and is derived from it.
NODE_TYPES: Mapping[str, int] = MappingProxyType({"mean": swe.MEAN_NODE, "true": swe.TRUE_NODE})

#: Major aspects with their exact angular distance in degrees.
MAJOR_ASPECTS: tuple[tuple[str, float], ...] = (
    ("conjunction", 0.0),
    ("sextile", 60.0),
    ("square", 90.0),
    ("trine", 120.0),
    ("opposition", 180.0),
)

DEFAULT_ORBS: Mapping[str, float] = MappingProxyType(
    {
        "conjunction": 8.0,
        "sextile": 6.0,
        "square": 7.0,
        "trine": 8.0,
        "opposition": 8.0,
    }
)

#: |speed| below this value (degrees per day) marks a station.
STATIONARY_SPEED_THRESHOLD = 0.05

#: House codes accepted by ``swe.houses_ex``; anything else would silently fall
#: back to Placidus inside the library, so the code is validated up front.
HOUSE_SYSTEMS: Mapping[str, str] = MappingProxyType(
    {
        "P": "Placidus",
        "K": "Koch",
        "O": "Porphyry",
        "R": "Regiomontanus",
        "C": "Campanus",
        "E": "Equal (Asc)",
        "W": "Whole sign",
        "H": "Horizon/Azimuth",
        "B": "Alcabitius",
        "M": "Morinus",
        "X": "Axial rotation",
        "T": "Polich/Page",
        "V": "Vehlow equal",
        "N": "Whole sign from 0° Aries",
        "F": "Carter poli-equatorial",
    }
)


@dataclass(frozen=True)
class ZodiacPoint:
    """A degree on the ecliptic with its sign and house."""

    longitude: float
    sign: str
    house: int


@dataclass(frozen=True)
class PlanetPosition(ZodiacPoint):
    """A body placement: ecliptic longitude, sign, house, motion state."""

    speed: float
    is_retrograde: bool
    is_stationary: bool


@dataclass(frozen=True)
class Aspect:
    """A major aspect between two bodies."""

    body_a: str
    body_b: str
    aspect: str
    exact_angle: float
    separation: float
    orb: float


@dataclass(frozen=True)
class NatalChart:
    """The full output of the calculator: angles, cusps, bodies and aspects."""

    dt_utc: datetime
    latitude: float
    longitude: float
    house_system: str
    julian_day: float
    cusps: tuple[float, ...]
    asc: ZodiacPoint
    mc: ZodiacPoint
    planets: Mapping[str, PlanetPosition]
    aspects: tuple[Aspect, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the chart."""
        return {
            "dt_utc": self.dt_utc.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "house_system": self.house_system,
            "julian_day": self.julian_day,
            "cusps": list(self.cusps),
            "asc": _point_payload(self.asc),
            "mc": _point_payload(self.mc),
            "planets": {
                name: {
                    **_point_payload(position),
                    "speed": position.speed,
                    "is_retrograde": position.is_retrograde,
                    "is_stationary": position.is_stationary,
                }
                for name, position in self.planets.items()
            },
            "aspects": [
                {
                    "body_a": aspect.body_a,
                    "body_b": aspect.body_b,
                    "aspect": aspect.aspect,
                    "exact_angle": aspect.exact_angle,
                    "separation": aspect.separation,
                    "orb": aspect.orb,
                }
                for aspect in self.aspects
            ],
        }


def normalize_longitude(longitude: float) -> float:
    """Wrap an ecliptic longitude into ``0 <= longitude < 360``."""
    return float(longitude) % 360.0


def sign_of(longitude: float) -> str:
    """Return the zodiac sign that contains the longitude."""
    return ZODIAC_SIGNS[int(normalize_longitude(longitude) // ZODIAC_SIGN_SIZE)]


def house_of(longitude: float, cusps: Sequence[float]) -> int:
    """Return the house number (1-12) that contains the longitude.

    Houses run in zodiacal order from one cusp to the next; the ring may cross
    0° Aries, which the wrap-around branch handles. Degenerate or overlapping
    cusp rings fall back to the cusp with the smallest forward arc.
    """
    if len(cusps) != HOUSE_COUNT:
        raise ValueError(f"cusps must contain 12 values, got {len(cusps)}")
    degrees = normalize_longitude(longitude)
    for index in range(len(cusps)):
        start = normalize_longitude(cusps[index])
        end = normalize_longitude(cusps[(index + 1) % len(cusps)])
        if start <= end:
            if start <= degrees < end:
                return index + 1
        elif degrees >= start or degrees < end:
            return index + 1

    forward_arcs = [(degrees - normalize_longitude(cusp)) % 360.0 for cusp in cusps]
    return min(range(len(cusps)), key=lambda index: forward_arcs[index]) + 1


def angular_separation(first: float, second: float) -> float:
    """Return the shortest distance between two longitudes, 0-180 degrees."""
    difference = abs(normalize_longitude(first) - normalize_longitude(second)) % 360.0
    return 360.0 - difference if difference > 180.0 else difference


def calculate_aspects(
    planets: Mapping[str, PlanetPosition],
    orbs: Mapping[str, float] | None = None,
) -> tuple[Aspect, ...]:
    """Find every pair forming a major aspect within its orb.

    Only the closest major aspect per pair is kept; results are ordered by orb
    and then by body names so the output is stable between runs. The North and
    South Nodes are one axis, so their permanent opposition is not reported.
    """
    active_orbs = _resolve_orbs(orbs)
    names = tuple(planets)
    found: list[Aspect] = []
    for index, body_a in enumerate(names):
        for body_b in names[index + 1 :]:
            if {body_a, body_b} == {"NorthNode", "SouthNode"}:
                continue
            separation = angular_separation(planets[body_a].longitude, planets[body_b].longitude)
            best: tuple[str, float, float] | None = None
            for aspect, exact_angle in MAJOR_ASPECTS:
                orb = abs(separation - exact_angle)
                if orb <= active_orbs[aspect] and (best is None or orb < best[2]):
                    best = (aspect, exact_angle, orb)
            if best is not None:
                aspect, exact_angle, orb = best
                found.append(
                    Aspect(
                        body_a=body_a,
                        body_b=body_b,
                        aspect=aspect,
                        exact_angle=exact_angle,
                        separation=separation,
                        orb=orb,
                    )
                )
    return tuple(sorted(found, key=lambda item: (item.orb, item.body_a, item.body_b)))


def calculate_natal_chart(
    dt_utc: datetime,
    lat: float,
    lon: float,
    house_system: str = "P",
    *,
    node_type: str = "mean",
    orbs: Mapping[str, float] | None = None,
) -> NatalChart:
    """Build a natal chart for a UTC instant and a point on Earth.

    ``lat``/``lon`` are geographical degrees (east longitude positive), matching
    the arguments of ``swe.houses_ex``. The returned angles are the true
    Ascendant and Midheaven: for quadrant systems they coincide with cusps 1 and
    10, while equal and whole-sign systems keep their own cusp grid.
    """
    moment = _utc_datetime(dt_utc)
    latitude = _coordinate(lat, "lat", 90.0)
    longitude = _coordinate(lon, "lon", 180.0)
    system = _house_system(house_system)
    node = _node_flavour(node_type)

    julian_day = _julian_day(moment)
    cusps, asc, mc = _cusps_and_angles(julian_day, latitude, longitude, system)

    planets: dict[str, PlanetPosition] = {}
    for name, body in BODIES:
        planets[name] = _planet_position(julian_day, body, name, cusps)

    north_node = _planet_position(julian_day, NODE_TYPES[node], "NorthNode", cusps)
    planets["NorthNode"] = north_node
    south_longitude = normalize_longitude(north_node.longitude + 180.0)
    planets["SouthNode"] = PlanetPosition(
        longitude=south_longitude,
        sign=sign_of(south_longitude),
        house=house_of(south_longitude, cusps),
        speed=north_node.speed,
        is_retrograde=north_node.is_retrograde,
        is_stationary=north_node.is_stationary,
    )

    return NatalChart(
        dt_utc=moment,
        latitude=latitude,
        longitude=longitude,
        house_system=system,
        julian_day=julian_day,
        cusps=cusps,
        asc=ZodiacPoint(longitude=asc, sign=sign_of(asc), house=1),
        mc=ZodiacPoint(longitude=mc, sign=sign_of(mc), house=10),
        planets=MappingProxyType(planets),
        aspects=calculate_aspects(planets, orbs=orbs),
    )


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("dt_utc must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("dt_utc must be timezone-aware and expressed in UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"dt_utc must be UTC (offset 0), got {value.utcoffset()}")
    return value.astimezone(timezone.utc)


def _coordinate(value: float, name: str, limit: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if not -limit <= number <= limit:
        raise ValueError(f"{name} must be within [-{limit:g}, {limit:g}], got {number}")
    return number


def _house_system(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("house_system must be a single-letter Swiss Ephemeris code")
    code = value.strip().upper()
    if code not in HOUSE_SYSTEMS:
        supported = ", ".join(f"{key} ({name})" for key, name in HOUSE_SYSTEMS.items())
        raise ValueError(f"unsupported house system {value!r}; supported: {supported}")
    return code


def _node_flavour(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("node_type must be 'mean' or 'true'")
    code = value.strip().lower()
    if code not in NODE_TYPES:
        raise ValueError(f"unsupported node_type {value!r}; supported: mean, true")
    return code


def _resolve_orbs(orbs: Mapping[str, float] | None) -> Mapping[str, float]:
    if orbs is None:
        return DEFAULT_ORBS
    resolved: dict[str, float] = {}
    for aspect, _exact_angle in MAJOR_ASPECTS:
        supplied = orbs.get(aspect, DEFAULT_ORBS[aspect])
        try:
            value = float(supplied)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"orb for {aspect} must be a number") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"orb for {aspect} must be a non-negative finite number")
        resolved[aspect] = value
    return resolved


def _julian_day(moment: datetime) -> float:
    hour = (
        moment.hour
        + moment.minute / 60.0
        + (moment.second + moment.microsecond / 1_000_000.0) / 3600.0
    )
    return swe.julday(moment.year, moment.month, moment.day, hour)


def _cusps_and_angles(
    julian_day: float,
    latitude: float,
    longitude: float,
    house_system: str,
) -> tuple[tuple[float, ...], float, float]:
    try:
        result = swe.houses_ex(julian_day, latitude, longitude, house_system.encode("ascii"), 0)
    except swe.Error as exc:
        raise AstronomyError(
            f"house system {house_system!r} is not computable at latitude {latitude:g}: {exc}"
        ) from exc

    raw_cusps = tuple(result[0])
    ascmc = tuple(result[1])
    # Builds differ: pysweph returns the C-style 13-element array with a leading
    # placeholder, pyswisseph returns 12 cusps. Keep the twelve real ones.
    cusps = tuple(normalize_longitude(float(value)) for value in raw_cusps[-HOUSE_COUNT:])
    if len(cusps) != HOUSE_COUNT:
        raise AstronomyError(f"expected 12 house cusps, got {len(cusps)}")
    return cusps, normalize_longitude(float(ascmc[0])), normalize_longitude(float(ascmc[1]))


def _planet_position(
    julian_day: float,
    body: int,
    name: str,
    cusps: Sequence[float],
) -> PlanetPosition:
    try:
        result = swe.calc_ut(julian_day, body, swe.FLG_SWIEPH | swe.FLG_SPEED)
    except swe.Error as exc:
        raise AstronomyError(f"Swiss Ephemeris failed for {name}: {exc}") from exc

    coordinates, status = result[0], int(result[1])
    if status < 0:
        detail = result[2] if len(result) > 2 else ""
        raise AstronomyError(f"Swiss Ephemeris failed for {name}: {str(detail).strip()}".rstrip(": "))

    longitude = normalize_longitude(float(coordinates[0]))
    speed = float(coordinates[3])
    return PlanetPosition(
        longitude=longitude,
        sign=sign_of(longitude),
        house=house_of(longitude, cusps),
        speed=speed,
        is_retrograde=speed < 0.0,
        is_stationary=abs(speed) < STATIONARY_SPEED_THRESHOLD,
    )


def _point_payload(point: ZodiacPoint) -> dict[str, Any]:
    return {"longitude": point.longitude, "sign": point.sign, "house": point.house}
