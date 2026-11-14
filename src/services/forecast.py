"""Forecast tool: transits of the current sky onto a stored natal profile.

``ForecastService.get_forecast_for_interval(telegram_id, interval, contact_id=None)``
is the function required by the TZ. It resolves the chart from the account
profile (owner) or from a contact owned by that account, so arbitrary birth
inputs cannot enter through the bot or a LangGraph tool call. The result is a
JSON-ready dict, so the same tool serves the Telegram handlers and a future
LangGraph node without reshaping.

Method: the interval is covered by a deterministic sampling grid; at every
sample the major transit aspects onto the natal points are collected within
tight transit orbs and aggregated (tightest orb, first/last sample). Slow
bodies dominate the report; the Moon is included only for the day interval,
where the grid is fine enough to catch its short transits. This is a strictly
algorithmic calculation - no invented "authorial methods".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

from storage.birth import BirthData
from storage.store import ChartStore, UserProfile

from .profiles import (
    CONTACT_PERMISSION_MESSAGE,
    PREMIUM_REQUIRED_MESSAGE,
    ContactPermissionError,
    PremiumRequiredError,
    ProfileError,
    birth_data_from_profile,
    has_premium,
    resolve_owner_profile,
)

INTERVAL_KEYS: tuple[str, ...] = ("day", "week", "month", "year")


@dataclass(frozen=True)
class IntervalPlan:
    """Sampling grid of one forecast interval."""

    days: int
    step: timedelta
    include_moon: bool


INTERVAL_PLANS: Mapping[str, IntervalPlan] = MappingProxyType(
    {
        "day": IntervalPlan(days=1, step=timedelta(hours=2), include_moon=True),
        "week": IntervalPlan(days=7, step=timedelta(hours=6), include_moon=False),
        "month": IntervalPlan(days=30, step=timedelta(hours=12), include_moon=False),
        "year": IntervalPlan(days=365, step=timedelta(days=10), include_moon=False),
    }
)

#: Transit orbs are tighter than natal ones: a transit is an event only while
#: it runs nearly exact.
TRANSIT_ORBS: Mapping[str, float] = MappingProxyType(
    {
        "conjunction": 3.0,
        "sextile": 2.0,
        "square": 3.0,
        "trine": 3.0,
        "opposition": 3.0,
    }
)

#: Bodies reported as transiting planets. The lunar nodes are skipped: their
#: transits are a slow background axis, not an event-grade signal.
TRANSIT_BODIES: tuple[str, ...] = (
    "Sun",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
)

#: Natal points the transits land on: the planets plus the angles.
NATAL_TARGET_NAMES: tuple[str, ...] = (
    "Sun",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
    "Ascendant",
    "Midheaven",
)


class IntervalError(ValueError):
    """Unknown forecast interval."""


def default_natal_builder(birth: BirthData) -> dict[str, Any]:
    """Default chart builder: the project calculator, JSON-ready."""
    from astronomy.calculator import calculate_natal_chart

    return calculate_natal_chart(
        birth.dt_utc, birth.latitude, birth.longitude, birth.house_system
    ).to_dict()


def default_longitude_sampler(moment: datetime) -> dict[str, float]:
    """Default transit sampler: Swiss Ephemeris longitudes of every body."""
    from astronomy.calculator import calculate_planet_longitudes

    return calculate_planet_longitudes(moment)


def default_now_provider() -> datetime:
    return datetime.now(timezone.utc)


def _natal_targets(natal: dict[str, Any]) -> dict[str, float]:
    """Name -> longitude for the natal points transits land on."""
    targets: dict[str, float] = {}
    planets = natal.get("planets") or {}
    for name in NATAL_TARGET_NAMES:
        if name in ("Ascendant", "Midheaven"):
            continue
        position = planets.get(name)
        if isinstance(position, dict) and position.get("longitude") is not None:
            targets[name] = float(position["longitude"])
    asc = natal.get("asc")
    mc = natal.get("mc")
    if isinstance(asc, dict) and asc.get("longitude") is not None:
        targets["Ascendant"] = float(asc["longitude"])
    if isinstance(mc, dict) and mc.get("longitude") is not None:
        targets["Midheaven"] = float(mc["longitude"])
    return targets


def _best_aspect(separation: float, orbs: Mapping[str, float]) -> tuple[str, float] | None:
    """The tightest major aspect of one angular separation, or ``None``."""
    from astronomy.calculator import MAJOR_ASPECTS

    best: tuple[str, float] | None = None
    for aspect, exact_angle in MAJOR_ASPECTS:
        orb = abs(separation - exact_angle)
        if orb <= orbs[aspect] and (best is None or orb < best[1]):
            best = (aspect, orb)
    return best


def _collect_transit_events(
    natal: dict[str, Any],
    plan: IntervalPlan,
    start: datetime,
    longitude_sampler: Callable[[datetime], dict[str, float]],
) -> list[dict[str, Any]]:
    """Sample the interval and aggregate transit events onto the natal points."""
    from astronomy.calculator import angular_separation, house_of, sign_of

    targets = _natal_targets(natal)
    cusps = tuple(float(value) for value in (natal.get("cusps") or ()))
    transit_names = [name for name in TRANSIT_BODIES if plan.include_moon or name != "Moon"]

    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    end = start + timedelta(days=plan.days)
    moment = start
    while moment <= end:
        longitudes = longitude_sampler(moment)
        for name in transit_names:
            longitude = longitudes.get(name)
            if longitude is None:
                continue
            for target_name, target_longitude in targets.items():
                separation = angular_separation(float(longitude), target_longitude)
                hit = _best_aspect(separation, TRANSIT_ORBS)
                if hit is None:
                    continue
                aspect, orb = hit
                key = (name, target_name, aspect)
                record = aggregate.get(key)
                if record is None:
                    record = {
                        "transit": name,
                        "natal": target_name,
                        "aspect": aspect,
                        "orb": orb,
                        "hits": 0,
                        "first_sample": moment,
                        "last_sample": moment,
                    }
                    aggregate[key] = record
                if orb < record["orb"]:
                    record["orb"] = orb
                record["hits"] += 1
                record["last_sample"] = moment
        moment += plan.step

    events = [
        {
            "transit": record["transit"],
            "natal": record["natal"],
            "aspect": record["aspect"],
            "orb": round(float(record["orb"]), 2),
            "hits": record["hits"],
            "first_sample": record["first_sample"].isoformat(),
            "last_sample": record["last_sample"].isoformat(),
        }
        for record in aggregate.values()
    ]
    events.sort(key=lambda item: (item["orb"], item["transit"], item["natal"]))
    return events


def _transit_snapshot(
    longitudes: dict[str, float],
    natal: dict[str, Any],
) -> list[dict[str, Any]]:
    """Current positions of the transiting bodies against the natal cusp ring."""
    from astronomy.calculator import house_of, sign_of

    cusps = tuple(float(value) for value in (natal.get("cusps") or ()))
    snapshot: list[dict[str, Any]] = []
    for name in TRANSIT_BODIES:
        longitude = longitudes.get(name)
        if longitude is None:
            continue
        entry: dict[str, Any] = {
            "name": name,
            "longitude": round(float(longitude), 4),
            "sign": sign_of(float(longitude)),
        }
        if cusps:
            entry["house"] = house_of(float(longitude), cusps)
        snapshot.append(entry)
    return snapshot


class ForecastService:
    """Transit forecasts resolved from stored profiles only."""

    def __init__(
        self,
        store: ChartStore,
        *,
        chart_builder: Callable[[BirthData], dict[str, Any]] | None = None,
        longitude_sampler: Callable[[datetime], dict[str, float]] | None = None,
        now_provider: Callable[[], datetime] | None = None,
        max_events: int = 15,
    ) -> None:
        """``chart_builder`` / ``longitude_sampler`` / ``now_provider`` are DI
        seams for tests; production calls never pass birth data through them -
        the data always comes from the stored profile of the account."""
        if max_events < 1:
            raise ValueError("max_events must be greater than zero")
        self.store = store
        self.max_events = max_events
        self._chart_builder = chart_builder or default_natal_builder
        self._longitude_sampler = longitude_sampler or default_longitude_sampler
        self._now_provider = now_provider or default_now_provider

    async def get_forecast_for_interval(
        self,
        telegram_id: int | str,
        interval: str,
        contact_id: int | None = None,
    ) -> dict[str, Any]:
        """Forecast for the owner's profile, or for one of their contacts.

        This is the TZ function. ``contact_id=None`` always means the account
        owner; a non-null contact id is accepted only from a premium account
        and only for a contact that belongs to that account.
        """
        plan = INTERVAL_PLANS.get(interval)
        if plan is None:
            supported = ", ".join(INTERVAL_KEYS)
            raise IntervalError(f"Неизвестный интервал {interval!r}; поддерживаются: {supported}")

        try:
            telegram = int(telegram_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"telegram_id must be an integer, got {telegram_id!r}") from exc

        profile = await resolve_owner_profile(self.store, telegram_id=telegram)
        person, birth, record = await self._resolve_person(profile, telegram, contact_id)
        natal = await asyncio.to_thread(self._chart_builder, birth)
        now = self._now_provider()
        events = await asyncio.to_thread(
            _collect_transit_events, natal, plan, now, self._longitude_sampler
        )
        snapshot = await asyncio.to_thread(
            _transit_snapshot, self._longitude_sampler(now), natal
        )
        return {
            "tool": "forecast",
            "telegram_id": telegram,
            "interval": interval,
            "person": person,
            "birth": {
                "city": birth.city,
                "timezone": birth.timezone,
                "datetime_local": birth.datetime_local,
                "time_unknown": bool(record.birth_time_unknown),
            },
            "range": {
                "start": now.isoformat(),
                "end": (now + timedelta(days=plan.days)).isoformat(),
                "days": plan.days,
            },
            "generated_at": now.isoformat(),
            "now_transits": snapshot,
            "events": events[: self.max_events],
            "event_count": len(events),
        }

    async def _resolve_person(
        self,
        profile: UserProfile,
        telegram: int,
        contact_id: int | None,
    ) -> tuple[dict[str, Any], BirthData, Any]:
        """Resolve whose chart the forecast is built for (the TZ permission rule).

        Premium is checked before the contact lookup so a foreign contact id
        cannot even be probed by non-premium callers.
        """
        if contact_id is None:
            name = profile.display_name or profile.first_name or "Владелец"
            person = {"kind": "owner", "name": name, "relation": None}
            return person, birth_data_from_profile(profile), profile

        if not has_premium(profile, telegram):
            raise PremiumRequiredError(PREMIUM_REQUIRED_MESSAGE)
        contact = await self.store.get_contact(int(contact_id), user_id=profile.user_id)
        if contact is None:
            raise ContactPermissionError(CONTACT_PERMISSION_MESSAGE)
        person = {"kind": "contact", "name": contact.name, "relation": contact.relation}
        return person, birth_data_from_profile(contact), contact
