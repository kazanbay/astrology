"""Synastry tool: cross aspects between the owner's chart and a contact's chart.

``SynastryService.check_synastry(telegram_id, contact_id)`` is the function
required by the TZ. Both charts are resolved from stored profiles only (the
account owner and a ``user_contacts`` row owned by that account), and the
feature is premium-gated. The result is a JSON-ready dict for the Telegram
handlers and future LangGraph tools.

Classical core only: major planet-to-planet aspects within the calculator's
default orbs, ordered by orb. Nodes and angles are skipped - the aspect grid
of planet-to-planet contacts is the deterministic, book-grounded layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from storage.birth import BirthData
from storage.store import ChartStore

from .forecast import default_natal_builder, default_now_provider
from .profiles import (
    CONTACT_PERMISSION_MESSAGE,
    PREMIUM_REQUIRED_MESSAGE,
    ContactPermissionError,
    PremiumRequiredError,
    birth_data_from_profile,
    has_premium,
    resolve_owner_profile,
)

#: Bodies used in synastry, both sides.
SYNASTRY_BODIES: tuple[str, ...] = (
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

#: Placements shown in the summary block of the report.
HIGHLIGHT_BODIES: tuple[str, ...] = ("Sun", "Moon", "Venus", "Mars")


def _planet_longitudes(chart: dict[str, Any]) -> dict[str, float]:
    """Name -> longitude of the synastry bodies of one chart."""
    planets = chart.get("planets") or {}
    result: dict[str, float] = {}
    for name in SYNASTRY_BODIES:
        position = planets.get(name)
        if isinstance(position, dict) and position.get("longitude") is not None:
            result[name] = float(position["longitude"])
    return result


def _cross_aspects(chart_a: dict[str, Any], chart_b: dict[str, Any]) -> list[dict[str, Any]]:
    """Major aspects between chart A's planets and chart B's planets, by orb."""
    from astronomy.calculator import DEFAULT_ORBS, MAJOR_ASPECTS, angular_separation

    points_a = _planet_longitudes(chart_a)
    points_b = _planet_longitudes(chart_b)
    events: list[dict[str, Any]] = []
    for name_a, longitude_a in points_a.items():
        for name_b, longitude_b in points_b.items():
            separation = angular_separation(longitude_a, longitude_b)
            best: tuple[str, float, float] | None = None
            for aspect, exact_angle in MAJOR_ASPECTS:
                orb = abs(separation - exact_angle)
                if orb <= DEFAULT_ORBS[aspect] and (best is None or orb < best[2]):
                    best = (aspect, exact_angle, orb)
            if best is None:
                continue
            aspect, _exact_angle, orb = best
            events.append(
                {
                    "body_a": name_a,
                    "body_b": name_b,
                    "aspect": aspect,
                    "separation": round(separation, 2),
                    "orb": round(orb, 2),
                }
            )
    events.sort(key=lambda item: (item["orb"], item["body_a"], item["body_b"]))
    return events


def _positions(chart: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Sign and degree of every synastry body for the summary block."""
    planets = chart.get("planets") or {}
    positions: dict[str, dict[str, Any]] = {}
    for name in SYNASTRY_BODIES:
        position = planets.get(name)
        if isinstance(position, dict) and position.get("sign"):
            positions[name] = {
                "sign": str(position["sign"]),
                "longitude": round(float(position.get("longitude") or 0.0), 1),
            }
    return positions


class SynastryService:
    """Compatibility reports resolved from stored profiles only."""

    def __init__(
        self,
        store: ChartStore,
        *,
        chart_builder: Callable[[BirthData], dict[str, Any]] | None = None,
        now_provider: Callable[[], datetime] | None = None,
        max_events: int = 20,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be greater than zero")
        self.store = store
        self.max_events = max_events
        self._chart_builder = chart_builder or default_natal_builder
        self._now_provider = now_provider or default_now_provider

    async def check_synastry(self, telegram_id: int | str, contact_id: int) -> dict[str, Any]:
        """Compatibility between the owner's chart and one contact's chart.

        This is the TZ function: the contact must belong to the account and
        the account must have premium access.
        """
        try:
            telegram = int(telegram_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"telegram_id must be an integer, got {telegram_id!r}") from exc
        try:
            contact_key = int(contact_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"contact_id must be an integer, got {contact_id!r}") from exc

        profile = await resolve_owner_profile(self.store, telegram_id=telegram)
        # The owner's own natal data is validated first: a profile without
        # birth data must get the onboarding hint, not the premium gate.
        owner_birth = birth_data_from_profile(profile)
        if not has_premium(profile, telegram):
            raise PremiumRequiredError(PREMIUM_REQUIRED_MESSAGE)
        contact = await self.store.get_contact(contact_key, user_id=profile.user_id)
        if contact is None:
            raise ContactPermissionError(CONTACT_PERMISSION_MESSAGE)
        contact_birth = birth_data_from_profile(contact)
        owner_chart = await asyncio.to_thread(self._chart_builder, owner_birth)
        contact_chart = await asyncio.to_thread(self._chart_builder, contact_birth)
        events = await asyncio.to_thread(_cross_aspects, owner_chart, contact_chart)
        now = self._now_provider()
        owner_name = profile.display_name or profile.first_name or "Владелец"
        return {
            "tool": "synastry",
            "telegram_id": telegram,
            "person_a": {
                "kind": "owner",
                "name": owner_name,
                "positions": _positions(owner_chart),
            },
            "person_b": {
                "kind": "contact",
                "name": contact.name,
                "relation": contact.relation,
                "positions": _positions(contact_chart),
            },
            "events": events[: self.max_events],
            "event_count": len(events),
            "generated_at": now.isoformat(),
        }
