"""Premium plans, Telegram Stars prices and status texts.

The offer lives in ``config/premium_plans.json`` (``PREMIUM_PLANS_PATH``
overrides the location), so changing prices or adding a term never touches the
code. The catalog is validated on load: a broken file disables the shop but
never crashes the bot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANS_PATH = PROJECT_ROOT / "config" / "premium_plans.json"
PLANS_PATH_ENV = "PREMIUM_PLANS_PATH"


class PremiumConfigError(RuntimeError):
    """The premium plans file is missing or invalid."""


@dataclass(frozen=True)
class PremiumPlan:
    """One purchasable term: months of access for a number of Stars."""

    key: str
    months: int
    stars: int
    title: str


@dataclass(frozen=True)
class PremiumCatalog:
    """Validated plans plus the referral reward."""

    plans: dict[str, PremiumPlan]
    referral_bonus_months: int

    def get(self, key: str) -> PremiumPlan | None:
        return self.plans.get(str(key))

    def ordered(self) -> list[PremiumPlan]:
        """Plans sorted by term: shortest first, the way the keyboard shows them."""
        return sorted(self.plans.values(), key=lambda plan: (plan.months, plan.key))


def _as_int(value: Any, field: str, *, minimum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PremiumConfigError(f"{field} must be an integer, got {value!r}") from exc
    if number < minimum:
        raise PremiumConfigError(f"{field} must be >= {minimum}, got {number}")
    return number


def load_catalog(path: str | Path | None = None) -> PremiumCatalog:
    """Read and validate the premium plans file."""
    resolved = Path(path or os.getenv(PLANS_PATH_ENV, "").strip() or DEFAULT_PLANS_PATH)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PremiumConfigError(f"premium plans file not found: {resolved}") from exc
    except ValueError as exc:
        raise PremiumConfigError(f"premium plans file is not valid JSON: {resolved}") from exc
    if not isinstance(raw, dict):
        raise PremiumConfigError("premium plans file must contain a JSON object")
    plans_raw = raw.get("plans")
    if not isinstance(plans_raw, dict) or not plans_raw:
        raise PremiumConfigError("'plans' must be a non-empty object")
    plans: dict[str, PremiumPlan] = {}
    for key, value in plans_raw.items():
        if not isinstance(value, dict):
            raise PremiumConfigError(f"plan {key!r} must be an object")
        plans[str(key)] = PremiumPlan(
            key=str(key),
            months=_as_int(value.get("months"), f"plan {key!r} months", minimum=1),
            stars=_as_int(value.get("stars"), f"plan {key!r} stars", minimum=1),
            title=str(value.get("title") or f"{value.get('months')} мес."),
        )
    bonus = _as_int(
        raw.get("referral_bonus_months", 1), "referral_bonus_months", minimum=0
    )
    return PremiumCatalog(plans=plans, referral_bonus_months=bonus)


def _local_moment(profile: Any, moment: datetime) -> datetime:
    """Shift an instant into the profile's zone for display."""
    tz_name = getattr(profile, "birth_timezone", None)
    target: Any = timezone.utc
    if tz_name:
        try:
            target = ZoneInfo(str(tz_name))
        except Exception:  # pragma: no cover - zones come from timezonefinder
            target = timezone.utc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(target)


def format_premium_status(profile: Any, *, now: datetime | None = None) -> str:
    """One-line premium state of the account."""
    until = getattr(profile, "premium_until", None)
    if getattr(profile, "is_premium", False):
        return "👑 Премиум активен бессрочно (ручное начисление)."
    moment_now = now or datetime.now(timezone.utc)
    if until is not None:
        moment = _local_moment(profile, until)
        if moment > moment_now:
            return f"👑 Премиум активен до {moment.strftime('%d.%m.%Y')}."
    return (
        "👑 Премиум не активен. Бесплатно доступны разбор карты и вопросы по книгам; "
        "прогнозы для других людей и синастрия — в премиуме."
    )


def format_invite_text(
    *,
    link: str,
    invited: int,
    bonus_months: int,
    reward_months: int,
) -> str:
    """Referral block for /invite and the premium screen."""
    lines = [
        "🔗 Ваша реферальная ссылка:",
        link,
        "",
        f"🎁 За каждого друга, который придёт по вашей ссылке, вы получаете "
        f"{reward_months} мес. премиума — бесплатно.",
        f"Уже приглашено: {invited} · начислено: {bonus_months} мес.",
    ]
    return "\n".join(lines)


SOURCE_LABELS_RU: dict[str, str] = {
    "manual": "вручную",
    "referral": "реферал",
    "telegram_stars": "оплата звёздами",
    "other": "прочее",
}


def format_premium_history(
    events: list[Any], profile: Any, *, max_items: int = 10
) -> str:
    """Audit trail of premium grants: when, why, for how long, by what."""
    if not events:
        return "📜 История премиума пуста."
    lines = ["📜 История премиума:"]
    for index, event in enumerate(events[:max_items], start=1):
        source = SOURCE_LABELS_RU.get(str(event.source), str(event.source))
        created = event.created_at
        when_text = _local_moment(profile, created).strftime("%d.%m.%Y") if created else "—"
        line = f"{index}. {when_text} — {source}: +{event.months} мес."
        if event.stars:
            line += f", {event.stars} ⭐"
        if event.plan_key:
            line += f" ({event.plan_key})"
        if event.premium_until_after is not None:
            until = _local_moment(profile, event.premium_until_after).strftime("%d.%m.%Y")
            line += f" → до {until}"
        if event.reason:
            line += f" — {event.reason}"
        lines.append(line)
    return "\n".join(lines)
