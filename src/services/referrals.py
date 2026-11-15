"""Referral links: short codes, first-touch attribution and the inviter bonus.

Attribution is stored in ``referrals``: one row per invited account (the unique
index makes a double reward impossible), written only on the referee's first
contact with the bot. The bonus is granted to the inviter immediately after the
row is created.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

from storage.store import ChartStore

logger = logging.getLogger(__name__)

REFERRAL_PREFIX = "ref_"
CODE_LENGTH = 10
CODE_ATTEMPTS = 5


def referral_link(bot_username: str, code: str) -> str:
    """Public deep link that carries the inviter's code."""
    return f"https://t.me/{bot_username}?start={REFERRAL_PREFIX}{code}"


def parse_referral_payload(payload: str | None) -> str | None:
    """Extract the referral code from a ``/start`` payload, if it is one."""
    text = (payload or "").strip()
    if not text.startswith(REFERRAL_PREFIX):
        return None
    code = text[len(REFERRAL_PREFIX):].strip()
    return code or None


async def get_or_create_referral_code(store: ChartStore, user_id: int) -> str:
    """Return the account's code, generating a unique one on first use."""
    code = await store.get_referral_code(user_id)
    if code:
        return code
    for _ in range(CODE_ATTEMPTS):
        candidate = secrets.token_urlsafe(8)[:CODE_LENGTH]
        created = await store.ensure_referral_code(user_id, candidate)
        if created:
            return created
        current = await store.get_referral_code(user_id)
        if current:
            return current
    raise RuntimeError("could not allocate a referral code")


@dataclass(frozen=True)
class ReferralOutcome:
    """What a ``/start ref_...`` produced."""

    status: str  # "granted" | "skipped"
    reason: str | None = None
    referrer_telegram_id: int | None = None
    bonus_months: int = 0


async def apply_referral(
    store: ChartStore,
    *,
    referee_user_id: int,
    code: str,
    bonus_months: int,
) -> ReferralOutcome:
    """Attribute a new account to the inviter and pay the bonus once."""
    referrer = await store.find_user_by_referral_code(code)
    if referrer is None:
        return ReferralOutcome(status="skipped", reason="unknown_code")
    referrer_user_id, referrer_telegram_id = referrer
    if referrer_user_id == referee_user_id:
        return ReferralOutcome(status="skipped", reason="self_referral")
    created = await store.create_referral(
        referrer_user_id=referrer_user_id,
        referee_user_id=referee_user_id,
        bonus_months=bonus_months,
    )
    if not created:
        return ReferralOutcome(status="skipped", reason="already_referred")
    await store.grant_premium(
        referrer_user_id,
        bonus_months,
        source="referral",
        actor_user_id=referee_user_id,
        reference=code,
        reason="Реферальный бонус за приглашённого пользователя",
    )
    return ReferralOutcome(
        status="granted",
        referrer_telegram_id=referrer_telegram_id,
        bonus_months=bonus_months,
    )
