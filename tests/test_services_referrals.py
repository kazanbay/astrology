"""Unit tests of referral codes and attribution."""

import asyncio

from services.referrals import (
    apply_referral,
    get_or_create_referral_code,
    parse_referral_payload,
    referral_link,
)


class FakeReferralStore:
    """In-memory store for the referral layer."""

    def __init__(self, referrer=None, code=None):
        self.code = code
        self.referrer = referrer  # (user_id, telegram_id)
        self.referrals = []
        self.grants = []

    async def get_referral_code(self, user_id):
        return self.code

    async def ensure_referral_code(self, user_id, code):
        if self.code is not None:
            return None
        self.code = code
        return code

    async def find_user_by_referral_code(self, code):
        return self.referrer

    async def create_referral(self, *, referrer_user_id, referee_user_id, bonus_months):
        key = (referrer_user_id, referee_user_id)
        if key in self.referrals:
            return False
        self.referrals.append(key)
        return True

    async def grant_premium(self, user_id, months, **kwargs):
        self.grants.append((user_id, months, kwargs))
        return None


def test_referral_link_and_payload():
    assert referral_link("astro_bot", "abc123") == "https://t.me/astro_bot?start=ref_abc123"
    assert parse_referral_payload("ref_abc123") == "abc123"
    assert parse_referral_payload("other") is None
    assert parse_referral_payload("ref_") is None
    assert parse_referral_payload(None) is None


def test_code_is_created_once_and_reused():
    store = FakeReferralStore()

    first = asyncio.run(get_or_create_referral_code(store, 1))
    second = asyncio.run(get_or_create_referral_code(store, 1))

    assert first == second
    assert 0 < len(first) <= 16


def test_apply_referral_grants_the_bonus():
    store = FakeReferralStore(referrer=(5, 555))

    outcome = asyncio.run(apply_referral(store, referee_user_id=1, code="c", bonus_months=1))

    assert outcome.status == "granted"
    assert outcome.referrer_telegram_id == 555
    assert store.grants[0][0] == 5
    assert store.grants[0][1] == 1
    assert store.grants[0][2]["source"] == "referral"
    assert store.grants[0][2]["actor_user_id"] == 1
    assert store.grants[0][2]["reference"] == "c"

    again = asyncio.run(apply_referral(store, referee_user_id=1, code="c", bonus_months=1))

    assert again.status == "skipped"
    assert again.reason == "already_referred"
    assert len(store.grants) == 1


def test_self_and_unknown_referrals_are_skipped():
    store = FakeReferralStore(referrer=(1, 42))

    outcome = asyncio.run(apply_referral(store, referee_user_id=1, code="c", bonus_months=1))

    assert outcome.status == "skipped" and outcome.reason == "self_referral"
    assert store.grants == []

    unknown = asyncio.run(
        apply_referral(FakeReferralStore(), referee_user_id=2, code="c", bonus_months=1)
    )

    assert unknown.status == "skipped" and unknown.reason == "unknown_code"
