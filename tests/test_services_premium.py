"""Unit tests of the premium plans config and status texts."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.premium import (
    PremiumConfigError,
    format_invite_text,
    format_premium_history,
    format_premium_status,
    load_catalog,
)
from storage.store import UserProfile


def write_plans(tmp_path, payload):
    path = tmp_path / "plans.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_catalog_reads_plans_and_sorts_by_months(tmp_path):
    path = write_plans(
        tmp_path,
        {
            "referral_bonus_months": 2,
            "plans": {
                "12m": {"months": 12, "stars": 800, "title": "Год"},
                "1m": {"months": 1, "stars": 100},
            },
        },
    )

    catalog = load_catalog(path)

    assert catalog.referral_bonus_months == 2
    assert [plan.months for plan in catalog.ordered()] == [1, 12]
    assert catalog.get("1m").stars == 100
    assert catalog.get("1m").title == "1 мес."
    assert catalog.get("nope") is None


def test_env_override_sets_the_path(tmp_path, monkeypatch):
    path = write_plans(tmp_path, {"plans": {"1m": {"months": 1, "stars": 5}}})
    monkeypatch.setenv("PREMIUM_PLANS_PATH", str(path))

    catalog = load_catalog()

    assert catalog.get("1m").stars == 5


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"plans": {}},
        {"plans": {"1m": {"months": 0, "stars": 10}}},
        {"plans": {"1m": {"months": 1, "stars": 0}}},
        {"plans": "nope"},
        {"plans": {"1m": {"months": 1, "stars": 5}}, "referral_bonus_months": -1},
    ],
)
def test_invalid_config_is_rejected(tmp_path, payload):
    path = write_plans(tmp_path, payload)

    with pytest.raises(PremiumConfigError):
        load_catalog(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(PremiumConfigError):
        load_catalog(tmp_path / "missing.json")


def profile(**overrides):
    fields = dict(user_id=1, telegram_id=42, is_premium=False, birth_timezone="Europe/Moscow")
    fields.update(overrides)
    return UserProfile(**fields)


def test_status_texts():
    future = datetime.now(timezone.utc) + timedelta(days=5)
    past = datetime.now(timezone.utc) - timedelta(days=5)

    assert "до" in format_premium_status(profile(premium_until=future))
    assert "не активен" in format_premium_status(profile(premium_until=past))
    assert "не активен" in format_premium_status(profile())
    assert "бессрочно" in format_premium_status(profile(is_premium=True))


def test_invite_text_mentions_reward_and_stats():
    text = format_invite_text(
        link="https://t.me/bot?start=ref_x",
        invited=2,
        bonus_months=2,
        reward_months=1,
    )

    assert "https://t.me/bot?start=ref_x" in text
    assert "1 мес" in text
    assert "приглашено: 2" in text
    assert "2 мес" in text


def test_shipped_catalog_is_valid():
    """The repository config must always parse (it ships inside the image)."""
    from services.premium import DEFAULT_PLANS_PATH

    catalog = load_catalog(DEFAULT_PLANS_PATH)

    assert catalog.get("1m") is not None
    assert [plan.months for plan in catalog.ordered()] == sorted(
        plan.months for plan in catalog.ordered()
    )


def test_premium_history_lists_source_term_and_period():
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    events = [
        SimpleNamespace(
            source="telegram_stars",
            months=3,
            stars=250,
            plan_key="3m",
            reason="Тариф «3 месяца» (3m)",
            created_at=now,
            premium_until_after=now + timedelta(days=90),
        ),
        SimpleNamespace(
            source="referral",
            months=1,
            stars=None,
            plan_key=None,
            reason=None,
            created_at=now - timedelta(days=10),
            premium_until_after=now - timedelta(days=10) + timedelta(days=30),
        ),
    ]

    text = format_premium_history(events, profile())

    assert text.startswith("📜 История премиума:")
    assert "оплата звёздами: +3 мес., 250 ⭐" in text
    assert "3m" in text
    assert "реферал: +1 мес." in text


def test_empty_history_text():
    assert format_premium_history([], profile()) == "📜 История премиума пуста."
