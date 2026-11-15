"""Unit tests of the digest schedule (Moscow time, no clock dependency)."""

from datetime import datetime
from zoneinfo import ZoneInfo

from services.digest import next_daily, next_weekly

MSK = ZoneInfo("Europe/Moscow")


def msk(*args: int) -> datetime:
    return datetime(*args, tzinfo=MSK)


def test_next_daily_same_day_before_21():
    assert next_daily(msk(2026, 11, 15, 20, 59)) == msk(2026, 11, 15, 21, 0)


def test_next_daily_after_21_rolls_to_tomorrow():
    assert next_daily(msk(2026, 11, 15, 21, 0, 1)) == msk(2026, 11, 16, 21, 0)


def test_next_weekly_sunday_before_20():
    assert next_weekly(msk(2026, 11, 15, 19, 59)) == msk(2026, 11, 15, 20, 0)


def test_next_weekly_monday_rolls_to_next_sunday():
    assert next_weekly(msk(2026, 11, 16, 10, 0)) == msk(2026, 11, 22, 20, 0)


def test_next_weekly_sunday_after_20_rolls_a_week():
    assert next_weekly(msk(2026, 11, 15, 20, 1)) == msk(2026, 11, 22, 20, 0)
