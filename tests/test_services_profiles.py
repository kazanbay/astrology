"""Unit tests for profile persistence, premium permissions and the /contact parser."""

import asyncio
from datetime import date, datetime, time, timedelta, timezone

import pytest

from services.profiles import (
    CONTACT_COMMAND_USAGE,
    DEFAULT_PROFILE_CHANGE_LIMIT,
    ContactCommandInput,
    ProfileChangeLimitError,
    ProfileError,
    birth_data_from_profile,
    changes_used_this_month,
    ensure_birth_change_allowed,
    has_premium,
    natal_data_present,
    parse_contact_command,
    premium_telegram_ids,
    profile_change_limit,
    profile_changes_note,
    resolve_owner_profile,
    save_owner_profile,
)
from storage.birth import BirthData, BirthInputError
from storage.store import NatalProfile, UserProfile

OWNER_ID = 1


def owner_profile(**overrides) -> UserProfile:
    fields = dict(
        user_id=OWNER_ID,
        telegram_id=42,
        display_name="Анна",
        is_premium=False,
        birth_date=date(1990, 8, 15),
        birth_time=time(12, 30),
        birth_time_unknown=False,
        birth_lat=55.7558,
        birth_lon=37.6173,
        birth_timezone="Europe/Moscow",
        birth_city="Москва",
        birth_chart_hash="a" * 64,
        profile_changes_count=0,
        profile_changes_period=None,
    )
    fields.update(overrides)
    return UserProfile(**fields)


class FakeProfileStore:
    """Minimal ``ChartStore`` subset used by the profile layer."""

    def __init__(self, profile: UserProfile | None = None):
        self.profile = profile

    async def get_user_profile(self, *, telegram_id=None, email=None):
        if telegram_id is None and email is None:
            raise ValueError("identity required")
        return self.profile

    async def save_user_natal_profile(
        self,
        user_id,
        profile: NatalProfile,
        *,
        count_change: bool = False,
        max_changes_per_month: int | None = None,
    ) -> bool:
        if self.profile is None or self.profile.user_id != user_id:
            return False
        count = self.profile.profile_changes_count
        period = self.profile.profile_changes_period
        if count_change:
            today = datetime.now(timezone.utc).date()
            same_month = (
                period is not None and (period.year, period.month) == (today.year, today.month)
            )
            used = count if same_month else 0
            if max_changes_per_month is not None and used >= max_changes_per_month:
                return False
            count = used + 1
            period = today.replace(day=1)
        self.profile = owner_profile(
            user_id=self.profile.user_id,
            telegram_id=self.profile.telegram_id,
            display_name=self.profile.display_name,
            is_premium=self.profile.is_premium,
            birth_date=profile.birth_date,
            birth_time=profile.birth_time,
            birth_time_unknown=profile.birth_time_unknown,
            birth_lat=profile.birth_lat,
            birth_lon=profile.birth_lon,
            birth_timezone=profile.birth_timezone,
            birth_city=profile.birth_city,
            birth_chart_hash=profile.birth_chart_hash,
            profile_changes_count=count,
            profile_changes_period=period,
        )
        return True


BIRTH = BirthData(
    dt_utc=datetime(1990, 8, 15, 8, 30, tzinfo=timezone.utc),
    latitude=55.7558,
    longitude=37.6173,
    city="Москва, Россия",
    timezone="Europe/Moscow",
    datetime_local="1990-08-15T12:30:00",
)

CHANGED_BIRTH = BirthData(
    dt_utc=datetime(1990, 8, 15, 9, 30, tzinfo=timezone.utc),
    latitude=55.7558,
    longitude=37.6173,
    city="Москва, Россия",
    timezone="Europe/Moscow",
    datetime_local="1990-08-15T13:30:00",
)


def money_month_start() -> date:
    return datetime.now(timezone.utc).date().replace(day=1)


def test_save_owner_profile_persists_every_field():
    store = FakeProfileStore(profile=owner_profile(birth_date=None, birth_time=None))

    asyncio.run(
        save_owner_profile(store, OWNER_ID, BIRTH, time_unknown=False, profile=store.profile)
    )

    profile = store.profile
    assert profile.birth_date == date(1990, 8, 15)
    assert profile.birth_time == time(12, 30)
    assert profile.birth_time_unknown is False
    assert profile.birth_lat == 55.7558
    assert profile.birth_lon == 37.6173
    assert profile.birth_timezone == "Europe/Moscow"
    assert profile.birth_city == "Москва, Россия"
    assert len(profile.birth_chart_hash) == 64


def test_save_owner_profile_marks_unknown_time():
    store = FakeProfileStore(profile=owner_profile())

    asyncio.run(
        save_owner_profile(store, OWNER_ID, BIRTH, time_unknown=True, profile=store.profile)
    )

    assert store.profile.birth_time_unknown is True
    assert store.profile.birth_time == time(12, 30)


def test_birth_data_from_profile_converts_local_time():
    birth = birth_data_from_profile(owner_profile())

    # 12:30 Europe/Moscow on 1990-08-15 is 08:30 UTC (matches the wizard test).
    assert birth.dt_utc.isoformat() == "1990-08-15T08:30:00+00:00"
    assert birth.city == "Москва"
    assert birth.timezone == "Europe/Moscow"
    assert birth.datetime_local == "1990-08-15T12:30:00"


def test_birth_data_from_unknown_time_falls_back_to_noon():
    profile = owner_profile(birth_time=None, birth_time_unknown=True)

    birth = birth_data_from_profile(profile)

    assert birth.dt_utc.isoformat() == "1990-08-15T08:00:00+00:00"
    assert birth.datetime_local == "1990-08-15T12:00:00"


def test_birth_data_requires_natal_fields():
    empty = owner_profile(birth_date=None, birth_lat=None)
    assert natal_data_present(empty) is False
    with pytest.raises(ProfileError):
        birth_data_from_profile(empty)


def test_resolve_owner_profile_found_and_missing():
    store = FakeProfileStore(profile=owner_profile())
    profile = asyncio.run(resolve_owner_profile(store, telegram_id=42))
    assert profile.user_id == OWNER_ID

    with pytest.raises(ProfileError):
        asyncio.run(resolve_owner_profile(FakeProfileStore(profile=None), telegram_id=42))

    with pytest.raises(ProfileError):
        asyncio.run(resolve_owner_profile(store))


def test_has_premium_uses_database_flag():
    assert has_premium(owner_profile(is_premium=True)) is True
    assert has_premium(owner_profile(is_premium=False)) is False


def test_has_premium_uses_the_paid_period():
    future = datetime.now(timezone.utc) + timedelta(days=10)
    past = datetime.now(timezone.utc) - timedelta(days=1)

    assert has_premium(owner_profile(premium_until=future)) is True
    assert has_premium(owner_profile(premium_until=past)) is False


def test_has_premium_environment_override(monkeypatch):
    monkeypatch.setenv("PREMIUM_TELEGRAM_IDS", " 42, abc;99 ")
    assert premium_telegram_ids() == frozenset({42, 99})
    assert has_premium(owner_profile(is_premium=False)) is True

    monkeypatch.setenv("PREMIUM_TELEGRAM_IDS", "")
    assert premium_telegram_ids() == frozenset()
    assert has_premium(owner_profile(is_premium=False)) is False


def test_parse_contact_command_full_form():
    parsed = parse_contact_command("Мария | супруга | 15.03.1992 14:30 | Казань")

    assert isinstance(parsed, ContactCommandInput)
    assert parsed.name == "Мария"
    assert parsed.relation == "супруга"
    assert parsed.birth_date == date(1992, 3, 15)
    assert parsed.birth_time == time(14, 30)
    assert parsed.time_unknown is False
    assert parsed.city == "Казань"


def test_parse_contact_command_without_relation_and_time():
    parsed = parse_contact_command("Олег | 02.11.1985 | Санкт-Петербург")
    assert parsed.relation is None
    assert parsed.birth_time is None
    assert parsed.time_unknown is True

    marked = parse_contact_command("Пётр | друг | 02.11.1985 не знаю | Казань")
    assert marked.time_unknown is True


def test_parse_contact_command_rejects_bad_input():
    with pytest.raises(BirthInputError):
        parse_contact_command("")
    with pytest.raises(BirthInputError):
        parse_contact_command("Мария | 15.03.1992")
    with pytest.raises(BirthInputError):
        parse_contact_command(" | 15.03.1992 | Казань")
    with pytest.raises(BirthInputError):
        parse_contact_command("Мария | 15.13.1992 | Казань")
    with pytest.raises(BirthInputError):
        parse_contact_command("Мария | 15.03.1992 | ")


def test_usage_hint_is_stable():
    assert CONTACT_COMMAND_USAGE == "/contact Мария | супруга | 15.03.1992 14:30 | Казань"


def test_unchanged_profile_spends_no_change():
    store = FakeProfileStore(profile=owner_profile())

    assert asyncio.run(save_owner_profile(store, OWNER_ID, BIRTH, profile=store.profile)) == "saved"
    assert store.profile.profile_changes_count == 1

    assert (
        asyncio.run(save_owner_profile(store, OWNER_ID, BIRTH, profile=store.profile))
        == "unchanged"
    )
    assert store.profile.profile_changes_count == 1


def test_change_limit_blocks_further_changes(monkeypatch):
    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "1")
    store = FakeProfileStore(profile=owner_profile())

    assert (
        asyncio.run(save_owner_profile(store, OWNER_ID, CHANGED_BIRTH, profile=store.profile))
        == "saved"
    )
    with pytest.raises(ProfileChangeLimitError):
        asyncio.run(save_owner_profile(store, OWNER_ID, BIRTH, profile=store.profile))


def test_limit_resets_in_a_new_month(monkeypatch):
    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "1")
    store = FakeProfileStore(
        profile=owner_profile(profile_changes_count=5, profile_changes_period=date(2000, 1, 1))
    )

    assert (
        asyncio.run(save_owner_profile(store, OWNER_ID, CHANGED_BIRTH, profile=store.profile))
        == "saved"
    )
    assert store.profile.profile_changes_count == 1


def test_zero_limit_disables_the_cap(monkeypatch):
    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "0")
    store = FakeProfileStore(
        profile=owner_profile(
            profile_changes_count=99, profile_changes_period=money_month_start()
        )
    )

    assert (
        asyncio.run(save_owner_profile(store, OWNER_ID, CHANGED_BIRTH, profile=store.profile))
        == "saved"
    )


def test_profile_change_limit_env_parsing(monkeypatch):
    monkeypatch.delenv("PROFILE_CHANGES_PER_MONTH", raising=False)
    assert profile_change_limit() == DEFAULT_PROFILE_CHANGE_LIMIT

    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "5")
    assert profile_change_limit() == 5

    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "0")
    assert profile_change_limit() == 0

    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "abc")
    assert profile_change_limit() == DEFAULT_PROFILE_CHANGE_LIMIT


def test_ensure_change_allowed_skips_unchanged_and_initial(monkeypatch):
    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "3")
    crowded = owner_profile(
        profile_changes_count=3,
        profile_changes_period=money_month_start(),
        birth_city="Москва, Россия",
    )

    # The stored data already matches: nothing to spend, nothing to raise.
    ensure_birth_change_allowed(crowded, BIRTH, time_unknown=False)
    # The very first save is always allowed.
    ensure_birth_change_allowed(owner_profile(birth_date=None), CHANGED_BIRTH, time_unknown=False)

    with pytest.raises(ProfileChangeLimitError):
        ensure_birth_change_allowed(crowded, CHANGED_BIRTH, time_unknown=False)


def test_changes_counter_and_note(monkeypatch):
    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "3")
    profile = owner_profile(
        profile_changes_count=2, profile_changes_period=money_month_start()
    )

    assert changes_used_this_month(profile) == 2
    assert profile_changes_note(profile) == (
        "🔄 Изменений данных рождения в этом месяце: 2 из 3"
    )

    old = owner_profile(profile_changes_count=2, profile_changes_period=date(2000, 1, 1))
    assert changes_used_this_month(old) == 0

    monkeypatch.setenv("PROFILE_CHANGES_PER_MONTH", "0")
    assert profile_changes_note(profile) is None
