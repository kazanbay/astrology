"""Unit tests of the forecast tool: permissions, intervals, event aggregation."""

import asyncio
from datetime import date, datetime, time, timedelta, timezone

import pytest

from services.forecast import ForecastService, IntervalError
from services.profiles import ContactPermissionError, PremiumRequiredError, ProfileError
from storage.store import ContactProfile, UserProfile

FIXED_NOW = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def owner_profile(**overrides) -> UserProfile:
    fields = dict(
        user_id=1,
        telegram_id=42,
        display_name="Анна",
        is_premium=False,
        birth_date=date(1990, 8, 15),
        birth_time=time(12, 0),
        birth_time_unknown=False,
        birth_lat=55.7558,
        birth_lon=37.6173,
        birth_timezone="Europe/Moscow",
        birth_city="Москва",
        birth_chart_hash="a" * 64,
    )
    fields.update(overrides)
    return UserProfile(**fields)


def contact_profile(contact_id=7, user_id=1, name="Мария") -> ContactProfile:
    return ContactProfile(
        contact_id=contact_id,
        user_id=user_id,
        name=name,
        relation="супруга",
        birth_date=date(1992, 3, 15),
        birth_time=time(14, 30),
        birth_time_unknown=False,
        birth_lat=55.79,
        birth_lon=49.11,
        birth_timezone="Europe/Moscow",
        birth_city="Казань",
    )


class FakeStore:
    def __init__(self, profile: UserProfile, contacts=()):
        self.profile = profile
        self.contacts = {contact.contact_id: contact for contact in contacts}

    async def get_user_profile(self, *, telegram_id=None, email=None):
        if self.profile is not None and self.profile.telegram_id == telegram_id:
            return self.profile
        return None

    async def get_contact(self, contact_id, user_id):
        contact = self.contacts.get(contact_id)
        if contact is not None and contact.user_id == user_id:
            return contact
        return None


def fixed_chart(birth):
    """Natal chart stub: Sun at 0° Aries, Moon at 90° (Cancer)."""
    return {
        "dt_utc": birth.dt_utc.isoformat(),
        "latitude": birth.latitude,
        "longitude": birth.longitude,
        "house_system": "P",
        "cusps": [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0],
        "asc": {"longitude": 0.0, "sign": "Aries", "house": 1},
        "mc": {"longitude": 90.0, "sign": "Cancer", "house": 10},
        "planets": {
            "Sun": {
                "longitude": 0.0,
                "sign": "Aries",
                "house": 1,
                "speed": 1.0,
                "is_retrograde": False,
                "is_stationary": False,
            },
            "Moon": {
                "longitude": 90.0,
                "sign": "Cancer",
                "house": 4,
                "speed": 13.0,
                "is_retrograde": False,
                "is_stationary": False,
            },
        },
        "aspects": [],
    }


def sampler_factory(sun_speed=0.5, sun_start=1.5, moon_speed=-14.0, moon_start=89.0):
    """Deterministic fake ephemeris: linear motion from FIXED_NOW.

    The Moon moves away from the natal Moon (89° -> 75°/day 1), so the minimal
    Moon orb stays at the first sample (1.0°) and the assertion is stable.
    """

    def sampler(moment: datetime) -> dict[str, float]:
        days = (moment - FIXED_NOW).total_seconds() / 86400.0
        return {
            "Sun": (sun_start + sun_speed * days) % 360.0,
            "Moon": (moon_start + moon_speed * days) % 360.0,
        }

    return sampler


def build_service(store, **overrides) -> ForecastService:
    return ForecastService(
        store,
        chart_builder=fixed_chart,
        longitude_sampler=sampler_factory(**overrides.pop("sampler", {})),
        now_provider=lambda: FIXED_NOW,
        **overrides,
    )


def test_invalid_interval_is_rejected():
    store = FakeStore(owner_profile())
    service = build_service(store)

    with pytest.raises(IntervalError):
        asyncio.run(service.get_forecast_for_interval(42, "decade"))


def test_bad_telegram_id_is_rejected():
    store = FakeStore(owner_profile())
    service = build_service(store)

    with pytest.raises(ValueError):
        asyncio.run(service.get_forecast_for_interval("abc", "day"))


def test_missing_profile_raises():
    service = build_service(FakeStore(owner_profile(telegram_id=999)))

    with pytest.raises(ProfileError):
        asyncio.run(service.get_forecast_for_interval(42, "day"))


def test_profile_without_natal_data_raises():
    empty = UserProfile(user_id=1, telegram_id=42, display_name="Анна")
    service = build_service(FakeStore(empty))

    with pytest.raises(ProfileError):
        asyncio.run(service.get_forecast_for_interval(42, "day"))


def test_owner_day_forecast_includes_moon_and_is_deterministic():
    store = FakeStore(owner_profile())
    service = build_service(store)

    first = asyncio.run(service.get_forecast_for_interval(42, "day"))
    second = asyncio.run(service.get_forecast_for_interval(42, "day"))

    assert first["tool"] == "forecast"
    assert first["interval"] == "day"
    assert first["person"]["kind"] == "owner"
    assert first["person"]["name"] == "Анна"
    assert first == second  # injected now and sampler make the tool deterministic

    pairs = {(event["transit"], event["natal"], event["aspect"]) for event in first["events"]}
    assert ("Sun", "Sun", "conjunction") in pairs
    assert ("Moon", "Moon", "conjunction") in pairs

    # The transit snapshot carries signs and houses against the natal cusps.
    snapshot = {entry["name"]: entry for entry in first["now_transits"]}
    assert snapshot["Sun"]["sign"] == "Aries"
    assert snapshot["Sun"]["house"] == 1


def test_week_forecast_excludes_moon():
    store = FakeStore(owner_profile())
    service = build_service(store)

    result = asyncio.run(service.get_forecast_for_interval(42, "week"))

    assert result["range"]["days"] == 7
    assert {event["transit"] for event in result["events"]} == {"Sun"}


def test_events_are_sorted_by_orb():
    store = FakeStore(owner_profile())
    service = build_service(store)

    result = asyncio.run(service.get_forecast_for_interval(42, "day"))

    orbs = [event["orb"] for event in result["events"]]
    assert orbs == sorted(orbs)
    # The Moon starts 1° from the natal Moon, the Sun 1.5° from the natal Sun.
    assert result["events"][0]["transit"] == "Moon"
    assert result["events"][0]["orb"] == pytest.approx(1.0)


def test_max_events_caps_the_report():
    store = FakeStore(owner_profile())
    service = build_service(store, max_events=1)

    result = asyncio.run(service.get_forecast_for_interval(42, "day"))

    assert len(result["events"]) == 1
    assert result["event_count"] > 1


def test_contact_forecast_requires_premium():
    store = FakeStore(owner_profile(is_premium=False), contacts=[contact_profile()])
    service = build_service(store)

    with pytest.raises(PremiumRequiredError):
        asyncio.run(service.get_forecast_for_interval(42, "week", contact_id=7))


def test_contact_forecast_for_premium_owner():
    store = FakeStore(owner_profile(is_premium=True), contacts=[contact_profile()])
    service = build_service(store)

    result = asyncio.run(service.get_forecast_for_interval(42, "week", contact_id=7))

    assert result["person"] == {"kind": "contact", "name": "Мария", "relation": "супруга"}
    assert result["birth"]["city"] == "Казань"


def test_foreign_contact_is_rejected_even_for_premium():
    store = FakeStore(owner_profile(is_premium=True), contacts=[contact_profile(user_id=2)])
    service = build_service(store)

    with pytest.raises(ContactPermissionError):
        asyncio.run(service.get_forecast_for_interval(42, "week", contact_id=7))


def test_unknown_contact_is_rejected():
    store = FakeStore(owner_profile(is_premium=True))
    service = build_service(store)

    with pytest.raises(ContactPermissionError):
        asyncio.run(service.get_forecast_for_interval(42, "week", contact_id=99))


def test_unknown_birth_time_is_reported_in_the_result():
    profile = owner_profile(birth_time=None, birth_time_unknown=True)
    store = FakeStore(profile)
    service = build_service(store)

    result = asyncio.run(service.get_forecast_for_interval(42, "day"))

    assert result["birth"]["time_unknown"] is True


def test_year_and_month_ranges():
    store = FakeStore(owner_profile())
    service = build_service(store)

    month = asyncio.run(service.get_forecast_for_interval(42, "month"))
    year = asyncio.run(service.get_forecast_for_interval(42, "year"))

    assert month["range"]["days"] == 30
    assert year["range"]["days"] == 365


def test_max_events_must_be_positive():
    with pytest.raises(ValueError):
        build_service(FakeStore(owner_profile()), max_events=0)
