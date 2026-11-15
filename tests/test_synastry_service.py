"""Unit tests of the synastry tool: premium gate, ownership, cross aspects."""

import asyncio
from datetime import date, time

import pytest

from services.profiles import ContactPermissionError, PremiumRequiredError, ProfileError
from services.synastry import SynastryService
from storage.store import ContactProfile, UserProfile

OWNER_LAT = 55.7558
CONTACT_LAT = 55.79


def owner_profile(**overrides) -> UserProfile:
    fields = dict(
        user_id=1,
        telegram_id=42,
        display_name="Анна",
        is_premium=True,
        birth_date=date(1990, 8, 15),
        birth_time=time(12, 0),
        birth_time_unknown=False,
        birth_lat=OWNER_LAT,
        birth_lon=37.6173,
        birth_timezone="Europe/Moscow",
        birth_city="Москва",
        birth_chart_hash="a" * 64,
    )
    fields.update(overrides)
    return UserProfile(**fields)


def contact_profile(contact_id=7, user_id=1) -> ContactProfile:
    return ContactProfile(
        contact_id=contact_id,
        user_id=user_id,
        name="Мария",
        relation="супруга",
        birth_date=date(1992, 3, 15),
        birth_time=time(14, 30),
        birth_time_unknown=False,
        birth_lat=CONTACT_LAT,
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


def chart_for(birth):
    """Deterministic charts, geometry chosen for exact aspects.

    owner: Sun 0°, Moon 120°; contact: Sun 60°, Moon 190°, Venus 5°.
    Expected cross aspects: Moon-Sun sextile (orb 0), Sun-Sun sextile (orb 0),
    Sun-Venus conjunction (orb 5), Moon-Venus trine (orb 5).
    """
    if abs(birth.latitude - CONTACT_LAT) < 0.01:
        longitudes = {"Sun": 60.0, "Moon": 190.0, "Venus": 5.0}
    else:
        longitudes = {"Sun": 0.0, "Moon": 120.0}
    return {
        "dt_utc": birth.dt_utc.isoformat(),
        "latitude": birth.latitude,
        "longitude": birth.longitude,
        "house_system": "P",
        "cusps": [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0],
        "asc": {"longitude": 0.0, "sign": "Aries", "house": 1},
        "mc": {"longitude": 90.0, "sign": "Cancer", "house": 10},
        "planets": {
            name: {
                "longitude": longitude,
                "sign": "stub",
                "house": 1,
                "speed": 1.0,
                "is_retrograde": False,
                "is_stationary": False,
            }
            for name, longitude in longitudes.items()
        },
        "aspects": [],
    }


def build_service(store, **overrides) -> SynastryService:
    return SynastryService(store, chart_builder=chart_for, **overrides)


def test_synastry_report_shape_and_top_event():
    store = FakeStore(owner_profile(), contacts=[contact_profile()])
    service = build_service(store)

    result = asyncio.run(service.check_synastry(42, 7))

    assert result["tool"] == "synastry"
    assert result["person_a"]["kind"] == "owner"
    assert result["person_a"]["name"] == "Анна"
    assert result["person_b"]["name"] == "Мария"
    assert result["person_b"]["relation"] == "супруга"

    orbs = [event["orb"] for event in result["events"]]
    assert orbs == sorted(orbs)
    # Moon 90° (owner) vs Sun 120° (contact): exact sextile, the tightest aspect.
    assert result["events"][0]["body_a"] == "Moon"
    assert result["events"][0]["body_b"] == "Sun"
    assert result["events"][0]["aspect"] == "sextile"
    assert result["events"][0]["orb"] == pytest.approx(0.0)

    pairs = {(event["body_a"], event["body_b"]) for event in result["events"]}
    assert ("Sun", "Venus") in pairs  # 0° vs 5°: conjunction within the orb
    assert ("Sun", "Sun") in pairs  # 0° vs 120°: trine

    positions = result["person_b"]["positions"]
    # Only the bodies present in the stub chart are reported.
    assert set(positions) == {"Sun", "Moon", "Venus"}

    # The pair identity feeds the interpretation cache key.
    assert len(result["chart_hash_a"]) == 64
    assert len(result["chart_hash_b"]) == 64
    assert result["natal_a"]["planets"]["Sun"]["sign"] == "stub"


def test_synastry_requires_premium():
    store = FakeStore(owner_profile(is_premium=False), contacts=[contact_profile()])
    service = build_service(store)

    with pytest.raises(PremiumRequiredError):
        asyncio.run(service.check_synastry(42, 7))


def test_synastry_rejects_foreign_contact():
    store = FakeStore(owner_profile(), contacts=[contact_profile(user_id=2)])
    service = build_service(store)

    with pytest.raises(ContactPermissionError):
        asyncio.run(service.check_synastry(42, 7))


def test_synastry_rejects_unknown_contact():
    store = FakeStore(owner_profile())
    service = build_service(store)

    with pytest.raises(ContactPermissionError):
        asyncio.run(service.check_synastry(42, 99))


def test_synastry_requires_owner_profile():
    empty = UserProfile(user_id=1, telegram_id=42)
    service = build_service(FakeStore(empty))

    with pytest.raises(ProfileError):
        asyncio.run(service.check_synastry(42, 7))


def test_synastry_validates_arguments():
    store = FakeStore(owner_profile())
    service = build_service(store)

    with pytest.raises(ValueError):
        asyncio.run(service.check_synastry("abc", 7))
    with pytest.raises(ValueError):
        asyncio.run(service.check_synastry(42, "abc"))
