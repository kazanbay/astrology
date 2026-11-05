import pytest

from storage.birth import BirthInputError
from storage.hashing import CHART_HASH_LENGTH, generate_chart_hash


BIRTH_UTC = "1990-08-15T08:00:00+00:00"
MOSCOW = (55.7558, 37.6173)


def build_hash(**overrides):
    payload = {
        "dt_utc": BIRTH_UTC,
        "lat": MOSCOW[0],
        "lon": MOSCOW[1],
        "house_system": "P",
        "language": "ru",
    }
    payload.update(overrides)
    return generate_chart_hash(**payload)


def test_hash_is_sha256_hex_of_fixed_length():
    value = build_hash()

    assert len(value) == CHART_HASH_LENGTH == 64
    assert all(character in "0123456789abcdef" for character in value)


def test_hash_is_deterministic():
    assert build_hash() == build_hash()


def test_hash_accepts_equivalent_instant_forms():
    from datetime import datetime, timedelta, timezone

    aware = datetime(1990, 8, 15, 12, 0, tzinfo=timezone(timedelta(hours=4)))

    assert build_hash(dt_utc=aware) == build_hash(dt_utc="1990-08-15T08:00:00Z")
    assert build_hash(dt_utc=datetime(1990, 8, 15, 8, 0, tzinfo=timezone.utc)) == build_hash()


def test_hash_rounds_coordinates_to_four_decimals():
    # Geocoders return slightly different coordinates for the same city; those
    # micro-offsets must not split one chart into several cache rows.
    assert build_hash(lat=55.75583, lon=37.61734) == build_hash()


def test_hash_distinguishes_fourth_decimal():
    assert build_hash(lat=55.7559) != build_hash()


def test_hash_depends_on_house_system_language_and_instant():
    base = build_hash()

    assert build_hash(house_system="W") != base
    assert build_hash(language="en") != base
    assert build_hash(dt_utc="1990-08-15T08:00:01+00:00") != base


def test_hash_normalizes_house_system_and_language_case():
    assert build_hash(house_system="p", language="RU") == build_hash()


def test_hash_collapses_negative_zero():
    assert build_hash(lat=-0.0, lon=0.0) == build_hash(lat=0.0, lon=0.0)


def test_hash_rejects_naive_datetime():
    with pytest.raises(BirthInputError):
        build_hash(dt_utc="1990-08-15T08:00:00")


def test_hash_rejects_coordinates_out_of_range():
    with pytest.raises(BirthInputError):
        build_hash(lat=91.0)
    with pytest.raises(BirthInputError):
        build_hash(lon=181.0)


def test_hash_rejects_unknown_house_system_code():
    with pytest.raises(BirthInputError):
        build_hash(house_system="Placidus")
