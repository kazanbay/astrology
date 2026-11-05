from datetime import datetime, timedelta, timezone

import pytest

from storage.birth import (
    BirthData,
    BirthInputError,
    canonical_utc_string,
    local_to_utc,
    parse_birth_command,
    parse_local_datetime,
)


def test_parse_birth_command_reads_date_time_and_city():
    moment, city = parse_birth_command("1990-08-15 12:00 Москва")

    assert moment == datetime(1990, 8, 15, 12, 0)
    assert moment.tzinfo is None
    assert city == "Москва"


def test_parse_birth_command_accepts_iso_datetime_and_multiword_city():
    moment, city = parse_birth_command("1990-08-15T12:00:00 Нижний Новгород")

    assert moment == datetime(1990, 8, 15, 12, 0)
    assert city == "Нижний Новгород"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        "   ",
        "1990-08-15",
        "1990-08-15 12:00",
        "1990-08-15 Москва",
        "1990-08-15T12:00:00+04:00 Москва",
        "not-a-date 12:00 Москва",
    ],
)
def test_parse_birth_command_rejects_incomplete_payloads(payload):
    with pytest.raises(BirthInputError):
        parse_birth_command(payload)


def test_local_to_utc_applies_historical_rules():
    utc, offset, notes = local_to_utc(datetime(1990, 8, 15, 12, 0), "Europe/Moscow")

    assert utc == datetime(1990, 8, 15, 8, 0, tzinfo=timezone.utc)
    assert offset == timedelta(hours=4)  # Moscow summer time in 1990
    assert notes == []


def test_local_to_utc_rejects_unknown_zone():
    with pytest.raises(BirthInputError):
        local_to_utc(datetime(1990, 8, 15, 12, 0), "Mars/Olympus")


def test_parse_local_datetime_rejects_explicit_offset():
    with pytest.raises(BirthInputError):
        parse_local_datetime("1990-08-15T12:00:00+04:00")


def test_canonical_utc_string_keeps_microseconds_only_when_present():
    assert canonical_utc_string("1990-08-15T08:00:00Z") == "1990-08-15T08:00:00+00:00"
    assert (
        canonical_utc_string(datetime(1990, 8, 15, 8, 0, 0, 250000, tzinfo=timezone.utc))
        == "1990-08-15T08:00:00.250000+00:00"
    )


def test_birth_data_normalizes_codes():
    birth = BirthData(
        dt_utc="1990-08-15T12:00:00+04:00",
        latitude=55.7558,
        longitude=37.6173,
        house_system="p",
        language="RU",
    )

    assert birth.dt_utc == datetime(1990, 8, 15, 8, 0, tzinfo=timezone.utc)
    assert birth.house_system == "P"
    assert birth.language == "ru"


def test_birth_data_input_params_contain_only_known_fields():
    birth = BirthData(dt_utc="1990-08-15T08:00:00Z", latitude=55.7558, longitude=37.6173)

    params = birth.input_params()

    assert params == {
        "dt_utc": "1990-08-15T08:00:00+00:00",
        "latitude": 55.7558,
        "longitude": 37.6173,
        "house_system": "P",
        "language": "ru",
    }

    detailed = BirthData(
        dt_utc="1990-08-15T08:00:00Z",
        latitude=55.7558,
        longitude=37.6173,
        city="Москва",
        timezone="Europe/Moscow",
        datetime_local="1990-08-15T12:00:00",
    ).input_params()
    assert detailed["city"] == "Москва"
    assert detailed["timezone"] == "Europe/Moscow"
    assert detailed["datetime_local"] == "1990-08-15T12:00:00"


def test_birth_data_rejects_invalid_input():
    with pytest.raises(BirthInputError):
        BirthData(dt_utc="1990-08-15T08:00:00", latitude=55.7, longitude=37.6)
    with pytest.raises(BirthInputError):
        BirthData(dt_utc="1990-08-15T08:00:00Z", latitude=95.0, longitude=37.6)
    with pytest.raises(BirthInputError):
        BirthData(dt_utc="1990-08-15T08:00:00Z", latitude=55.7, longitude=37.6, language="")
