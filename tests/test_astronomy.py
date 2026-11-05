import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from astronomy.calculator import (
    DEFAULT_ORBS,
    MAJOR_ASPECTS,
    ZODIAC_SIGNS,
    AstronomyError,
    angular_separation,
    calculate_aspects,
    calculate_natal_chart,
    house_of,
    sign_of,
)

MOSCOW = (55.7558, 37.6173)
BIRTH = datetime(1990, 8, 15, 12, 0, tzinfo=timezone.utc)
PLANET_FIELDS = {"longitude", "sign", "house", "speed", "is_retrograde", "is_stationary"}


def make_chart(moment=BIRTH, lat=55.7558, lon=37.6173, house_system="P"):
    return calculate_natal_chart(moment, lat, lon, house_system)


def test_returns_twelve_cusps_and_matching_angles():
    chart = make_chart()

    assert len(chart.cusps) == 12
    assert all(0.0 <= cusp < 360.0 for cusp in chart.cusps)
    assert chart.cusps[0] == pytest.approx(chart.asc.longitude)
    assert chart.cusps[9] == pytest.approx(chart.mc.longitude)
    assert (chart.asc.house, chart.mc.house) == (1, 10)


def test_reports_ten_bodies_plus_both_lunar_nodes():
    chart = make_chart()

    assert list(chart.planets) == [
        "Sun",
        "Moon",
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
        "NorthNode",
        "SouthNode",
    ]


def test_planet_payload_has_exactly_the_contract_fields():
    chart = make_chart()
    payload = chart.to_dict()

    for name, planet in payload["planets"].items():
        assert set(planet) == PLANET_FIELDS, name
        assert isinstance(planet["house"], int)
        assert 1 <= planet["house"] <= 12
    assert set(payload["asc"]) == {"longitude", "sign", "house"}
    assert set(payload["mc"]) == {"longitude", "sign", "house"}
    json.dumps(payload)


def test_sign_and_house_are_derived_from_longitude():
    chart = make_chart()

    for name, planet in chart.planets.items():
        assert planet.sign == ZODIAC_SIGNS[int(planet.longitude // 30.0)], name
        assert planet.house == house_of(planet.longitude, chart.cusps), name


def test_angles_match_independent_sidereal_time_formula():
    chart = make_chart()
    julian_day = chart.julian_day
    centuries = (julian_day - 2451545.0) / 36525.0
    # Meeus, Astronomical Algorithms, 12.4: GMST in degrees.
    gmst = (
        280.46061837
        + 360.98564736629 * (julian_day - 2451545.0)
        + 0.000387933 * centuries**2
        - centuries**3 / 38710000.0
    )
    ramc = math.radians((gmst + MOSCOW[1]) % 360.0)
    obliquity = math.radians(
        23.4392911 - 0.0130042 * centuries - 1.64e-7 * centuries**2 + 5.04e-7 * centuries**3
    )
    latitude = math.radians(MOSCOW[0])

    mc = math.degrees(math.atan2(math.sin(ramc), math.cos(ramc) * math.cos(obliquity))) % 360.0
    asc = math.degrees(
        math.atan2(
            math.cos(ramc),
            -(math.sin(ramc) * math.cos(obliquity) + math.tan(latitude) * math.sin(obliquity)),
        )
    ) % 360.0

    assert chart.mc.longitude == pytest.approx(mc, abs=0.02)
    assert chart.asc.longitude == pytest.approx(asc, abs=0.02)


def test_golden_chart_for_moscow_1990():
    chart = make_chart()

    assert chart.julian_day == pytest.approx(2448119.0)
    assert chart.asc.longitude == pytest.approx(240.556767, abs=1e-6)
    assert chart.mc.longitude == pytest.approx(181.366556, abs=1e-6)

    expected = {
        "Sun": 142.405383,
        "Moon": 76.819899,
        "Saturn": 289.861672,
        "Pluto": 225.087742,
        "NorthNode": 306.468225,
    }
    for name, longitude in expected.items():
        assert chart.planets[name].longitude == pytest.approx(longitude, abs=0.01), name


def test_sun_hits_zero_longitude_at_the_2024_vernal_equinox():
    chart = calculate_natal_chart(datetime(2024, 3, 20, 3, 6, tzinfo=timezone.utc), 0.0, 0.0)

    assert angular_separation(chart.planets["Sun"].longitude, 0.0) < 0.01


def test_solar_eclipse_places_sun_and_moon_at_19_aries():
    chart = calculate_natal_chart(datetime(2024, 4, 8, 18, 18, tzinfo=timezone.utc), 0.0, 0.0)
    sun = chart.planets["Sun"]
    moon = chart.planets["Moon"]

    assert sun.longitude == pytest.approx(19.4, abs=0.05)
    assert (sun.sign, moon.sign) == ("Aries", "Aries")
    assert angular_separation(sun.longitude, moon.longitude) < 0.1


def test_sun_longitude_at_the_j2000_epoch():
    chart = calculate_natal_chart(datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc), 0.0, 0.0)

    assert chart.julian_day == pytest.approx(2451545.0)
    assert chart.planets["Sun"].longitude == pytest.approx(280.37, abs=0.05)
    assert chart.planets["Sun"].sign == "Capricorn"


def test_mercury_is_retrograde_in_april_2024():
    chart = make_chart(datetime(2024, 4, 10, 12, 0, tzinfo=timezone.utc))
    mercury = chart.planets["Mercury"]

    assert mercury.speed < 0
    assert mercury.is_retrograde is True


def test_uranus_is_near_station_in_august_1990():
    uranus = make_chart().planets["Uranus"]

    assert abs(uranus.speed) < 0.05
    assert uranus.is_stationary is True
    assert uranus.is_retrograde is True


def test_sun_is_never_retrograde_or_stationary():
    for month in (1, 4, 7, 10):
        sun = make_chart(datetime(2023, month, 15, 12, 0, tzinfo=timezone.utc)).planets["Sun"]
        assert sun.is_retrograde is False
        assert sun.is_stationary is False


def test_motion_flags_are_consistent_with_speed():
    chart = make_chart()

    for name, planet in chart.planets.items():
        assert planet.is_retrograde is (planet.speed < 0), name
        assert planet.is_stationary is (abs(planet.speed) < 0.05), name


def test_whole_sign_houses_start_at_sign_boundaries_and_keep_planets():
    tropical = make_chart()
    whole_sign = make_chart(house_system="W")

    assert whole_sign.cusps[0] == pytest.approx(240.0)
    assert all(cusp % 30.0 == pytest.approx(0.0) for cusp in whole_sign.cusps)
    assert whole_sign.asc.longitude == pytest.approx(tropical.asc.longitude)
    for name, planet in whole_sign.planets.items():
        assert planet.longitude == pytest.approx(tropical.planets[name].longitude)


def test_equal_house_system_produces_even_thirty_degree_houses():
    chart = make_chart(house_system="E")

    for first, second in zip(chart.cusps, chart.cusps[1:]):
        assert (second - first) % 360.0 == pytest.approx(30.0)


def test_lunar_nodes_are_opposite_and_not_aspect_each_other():
    chart = make_chart()
    north = chart.planets["NorthNode"]
    south = chart.planets["SouthNode"]

    assert angular_separation(north.longitude, south.longitude) == pytest.approx(180.0)
    assert south.house == house_of(south.longitude, chart.cusps)
    pairs = {frozenset((aspect.body_a, aspect.body_b)) for aspect in chart.aspects}
    assert frozenset(("NorthNode", "SouthNode")) not in pairs


def test_aspects_are_major_and_within_configured_orbs():
    chart = make_chart()

    for aspect in chart.aspects:
        assert aspect.aspect in dict(MAJOR_ASPECTS)
        assert aspect.orb <= DEFAULT_ORBS[aspect.aspect] + 1e-9
        separation = angular_separation(
            chart.planets[aspect.body_a].longitude, chart.planets[aspect.body_b].longitude
        )
        assert aspect.separation == pytest.approx(separation)
        assert aspect.orb == pytest.approx(abs(separation - aspect.exact_angle))


def test_narrow_orbs_shorten_the_aspect_list():
    chart = make_chart()
    zero_orbs = {name: 0.0 for name, _angle in MAJOR_ASPECTS}
    strict = calculate_aspects(chart.planets, orbs=zero_orbs)

    assert len(strict) <= len(chart.aspects)
    assert all(aspect.orb == pytest.approx(0.0, abs=1e-9) for aspect in strict)


def test_house_of_handles_the_wrap_around_zero_aries():
    cusps = tuple((350.0 + 30.0 * index) % 360.0 for index in range(12))

    assert house_of(355.0, cusps) == 1
    assert house_of(10.0, cusps) == 1
    assert house_of(30.0, cusps) == 2
    assert house_of(349.9, cusps) == 12


def test_sign_of_handles_boundaries():
    assert sign_of(0.0) == "Aries"
    assert sign_of(29.999) == "Aries"
    assert sign_of(30.0) == "Taurus"
    assert sign_of(359.999) == "Pisces"
    assert sign_of(360.0) == "Aries"
    assert sign_of(-1.0) == "Pisces"


def test_rejects_naive_datetime():
    with pytest.raises(ValueError):
        calculate_natal_chart(datetime(1990, 8, 15, 12, 0), *MOSCOW)


def test_rejects_non_utc_offset():
    with pytest.raises(ValueError):
        calculate_natal_chart(BIRTH.astimezone(timezone(timedelta(hours=3))), *MOSCOW)


def test_accepts_a_zero_offset_named_timezone():
    chart = calculate_natal_chart(
        datetime(1990, 8, 15, 12, 0, tzinfo=timezone(timedelta(0), "UTC")), *MOSCOW
    )

    assert chart.dt_utc == BIRTH


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(90.1, 0.0), (-90.1, 0.0), (0.0, 180.1), (0.0, -180.1), (float("nan"), 0.0)],
)
def test_rejects_impossible_coordinates(lat, lon):
    with pytest.raises(ValueError):
        calculate_natal_chart(BIRTH, lat, lon)


def test_rejects_unknown_house_system():
    with pytest.raises(ValueError, match="unsupported house system"):
        calculate_natal_chart(BIRTH, *MOSCOW, "Z")


def test_rejects_unknown_node_type():
    with pytest.raises(ValueError, match="unsupported node_type"):
        calculate_natal_chart(BIRTH, *MOSCOW, node_type="osculating")


def test_true_node_differs_from_mean_node_but_stays_close():
    mean_node = make_chart().planets["NorthNode"]
    true_node = calculate_natal_chart(BIRTH, *MOSCOW, node_type="true").planets["NorthNode"]

    assert angular_separation(mean_node.longitude, true_node.longitude) < 2.0


def test_placidus_fails_above_the_polar_circle():
    with pytest.raises(AstronomyError, match="not computable"):
        calculate_natal_chart(BIRTH, 80.0, 37.6173)


def test_same_input_gives_identical_chart():
    assert make_chart() == make_chart()
