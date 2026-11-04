"""Extract canonical planet and house tags from astrology book fragments."""

from __future__ import annotations

import re


PLANET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Sun", re.compile(r"(?<!\w)(?:sun|солнц\w*)(?!\w)", re.IGNORECASE)),
    ("Moon", re.compile(r"(?<!\w)(?:moon|лун\w*)(?!\w)", re.IGNORECASE)),
    ("Mercury", re.compile(r"(?<!\w)(?:mercury|меркур\w*)(?!\w)", re.IGNORECASE)),
    ("Venus", re.compile(r"(?<!\w)(?:venus|венер\w*)(?!\w)", re.IGNORECASE)),
    ("Mars", re.compile(r"(?<!\w)(?:mars|марс\w*)(?!\w)", re.IGNORECASE)),
    ("Jupiter", re.compile(r"(?<!\w)(?:jupiter|юпитер\w*)(?!\w)", re.IGNORECASE)),
    ("Saturn", re.compile(r"(?<!\w)(?:saturn|сатурн\w*)(?!\w)", re.IGNORECASE)),
    ("Uranus", re.compile(r"(?<!\w)(?:uranus|уран\w*)(?!\w)", re.IGNORECASE)),
    ("Neptune", re.compile(r"(?<!\w)(?:neptune|нептун\w*)(?!\w)", re.IGNORECASE)),
    ("Pluto", re.compile(r"(?<!\w)(?:pluto|плутон\w*)(?!\w)", re.IGNORECASE)),
)

ROMAN_HOUSES = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}
HOUSE_TOKEN = r"(?:12|11|10|[1-9]|XII|XI|IX|VIII|VII|VI|IV|V|III|II|I|X)"
HOUSE_WORD = r"(?:дом(?:а|е|у|ом|ов|ах)?|house)"
HOUSE_PATTERNS = (
    re.compile(rf"{HOUSE_WORD}\s*(?:№|no\.?\s*)?({HOUSE_TOKEN})(?!\w)", re.IGNORECASE),
    re.compile(rf"(?<!\w)({HOUSE_TOKEN})\s*(?:-(?:й|м)\s*)?{HOUSE_WORD}(?!\w)", re.IGNORECASE),
)


def roman_or_integer(value: str) -> int | None:
    """Convert an Arabic or Roman house/volume number from 1 to 12."""
    value = value.strip().upper()
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 12 else None
    return ROMAN_HOUSES.get(value)


def extract_meta_tags(section_title: str, text: str) -> dict[str, list[object]]:
    """Extract canonical planet names and house numbers from a section."""
    searchable = f"{section_title}\n{text}"
    planets = [name for name, pattern in PLANET_PATTERNS if pattern.search(searchable)]
    houses: set[int] = set()
    for pattern in HOUSE_PATTERNS:
        for match in pattern.finditer(searchable):
            house = roman_or_integer(match.group(1))
            if house is not None:
                houses.add(house)
    return {"planets": planets, "houses": sorted(houses)}
