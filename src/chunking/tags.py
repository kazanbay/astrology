"""Extract canonical astrology entities, language and school for Qdrant payloads."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


DEFAULT_SCHOOL = "vronsky"

# Patterns are strings so callers can choose case-sensitive matching for content.
PLANET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Sun", r"(?<!\w)(?:sun|солнц\w*)(?!\w)"),
    ("Moon", r"(?<!\w)(?:moon|лун\w*)(?!\w)"),
    ("Mercury", r"(?<!\w)(?:mercury|меркур\w*)(?!\w)"),
    ("Venus", r"(?<!\w)(?:venus|венер\w*)(?!\w)"),
    ("Mars", r"(?<!\w)(?:mars|марс\w*)(?!\w)"),
    ("Jupiter", r"(?<!\w)(?:jupiter|юпитер\w*)(?!\w)"),
    ("Saturn", r"(?<!\w)(?:saturn|сатурн\w*)(?!\w)"),
    ("Uranus", r"(?<!\w)(?:uranus|уран\w*)(?!\w)"),
    ("Neptune", r"(?<!\w)(?:neptune|нептун\w*)(?!\w)"),
    ("Pluto", r"(?<!\w)(?:pluto|плутон\w*)(?!\w)"),
    ("Chiron", r"(?<!\w)(?:chiron|хирон\w*)(?!\w)"),
    ("Proserpina", r"(?<!\w)(?:proserpina|прозерпин\w*)(?!\w)"),
    ("Lilith", r"(?<!\w)(?:lilith|лилит\w*)(?!\w)"),
    ("NorthNode", r"(?<!\w)(?:north\s*node|северн\w*\s+узел\w*|раху)(?!\w)"),
    ("SouthNode", r"(?<!\w)(?:south\s*node|южн\w*\s+узел\w*|кету)(?!\w)"),
)

SIGN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Aries", r"(?<!\w)(?:овен|овн[а-яё]{0,3}|aries)(?!\w)"),
    ("Taurus", r"(?<!\w)(?:телец|тельц[а-яё]{0,3}|taurus)(?!\w)"),
    ("Gemini", r"(?<!\w)(?:близнец[а-яё]{0,3}|gemini)(?!\w)"),
    ("Cancer", r"(?<!\w)(?:рак(?:а|е|у|ом|и|ов)?|cancer)(?!\w)"),
    ("Leo", r"(?<!\w)(?:лев|льв[а-яё]{0,3}|leo)(?!\w)"),
    ("Virgo", r"(?<!\w)(?:дева|дев[а-яё]{0,3}|virgo)(?!\w)"),
    ("Libra", r"(?<!\w)(?:весы|весов|весам|весах|весами|libra)(?!\w)"),
    ("Scorpio", r"(?<!\w)(?:скорпион[а-яё]{0,3}|scorpio)(?!\w)"),
    ("Sagittarius", r"(?<!\w)(?:стрелец|стрельц[а-яё]{0,3}|sagittarius)(?!\w)"),
    ("Capricorn", r"(?<!\w)(?:козерог[а-яё]{0,3}|capricorn)(?!\w)"),
    ("Aquarius", r"(?<!\w)(?:водолей|водоле[а-яё]{0,3}|aquarius)(?!\w)"),
    ("Pisces", r"(?<!\w)(?:рыбы|рыбам|рыбах|рыбами|pisces)(?!\w)"),
)

ASPECT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("conjunction", r"(?<!\w)(?:conjunction|соединени\w*|конъюнкци\w*)(?!\w)"),
    ("square", r"(?<!\w)(?:square|квадрат\w*|квадратур\w*)(?!\w)"),
    ("trine", r"(?<!\w)(?:trine|трин\w*)(?!\w)"),
    ("opposition", r"(?<!\w)(?:opposition|оппозици\w*|противостояни\w*)(?!\w)"),
    ("sextile", r"(?<!\w)(?:sextile|секстил\w*)(?!\w)"),
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
HOUSE_WORD = r"(?:дом(?:а|е|у|ом|ов|ах)?|пол(?:е|я|ю|ем|ей|ях)?|house)"
HOUSE_PATTERNS = (
    re.compile(rf"{HOUSE_WORD}\s*(?:№|no\.?\s*)?({HOUSE_TOKEN})(?!\w)", re.IGNORECASE),
    re.compile(rf"(?<!\w)({HOUSE_TOKEN})\s*(?:-(?:й|м)\s*)?{HOUSE_WORD}(?!\w)", re.IGNORECASE),
)

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True)
class ChunkMetadata:
    """Scalar, English-language metadata stored at the Qdrant payload root."""

    language: str
    planet: str | None
    sign: str | None
    house: int | None
    aspect: str | None
    school: str


def roman_or_integer(value: str) -> int | None:
    """Convert an Arabic or Roman house/volume number from 1 to 12."""
    value = value.strip().upper()
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 12 else None
    return ROMAN_HOUSES.get(value)


def normalize_language(value: object) -> str | None:
    """Normalize ``ru``, ``ru-RU`` or ``RU`` to a two-letter language code."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-").split("-")[0]
    return code if _LANGUAGE_CODE.match(code) else None


def detect_language(text: str) -> str:
    """Return ``ru``, ``en`` or ``zh`` from the script used in the text."""
    if not text:
        return "ru"
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    cjk = len(_CJK.findall(text))
    if cjk > 0 and cjk >= max(cyrillic, latin):
        return "zh"
    if latin > cyrillic:
        return "en"
    return "ru"


def infer_school(author: str | None, default: str = DEFAULT_SCHOOL) -> str:
    """Resolve the school identifier; the current corpus is Vronsky's school."""
    normalized = (author or "").lower()
    if "вронск" in normalized or "vronsky" in normalized:
        return "vronsky"
    return default


def _collect(text: str, patterns: tuple[tuple[str, str], ...], *, ignore_case: bool) -> list[tuple[int, str]]:
    if not text:
        return []
    flags = re.IGNORECASE if ignore_case else 0
    matches: list[tuple[int, str]] = []
    for canonical, pattern in patterns:
        matches.extend((match.start(), canonical) for match in re.finditer(pattern, text, flags))
    return sorted(matches)


def _pick_entity(
    title: str,
    text: str,
    patterns: tuple[tuple[str, str], ...],
    *,
    content_case_sensitive: bool,
) -> str | None:
    matches = _collect(title, patterns, ignore_case=True)
    if not matches:
        matches = _collect(text, patterns, ignore_case=not content_case_sensitive)
    if not matches:
        return None

    counts = Counter(canonical for _position, canonical in matches)
    first_position: dict[str, int] = {}
    for position, canonical in matches:
        first_position.setdefault(canonical, position)
    return max(counts, key=lambda canonical: (counts[canonical], -first_position[canonical]))


def _collect_houses(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    matches: list[tuple[int, int]] = []
    for pattern in HOUSE_PATTERNS:
        for match in pattern.finditer(text):
            house = roman_or_integer(match.group(1))
            if house is not None:
                matches.append((match.start(), house))
    return sorted(matches)


def _pick_house(title: str, text: str) -> int | None:
    matches = _collect_houses(title) or _collect_houses(text)
    if not matches:
        return None
    counts = Counter(house for _position, house in matches)
    first_position: dict[int, int] = {}
    for position, house in matches:
        first_position.setdefault(house, position)
    return max(counts, key=lambda house: (counts[house], -first_position[house]))


def extract_chunk_metadata(
    section_title: str,
    text: str,
    *,
    language: str | None = None,
    school: str = DEFAULT_SCHOOL,
) -> ChunkMetadata:
    """Extract one scalar value per entity field, preferring the section title."""
    title = section_title or ""
    return ChunkMetadata(
        language=normalize_language(language) or detect_language(f"{title}\n{text}"),
        planet=_pick_entity(title, text, PLANET_PATTERNS, content_case_sensitive=False),
        sign=_pick_entity(title, text, SIGN_PATTERNS, content_case_sensitive=True),
        house=_pick_house(title, text),
        aspect=_pick_entity(title, text, ASPECT_PATTERNS, content_case_sensitive=False),
        school=school.strip().lower() or DEFAULT_SCHOOL,
    )


def extract_meta_tags(section_title: str, text: str) -> dict[str, list[object]]:
    """Legacy PostgreSQL tags: all planets and house numbers found in a section."""
    searchable = f"{section_title}\n{text}"
    planets = [
        name
        for name, pattern in PLANET_PATTERNS
        if re.search(pattern, searchable, re.IGNORECASE)
    ]
    houses = sorted({house for _position, house in _collect_houses(searchable)})
    return {"planets": planets, "houses": houses}
