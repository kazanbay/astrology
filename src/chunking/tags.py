"""Extract canonical astrology entities, language and school for Qdrant payloads.

Structural metadata is read from the section title only: a payload must describe
the subject of its section, not incidental mentions in the body. This is what
makes exact-match routing trustworthy — a paragraph that merely says "Сатурн"
no longer claims to be a Saturn article.

Titles that name two bodies ("ВЕНЕРА – УРАН") or two signs/houses ("Восходящий
Лунный Узел в знаке Овна. Нисходящий Лунный Узел в знаке Весов") keep the
second entity in ``planet_b`` / ``sign_b`` / ``house_b`` so pair lookups stay
order-invariant without losing information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_SCHOOL = "vronsky"

# Patterns are strings so callers can choose case-sensitive matching for content.
PLANET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Sun", r"(?<!\w)(?:sun|солнц\w*)(?!\w)"),
    ("Moon", r"(?<!\w)(?:moon|лун\w*)(?!\w)(?!\s+узел)"),
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
    # The nodes are named in Russian as "Восходящий/Нисходящий Лунный Узел";
    # these patterns must run before the generic Moon pattern can claim the
    # adjective "Лунный".
    (
        "NorthNode",
        r"(?<!\w)(?:north\s*node|восходящ\w*\s+лунн\w*\s+узел\w*|северн\w*\s+узел\w*|раху)(?!\w)",
    ),
    (
        "SouthNode",
        r"(?<!\w)(?:south\s*node|нисходящ\w*\s+лунн\w*\s+узел\w*|южн\w*\s+узел\w*|кету)(?!\w)",
    ),
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
    ("trine", r"(?<!\w)(?:trine|трин(?:а|у|ом|е|ы|ов)?|тригон(?:а|у|ом|е)?)(?!\w)"),
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

TRANSIT_PATTERN = re.compile(r"транзит", re.IGNORECASE)

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True)
class ChunkMetadata:
    """Scalar, English-language metadata stored at the Qdrant payload root."""

    language: str
    planet: str | None
    planet_b: str | None
    sign: str | None
    sign_b: str | None
    house: int | None
    house_b: int | None
    aspect: str | None
    school: str
    is_transit: bool


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


def _ordered_entities(text: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    """Return distinct canonical names in the order they first appear in ``text``."""
    ordered: list[str] = []
    for _position, canonical in _collect(text, patterns, ignore_case=True):
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


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


def _ordered_houses(text: str) -> list[int]:
    ordered: list[int] = []
    for _position, house in _collect_houses(text):
        if house not in ordered:
            ordered.append(house)
    return ordered


def extract_chunk_metadata(
    section_title: str,
    text: str,
    *,
    language: str | None = None,
    school: str = DEFAULT_SCHOOL,
) -> ChunkMetadata:
    """Extract structural metadata from the section title.

    ``text`` is used only to detect the language: entity tags must describe the
    section's own subject, so body mentions are deliberately ignored.
    """
    title = section_title or ""
    planets = _ordered_entities(title, PLANET_PATTERNS)
    signs = _ordered_entities(title, SIGN_PATTERNS)
    houses = _ordered_houses(title)
    aspects = _ordered_entities(title, ASPECT_PATTERNS)
    return ChunkMetadata(
        language=normalize_language(language) or detect_language(f"{title}\n{text}"),
        planet=planets[0] if planets else None,
        planet_b=planets[1] if len(planets) > 1 else None,
        sign=signs[0] if signs else None,
        sign_b=signs[1] if len(signs) > 1 else None,
        house=houses[0] if houses else None,
        house_b=houses[1] if len(houses) > 1 else None,
        aspect=aspects[0] if aspects else None,
        school=(school or "").strip().lower() or DEFAULT_SCHOOL,
        is_transit=bool(TRANSIT_PATTERN.search(title)),
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
