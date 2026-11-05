"""Exact structural routing: natal-chart elements to knowledge-base chunks.

The router implements deterministic payload filtering over Qdrant (exact match
on scalar metadata) and never touches vector search. Each chart element
(planet in sign, planet in house, major aspect) becomes one structured query;
the query checks every allowed metadata spelling of the element and returns the
best ``per_element`` chunks ranked by how explicitly the section title names it.

Payload schema used by the queries (filled from section titles only):

* ``planet`` / ``planet_b`` - first and second body named in the title;
* ``sign`` / ``sign_b`` - first and second sign named in the title;
* ``house`` / ``house_b`` - first and second house named in the title;
* ``is_transit`` - the section describes a transit, not a natal placement.

Aspect chapters in the corpus are named by pair ("ВЕНЕРА – УРАН") and describe
the pair as a whole; the source does not tag a specific aspect type per chunk.
Aspect lookups therefore return the pair chapter and report ``coverage="pair"``.

Production defaults to a single candidate per element; manual review and
evaluation pass a larger number to compare the alternatives side by side.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:  # support both `src.rag` and top-level `rag` imports
    from ..chunking.tags import (
        ASPECT_PATTERNS,
        PLANET_PATTERNS,
        SIGN_PATTERNS,
        extract_chunk_metadata,
        infer_school,
    )
except ImportError:  # pragma: no cover - depends on sys.path layout
    from chunking.tags import (
        ASPECT_PATTERNS,
        PLANET_PATTERNS,
        SIGN_PATTERNS,
        extract_chunk_metadata,
        infer_school,
    )


DEFAULT_COLLECTION = "vronsky_structural_xml"
DEFAULT_LANGUAGE = "ru"

#: Upper bound on chunks fetched per query variant before ranking. Exact-match
#: filters may legitimately match many sections (several volumes, split parts);
#: the cap keeps a lookup bounded while still ranking the alternatives instead
#: of trusting raw scroll order.
MAX_CANDIDATES_TO_FETCH = 100

KIND_PLANET_IN_SIGN = "planet_in_sign"
KIND_PLANET_IN_HOUSE = "planet_in_house"
KIND_ASPECT = "aspect"

COVERAGE_SINGLE = "single_element"
COVERAGE_PAIR = "pair_chapter"

#: Bodies that appear as the second entity of a two-subject section title.
SECONDARY_BODIES = frozenset({"NorthNode", "SouthNode"})

_PLANET_CANONICAL = {name.lower(): name for name, _pattern in PLANET_PATTERNS}
_SIGN_CANONICAL = {name.lower(): name for name, _pattern in SIGN_PATTERNS}
_ASPECT_CANONICAL = {name.lower(): name for name, _pattern in ASPECT_PATTERNS}
ALLOWED_SIGNS: tuple[str, ...] = tuple(name for name, _pattern in SIGN_PATTERNS)
ALLOWED_ASPECTS: tuple[str, ...] = tuple(name for name, _pattern in ASPECT_PATTERNS)

FieldPair = tuple[str, Any]


def canonical_planet(value: str) -> str:
    """Return the canonical English body name used in Qdrant payloads."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("planet name must be a non-empty string")
    normalized = value.strip()
    return _PLANET_CANONICAL.get(normalized.lower(), normalized)


def canonical_sign(value: str) -> str:
    """Return the canonical English sign name; unknown signs are rejected."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sign name must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in _SIGN_CANONICAL:
        raise ValueError(f"unknown sign {value!r}; allowed: {', '.join(ALLOWED_SIGNS)}")
    return _SIGN_CANONICAL[normalized]


def canonical_aspect(value: str) -> str:
    """Return the canonical English aspect name; unknown aspects are rejected."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aspect name must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in _ASPECT_CANONICAL:
        raise ValueError(f"unknown aspect {value!r}; allowed: {', '.join(ALLOWED_ASPECTS)}")
    return _ASPECT_CANONICAL[normalized]


def canonical_aspect_pair(body_a: str, body_b: str) -> tuple[str, str]:
    """Sort an aspect pair lexicographically so lookups are order-invariant."""
    first = canonical_planet(body_a)
    second = canonical_planet(body_b)
    if first == second:
        raise ValueError(f"an aspect needs two different bodies, got {first!r} twice")
    return tuple(sorted((first, second)))  # type: ignore[return-value]


@dataclass(frozen=True)
class QueryVariant:
    """One exact-match spelling of an element plus the fields used for ranking."""

    conditions: tuple[FieldPair, ...]
    rank_fields: tuple[str, ...]

    @property
    def label(self) -> str:
        return " AND ".join(f"{key}={value}" for key, value in self.conditions)


@dataclass(frozen=True)
class ElementQuery:
    """One discrete chart element rendered as exact-match payload filters."""

    key: str
    kind: str
    variants: tuple[QueryVariant, ...]
    display: str
    coverage: str = COVERAGE_SINGLE

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(f"element query {self.key!r} has no filter variants")

    @property
    def filter_label(self) -> str:
        """Human-readable label of the exact-match request."""
        labels = [variant.label for variant in self.variants]
        if len(labels) == 1:
            return labels[0]
        return "[" + " OR ".join(f"({label})" for label in labels) + "]"


@dataclass(frozen=True)
class Candidate:
    """One knowledge-base chunk returned for an element query."""

    chunk_id: str
    title: str
    content: str
    section_title: str | None
    section_path: tuple[str, ...]
    book_id: str | None
    chunk_index: int | None
    #: How many of the matched variant's fields are named by the section title
    #: itself. Higher tiers are more trustworthy.
    match_tier: int
    #: Number of the matched variant's fields that the payload actually fills.
    rank_total: int
    #: A natal chart must not resolve to a transit section (same tags, wrong
    #: reading), so transits rank after natal sections of the same tier.
    is_transit: bool = False

    def sort_key(self) -> tuple[int, bool, str, int]:
        return (
            -self.match_tier,
            self.is_transit,
            self.book_id or "",
            self.chunk_index if isinstance(self.chunk_index, int) else 0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "section_title": self.section_title,
            "section_path": list(self.section_path),
            "book_id": self.book_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "match_tier": self.match_tier,
            "rank_total": self.rank_total,
            "is_transit": self.is_transit,
        }


@dataclass(frozen=True)
class ElementMatch:
    """The outcome of one structural lookup."""

    query: ElementQuery
    candidates: tuple[Candidate, ...] = ()
    #: Unique chunks retrieved for the query across all variants. Bounded by
    #: ``MAX_CANDIDATES_TO_FETCH`` per variant, so it may under-count when a
    #: filter matches more sections than the cap.
    total_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        """TZ-compatible mapping: flat top-1 keys plus the candidate list."""
        top = self.candidates[0] if self.candidates else None
        return {
            "chunk_id": top.chunk_id if top else None,
            "title": top.title if top else None,
            "content": top.content if top else None,
            "kind": self.query.kind,
            "element": self.query.display,
            "coverage": self.query.coverage,
            "filter": self.query.filter_label,
            "total_found": self.total_found,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _house_number(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("house must be an integer from 1 to 12")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"house must be an integer from 1 to 12, got {value!r}") from exc
    if not 1 <= number <= 12:
        raise ValueError(f"house must be an integer from 1 to 12, got {number}")
    return number


def _position_variants(planet: str, field: str, value: Any) -> tuple[QueryVariant, ...]:
    """Primary spelling plus the secondary one for bodies that head two-subject titles."""
    variants = [
        QueryVariant(conditions=(("planet", planet), (field, value)), rank_fields=("planet", field))
    ]
    if planet in SECONDARY_BODIES:
        variants.append(
            QueryVariant(
                conditions=(("planet_b", planet), (f"{field}_b", value)),
                rank_fields=(f"{field}_b",),
            )
        )
    return tuple(variants)


def _planet_position_fields(raw_name: str, position: Any) -> list[ElementQuery]:
    planet = canonical_planet(raw_name)
    if not isinstance(position, Mapping):
        raise ValueError(f"planets[{raw_name!r}] must be a mapping with 'sign'/'house'")
    queries: list[ElementQuery] = []
    sign = position.get("sign")
    house = position.get("house")
    if sign:
        sign_name = canonical_sign(str(sign))
        queries.append(
            ElementQuery(
                key=f"{planet}_in_{sign_name}",
                kind=KIND_PLANET_IN_SIGN,
                variants=_position_variants(planet, "sign", sign_name),
                display=f"{planet} in {sign_name}",
            )
        )
    if house is not None:
        number = _house_number(house)
        queries.append(
            ElementQuery(
                key=f"{planet}_in_House_{number}",
                kind=KIND_PLANET_IN_HOUSE,
                variants=_position_variants(planet, "house", number),
                display=f"{planet} in house {number}",
            )
        )
    return queries


def decompose_chart(chart_data: Mapping[str, Any]) -> list[ElementQuery]:
    """Split a chart dictionary into exact-match element queries.

    ``chart_data`` is the output of ``NatalChart.to_dict()`` (or any mapping
    with the same shape). Aspect pairs are canonicalized to a lexicographic
    order so ``[Mercury, Saturn]`` and ``[Saturn, Mercury]`` share one key; the
    pair is matched in both stored orders.
    """
    if not isinstance(chart_data, Mapping):
        raise ValueError("chart_data must be a mapping (the output of NatalChart.to_dict())")
    planets = chart_data.get("planets")
    if not isinstance(planets, Mapping) or not planets:
        raise ValueError("chart_data['planets'] must be a non-empty mapping of body placements")

    queries: list[ElementQuery] = []
    for raw_name, position in planets.items():
        queries.extend(_planet_position_fields(str(raw_name), position))

    aspects = chart_data.get("aspects") or []
    if not isinstance(aspects, Sequence) or isinstance(aspects, (str, bytes)):
        raise ValueError("chart_data['aspects'] must be a list of aspect mappings")
    for index, raw in enumerate(aspects):
        if not isinstance(raw, Mapping):
            raise ValueError(f"aspects[{index}] must be a mapping with 'body_a'/'body_b'/'aspect'")
        missing = [field for field in ("body_a", "body_b", "aspect") if raw.get(field) is None]
        if missing:
            raise ValueError(f"aspects[{index}] is missing fields: {', '.join(missing)}")
        first, second = canonical_aspect_pair(str(raw["body_a"]), str(raw["body_b"]))
        aspect_name = canonical_aspect(str(raw["aspect"]))
        queries.append(
            ElementQuery(
                key=f"{first}_{second}_{aspect_name}",
                kind=KIND_ASPECT,
                variants=(
                    QueryVariant(
                        conditions=(("planet", first), ("planet_b", second)),
                        rank_fields=("planet", "planet_b"),
                    ),
                    QueryVariant(
                        conditions=(("planet", second), ("planet_b", first)),
                        rank_fields=("planet", "planet_b"),
                    ),
                ),
                display=f"{first} – {second} ({aspect_name})",
                coverage=COVERAGE_PAIR,
            )
        )
    return queries


def metadata_pairs(metadata_filters: Mapping[str, Any] | None) -> tuple[FieldPair, ...]:
    """Convert context filters into exact payload pairs (school/author/language/book_id)."""
    if not metadata_filters:
        return ()
    pairs: list[FieldPair] = []
    school = metadata_filters.get("school")
    author = metadata_filters.get("author")
    if not school and author:
        school = infer_school(str(author))
    if school:
        pairs.append(("school", str(school).strip().lower()))
    language = metadata_filters.get("language")
    if language:
        pairs.append(("language", str(language)))
    book_id = metadata_filters.get("book_id")
    if book_id is not None and str(book_id) != "":
        pairs.append(("book_id", str(book_id)))
    return tuple(pairs)


def _is_transit_section(section_title: str | None) -> bool:
    return bool(section_title and re.search(r"транзит", section_title, re.IGNORECASE))


def _title_tier(payload: Mapping[str, Any], rank_fields: Sequence[str]) -> tuple[int, int]:
    """Return (matched, available) counts of the variant's fields re-derived from the title."""
    section_title = payload.get("section_title")
    if not section_title or not rank_fields:
        return 0, 0
    title_meta = extract_chunk_metadata(
        str(section_title),
        "",
        language=payload.get("language"),
        school=str(payload.get("school") or "vronsky"),
    )
    matched = 0
    available = 0
    for field_name in rank_fields:
        expected = payload.get(field_name)
        if expected is None:
            continue
        available += 1
        actual = getattr(title_meta, field_name, None)
        if actual is not None and actual == expected:
            matched += 1
    return matched, available


def _build_filter(variant: QueryVariant, extra_pairs: Sequence[FieldPair]) -> Any:
    from qdrant_client import models

    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in tuple(variant.conditions) + tuple(extra_pairs)
    ]
    return models.Filter(must=conditions)


class StructuralRouter:
    """Exact-match routing over one structural Qdrant collection."""

    def __init__(self, *, client: Any | None = None, collection_name: str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        self.collection_name = collection_name or DEFAULT_COLLECTION
        if client is not None:
            self.client = client
            return

        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is missing from the environment or .env")
        if not api_key:
            raise RuntimeError("QDRANT_API_KEY is missing from the environment or .env")
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=120)

    def lookup(
        self,
        query: ElementQuery,
        *,
        limit: int = 1,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> ElementMatch:
        """Return up to ``limit`` ranked candidates for one element query."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        extra_pairs = metadata_pairs(metadata_filters)
        fetch_limit = max(limit, MAX_CANDIDATES_TO_FETCH)

        seen: dict[str, Candidate] = {}
        for variant in query.variants:
            points, _offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=_build_filter(variant, extra_pairs),
                limit=fetch_limit,
                with_payload=True,
                with_vectors=False,
            )
            for point in points or []:
                payload = point.payload or {}
                chunk_id = str(payload.get("chunk_id") or point.id)
                if chunk_id in seen:
                    continue
                section_title = (
                    str(payload["section_title"]) if payload.get("section_title") else None
                )
                tier, available = _title_tier(payload, variant.rank_fields)
                seen[chunk_id] = Candidate(
                    chunk_id=chunk_id,
                    title=str(payload.get("title") or ""),
                    content=str(payload.get("content") or ""),
                    section_title=section_title,
                    section_path=tuple(str(part) for part in (payload.get("section_path") or []) if part),
                    book_id=str(payload["book_id"]) if payload.get("book_id") is not None else None,
                    chunk_index=payload.get("chunk_index")
                    if isinstance(payload.get("chunk_index"), int)
                    else None,
                    match_tier=tier,
                    rank_total=available,
                    is_transit=_is_transit_section(section_title),
                )

        ranked = sorted(seen.values(), key=lambda candidate: candidate.sort_key())
        return ElementMatch(query=query, candidates=tuple(ranked[:limit]), total_found=len(ranked))

    def lookup_all(
        self,
        queries: Sequence[ElementQuery],
        *,
        per_element: int = 1,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> dict[str, ElementMatch]:
        """Look up every element query, keyed by the element key."""
        return {
            query.key: self.lookup(query, limit=per_element, metadata_filters=metadata_filters)
            for query in queries
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()


def get_structural_interpretations(
    chart_data: Mapping[str, Any],
    metadata_filters: Mapping[str, Any] | None = None,
    *,
    per_element: int = 1,
    client: Any | None = None,
    collection_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Map every discrete chart element to its structural knowledge-base chunk.

    ``metadata_filters`` accepts ``school``, ``author``, ``book_id`` and
    ``language`` (default ``"ru"``). ``per_element=1`` (production) returns
    exactly one best chunk per element; larger values return up to that many
    ranked candidates for manual review and evaluation.
    """
    if per_element < 1:
        raise ValueError("per_element must be greater than zero")
    filters = dict(metadata_filters or {})
    filters.setdefault("language", DEFAULT_LANGUAGE)

    queries = decompose_chart(chart_data)
    router = StructuralRouter(client=client, collection_name=collection_name)
    try:
        matches = router.lookup_all(queries, per_element=per_element, metadata_filters=filters)
    finally:
        if client is None:
            router.close()
    return {key: match.to_dict() for key, match in matches.items()}
