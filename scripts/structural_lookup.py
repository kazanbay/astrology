"""Manual-review report: N structural candidates per chart element.

Prints the exact payload filter used for every chart element and lists up to N
ranked chunks per element, so the "one chunk is enough per element" assumption
can be checked by eye. Production routing (one candidate) lives in
src/rag/structural_router.py; this script is the debug companion.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.structural_router import (  # noqa: E402
    DEFAULT_COLLECTION,
    StructuralRouter,
    canonical_planet,
    canonical_sign,
    decompose_chart,
    metadata_pairs,
)

DEFAULT_MD_OUT = PROJECT_ROOT / "reports" / "structural_lookup.md"
SNIPPET_CHARS = 600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Структурный точный поиск: показать N кандидатов на элемент карты",
    )
    chart = parser.add_argument_group("натальная карта (через калькулятор)")
    chart.add_argument("--utc", help="момент UTC, например 1990-08-15T12:00:00+00:00")
    chart.add_argument("--lat", type=float, help="широта, -90..90")
    chart.add_argument("--lon", type=float, help="долгота (восточная положительная), -180..180")
    chart.add_argument("--house-system", default="P", help="код системы домов (по умолчанию P)")
    chart.add_argument("--node-type", default="mean", choices=("mean", "true"))

    manual = parser.add_argument_group("ручные элементы (без калькулятора)")
    manual.add_argument(
        "--planet-sign",
        nargs=2,
        action="append",
        metavar=("ПЛАНЕТА", "ЗНАК"),
        help="например: --planet-sign Sun Leo",
    )
    manual.add_argument(
        "--planet-house",
        nargs=2,
        action="append",
        metavar=("ПЛАНЕТА", "ДОМ"),
        help="например: --planet-house Saturn 2",
    )
    manual.add_argument(
        "--aspect",
        nargs=3,
        action="append",
        metavar=("ТЕЛО_A", "ТЕЛО_B", "АСПЕКТ"),
        help="порядок планет не важен, например: --aspect Mercury Saturn trine",
    )

    parser.add_argument("--top", type=int, default=5, help="кандидатов на элемент (по умолчанию 5)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="имя коллекции Qdrant")
    parser.add_argument("--school", help="школа, например vronsky")
    parser.add_argument("--author", help="автор; школа выводится из имени, если --school не задан")
    parser.add_argument("--language", default="ru", help="язык чанков (по умолчанию ru)")
    parser.add_argument("--book-id", help="ограничиться одной книгой")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT, help="куда писать Markdown-отчёт")
    parser.add_argument("--no-md", action="store_true", help="не писать Markdown-отчёт")
    parser.add_argument("--json-out", type=Path, help="полный отчёт в JSON (опционально)")
    return parser


def build_chart_data(args: argparse.Namespace) -> dict[str, Any]:
    """Build the chart dictionary either from the calculator or from manual elements."""
    if args.utc and (args.planet_sign or args.planet_house or args.aspect):
        raise SystemExit("Используйте либо --utc, либо ручные элементы, но не оба сразу")

    if args.utc:
        from astronomy.calculator import AstronomyError, calculate_natal_chart

        if args.lat is None or args.lon is None:
            raise SystemExit("--utc требует --lat и --lon")
        try:
            moment = datetime.fromisoformat(args.utc)
        except ValueError as exc:
            raise SystemExit(f"Не удалось разобрать --utc: {exc}") from exc
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise SystemExit("--utc должен содержать смещение UTC, например 1990-08-15T12:00:00+00:00")
        try:
            chart = calculate_natal_chart(
                moment, args.lat, args.lon, args.house_system, node_type=args.node_type
            )
        except (AstronomyError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        return chart.to_dict()

    planets: dict[str, dict[str, Any]] = {}
    for name, sign in args.planet_sign or []:
        planets.setdefault(canonical_planet(name), {})["sign"] = canonical_sign(sign)
    for name, house in args.planet_house or []:
        planets.setdefault(canonical_planet(name), {})["house"] = int(house)
    aspects = [
        {"body_a": first, "body_b": second, "aspect": aspect}
        for first, second, aspect in args.aspect or []
    ]
    if not planets and not aspects:
        raise SystemExit("Укажите --utc с координатами или хотя бы один ручной элемент")
    return {"planets": planets, "aspects": aspects}


def _candidate_block(candidate: dict[str, Any]) -> list[str]:
    snippet = " ".join(str(candidate.get("content") or "").split())
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[: SNIPPET_CHARS - 1].rstrip() + "…"
    section = candidate.get("section_title") or "—"
    header = (
        f"{candidate['match_tier']}/{candidate.get('rank_total') or 0}"
        f" «{section}» — {candidate['title'] or 'без названия'}"
    )
    meta = (
        f"chunk_id `{candidate['chunk_id']}`"
        f" · book `{candidate.get('book_id') or '—'}`"
        f" · chunk_index {candidate.get('chunk_index') if candidate.get('chunk_index') is not None else '—'}"
    )
    return [f"1. **[{header}]**", f"   {meta}", f"   > {snippet}"]


def render_markdown(
    *,
    matches: list[Any],
    collection: str,
    top: int,
    filters: dict[str, Any],
    per_variant_note: int,
) -> str:
    pairs = ", ".join(f"{key}={value}" for key, value in metadata_pairs(filters)) or "нет"
    lines = [
        "# Структурный поиск: ручная проверка кандидатов",
        "",
        f"- Дата: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Коллекция: `{collection}`",
        f"- Кандидатов на элемент: {top}",
        f"- Фильтры метаданных: {pairs}",
        f"- Элементов в запросе: {len(matches)}",
        "",
        "> Ранжирование кандидатов: `tier` — сколько полей элемента названо в заголовке раздела"
        " (2/2 — заголовок называет элемент целиком, 1/2 — частично, 0/2 — метаданные взяты"
        " из текста). Внутри одного tier — порядок книги. Число найденных ограничено выборкой"
        f" до {per_variant_note} точек на вариант запроса.",
    ]
    for match in matches:
        query = match.query
        lines += [
            "",
            f"## `{query.key}` — {query.display} ({query.kind})",
            "",
            f"Точный фильтр: `{query.filter_label}`",
            f"Найдено уникальных чанков: **{match.total_found}**; показано: {len(match.candidates)}",
            "",
        ]
        if not match.candidates:
            lines.append("Кандидатов нет — элемент не найден по точным тегам.")
            continue
        for rank, candidate in enumerate(match.candidates, start=1):
            block = _candidate_block(candidate.to_dict())
            block[0] = block[0].replace("1.", f"{rank}.", 1)
            lines.extend(block)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be greater than zero")

    chart_data = build_chart_data(args)
    queries = decompose_chart(chart_data)
    filters = {
        "school": args.school,
        "author": args.author,
        "language": args.language,
        "book_id": args.book_id,
    }

    router = StructuralRouter(collection_name=args.collection)
    try:
        matches = [
            router.lookup(query, limit=args.top, metadata_filters=filters) for query in queries
        ]
    finally:
        router.close()

    for index, match in enumerate(matches, start=1):
        print(f"[{index}/{len(matches)}] {match.query.key} ({match.query.kind})")
        print(f"    фильтр: {match.query.filter_label}")
        print(f"    найдено: {match.total_found}; показано: {len(match.candidates)}")
        for rank, candidate in enumerate(match.candidates, start=1):
            section = candidate.section_title or "—"
            print(
                f"      {rank}. [tier {candidate.match_tier}/{candidate.rank_total}]"
                f" «{section}» ({candidate.chunk_id})"
            )
        if not match.candidates:
            print("      — ничего не найдено —")

    if not args.no_md:
        report = render_markdown(
            matches=matches,
            collection=args.collection,
            top=args.top,
            filters=filters,
            per_variant_note=100,
        )
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(report, encoding="utf-8")
        print(f"\nMarkdown: {args.md_out}")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection": args.collection,
            "per_element": args.top,
            "metadata_filters": dict(metadata_pairs(filters)),
            "elements": [
                {
                    "key": match.query.key,
                    "kind": match.query.kind,
                    "display": match.query.display,
                    "filter": match.query.filter_label,
                    "total_found": match.total_found,
                    "candidates": [candidate.to_dict() for candidate in match.candidates],
                }
                for match in matches
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
