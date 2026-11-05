"""Local birth data -> geocoding -> timezone -> natal chart -> structural chunks.

End-to-end pipeline for manual review:

1. Nominatim (OpenStreetMap) geocodes the city into latitude/longitude.
2. ``timezonefinder`` resolves the IANA time zone for those coordinates.
3. The naive local datetime is converted to UTC using historical tz rules.
4. ``calculate_natal_chart`` (pyswisseph) builds the chart dictionary.
5. ``get_structural_interpretations`` queries Qdrant by exact payload tags.

Every stage prints what it produced, so a failed step is visible without a
debugger. The script exits with code 0 only when the structural lookup returned
at least one section.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astronomy.calculator import AstronomyError, calculate_natal_chart  # noqa: E402
from rag.structural_router import (  # noqa: E402
    DEFAULT_COLLECTION,
    get_structural_interpretations,
)
from storage.birth import (  # noqa: E402
    BirthInputError,
    detect_timezone,
    geocode_city,
    local_to_utc,
)

PREVIEW_CHARS = 320

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - encoding is a convenience, not a requirement
            pass


#: Birth-input failures now come from the shared storage layer; the script
#: keeps the short name so its error handling and messages stay unchanged.
PipelineError = BirthInputError


def _stage(index: int, title: str) -> float:
    print(f"\n=== {index}. {title} ===")
    return time.perf_counter()


def _done(started: float, detail: str = "") -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000
    suffix = f" — {detail}" if detail else ""
    print(f"  OK ({elapsed_ms:.0f} мс){suffix}")


def _format_offset(offset: timedelta) -> str:
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"UTC{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def build_chart(dt_utc: datetime, lat: float, lon: float, house_system: str) -> Any:
    try:
        return calculate_natal_chart(dt_utc, lat, lon, house_system)
    except AstronomyError as exc:
        raise PipelineError(f"Swiss Ephemeris: {exc}") from exc
    except ValueError as exc:
        raise PipelineError(f"Некорректные параметры карты: {exc}") from exc


def _preview_text(content: str, limit: int = PREVIEW_CHARS) -> str:
    snippet = " ".join(str(content or "").split())
    if len(snippet) > limit:
        snippet = snippet[: limit - 1].rstrip() + "…"
    return snippet or "—"


def run(args: argparse.Namespace) -> int:
    print("Входные данные:")
    print(f"  дата/время (локальное): {args.datetime_local}")
    print(f"  город: {args.city}")
    print(f"  система домов: {args.house_system}")

    moment_local = datetime.fromisoformat(args.datetime_local)
    if moment_local.tzinfo is not None and moment_local.utcoffset() is not None:
        raise PipelineError(
            "Указывайте локальное время без смещения: таймзона определяется по городу"
        )

    started = _stage(1, "Геокодинг (Nominatim / OpenStreetMap)")
    if args.lat is not None:
        geo = {
            "lat": args.lat,
            "lon": args.lon,
            "display_name": f"координаты из аргументов ({args.city})",
            "place_type": "coords",
        }
    else:
        geo = geocode_city(args.city, timeout=args.timeout)
    print(f"  место: {geo['display_name']}")
    print(f"  координаты: {geo['lat']:.4f}, {geo['lon']:.4f} (тип: {geo['place_type']})")
    _done(started)

    started = _stage(2, "Таймзона (timezonefinder)")
    tz_name = args.timezone or detect_timezone(geo["lat"], geo["lon"])
    print(f"  IANA-таймзона: {tz_name}")
    _done(started)

    started = _stage(3, "Перевод локального времени в UTC")
    utc, offset, notes = local_to_utc(moment_local, tz_name)
    print(f"  локальное время: {moment_local.isoformat()} ({_format_offset(offset)})")
    print(f"  UTC: {utc.isoformat()}")
    for note in notes:
        print(f"  ВНИМАНИЕ: {note}")
    _done(started)

    started = _stage(4, "Астрономический расчёт (pyswisseph)")
    chart = build_chart(utc, geo["lat"], geo["lon"], args.house_system)
    chart_dict = chart.to_dict()
    planets = chart_dict.get("planets") or {}
    aspects = chart_dict.get("aspects") or []
    cusps = chart_dict.get("cusps") or []
    if not planets:
        raise PipelineError("Калькулятор вернул пустой список планет")
    if len(cusps) != 12:
        raise PipelineError(f"Ожидалось 12 куспидов домов, получено {len(cusps)}")
    print(f"  планет: {len(planets)}; домов (куспидов): {len(cusps)}; аспектов: {len(aspects)}")
    print(
        f"  ASC: {chart.asc.sign} {chart.asc.longitude:.2f}°; "
        f"MC: {chart.mc.sign} {chart.mc.longitude:.2f}°"
    )
    _done(started, f"ключи карты: {', '.join(sorted(chart_dict))}")

    started = _stage(5, "Структурный поиск по базе Вронского (Qdrant, точные метаданные)")
    result = get_structural_interpretations(
        chart_dict,
        {"language": args.language},
        per_element=args.top,
        collection_name=args.collection,
    )
    elements = len(result)
    with_hits = [key for key, entry in result.items() if entry["candidates"]]
    empty = [key for key, entry in result.items() if not entry["candidates"]]
    total_sections = sum(int(entry.get("total_found") or 0) for entry in result.values())
    unique_sections = {
        candidate["chunk_id"]
        for entry in result.values()
        for candidate in entry["candidates"]
    }
    print(f"  коллекция: {args.collection}; кандидатов на элемент: {args.top}")
    print(
        f"  элементов карты: {elements}; с найденными статьями: {len(with_hits)}; "
        f"без статей: {len(empty)}"
    )
    print(f"  точных совпадений по всем элементам: {total_sections} (с повторами)")
    print(f"  уникальных разделов в выборке: {len(unique_sections)}")
    if empty:
        shown = ", ".join(empty[:10]) + ("…" if len(empty) > 10 else "")
        print(f"  нет точного совпадения: {shown}")
    _done(started)
    if total_sections == 0:
        raise PipelineError(
            "Точный поиск не вернул ни одного раздела — проверьте коллекцию и фильтры метаданных"
        )

    print("\n=== ИТОГО ===")
    print(
        f"Найдено разделов в базе знаний Вронского: {len(unique_sections)} уникальных "
        f"(top-{args.top} на элемент); всего точных совпадений: {total_sections}"
    )

    previews: list[tuple[str, dict[str, Any]]] = []
    for key, entry in result.items():
        if entry["candidates"]:
            previews.append((key, entry["candidates"][0]))
    limit = min(len(previews), args.preview)
    print(f"\nПревью первых {limit} чанков:")
    for index, (key, candidate) in enumerate(previews[:limit], start=1):
        section = candidate.get("section_title") or "—"
        print(f"\n  [{index}] {key} — «{section}»")
        print(f"      chunk_id: {candidate.get('chunk_id')}; book: {candidate.get('book_id')}")
        print(f"      {_preview_text(candidate.get('content'))}")

    if args.json_out:
        report = {
            "input": {
                "datetime_local": args.datetime_local,
                "city": args.city,
                "house_system": args.house_system,
            },
            "geocoding": geo,
            "timezone": tz_name,
            "utc": utc.isoformat(),
            "chart": {
                "planets": len(planets),
                "houses": len(cusps),
                "aspects": len(aspects),
                "asc": {"sign": chart.asc.sign, "longitude": chart.asc.longitude},
                "mc": {"sign": chart.mc.sign, "longitude": chart.mc.longitude},
            },
            "structural": {
                "collection": args.collection,
                "per_element": args.top,
                "elements": elements,
                "elements_with_hits": len(with_hits),
                "elements_without_hits": empty,
                "total_sections": total_sections,
                "unique_sections": len(unique_sections),
                "elements_detail": result,
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON: {args.json_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Локальные данные рождения → координаты → таймзона → натальная карта → точные чанки Вронского",
    )
    parser.add_argument("datetime_local", help="локальное время без смещения, например 1990-08-15T12:00:00")
    parser.add_argument("city", help="город, например Москва")
    parser.add_argument("--house-system", default="P", help="код системы домов (по умолчанию P — Плацидус)")
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="кандидатов на элемент: 1 — продакшен, больше — для ручной проверки",
    )
    parser.add_argument("--preview", type=int, default=5, help="сколько чанков показать в превью")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="коллекция Qdrant")
    parser.add_argument("--language", default="ru", help="язык чанков (по умолчанию ru)")
    parser.add_argument("--timeout", type=float, default=15.0, help="таймаут Nominatim, секунды")
    parser.add_argument("--lat", type=float, help="широта (пропускает геокодинг)")
    parser.add_argument("--lon", type=float, help="долгота (пропускает геокодинг)")
    parser.add_argument("--timezone", help="IANA-таймзона (пропускает timezonefinder)")
    parser.add_argument("--json-out", type=Path, help="сохранить полный отчёт в JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top должен быть больше нуля")
    if args.preview < 1:
        parser.error("--preview должен быть больше нуля")
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat и --lon задаются вместе")
    if args.timezone and args.lat is None:
        print("Примечание: --timezone задан, таймзона по координатам не определяется")

    try:
        return run(args)
    except PipelineError as exc:
        print(f"\nОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
