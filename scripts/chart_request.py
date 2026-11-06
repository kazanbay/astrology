"""Local birth data -> UTC -> cache-aside chart request -> stored history.

The script is the backend entry point the Telegram bot uses: it resolves the
place, converts local time to UTC, asks ``ChartPipeline`` for the chart and
prints whether PostgreSQL served the result from ``charts_cache`` or the chart
had to be computed. Run it twice with the same input to see a miss and a hit.

    uv run python scripts/chart_request.py "1990-08-15T12:00:00" "Москва"

Exit code 0 means the request was persisted and the structural lookup found at
least one section.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astronomy.calculator import AstronomyError  # noqa: E402
from generator.report import ChartReportService  # noqa: E402
from generator.synthesizer import SynthesisError  # noqa: E402
from rag.structural_router import DEFAULT_COLLECTION  # noqa: E402
from storage.birth import (  # noqa: E402
    BirthData,
    BirthInputError,
    detect_timezone,
    geocode_city,
    local_to_utc,
    parse_local_datetime,
)
from storage.pipeline import ChartPipeline  # noqa: E402
from storage.presentation import format_chart_reply  # noqa: E402
from storage.store import PostgresChartStore  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - encoding is a convenience, not a requirement
            pass


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


async def run(args: argparse.Namespace) -> int:
    print("Входные данные:")
    print(f"  дата/время (локальное): {args.datetime_local}")
    print(f"  город: {args.city}")
    print(f"  система домов: {args.house_system}; язык чанков: {args.language}")

    moment_local = parse_local_datetime(args.datetime_local)

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

    birth = BirthData(
        dt_utc=utc,
        latitude=geo["lat"],
        longitude=geo["lon"],
        house_system=args.house_system,
        language=args.language,
        city=str(geo["display_name"]),
        timezone=tz_name,
        datetime_local=moment_local.isoformat(),
    )

    started = _stage(4, "Кэш-адресация: users -> chart_hash -> charts_cache -> история")
    store = await PostgresChartStore.connect()
    pipeline = ChartPipeline(
        store,
        collection_name=args.collection,
        per_element=args.top,
    )
    report_result = None
    try:
        result = await pipeline.load(
            birth,
            telegram_id=args.telegram_id,
            email=args.email,
            profile_name=args.profile_name,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        status = "ПОПАДАНИЕ" if result.cache_hit else "ПРОМАХ"
        print(f"  chart_hash: {result.chart_hash}")
        print(
            f"  кэш: {status}; карта {'взята из charts_cache' if result.cache_hit else 'рассчитана и сохранена'}; "
            f"{'новая строка' if result.cache_written else 'строка уже существовала'}"
        )
        print(f"  user_id: {result.user_id}; request_id: {result.request_id}; профиль: {result.profile_name}")
        print(f"  время запроса: {elapsed_ms:.0f} мс")
        _done(started)

        chart = result.chart
        planets = chart.get("planets") or {}
        aspects = chart.get("aspects") or []
        cusps = chart.get("cusps") or []
        print("\n=== 5. Карта ===")
        print(f"  планет: {len(planets)}; домов (куспидов): {len(cusps)}; аспектов: {len(aspects)}")
        asc = chart.get("asc") or {}
        mc = chart.get("mc") or {}
        print(f"  ASC: {asc.get('sign')} {float(asc.get('longitude') or 0):.2f}°")
        print(f"  MC: {mc.get('sign')} {float(mc.get('longitude') or 0):.2f}°")

        chunks = result.chunks
        answered = {
            str(element.get("key"))
            for chunk in chunks
            for element in chunk.get("elements") or []
            if element.get("key")
        }
        print("\n=== 6. Структурные разделы (сырые чанки из charts_cache) ===")
        print(f"  элементов с найденными статьями: {len(answered)}")
        print(f"  уникальных разделов: {len(chunks)}")
        for index, chunk in enumerate(chunks[: args.preview], start=1):
            title = chunk.get("section_title") or chunk.get("title") or "—"
            elements = ", ".join(str(item.get("key")) for item in chunk.get("elements") or [])
            print(f"\n  [{index}] «{title}»")
            print(f"      chunk_id: {chunk.get('chunk_id')}; book: {chunk.get('book_id')}")
            print(f"      элементы: {elements or '—'}")
            preview = " ".join(str(chunk.get("content") or "").split())
            print(f"      {preview[:320]}{'…' if len(preview) > 320 else ''}")

        print("\n=== 7. Ответ для бота ===")
        print(format_chart_reply(result))

        print("\n=== 8. История пользователя ===")
        history = await pipeline.history(result.user_id, limit=args.history_limit)
        for index, record in enumerate(history, start=1):
            place = record.input_params.get("city") or "—"
            moment = record.input_params.get("dt_utc") or "—"
            print(
                f"  [{index}] {record.requested_at:%d.%m.%Y %H:%M} · {record.profile_name} · "
                f"{place} · {moment} · {record.chart_hash[:8]}"
            )

        if args.report:
            print("\n=== 9. LLM-разбор (кэш llm_generations) ===")
            report_service = ChartReportService(store)
            started_report = time.perf_counter()
            try:
                report_result = await report_service.get_or_generate(result)
            except SynthesisError as exc:
                print(f"  ОШИБКА генерации: {exc}")
            else:
                print(
                    f"  модель: {report_result.model_name}; промпт: {report_result.prompt_version}; "
                    f"temperature={report_result.temperature}"
                )
                print(f"  generation_id: {report_result.generation_id}")
                print(
                    f"  кэш: {'ПОПАДАНИЕ' if report_result.cache_hit else 'ПРОМАХ'}; "
                    f"{time.perf_counter() - started_report:.1f} с; "
                    f"{len(report_result.text)} символов"
                )
                print("\n" + report_result.text)
    finally:
        await store.close()

    if args.json_out:
        report = {
            "input": {
                "datetime_local": args.datetime_local,
                "city": args.city,
                "house_system": args.house_system,
                "language": args.language,
            },
            "geocoding": geo,
            "timezone": tz_name,
            "utc": utc.isoformat(),
            "cache": {
                "chart_hash": result.chart_hash,
                "hit": result.cache_hit,
                "written": result.cache_written,
                "user_id": result.user_id,
                "request_id": result.request_id,
            },
            "chart": {
                "planets": len(result.chart.get("planets") or {}),
                "houses": len(result.chart.get("cusps") or []),
                "aspects": len(result.chart.get("aspects") or []),
            },
            "chunks": result.chunks,
            "report": None
            if report_result is None
            else {
                "generation_id": report_result.generation_id,
                "model_name": report_result.model_name,
                "prompt_version": report_result.prompt_version,
                "temperature": report_result.temperature,
                "cache_hit": report_result.cache_hit,
                "chars": len(report_result.text),
            },
            "history": [
                {
                    "request_id": record.request_id,
                    "chart_hash": record.chart_hash,
                    "profile_name": record.profile_name,
                    "requested_at": record.requested_at.isoformat(),
                    "input_params": record.input_params,
                }
                for record in history
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json_out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Данные рождения -> UTC -> кэш карты в PostgreSQL -> структурные чанки",
    )
    parser.add_argument("datetime_local", help="локальное время без смещения, например 1990-08-15T12:00:00")
    parser.add_argument("city", help="город, например Москва")
    parser.add_argument("--house-system", default="P", help="код системы домов (по умолчанию P — Плацидус)")
    parser.add_argument("--language", default="ru", help="язык чанков (по умолчанию ru)")
    parser.add_argument("--top", type=int, default=1, help="кандидатов на элемент: 1 — продакшен")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="коллекция Qdrant")
    parser.add_argument(
        "--telegram-id",
        type=int,
        default=0,
        help="Telegram ID пользователя; 0 — служебный профиль CLI (по умолчанию 0)",
    )
    parser.add_argument("--email", help="email вместо Telegram ID (для будущих клиентов)")
    parser.add_argument("--profile-name", default="Я", help="имя профиля в истории (по умолчанию «Я»)")
    parser.add_argument("--preview", type=int, default=3, help="сколько чанков показать в превью")
    parser.add_argument("--history-limit", type=int, default=5, help="сколько записей истории показать")
    parser.add_argument("--timeout", type=float, default=15.0, help="таймаут Nominatim, секунды")
    parser.add_argument("--lat", type=float, help="широта (пропускает геокодинг)")
    parser.add_argument("--lon", type=float, help="долгота (пропускает геокодинг)")
    parser.add_argument("--timezone", help="IANA-таймзона (пропускает timezonefinder)")
    parser.add_argument("--json-out", type=Path, help="сохранить полный отчёт в JSON")
    parser.add_argument(
        "--report",
        action="store_true",
        help="сгенерировать LLM-разбор карты (с проверкой кэша llm_generations)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top должен быть больше нуля")
    if args.preview < 1:
        parser.error("--preview должен быть больше нуля")
    if args.history_limit < 1:
        parser.error("--history-limit должен быть больше нуля")
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat и --lon задаются вместе")
    if args.lat is None and args.timezone:
        print("Примечание: --timezone задан, таймзона по координатам не определяется")

    try:
        return asyncio.run(run(args))
    except (BirthInputError, AstronomyError) as exc:
        print(f"\nОШИБКА: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as exc:
        print(f"\nОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
