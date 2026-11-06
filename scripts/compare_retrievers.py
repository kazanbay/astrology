"""Compare retrieval quality of the four chunking collections on the same queries."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.chunking.strategies import (  # noqa: E402
    COLLECTION_NAMES,
    STRATEGIES,
    collection_name,
)

DEFAULT_QUERIES_FILE = PROJECT_ROOT / "data" / "eval_queries.txt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    title: str
    section: str
    content: str
    parent_id: str | None
    parent_chars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "title": self.title,
            "section": self.section,
            "parent_id": self.parent_id,
            "parent_chars": self.parent_chars,
            "content": self.content,
        }


def _parse_collections(value: str) -> list[str]:
    aliases: dict[str, str] = {}
    for strategy, collection in COLLECTION_NAMES.items():
        aliases[strategy.lower()] = strategy
        aliases[collection.lower()] = strategy
    selected: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item not in aliases:
            raise argparse.ArgumentTypeError(
                f"Unknown collection {raw!r}; choose from: {', '.join(STRATEGIES)}"
            )
        strategy = aliases[item]
        if strategy not in selected:
            selected.append(strategy)
    if not selected:
        raise argparse.ArgumentTypeError("At least one collection must be selected")
    return selected


def _read_queries(path: Path, inline: list[str]) -> list[str]:
    queries = [value.strip() for value in inline if value.strip()]
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                queries.append(value)
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            unique.append(query)
    return unique


def _section(payload: dict[str, Any]) -> str:
    path = payload.get("section_path") or payload.get("hierarchy_path") or []
    if isinstance(path, (list, tuple)):
        return " > ".join(str(part) for part in path if part)
    return str(path or "")


def _client() -> Any:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=120, check_compatibility=False)


def _search(client: Any, collection: str, vector: list[float], top_k: int) -> tuple[list[Hit], float]:
    started = time.perf_counter()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    hits: list[Hit] = []
    for point in response.points:
        payload = point.payload or {}
        parent = payload.get("parent_content")
        hits.append(
            Hit(
                chunk_id=str(payload.get("chunk_id") or point.id),
                score=float(point.score),
                title=str(payload.get("title") or ""),
                section=_section(payload),
                content=str(payload.get("content") or ""),
                parent_id=payload.get("parent_id"),
                parent_chars=len(parent) if isinstance(parent, str) else 0,
            )
        )
    return hits, latency_ms


def _aggregate(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for strategy, rows in results.items():
        if not rows:
            continue
        top1 = [row["hits"][0]["score"] for row in rows if row["hits"]]
        average = [
            statistics.mean(hit["score"] for hit in row["hits"]) for row in rows if row["hits"]
        ]
        latencies = [row["latency_ms"] for row in rows]
        sections = [len({hit["section"] for hit in row["hits"]}) for row in rows if row["hits"]]
        unique = [row["unique_share"] for row in rows]
        aggregate[strategy] = {
            "mean_top1_score": round(statistics.mean(top1), 4) if top1 else 0.0,
            "mean_topk_score": round(statistics.mean(average), 4) if average else 0.0,
            "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
            "mean_sections": round(statistics.mean(sections), 2) if sections else 0.0,
            "mean_unique_share": round(statistics.mean(unique), 4) if unique else 0.0,
            "queries_with_hits": len(top1),
        }
    return aggregate


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Сравнение стратегий чанкинга",
        "",
        f"Дата: {report['generated_at']}",
        f"Модель эмбеддингов: `{report['embedding_model']}`",
        f"Запросов: {len(report['queries'])}; top-k: {report['top_k']}",
        "",
        "> Cosine score зависит от размера чанка и не является абсолютной метрикой качества. "
        "Таблица показывает сигнал; итоговый выбор подтверждайте ручной разметкой или RAGAS.",
        "",
        "## Сводка",
        "",
        "| Коллекция | Точек | Top-1 | Средний top-k | Latency, мс | Уникальных чанков |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in report["selected"]:
        collection = report["collections"][strategy]["name"]
        stats = report["aggregate"].get(strategy)
        points = report["collections"][strategy].get("points")
        if not stats:
            lines.append(f"| `{collection}` | {points or 0} | — | — | — | — |")
            continue
        lines.append(
            f"| `{collection}` | {points if points is not None else '—'} | "
            f"{stats['mean_top1_score']:.4f} | {stats['mean_topk_score']:.4f} | "
            f"{stats['mean_latency_ms']:.1f} | {stats['mean_unique_share']:.2%} |"
        )

    lines += ["", "## По запросам", "", "| Запрос | Лучший top-1 | Scores по коллекциям |", "|---|---|---|"]
    for row in report["per_query"]:
        scores = ", ".join(
            f"{strategy}={row['results'][strategy]['hits'][0]['score']:.4f}"
            if row["results"][strategy]["hits"]
            else f"{strategy}=—"
            for strategy in report["selected"]
        )
        lines.append(f"| {row['query']} | {row['winner'] or '—'} | {scores} |")

    lines += ["", "## Выдача", ""]
    for row in report["per_query"]:
        lines.append(f"### {row['query']}")
        lines.append("")
        for strategy in report["selected"]:
            result = row["results"][strategy]
            collection = report["collections"][strategy]["name"]
            lines.append(f"**`{collection}`** — {result['latency_ms']:.1f} мс")
            if result.get("error"):
                lines.append(f"\nОшибка: `{result['error']}`\n")
                continue
            if not result["hits"]:
                lines.append("\nНет результатов.\n")
                continue
            for rank, hit in enumerate(result["hits"][:3], start=1):
                snippet = " ".join(hit["content"].split())
                if len(snippet) > 240:
                    snippet = snippet[:239].rstrip() + "…"
                lines.append(
                    f"{rank}. **{hit['score']:.4f}** — {hit['title']} — {hit['section']}\n"
                    f"   {snippet}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Inline queries; the file is used when omitted")
    parser.add_argument("--queries-file", type=Path, default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--collections", type=_parse_collections, default=list(STRATEGIES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT_DIR / "retrieval_comparison.json")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_REPORT_DIR / "retrieval_comparison.md")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")

    queries = _read_queries(args.queries_file, args.query)
    if not queries:
        parser.error(f"No queries supplied and {args.queries_file} is empty or missing")

    from src.rag.embedding import get_embedding_backend

    model = get_embedding_backend()
    vectors = model.encode(
        queries,
        batch_size=min(16, len(queries)),
    )
    client = _client()
    try:
        collections: dict[str, dict[str, Any]] = {}
        for strategy in args.collections:
            name = collection_name(strategy)
            info: dict[str, Any] = {"name": name, "points": None, "error": None}
            try:
                if not client.collection_exists(name):
                    info["error"] = "collection does not exist"
                else:
                    info["points"] = client.count(collection_name=name, exact=True).count
            except Exception as exc:  # noqa: BLE001 - report per-collection failures
                info["error"] = f"{type(exc).__name__}: {exc}"
            collections[strategy] = info

        per_query: list[dict[str, Any]] = []
        raw_results: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in args.collections}
        for query, vector in zip(queries, vectors, strict=True):
            row: dict[str, Any] = {"query": query, "results": {}}
            ids_by_strategy: dict[str, set[str]] = {}
            for strategy in args.collections:
                info = collections[strategy]
                if info["error"]:
                    row["results"][strategy] = {"hits": [], "latency_ms": 0.0, "error": info["error"]}
                    ids_by_strategy[strategy] = set()
                    continue
                try:
                    hits, latency_ms = _search(client, info["name"], vector, args.top_k)
                    row["results"][strategy] = {
                        "hits": [hit.as_dict() for hit in hits],
                        "latency_ms": round(latency_ms, 1),
                    }
                    ids_by_strategy[strategy] = {hit.chunk_id for hit in hits}
                except Exception as exc:  # noqa: BLE001 - one collection must not stop the report
                    row["results"][strategy] = {
                        "hits": [],
                        "latency_ms": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    ids_by_strategy[strategy] = set()

            for strategy, ids in ids_by_strategy.items():
                others: set[str] = set()
                for other_strategy, other_ids in ids_by_strategy.items():
                    if other_strategy != strategy:
                        others |= other_ids
                unique = len(ids - others) / len(ids) if ids else 0.0
                row["results"][strategy]["unique_share"] = round(unique, 4)

            scored = [
                (row["results"][strategy]["hits"][0]["score"], strategy)
                for strategy in args.collections
                if row["results"][strategy]["hits"]
            ]
            row["winner"] = max(scored)[1] if scored else None
            per_query.append(row)
            for strategy in args.collections:
                raw_results[strategy].append(row["results"][strategy])
    finally:
        client.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model.model_name,
        "top_k": args.top_k,
        "selected": args.collections,
        "queries": queries,
        "collections": collections,
        "aggregate": _aggregate(raw_results),
        "per_query": per_query,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(_markdown(report), encoding="utf-8")

    print(f"{'strategy':<22} {'points':>7} {'top1':>7} {'topk':>7} {'ms':>7} {'unique':>7}")
    for strategy in args.collections:
        stats = report["aggregate"].get(strategy)
        name = collections[strategy]["name"]
        if not stats:
            print(f"{strategy:<22} {collections[strategy]['points'] or 0:>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7}")
            continue
        print(
            f"{strategy:<22} {collections[strategy]['points'] or 0:>7} "
            f"{stats['mean_top1_score']:>7.4f} {stats['mean_topk_score']:>7.4f} "
            f"{stats['mean_latency_ms']:>7.1f} {stats['mean_unique_share']:>6.1%}"
        )
    print(f"\nJSON: {args.json_out}\nMarkdown: {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
