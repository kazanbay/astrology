"""Build all chunking strategies for the FB2 corpus and load them into Qdrant Cloud."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.chunking.document import BookDocument, LoadReport, load_corpus  # noqa: E402
from src.chunking.tags import DEFAULT_SCHOOL  # noqa: E402
from src.chunking.strategies import (  # noqa: E402
    COLLECTION_NAMES,
    STRATEGIES,
    STRATEGY_DESCRIPTIONS,
    Chunk,
    build_all_chunks,
    collection_name,
)
from src.chunking.tokens import get_tokenizer  # noqa: E402
from src.indexing.qdrant_index import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ATTEMPTS,
    ensure_collection,
    upload_chunks,
)

LOGGER = logging.getLogger("build_strategy_collections")
DEFAULT_BOOKS_DIR = PROJECT_ROOT / "books" / "vronsky"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "strategy_chunks"


@dataclass(frozen=True)
class StrategyStats:
    strategy: str
    collection: str
    documents: int
    chunks: int
    parents: int
    total_tokens: int
    total_chars: int
    min_chars: int
    max_chars: int
    avg_chars: float
    avg_tokens: float


@dataclass
class StrategyRun:
    stats: StrategyStats
    upload: dict[str, Any] | None = None
    error: str | None = None


def _parse_strategies(value: str) -> list[str]:
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
                f"Unknown strategy {raw!r}; choose from: {', '.join(STRATEGIES)}"
            )
        strategy = aliases[item]
        if strategy not in selected:
            selected.append(strategy)
    if not selected:
        raise argparse.ArgumentTypeError("At least one strategy must be selected")
    return selected


def _stats(strategy: str, chunks: list[Chunk]) -> StrategyStats:
    chars = [chunk.char_count for chunk in chunks]
    tokens = [chunk.token_count for chunk in chunks]
    parents = {chunk.parent_id for chunk in chunks if chunk.parent_id}
    return StrategyStats(
        strategy=strategy,
        collection=collection_name(strategy),
        documents=len({chunk.book_id for chunk in chunks}),
        chunks=len(chunks),
        parents=len(parents),
        total_tokens=sum(tokens),
        total_chars=sum(chars),
        min_chars=min(chars, default=0),
        max_chars=max(chars, default=0),
        avg_chars=round(sum(chars) / len(chars), 1) if chars else 0.0,
        avg_tokens=round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
    )


def _print_summary(runs: Iterable[StrategyRun]) -> None:
    header = (
        f"{'strategy':<22} {'collection':<28} {'docs':>4} {'chunks':>7} "
        f"{'parents':>7} {'tokens':>9} {'avg_chars':>9} {'max_chars':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for run in runs:
        stats = run.stats
        print(
            f"{stats.strategy:<22} {stats.collection:<28} {stats.documents:>4} "
            f"{stats.chunks:>7} {stats.parents:>7} {stats.total_tokens:>9} "
            f"{stats.avg_chars:>9} {stats.max_chars:>9}"
        )
    print()


def _chunk_record(chunk: Chunk) -> dict[str, Any]:
    """Compact JSONL row; parent bodies live in a separate parents file."""
    return {
        "chunk_id": chunk.chunk_id,
        "strategy": chunk.strategy,
        "book_id": chunk.book_id,
        "title": chunk.title,
        "source": chunk.source,
        "section_path": list(chunk.section_path),
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        "part_index": chunk.part_index,
        "part_count": chunk.part_count,
        "parent_id": chunk.parent_id,
        "parent_token_count": chunk.parent_token_count,
        "language": chunk.language,
        "planet": chunk.planet,
        "sign": chunk.sign,
        "house": chunk.house,
        "aspect": chunk.aspect,
        "school": chunk.school,
        "content": chunk.content,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def dump_strategy(out_dir: Path, strategy: str, chunks: list[Chunk]) -> tuple[int, int]:
    """Write chunks and unique parent bodies for offline retrieval experiments."""
    collection = collection_name(strategy)
    chunks_path = out_dir / "chunks" / f"{collection}.jsonl"
    parents_path = out_dir / "parents" / f"{collection}.jsonl"

    parents: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        if not chunk.parent_id or chunk.parent_id in parents:
            continue
        parents[chunk.parent_id] = {
            "parent_id": chunk.parent_id,
            "strategy": chunk.strategy,
            "book_id": chunk.book_id,
            "title": chunk.title,
            "section_paths": [list(path) for path in chunk.parent_paths],
            "parent_token_count": chunk.parent_token_count,
            "content": chunk.parent_content,
        }

    chunk_count = _write_jsonl(chunks_path, (_chunk_record(chunk) for chunk in chunks))
    parent_count = _write_jsonl(parents_path, parents.values())
    LOGGER.info("Wrote %d chunks and %d parents for %s", chunk_count, parent_count, collection)
    return chunk_count, parent_count


def _qdrant_client() -> Any:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url:
        raise RuntimeError("QDRANT_URL is missing from the environment or .env")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY is missing from the environment or .env")
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=120, check_compatibility=False)


def _summary_payload(
    *,
    args: argparse.Namespace,
    report: LoadReport,
    runs: list[StrategyRun],
    embedding_model: str,
    vector_size: int | None,
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "generated_at": started_at,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "books_dir": str(args.books_dir),
        "school": args.school,
        "documents": len(report.documents),
        "load_errors": [{"source": source, "error": error} for source, error in report.errors],
        "embedding_model": embedding_model,
        "vector_size": vector_size,
        "dry_run": bool(args.dry_run),
        "strategies": {
            run.stats.strategy: {
                "description": STRATEGY_DESCRIPTIONS[run.stats.strategy],
                "collection": run.stats.collection,
                "stats": asdict(run.stats),
                "upload": run.upload,
                "error": run.error,
            }
            for run in runs
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-dir", type=Path, default=DEFAULT_BOOKS_DIR)
    parser.add_argument(
        "--school",
        default=DEFAULT_SCHOOL,
        help=f"School identifier stored in every payload (default: {DEFAULT_SCHOOL})",
    )
    parser.add_argument(
        "--only",
        type=_parse_strategies,
        default=list(STRATEGIES),
        help=f"Comma-separated strategies or collections (default: {', '.join(STRATEGIES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build chunks and write JSONL artifacts without loading the embedding model or Qdrant",
    )
    parser.add_argument("--recreate", action="store_true", help="Delete each target collection first")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Scroll target collections and skip point ids that are already present",
    )
    parser.add_argument("--limit-books", type=int, default=None, help="Debug: load only the first N documents")
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug: cap chunks per strategy")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Points per Qdrant upsert (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
        help="Texts per SentenceTransformer forward pass (default: 16)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Upload attempts per batch on network errors (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if args.embed_batch_size < 1:
        parser.error("--embed-batch-size must be greater than zero")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip() or "BAAI/bge-m3"

    report = load_corpus(args.books_dir, school=args.school)
    documents: list[BookDocument] = report.documents
    if args.limit_books is not None:
        if args.limit_books < 1:
            parser.error("--limit-books must be greater than zero")
        documents = documents[: args.limit_books]
    if not documents:
        for source, error in report.errors:
            LOGGER.error("Load error in %s: %s", source, error)
        LOGGER.error("No books loaded from %s", args.books_dir)
        return 1
    LOGGER.info("Loaded %d documents from %s", len(documents), args.books_dir)

    tokenizer = get_tokenizer(embedding_model)
    chunks_by_strategy = build_all_chunks(documents, args.only, tokenizer)

    runs: list[StrategyRun] = []
    for strategy, chunks in chunks_by_strategy.items():
        if args.max_chunks is not None:
            chunks = chunks[: args.max_chunks]
        stats = _stats(strategy, chunks)
        LOGGER.info(
            "Strategy %s: %d chunks, %d parents, %d tokens",
            strategy,
            stats.chunks,
            stats.parents,
            stats.total_tokens,
        )
        runs.append(StrategyRun(stats=stats))
        if args.dry_run:
            dump_strategy(args.out_dir, strategy, chunks)

    _print_summary(runs)

    vector_size: int | None = None
    if not args.dry_run:
        from src.rag.embedding import get_embedding_backend

        client = _qdrant_client()
        try:
            try:
                model = get_embedding_backend()
                vector_size = int(model.dimension)
            except Exception as exc:  # noqa: BLE001 - report instead of losing the summary
                LOGGER.exception("Embedding model failed")
                for run in runs:
                    run.error = f"{type(exc).__name__}: {exc}"
                model = None

            if model is not None:
                for run in runs:
                    strategy = run.stats.strategy
                    chunks = chunks_by_strategy[strategy]
                    if args.max_chunks is not None:
                        chunks = chunks[: args.max_chunks]
                    try:
                        ensure_collection(
                            client,
                            run.stats.collection,
                            vector_size=vector_size,
                            recreate=args.recreate,
                        )
                        result = upload_chunks(
                            client,
                            run.stats.collection,
                            chunks,
                            model,
                            embedding_model=embedding_model,
                            vector_size=vector_size,
                            batch_size=args.batch_size,
                            embed_batch_size=args.embed_batch_size,
                            skip_existing=args.skip_existing,
                            max_attempts=args.max_attempts,
                        )
                        run.upload = asdict(result)
                        LOGGER.info(
                            "Collection %s: uploaded %d, skipped %d, %.1fs",
                            run.stats.collection,
                            result.uploaded,
                            result.skipped,
                            result.seconds,
                        )
                    except Exception as exc:  # noqa: BLE001 - keep other collections running
                        run.error = f"{type(exc).__name__}: {exc}"
                        LOGGER.exception("Collection %s failed", run.stats.collection)
        finally:
            client.close()

    summary = _summary_payload(
        args=args,
        report=report,
        runs=runs,
        embedding_model=embedding_model,
        vector_size=vector_size,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started_perf,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)

    failed = any(run.error for run in runs) or bool(report.errors)
    if report.errors:
        LOGGER.warning("%d book(s) failed to load; see summary.json", len(report.errors))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
