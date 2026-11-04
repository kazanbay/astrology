"""Run retrieval and grounded answer generation from the command line."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.generator import RAGGenerator  # noqa: E402
from rag.retriever import QdrantRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the astrology RAG knowledge base")
    parser.add_argument("query", nargs="*", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks (default: 5)")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Вопрос: ").strip()
    if not query:
        parser.error("query must not be empty")
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")

    pipeline_started = time.perf_counter()
    retriever = QdrantRetriever()
    generator = None
    try:
        chunks, timings = retriever.search_with_timings(query, top_k=args.top_k)
        generation_started = time.perf_counter()
        generator = RAGGenerator()
        answer = generator.generate_answer(query, chunks)
        generation_seconds = time.perf_counter() - generation_started

        print("\nИсточники:")
        if chunks:
            for rank, chunk in enumerate(chunks, start=1):
                section = chunk.get("metadata") or {}
                print(f"{rank}. {chunk.get('title') or 'Без названия'} — score {chunk.get('score', 0.0):.4f}; раздел: {section}")
        else:
            print("Релевантные фрагменты не найдены.")
        print("\nОтвет:\n")
        print(answer)
        total_seconds = time.perf_counter() - pipeline_started
        measured_seconds = timings["embedding_seconds"] + timings["qdrant_seconds"] + generation_seconds
        print("\nВремя:")
        print(f"  Embedding: {timings['embedding_seconds']:.3f} с")
        print(f"  Qdrant: {timings['qdrant_seconds']:.3f} с")
        print(f"  Генерация: {generation_seconds:.3f} с")
        print(f"  Pipeline (embedding + Qdrant + LLM): {measured_seconds:.3f} с")
        print(f"  Полное время выполнения: {total_seconds:.3f} с")
    finally:
        retriever.close()
        if generator is not None:
            generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
