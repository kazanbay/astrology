"""Run a sample semantic query against the astrology_chunks collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import QdrantRetriever


def _section_label(metadata: dict[str, Any]) -> str:
    path = metadata.get("hierarchy_path") or []
    if path:
        return " > ".join(str(part) for part in path)
    return " > ".join(
        str(metadata[key])
        for key in ("chapter_title", "section_title")
        if metadata.get(key)
    ) or "—"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Search query; prompted when omitted")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Search query: ").strip()

    retriever = QdrantRetriever()
    try:
        results, timings = retriever.search_with_timings(query, top_k=5)
    finally:
        retriever.close()

    if not results:
        print("No matching chunks found.")
    for rank, result in enumerate(results, start=1):
            # Добавляем chunk_id и document_id в заголовок
            doc_id = result.get("document_id")
            chunk_id = result.get("chunk_id")
            
            print(f"\n#{rank}  score={result['score']:.4f}  [doc_id={doc_id}, chunk_id={chunk_id}]")
            print(f"{result['title']}  |  {_section_label(result['metadata'])}")
            content = result["content"]
            snippet = content if len(content) <= 1200 else content[:1199].rstrip() + "…"
            print(snippet)

    print(
        "\nLatency: "
        f"embedding={timings['embedding_seconds']:.3f}s, "
        f"Qdrant={timings['qdrant_seconds']:.3f}s, "
        f"total={timings['total_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
