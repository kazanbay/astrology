from pathlib import Path

from scripts.compare_retrievers import _aggregate, _read_queries, _section


def test_read_queries_combines_file_and_inline_without_duplicates(tmp_path: Path):
    path = tmp_path / "queries.txt"
    path.write_text("# комментарий\nПервый вопрос\n\nВторой вопрос\nПервый вопрос\n", encoding="utf-8")

    queries = _read_queries(path, ["Второй вопрос", "Третий вопрос"])

    assert queries == ["Второй вопрос", "Третий вопрос", "Первый вопрос"]


def test_section_prefers_section_path():
    assert _section({"section_path": ["Том 3", "Глава", "Раздел"]}) == "Том 3 > Глава > Раздел"
    assert _section({"hierarchy_path": ["Глава"]}) == "Глава"
    assert _section({}) == ""


def test_aggregate_computes_scores_latency_and_unique_share():
    rows = {
        "fixed_500": [
            {
                "hits": [
                    {"score": 0.8, "section": "A"},
                    {"score": 0.6, "section": "B"},
                ],
                "latency_ms": 10.0,
                "unique_share": 0.5,
            },
            {
                "hits": [{"score": 0.7, "section": "A"}],
                "latency_ms": 20.0,
                "unique_share": 1.0,
            },
        ],
        "semantic_paragraph": [
            {
                "hits": [{"score": 0.9, "section": "A"}],
                "latency_ms": 30.0,
                "unique_share": 0.0,
            }
        ],
    }

    aggregate = _aggregate(rows)

    assert aggregate["fixed_500"]["mean_top1_score"] == 0.75
    assert aggregate["fixed_500"]["mean_topk_score"] == 0.7
    assert aggregate["fixed_500"]["mean_latency_ms"] == 15.0
    assert aggregate["fixed_500"]["mean_unique_share"] == 0.75
    assert aggregate["fixed_500"]["queries_with_hits"] == 2
    assert aggregate["semantic_paragraph"]["mean_sections"] == 1.0
