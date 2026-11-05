import pytest

from storage.chunks import collect_unique_chunks, element_chunk_index, element_stats


def candidate(chunk_id, content="Текст раздела", **overrides):
    payload = {
        "chunk_id": chunk_id,
        "title": "Классическая астрология",
        "section_title": f"Раздел {chunk_id}",
        "section_path": ["Том 1", f"Раздел {chunk_id}"],
        "book_id": "book-1",
        "chunk_index": 7,
        "content": content,
        "match_tier": 2,
        "rank_total": 2,
        "is_transit": False,
    }
    payload.update(overrides)
    return payload


def match(kind, display, *candidates, coverage="single_element"):
    return {
        "kind": kind,
        "element": display,
        "coverage": coverage,
        "total_found": len(candidates),
        "candidates": list(candidates),
    }


def test_one_chunk_answering_two_elements_is_stored_once():
    structural = {
        "NorthNode_in_Aquarius": match("planet_in_sign", "NorthNode in Aquarius", candidate("node")),
        "SouthNode_in_Leo": match("planet_in_sign", "SouthNode in Leo", candidate("node")),
    }

    chunks = collect_unique_chunks(structural)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "node"
    assert [element["key"] for element in chunks[0]["elements"]] == [
        "NorthNode_in_Aquarius",
        "SouthNode_in_Leo",
    ]
    assert chunks[0]["content"] == "Текст раздела"


def test_chunk_order_follows_first_appearance_and_keeps_rank():
    structural = {
        "Sun_in_Leo": match("planet_in_sign", "Sun in Leo", candidate("a"), candidate("b")),
        "Moon_in_Gemini": match("planet_in_sign", "Moon in Gemini", candidate("b")),
    }

    chunks = collect_unique_chunks(structural)

    assert [chunk["chunk_id"] for chunk in chunks] == ["a", "b"]
    assert [element["rank"] for element in chunks[0]["elements"]] == [0]
    assert [(element["key"], element["rank"]) for element in chunks[1]["elements"]] == [
        ("Sun_in_Leo", 1),
        ("Moon_in_Gemini", 0),
    ]


def test_chunk_payload_keeps_routing_quality_fields():
    structural = {"Mercury_Saturn_trine": match("aspect", "Mercury - Saturn", candidate("pair", match_tier=1))}

    chunk = collect_unique_chunks(structural)[0]

    assert chunk["match_tier"] == 1
    assert chunk["rank_total"] == 2
    assert chunk["is_transit"] is False
    assert chunk["section_path"] == ["Том 1", "Раздел pair"]


def test_element_chunk_index_returns_best_candidate_per_element():
    structural = {
        "Sun_in_Leo": match("planet_in_sign", "Sun in Leo", candidate("a"), candidate("b")),
        "Moon_in_Gemini": match("planet_in_sign", "Moon in Gemini", candidate("b")),
    }
    chunks = collect_unique_chunks(structural)

    index = element_chunk_index(chunks)

    assert index["Sun_in_Leo"]["chunk_id"] == "a"
    assert index["Moon_in_Gemini"]["chunk_id"] == "b"
    assert element_stats(chunks) == {"chunks": 2, "elements_with_chunks": 2}


def test_empty_structural_result_produces_empty_payload():
    assert collect_unique_chunks({}) == []
    assert element_chunk_index([]) == {}
    assert element_stats([]) == {"chunks": 0, "elements_with_chunks": 0}


def test_invalid_structural_payload_is_rejected():
    with pytest.raises(ValueError):
        collect_unique_chunks([])
    with pytest.raises(ValueError):
        collect_unique_chunks({"Sun_in_Leo": "not a mapping"})
    with pytest.raises(ValueError):
        collect_unique_chunks({"Sun_in_Leo": {"candidates": [{"title": "no id"}]}})
