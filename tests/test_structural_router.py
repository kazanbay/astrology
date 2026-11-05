from types import SimpleNamespace

import pytest

from rag.structural_router import (
    COVERAGE_PAIR,
    MAX_CANDIDATES_TO_FETCH,
    StructuralRouter,
    canonical_aspect_pair,
    decompose_chart,
    get_structural_interpretations,
)


class FakeClient:
    """Records scroll calls and replays preloaded payloads."""

    def __init__(self, payloads=()):
        self.payloads = [dict(item) for item in payloads]
        self.scroll_calls = []

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        points = [
            SimpleNamespace(id=index, payload=dict(payload))
            for index, payload in enumerate(self.payloads, start=1)
        ]
        return points[: kwargs.get("limit")], None

    def close(self):
        pass


def payload(
    section_title,
    planet,
    *,
    planet_b=None,
    sign=None,
    sign_b=None,
    house=None,
    house_b=None,
    aspect=None,
    is_transit=False,
    chunk_index=0,
    content="текст раздела",
    book_id="book-1",
):
    return {
        "chunk_id": f"chunk-{chunk_index}",
        "title": "Вронский. Астрология",
        "content": content,
        "section_title": section_title,
        "section_path": [section_title] if section_title else [],
        "book_id": book_id,
        "chunk_index": chunk_index,
        "language": "ru",
        "school": "vronsky",
        "planet": planet,
        "planet_b": planet_b,
        "sign": sign,
        "sign_b": sign_b,
        "house": house,
        "house_b": house_b,
        "aspect": aspect,
        "is_transit": is_transit,
    }


def must_pairs(scroll_filter):
    return [(condition.key, condition.match.value) for condition in scroll_filter.must]


def single_query(chart):
    queries = decompose_chart(chart)
    assert len(queries) == 1
    return queries[0]


def test_decompose_chart_builds_sign_and_house_queries():
    queries = decompose_chart(
        {"planets": {"sun": {"sign": "leo", "house": 5}, "Saturn": {"house": 2}}, "aspects": []}
    )

    assert [query.key for query in queries] == [
        "Sun_in_Leo",
        "Sun_in_House_5",
        "Saturn_in_House_2",
    ]
    assert queries[0].kind == "planet_in_sign"
    assert queries[0].variants[0].conditions == (("planet", "Sun"), ("sign", "Leo"))
    assert queries[0].variants[0].rank_fields == ("planet", "sign")
    assert len(queries[0].variants) == 1  # Sun never heads a two-subject title
    assert queries[1].variants[0].conditions == (("planet", "Sun"), ("house", 5))
    assert queries[2].variants[0].conditions == (("planet", "Saturn"), ("house", 2))


def test_node_queries_also_check_the_secondary_spelling():
    queries = decompose_chart(
        {"planets": {"SouthNode": {"sign": "Leo", "house": 10}}, "aspects": []}
    )

    assert [variant.conditions for variant in queries[0].variants] == [
        (("planet", "SouthNode"), ("sign", "Leo")),
        (("planet_b", "SouthNode"), ("sign_b", "Leo")),
    ]
    assert [variant.conditions for variant in queries[1].variants] == [
        (("planet", "SouthNode"), ("house", 10)),
        (("planet_b", "SouthNode"), ("house_b", 10)),
    ]


def test_canonical_aspect_pair_sorts_and_rejects_same_body():
    assert canonical_aspect_pair("saturn", "Mercury") == ("Mercury", "Saturn")
    with pytest.raises(ValueError, match="two different bodies"):
        canonical_aspect_pair("Sun", "Sun")


def test_aspect_query_is_order_invariant_and_ignores_type_filter():
    first = decompose_chart(
        {
            "planets": {"Mercury": {}},
            "aspects": [{"body_a": "Mercury", "body_b": "Saturn", "aspect": "trine"}],
        }
    )[0]
    second = decompose_chart(
        {
            "planets": {"Mercury": {}},
            "aspects": [{"body_a": "Saturn", "body_b": "Mercury", "aspect": "trine"}],
        }
    )[0]

    assert first.key == "Mercury_Saturn_trine" == second.key
    assert first.variants == second.variants
    assert [variant.conditions for variant in first.variants] == [
        (("planet", "Mercury"), ("planet_b", "Saturn")),
        (("planet", "Saturn"), ("planet_b", "Mercury")),
    ]
    assert first.coverage == COVERAGE_PAIR


@pytest.mark.parametrize(
    "chart",
    [
        None,
        {},
        {"planets": {}},
        {"planets": {"Sun": {}}, "aspects": "trine"},
        {"planets": {"Sun": {"house": 13}}},
        {"planets": {"Sun": {"sign": "Марс"}}},
        {
            "planets": {"Sun": {}},
            "aspects": [{"body_a": "Sun", "body_b": "Sun", "aspect": "trine"}],
        },
        {
            "planets": {"Sun": {}},
            "aspects": [{"body_a": "Sun", "body_b": "Mars"}],
        },
    ],
)
def test_decompose_rejects_malformed_chart(chart):
    with pytest.raises(ValueError):
        decompose_chart(chart)


def test_lookup_sends_exact_filter_and_ranks_title_matches_first():
    payloads = [
        payload("Общая глава", "Sun", sign="Leo", chunk_index=1),
        payload("Солнце во Льве", "Sun", sign="Leo", chunk_index=2, content="целая глава"),
        payload("Солнце в Деве", "Sun", sign="Leo", chunk_index=3),
    ]
    client = FakeClient(payloads)
    router = StructuralRouter(client=client, collection_name="test-collection")
    query = single_query({"planets": {"Sun": {"sign": "Leo"}}, "aspects": []})

    match = router.lookup(query, limit=2)

    assert client.scroll_calls[0]["collection_name"] == "test-collection"
    assert client.scroll_calls[0]["limit"] == MAX_CANDIDATES_TO_FETCH
    assert must_pairs(client.scroll_calls[0]["scroll_filter"]) == [
        ("planet", "Sun"),
        ("sign", "Leo"),
    ]
    assert [candidate.chunk_index for candidate in match.candidates] == [2, 3]
    assert [candidate.match_tier for candidate in match.candidates] == [2, 1]
    assert match.candidates[0].content == "целая глава"
    assert match.total_found == 3


def test_lookup_returns_requested_number_of_candidates():
    payloads = [
        payload("Солнце во Льве", "Sun", sign="Leo", chunk_index=index)
        for index in range(4)
    ]
    router = StructuralRouter(client=FakeClient(payloads))
    query = single_query({"planets": {"Sun": {"sign": "Leo"}}, "aspects": []})

    match = router.lookup(query, limit=3)

    assert len(match.candidates) == 3
    assert match.total_found == 4
    assert match.to_dict()["candidates"][0]["chunk_id"] == "chunk-0"


def test_aspect_lookup_matches_the_pair_in_both_orders():
    payloads = [payload("ВЕНЕРА – УРАН", "Venus", planet_b="Uranus", chunk_index=7)]
    client = FakeClient(payloads)
    router = StructuralRouter(client=client, collection_name="test-collection")
    query = single_query(
        {
            "planets": {"Venus": {}, "Uranus": {}},
            "aspects": [{"body_a": "Uranus", "body_b": "Venus", "aspect": "conjunction"}],
        }
    )

    match = router.lookup(query, limit=1)

    assert len(client.scroll_calls) == 2
    assert must_pairs(client.scroll_calls[0]["scroll_filter"]) == [
        ("planet", "Uranus"),
        ("planet_b", "Venus"),
    ]
    assert must_pairs(client.scroll_calls[1]["scroll_filter"]) == [
        ("planet", "Venus"),
        ("planet_b", "Uranus"),
    ]
    assert match.total_found == 1
    assert match.candidates[0].chunk_index == 7
    assert match.to_dict()["coverage"] == COVERAGE_PAIR


def test_secondary_variant_finds_node_chapter():
    payloads = [
        payload(
            "Восходящий Лунный Узел в знаке Водолея. Нисходящий Лунный Узел в знаке Льва",
            "NorthNode",
            planet_b="SouthNode",
            sign="Aquarius",
            sign_b="Leo",
            chunk_index=3,
        )
    ]
    client = FakeClient(payloads)
    router = StructuralRouter(client=client)
    query = single_query({"planets": {"SouthNode": {"sign": "Leo"}}, "aspects": []})

    match = router.lookup(query, limit=1)

    assert len(client.scroll_calls) == 2
    assert must_pairs(client.scroll_calls[0]["scroll_filter"]) == [
        ("planet", "SouthNode"),
        ("sign", "Leo"),
    ]
    assert must_pairs(client.scroll_calls[1]["scroll_filter"]) == [
        ("planet_b", "SouthNode"),
        ("sign_b", "Leo"),
    ]
    assert match.total_found == 1
    assert match.candidates[0].chunk_index == 3


def test_lookup_prefers_natal_sections_over_transits():
    payloads = [
        payload(
            "2.2.10. ТРАНЗИТ СОЛНЦА ЧЕРЕЗ X ПОЛЕ ГОРОСКОПА",
            "Sun",
            house=10,
            is_transit=True,
            chunk_index=1,
            book_id="aaa-transits",
        ),
        payload(
            "2.3.10 Солнце в X поле гороскопа",
            "Sun",
            house=10,
            chunk_index=2,
            book_id="zzz-natal",
        ),
    ]
    router = StructuralRouter(client=FakeClient(payloads))
    query = single_query({"planets": {"Sun": {"house": 10}}, "aspects": []})

    match = router.lookup(query, limit=1)

    assert match.candidates[0].chunk_index == 2
    assert match.candidates[0].is_transit is False


def test_metadata_filters_reach_the_query():
    client = FakeClient([])
    router = StructuralRouter(client=client)
    query = single_query({"planets": {"Saturn": {"house": 2}}, "aspects": []})

    router.lookup(query, limit=1, metadata_filters={"school": "Hathor", "book_id": 5})

    assert must_pairs(client.scroll_calls[0]["scroll_filter"]) == [
        ("planet", "Saturn"),
        ("house", 2),
        ("school", "hathor"),
        ("book_id", "5"),
    ]


def test_interpretations_default_to_ru_and_author_school():
    client = FakeClient([payload("Сатурн во 2 доме", "Saturn", house=2)])

    get_structural_interpretations(
        {"planets": {"Saturn": {"house": 2}}, "aspects": []},
        {"author": "Сергей Вронский"},
        client=client,
        collection_name="test-collection",
    )

    pairs = must_pairs(client.scroll_calls[0]["scroll_filter"])
    assert ("language", "ru") in pairs
    assert ("school", "vronsky") in pairs


def test_lookup_returns_empty_match_for_unknown_element():
    router = StructuralRouter(client=FakeClient([]))
    query = single_query({"planets": {"Sun": {"sign": "Leo"}}, "aspects": []})

    match = router.lookup(query, limit=1)
    dumped = match.to_dict()

    assert match.candidates == ()
    assert match.total_found == 0
    assert dumped["chunk_id"] is None
    assert dumped["title"] is None
    assert dumped["content"] is None
    assert dumped["candidates"] == []


def test_interpretations_return_tz_shape_with_single_candidate():
    client = FakeClient(
        [
            payload("Общая глава", "Sun", sign="Leo", chunk_index=1),
            payload("Солнце во Льве", "Sun", sign="Leo", chunk_index=2),
        ]
    )

    result = get_structural_interpretations(
        {"planets": {"Sun": {"sign": "Leo", "house": 5}}, "aspects": []},
        client=client,
        collection_name="test-collection",
    )

    assert set(result) == {"Sun_in_Leo", "Sun_in_House_5"}
    entry = result["Sun_in_Leo"]
    assert entry["chunk_id"] == "chunk-2"
    assert entry["title"] == "Вронский. Астрология"
    assert entry["content"] == "текст раздела"
    assert len(entry["candidates"]) == 1
    assert entry["total_found"] == 2


def test_interpretations_return_multiple_candidates_in_debug_mode():
    client = FakeClient(
        [payload("Солнце во Льве", "Sun", sign="Leo", chunk_index=index) for index in range(3)]
    )

    result = get_structural_interpretations(
        {"planets": {"Sun": {"sign": "Leo"}}, "aspects": []},
        client=client,
        per_element=5,
    )

    entry = result["Sun_in_Leo"]
    assert len(entry["candidates"]) == 3
    assert [candidate["chunk_id"] for candidate in entry["candidates"]] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]


def test_interpretations_reject_invalid_per_element():
    with pytest.raises(ValueError, match="greater than zero"):
        get_structural_interpretations(
            {"planets": {"Sun": {"sign": "Leo"}}, "aspects": []},
            client=FakeClient(),
            per_element=0,
        )


def test_router_uses_injected_client_without_environment():
    client = FakeClient()
    router = StructuralRouter(client=client, collection_name="custom")

    assert router.collection_name == "custom"
    router.lookup(single_query({"planets": {"Sun": {"sign": "Leo"}}, "aspects": []}))
    assert client.scroll_calls[0]["collection_name"] == "custom"
