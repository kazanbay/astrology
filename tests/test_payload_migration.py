from indexing.payload_migration import METADATA_KEYS, rebuild_payload


def test_rebuild_payload_replaces_legacy_metadata_with_pair_fields():
    legacy = {
        "chunk_id": "chunk-1",
        "content": "текст раздела",
        "section_title": (
            "4.1.1. Восходящий Лунный Узел в знаке Овна. "
            "Нисходящий Лунный Узел в знаке Весов"
        ),
        "section_path": ["Том 4", "4.1.1. Восходящий Лунный Узел..."],
        "language": "ru",
        "school": "vronsky",
        "planet": "Moon",
        "sign": "Aries",
        "house": None,
        "aspect": "conjunction",
        "meta_tags": {"planets": ["Moon"], "houses": []},
    }

    rebuilt = rebuild_payload(legacy)

    assert rebuilt["planet"] == "NorthNode"
    assert rebuilt["planet_b"] == "SouthNode"
    assert rebuilt["sign"] == "Aries"
    assert rebuilt["sign_b"] == "Libra"
    assert rebuilt["aspect"] is None
    assert rebuilt["is_transit"] is False
    assert "meta_tags" not in rebuilt
    assert set(METADATA_KEYS) <= set(rebuilt)


def test_rebuild_payload_marks_transits():
    rebuilt = rebuild_payload(
        {
            "chunk_id": "chunk-2",
            "content": "текст",
            "section_title": "2.2.10. ТРАНЗИТ СОЛНЦА ЧЕРЕЗ X ПОЛЕ ГОРОСКОПА",
            "language": "ru",
            "school": "vronsky",
        }
    )

    assert rebuilt["planet"] == "Sun"
    assert rebuilt["house"] == 10
    assert rebuilt["is_transit"] is True
