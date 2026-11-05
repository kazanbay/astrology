from chunking.tags import extract_chunk_metadata


def meta(title, text=""):
    return extract_chunk_metadata(title, text)


def test_node_section_keeps_both_nodes_and_signs():
    extracted = meta(
        "4.1.1. Восходящий Лунный Узел в знаке Овна. Нисходящий Лунный Узел в знаке Весов"
    )

    assert (extracted.planet, extracted.planet_b) == ("NorthNode", "SouthNode")
    assert (extracted.sign, extracted.sign_b) == ("Aries", "Libra")


def test_lunar_node_is_not_tagged_as_moon():
    extracted = meta(
        "4.1.3. Восходящий Лунный Узел в знаке Близнецов. Нисходящий Лунный Узел в знаке Стрельца"
    )

    assert extracted.planet != "Moon"
    assert extracted.planet == "NorthNode"


def test_pair_title_keeps_the_second_body():
    extracted = meta("1.6. ВЕНЕРА – УРАН")

    assert (extracted.planet, extracted.planet_b) == ("Venus", "Uranus")
    assert extracted.sign is None
    assert extracted.aspect is None


def test_transit_flag_and_house_are_read_from_the_title():
    extracted = meta("2.2.10. ТРАНЗИТ СОЛНЦА ЧЕРЕЗ X ПОЛЕ ГОРОСКОПА")

    assert extracted.planet == "Sun"
    assert extracted.house == 10
    assert extracted.is_transit is True


def test_node_houses_keep_both_values():
    extracted = meta("4.2.4. Восходящий Лунный Узел в IV поле Нисходящий Лунный Узел в X поле")

    assert (extracted.house, extracted.house_b) == (4, 10)


def test_body_mentions_are_ignored_for_entities():
    extracted = meta("Общие замечания", "Сатурн в VII доме создаёт напряжение.")

    assert extracted.planet is None
    assert extracted.sign is None
    assert extracted.house is None


def test_explicit_aspect_in_title_is_kept():
    extracted = meta("Трин Меркурия к Сатурну")

    assert extracted.aspect == "trine"
    assert (extracted.planet, extracted.planet_b) == ("Mercury", "Saturn")


def test_language_is_still_detected_from_the_body():
    assert meta("", "Plain english text about the Moon").language == "en"
    assert meta("", "Луна в знаке Тельца").language == "ru"


def test_natal_placement_has_no_pair_fields():
    extracted = meta("3.3.2. Луна в знаке Тельца")

    assert extracted.planet_b is None
    assert extracted.sign_b is None
    assert extracted.house is None
