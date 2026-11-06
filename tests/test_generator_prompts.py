import re

from generator.prompts import ASTROLOGER_SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt


def test_prompt_is_versioned():
    assert re.fullmatch(r"v\d+\.\d+", PROMPT_VERSION)
    assert "TELEGRAM" in ASTROLOGER_SYSTEM_PROMPT
    assert "•" in ASTROLOGER_SYSTEM_PROMPT


def test_system_prompt_keeps_the_anti_hallucination_rules():
    assert "calculated_chart" in ASTROLOGER_SYSTEM_PROMPT
    assert "retrieved_chunks" in ASTROLOGER_SYSTEM_PROMPT
    assert "Никакой отсебятины" in ASTROLOGER_SYSTEM_PROMPT
    assert "звезды склоняют" in ASTROLOGER_SYSTEM_PROMPT


def test_user_prompt_embeds_chart_and_chunks():
    text = build_user_prompt(chart_json='{"planets": {}}', chunks="[1] «Раздел»")

    assert '{"planets": {}}' in text
    assert "[1] «Раздел»" in text
    assert "calculated_chart" in text
