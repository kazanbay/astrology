import pytest

from chunking.tokens import count_tokens, split_spans, split_text, trim_to_token_limit


def test_count_tokens_ignores_blank_text(word_tokenizer):
    assert count_tokens("", word_tokenizer) == 0
    assert count_tokens("   \n", word_tokenizer) == 0
    assert count_tokens("один два три", word_tokenizer) == 3


def test_split_text_respects_limit_and_overlap(word_tokenizer):
    text = " ".join(f"word{index}" for index in range(40))
    chunks = split_text(text, limit=10, overlap=2, tokenizer=word_tokenizer, prefer_boundary=False)

    assert all(count_tokens(chunk, word_tokenizer) <= 10 for chunk in chunks)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert chunks[-1].split()[-1] == "word39"


def test_split_spans_prefers_line_boundary(word_tokenizer):
    text = "a b c\nd e f g h"
    spans = split_spans(text, limit=4, overlap=0, tokenizer=word_tokenizer, prefer_boundary=True)
    assert len(spans) >= 2
    assert text[spans[0][0] : spans[0][1]] == "a b c"
    assert text[spans[1][0] : spans[1][1]].startswith("d e f")
    assert text[spans[0][1] : spans[1][0]] == "\n"


def test_trim_to_token_limit_returns_prefix(word_tokenizer):
    text = " ".join(f"word{index}" for index in range(20))
    trimmed = trim_to_token_limit(text, 5, word_tokenizer)
    assert count_tokens(trimmed, word_tokenizer) <= 5
    assert text.startswith(trimmed)
    assert trimmed != text


def test_invalid_limit_is_rejected(word_tokenizer):
    with pytest.raises(ValueError, match="greater than zero"):
        split_spans("текст", limit=0, overlap=0, tokenizer=word_tokenizer)
