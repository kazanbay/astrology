import re

import pytest

from chunking import chunker


class WordTokenizer:
    def __call__(self, text, *, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


@pytest.fixture
def fake_tokenizer(monkeypatch):
    tokenizer = WordTokenizer()
    monkeypatch.setattr(chunker, "_get_tokenizer", lambda: tokenizer)
    return tokenizer


def test_chunks_keep_h2_h3_hierarchy_in_content_and_metadata(fake_tokenizer):
    chunks = chunker.chunk_markdown(
        42,
        "Guide",
        "## Law\n\nIntro text.\n\n### Exemptions\n\nDetails here.",
    )
    assert len(chunks) == 2
    assert chunks[0]["content"].startswith("[Document: Guide | Section: Law]")
    assert chunks[0]["metadata"] == {"h2": "Law", "h3": None}
    assert "[Document: Guide | Section: Law > Exemptions]" in chunks[1]["content"]
    assert chunks[1]["metadata"] == {"h2": "Law", "h3": "Exemptions"}
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(chunk["document_id"] == 42 for chunk in chunks)


def test_long_section_splits_to_token_limit_with_overlap(fake_tokenizer, monkeypatch):
    monkeypatch.setattr(chunker, "CHUNK_SIZE", 40)
    monkeypatch.setattr(chunker, "CHUNK_OVERLAP", 8)
    words = [f"word{i}" for i in range(100)]
    chunks = chunker.chunk_markdown(1, "Title", "## Long\n\n" + " ".join(words))
    assert len(chunks) > 2
    assert all(chunk["token_count"] <= 40 for chunk in chunks)
    assert chunks[0]["content"].split()[-8:] == chunks[1]["content"].split()[5:13]


def test_empty_markdown_returns_no_chunks(fake_tokenizer):
    assert chunker.chunk_markdown(1, "Title", "\n  \n") == []


def test_non_string_markdown_is_rejected(fake_tokenizer):
    with pytest.raises(TypeError):
        chunker.chunk_markdown(1, "Title", None)
