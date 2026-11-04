import pytest

from chunking.document import BookDocument, Section
from chunking.strategies import (
    FIXED_STRATEGY,
    PARENT_CHILD_STRATEGY,
    SEMANTIC_STRATEGY,
    STRUCTURAL_STRATEGY,
    build_chunks,
    collection_name,
)
from chunking.tokens import count_tokens


def _document(paragraphs: tuple[str, ...], title: str = "Раздел") -> BookDocument:
    return BookDocument(
        source="fixture.fb2",
        book_id="fixture-book",
        title="Книга",
        author="Автор",
        series=None,
        volume="Том 1",
        sections=(Section(title=title, paragraphs=paragraphs),),
    )


def test_collection_names_match_the_four_target_collections():
    assert collection_name(FIXED_STRATEGY) == "vronsky_fixed_500"
    assert collection_name(SEMANTIC_STRATEGY) == "vronsky_semantic_paragraph"
    assert collection_name(STRUCTURAL_STRATEGY) == "vronsky_structural_xml"
    assert collection_name(PARENT_CHILD_STRATEGY) == "vronsky_parent_child"


def test_unknown_strategy_is_rejected(sample_document, word_tokenizer):
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        build_chunks(sample_document, "no_such_strategy", word_tokenizer)


def test_fixed_chunks_respect_token_limit_and_overlap(word_tokenizer):
    words = [f"w{index}" for index in range(1200)]
    document = _document((" ".join(words),), title="Глава")
    chunks = build_chunks(document, FIXED_STRATEGY, word_tokenizer)

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 500 for chunk in chunks)
    first = chunks[0].content.split()
    second = chunks[1].content.split()
    assert first[-50:] == second[:50]
    assert chunks[-1].content.split()[-1] == "w1199"


def test_fixed_chunk_ids_are_deterministic(sample_document, word_tokenizer):
    first = build_chunks(sample_document, FIXED_STRATEGY, word_tokenizer)
    second = build_chunks(sample_document, FIXED_STRATEGY, word_tokenizer)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_semantic_chunks_stay_within_character_budget_and_section(word_tokenizer):
    paragraphs = tuple(("Первый абзац. " * 20).strip() for _ in range(5))
    document = _document(paragraphs + ("Короткий финальный абзац.",), title="Раздел")
    chunks = build_chunks(document, SEMANTIC_STRATEGY, word_tokenizer)

    assert len(chunks) >= 2
    assert all(chunk.char_count <= 400 for chunk in chunks)
    assert all(chunk.section_path == ("Том 1", "Раздел") for chunk in chunks)
    assert "Короткий финальный абзац." in " ".join(chunk.content for chunk in chunks)


def test_semantic_merges_tiny_section_tail(word_tokenizer):
    document = _document(("А" * 390,), title="Раздел")
    chunks = build_chunks(document, SEMANTIC_STRATEGY, word_tokenizer)

    assert len(chunks) == 1
    assert chunks[0].char_count <= 450
    assert chunks[0].content.endswith("А")


def test_semantic_chunks_do_not_merge_different_sections(word_tokenizer):
    document = BookDocument(
        source="fixture.fb2",
        book_id="fixture-book",
        title="Книга",
        author=None,
        series=None,
        volume=None,
        sections=(
            Section(title="Первый", paragraphs=("Абзац первого раздела. " * 12,)),
            Section(title="Второй", paragraphs=("Абзац второго раздела. " * 12,)),
        ),
    )
    chunks = build_chunks(document, SEMANTIC_STRATEGY, word_tokenizer)
    assert {chunk.section_path for chunk in chunks} == {("Первый",), ("Второй",)}


def test_structural_is_one_chunk_per_section(sample_document, word_tokenizer):
    chunks = build_chunks(sample_document, STRUCTURAL_STRATEGY, word_tokenizer)
    paths = [chunk.section_path for chunk in chunks]

    assert paths == [
        ("Том 3", "Глава 1. Сатурн"),
        ("Том 3", "Глава 1. Сатурн", "1.1. Сатурн в Овне"),
        ("Том 3", "Глава 2. Луна"),
    ]
    assert all(chunk.content.startswith(chunk.section_path[-1]) for chunk in chunks)


def test_structural_soft_splits_oversized_section(word_tokenizer):
    paragraph = "Очень длинное предложение про астрологию. " * 150
    document = _document((paragraph,), title="Большой раздел")
    chunks = build_chunks(document, STRUCTURAL_STRATEGY, word_tokenizer)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 1500 for chunk in chunks)
    assert all(chunk.content.startswith("Большой раздел") for chunk in chunks)
    assert all(chunk.part_count == len(chunks) for chunk in chunks)
    assert [chunk.part_index for chunk in chunks] == list(range(len(chunks)))


def test_parent_child_links_children_to_a_large_parent(long_document, word_tokenizer):
    chunks = build_chunks(long_document, PARENT_CHILD_STRATEGY, word_tokenizer)

    assert chunks
    assert all(chunk.parent_id for chunk in chunks)
    assert all(chunk.parent_content for chunk in chunks)
    assert all(chunk.token_count <= 150 for chunk in chunks)
    assert all(chunk.parent_token_count is not None and chunk.parent_token_count >= 1000 for chunk in chunks)
    assert len({chunk.parent_id for chunk in chunks}) >= 1
    parents = {chunk.parent_id: chunk.parent_content for chunk in chunks}
    for chunk in chunks:
        assert parents[chunk.parent_id] == chunk.parent_content


def test_parent_child_children_cover_the_parent_text(long_document, word_tokenizer):
    chunks = build_chunks(long_document, PARENT_CHILD_STRATEGY, word_tokenizer)
    parent = chunks[0].parent_content
    joined = " ".join(chunk.content for chunk in chunks)

    assert parent is not None
    assert "Глава 1" in joined
    assert "Глава 3" in joined
    assert sum(chunk.char_count for chunk in chunks) >= len(parent)


def test_meta_tags_are_extracted_from_chunk_content(word_tokenizer):
    document = _document(("Сатурн в VII доме создаёт напряжение.",), title="Сатурн")
    chunks = build_chunks(document, STRUCTURAL_STRATEGY, word_tokenizer)
    assert chunks[0].meta_tags["planets"] == ["Saturn"]
    assert chunks[0].meta_tags["houses"] == [7]


def test_chunk_token_count_matches_content(word_tokenizer, sample_document):
    for strategy in (FIXED_STRATEGY, SEMANTIC_STRATEGY, STRUCTURAL_STRATEGY):
        for chunk in build_chunks(sample_document, strategy, word_tokenizer):
            assert chunk.token_count == count_tokens(chunk.content, word_tokenizer)
            assert chunk.char_count == len(chunk.content)
