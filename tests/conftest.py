import re

import pytest

from chunking.document import BookDocument, Section


class WordTokenizer:
    """Deterministic tokenizer: one token per whitespace-separated word."""

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=False,
        **kwargs,
    ):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


@pytest.fixture
def word_tokenizer():
    return WordTokenizer()


@pytest.fixture
def sample_document():
    sections = (
        Section(
            title="Глава 1. Сатурн",
            paragraphs=(
                "Сатурн в седьмом доме указывает на серьёзные партнёрские обязательства. " * 4,
                "Второй абзац главы объясняет ограничения и ответственность в браке. " * 4,
            ),
            children=(
                Section(
                    title="1.1. Сатурн в Овне",
                    paragraphs=(
                        "Сатурн в Овне требует самостоятельности и терпения. " * 4,
                        "Аспекты Сатурна к Марсу усиливают борьбу за независимость. " * 4,
                    ),
                ),
            ),
        ),
        Section(
            title="Глава 2. Луна",
            paragraphs=(
                "Луна в первом доме делает эмоции заметными для окружающих. " * 4,
                "Лунные фазы описывают циклы настроения и привычек. " * 4,
            ),
        ),
    )
    return BookDocument(
        source="sample.fb2",
        book_id="sample-book",
        title="Домология",
        author="С. А. Вронский",
        series="Классическая астрология",
        volume="Том 3",
        sections=sections,
    )


@pytest.fixture
def long_document():
    sections = []
    for chapter in range(1, 4):
        paragraphs = tuple(
            f"Глава {chapter}, абзац {paragraph}: " + "слово " * 120 for paragraph in range(1, 4)
        )
        sections.append(Section(title=f"Глава {chapter}", paragraphs=paragraphs))
    return BookDocument(
        source="long.fb2",
        book_id="long-book",
        title="Длинная книга",
        author="А. Автор",
        series=None,
        volume="Том 1",
        sections=tuple(sections),
    )
