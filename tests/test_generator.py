from types import SimpleNamespace

import pytest

from rag.generator import DEFAULT_MODEL, RAGGenerator, SYSTEM_PROMPT


class FakeCompletions:
    def __init__(self, content: str = "Ответ со ссылкой [источник](https://example.org)"):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def fake_client(content: str = "Ответ со ссылкой [источник](https://example.org)"):
    completions = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_format_context_keeps_titles_sections_and_source_links():
    context = RAGGenerator.format_context(
        [
            {
                "title": "Астрология",
                "chunk_id": 42,
                "metadata": {"h2": "Планеты", "h3": "Транзиты"},
                "content": "Подробности в [книге](https://example.org/book).",
            }
        ]
    )

    assert "Астрология" in context
    assert "Планеты > Транзиты" in context
    assert "ID фрагмента: 42" in context
    assert "https://example.org/book" in context


def test_generate_answer_sends_grounded_prompt_and_returns_text():
    client, completions = fake_client("Краткий ответ.")
    generator = RAGGenerator(client=client)
    chunks = [{"title": "Книга", "content": "Источник: https://example.org/book", "metadata": {"h2": "Планеты"}}]

    answer = generator.generate_answer("Что сказано в законе?", chunks)

    assert answer == "Краткий ответ."
    assert completions.kwargs["model"] == DEFAULT_MODEL
    messages = completions.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "опираясь на предоставленные фрагменты книг" in SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "Что говорится о планетах?" in messages[1]["content"]
    assert "https://example.org/book" in messages[1]["content"]


def test_empty_context_is_explicitly_marked():
    assert "Контекст не найден" in RAGGenerator.format_context([])


def test_rejects_blank_query():
    client, _ = fake_client()
    with pytest.raises(ValueError, match="non-empty"):
        RAGGenerator(client=client).generate_answer("  ", [])
