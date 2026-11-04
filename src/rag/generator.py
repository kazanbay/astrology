"""Grounded answer generation through the OpenRouter chat completions API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"


SYSTEM_PROMPT = """Ты — помощник по книгам об астрологии.
Отвечай на вопрос пользователя, опираясь на предоставленные фрагменты книг.
Не выдавай сведения, которых нет в контексте, за содержание источников. Если
найденных фрагментов недостаточно, прямо скажи об этом. Различай утверждения
авторов и общепринятые научные факты; не представляй астрологические трактовки
как научно доказанные закономерности. Пиши ясно и указывай раздел источника,
если он доступен."""

def _section_label(metadata: Any) -> str:
    if isinstance(metadata, dict):
        hierarchy = metadata.get("hierarchy_path")
        if isinstance(hierarchy, (list, tuple)) and hierarchy:
            return " > ".join(str(value).strip() for value in hierarchy if value)
        section_names = [metadata.get("chapter_title"), metadata.get("section_title")]
        if any(section_names):
            return " > ".join(str(value).strip() for value in section_names if value)
        parts = [str(value).strip() for value in metadata.values() if value]
        return " > ".join(parts)
    if isinstance(metadata, (list, tuple)):
        return " > ".join(str(value).strip() for value in metadata if value)
    return str(metadata).strip() if metadata else ""


class RAGGenerator:
    """Generate answers grounded in retrieved chunks using OpenRouter."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        if client is None:
            project_root = Path(__file__).resolve().parents[2]
            load_dotenv(project_root / ".env")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is missing from the environment or .env")
            from openai import OpenAI

            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                timeout=90,
            )
        self.client = client

    @staticmethod
    def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
        """Format chunk content and its existing source references for the model."""
        if not retrieved_chunks:
            return "Контекст не найден. Не делай выводов без источников."

        formatted: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = str(chunk.get("title") or "Без названия").strip()
            section = _section_label(chunk.get("metadata"))
            header = f"### Источник {index}: {title}"
            if section:
                header += f"\nРаздел: {section}"
            chunk_id = chunk.get("chunk_id")
            if chunk_id is not None:
                header += f"\nID фрагмента: {chunk_id}"
            content = str(chunk.get("content") or "").strip()
            # Include explicit source fields when available; URLs embedded in content
            # remain unchanged and are consequently available for citation as well.
            source_url = chunk.get("source_url") or chunk.get("url")
            if source_url and str(source_url) not in content:
                content = f"{content}\nИсточник: {source_url}".strip()
            formatted.append(f"{header}\n\n{content}".rstrip())
        return "\n\n---\n\n".join(formatted)

    def generate_answer(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        """Ask the configured model to answer using only the retrieved context."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        context = self.format_context(retrieved_chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Вопрос пользователя:\n{query.strip()}\n\nКонтекст базы знаний:\n{context}",
                },
            ],
            temperature=0.1,
            max_tokens=1600,
        )
        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an empty or invalid response") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("OpenRouter returned an empty answer")
        return answer.strip()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
