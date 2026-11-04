"""Telegram entry point for the astrology knowledge base assistant."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from rag.generator import RAGGenerator
from rag.retriever import QdrantRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GREETING = (
    "Здравствуйте! Я помощник базы знаний по астрологии. "
    "Задайте вопрос, и я поищу ответ в материалах книг."
)


def create_dispatcher(retriever: QdrantRetriever, generator: RAGGenerator) -> Dispatcher:
    """Create handlers bound to the single startup instances of the RAG services."""
    dispatcher = Dispatcher()
    router = Router()

    @router.message(CommandStart())
    async def start_handler(message: Message) -> None:
        await message.answer(GREETING)

    @router.message()
    async def text_handler(message: Message, bot: Bot) -> None:
        user_query = message.text
        if not user_query:
            return

        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        chunks = await asyncio.to_thread(retriever.search, user_query, top_k=5)
        answer = await asyncio.to_thread(generator.generate_answer, user_query, chunks)

        try:
            await message.answer(
                answer, 
                parse_mode=ParseMode.HTML, 
                disable_web_page_preview=True
            )
        except TelegramBadRequest:
            # Если вдруг HTML упадет, отправляем как чистый текст
            await message.answer(answer, disable_web_page_preview=True)

    dispatcher.include_router(router)
    return dispatcher


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from the environment or .env")

    # Build the potentially heavyweight clients once, before polling starts.
    retriever = QdrantRetriever()
    try:
        generator = RAGGenerator()
    except Exception:
        retriever.close()
        raise

    bot = Bot(token=token)
    dispatcher = create_dispatcher(retriever, generator)
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()
        retriever.close()
        generator.close()


if __name__ == "__main__":
    asyncio.run(main())
