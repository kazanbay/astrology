"""Initialize the PostgreSQL schema used by the astrology RAG project.

The module owns every table of the project: the knowledge base (schools,
authors, books, chunks) and the persistent layer of the chart pipeline
(users, charts_cache, llm_generations, user_chart_requests). All statements are
idempotent, so running the script on an existing database only adds what is
missing.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

import asyncpg

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from storage.config import resolve_database_url  # noqa: E402


DDL_STATEMENTS: Sequence[str] = (
    """CREATE TABLE IF NOT EXISTS schools (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL,
        description TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS authors (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        bio TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS books (
        id SERIAL PRIMARY KEY,
        school_id INT REFERENCES schools(id) ON DELETE SET NULL,
        title VARCHAR(255) NOT NULL,
        volume_num INT,
        volume_title VARCHAR(255),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS book_authors (
        book_id INT REFERENCES books(id) ON DELETE CASCADE,
        author_id INT REFERENCES authors(id) ON DELETE CASCADE,
        PRIMARY KEY (book_id, author_id)
    )""",
    """CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        book_id INT REFERENCES books(id) ON DELETE CASCADE,
        chapter_title TEXT,
        section_title TEXT,
        hierarchy_path TEXT[],
        raw_text TEXT NOT NULL,
        clean_text TEXT NOT NULL,
        meta_tags JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_chunks_meta_tags_gin ON chunks USING GIN (meta_tags)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks (book_id)",
    # --- Persistent layer of the natal chart pipeline -----------------------
    # 1. Universal users table: Telegram for the MVP, email is reserved for
    #    the future web/app clients.
    """CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        telegram_id BIGINT UNIQUE,
        email VARCHAR(255) UNIQUE,
        first_name VARCHAR(255),
        last_name VARCHAR(255),
        display_name VARCHAR(255),
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    # Existing databases may predate the onboarding flow.
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(255)",
    # 2. Cache shared by every user with the same birth data (twins included).
    #    chart_hash is SHA-256 over the canonical birth inputs and the RAG
    #    language, so one row holds both the ephemeris snapshot and the exact
    #    knowledge-base chunks retrieved for it.
    """CREATE TABLE IF NOT EXISTS charts_cache (
        chart_hash VARCHAR(64) PRIMARY KEY,
        input_params JSONB NOT NULL,
        calculated_chart JSONB NOT NULL,
        retrieved_chunks JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    # 3. Cached LLM reports: one row per (chart, prompt version, model, temperature).
    """CREATE TABLE IF NOT EXISTS llm_generations (
        generation_id VARCHAR(64) PRIMARY KEY,
        chart_hash VARCHAR(64) NOT NULL REFERENCES charts_cache(chart_hash) ON DELETE CASCADE,
        prompt_version VARCHAR(32) NOT NULL,
        model_name VARCHAR(128) NOT NULL,
        temperature NUMERIC(3, 2) NOT NULL,
        generated_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_llm_generations_chart_hash ON llm_generations (chart_hash)",
    # 4. Per-user request history: many profiles per user, one chart hash each.
    """CREATE TABLE IF NOT EXISTS user_chart_requests (
        request_id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        chart_hash VARCHAR(64) NOT NULL REFERENCES charts_cache(chart_hash) ON DELETE CASCADE,
        generation_id VARCHAR(64) REFERENCES llm_generations(generation_id) ON DELETE SET NULL,
        profile_name VARCHAR(100) DEFAULT 'Я',
        requested_at TIMESTAMP DEFAULT NOW()
    )""",
    # Databases created before the generation layer keep their rows; this adds
    # the column in place (idempotent).
    "ALTER TABLE user_chart_requests ADD COLUMN IF NOT EXISTS generation_id "
    "VARCHAR(64) REFERENCES llm_generations(generation_id) ON DELETE SET NULL",
    "CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users (telegram_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_requests_user_id ON user_chart_requests (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_requests_hash ON user_chart_requests (chart_hash)",
    # History is always read as "latest N of this user", which the single-column
    # index cannot serve without a sort.
    "CREATE INDEX IF NOT EXISTS idx_user_requests_user_requested_at "
    "ON user_chart_requests (user_id, requested_at DESC)",
)


async def healthcheck(connection: asyncpg.Connection) -> None:
    """Raise an exception if the database connection cannot run a query."""
    await connection.fetchval("SELECT 1")


async def init_db() -> None:
    """Create the RAG schema and indexes if they do not already exist."""
    database_url = resolve_database_url()
    connection = await asyncpg.connect(database_url)
    try:
        await healthcheck(connection)
        async with connection.transaction():
            for statement in DDL_STATEMENTS:
                await connection.execute(statement)
    finally:
        await connection.close()


def main() -> None:
    asyncio.run(init_db())
    print("Database connection verified; schema and indexes are ready.")


if __name__ == "__main__":
    main()
