"""Initialize the PostgreSQL schema used by the astrology RAG project."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence

import asyncpg
from dotenv import load_dotenv


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
)


def _get_database_url() -> str:
    """Load DATABASE_URL from .env or the process environment."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to the project's .env file "
            "or export it in the environment before initializing the database."
        )

    # Accept SQLAlchemy-style asyncpg URLs as well as native asyncpg URLs.
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = "postgresql://" + database_url.removeprefix("postgresql+asyncpg://")
    return database_url


async def healthcheck(connection: asyncpg.Connection) -> None:
    """Raise an exception if the database connection cannot run a query."""
    await connection.fetchval("SELECT 1")


async def init_db() -> None:
    """Create the RAG schema and indexes if they do not already exist."""
    database_url = _get_database_url()
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
