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
    # Profile change policy: the owner may change natal data a limited number
    # of times per calendar month (PROFILE_CHANGES_PER_MONTH). The period is
    # the first day of the month the counter belongs to (UTC).
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_changes_count INT NOT NULL DEFAULT 0",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_changes_period DATE",
    # --- Premium subscriptions, referrals and Telegram Stars payments -----
    # ``premium_until`` is the paid period; ``users.is_premium`` stays a
    # manual/permanent flag (gifts, staff). ``referral_code`` is the public
    # token of the invite link.
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS premium_until TIMESTAMPTZ",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS referral_code VARCHAR(16)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_code "
    "ON users (referral_code) WHERE referral_code IS NOT NULL",
    # One referee per account: the unique index makes a double reward impossible.
    """CREATE TABLE IF NOT EXISTS referrals (
        referral_id SERIAL PRIMARY KEY,
        referrer_user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        referee_user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        bonus_months INT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (referee_user_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_referrals_referrer ON referrals (referrer_user_id)",
    # Telegram Stars payments: charge_id is the idempotency key.
    """CREATE TABLE IF NOT EXISTS payments (
        payment_id SERIAL PRIMARY KEY,
        charge_id VARCHAR(128) UNIQUE NOT NULL,
        user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        plan_key VARCHAR(32) NOT NULL,
        months INT NOT NULL,
        stars INT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_payments_user ON payments (user_id)",
    # Audit trail of every premium grant: who received it, on what basis, for
    # how long and what the period became. ``source`` is one of manual /
    # referral / telegram_stars / other.
    """CREATE TABLE IF NOT EXISTS premium_events (
        event_id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        source VARCHAR(32) NOT NULL,
        months INT NOT NULL,
        stars INT,
        plan_key VARCHAR(32),
        actor_user_id INT REFERENCES users(user_id) ON DELETE SET NULL,
        reference VARCHAR(128),
        reason TEXT,
        premium_until_before TIMESTAMPTZ,
        premium_until_after TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_premium_events_user "
    "ON premium_events (user_id, created_at DESC)",
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
    #    prompt_version holds the Langfuse identity ("langfuse:<name>:v<n>"),
    #    which is longer than the original 32 characters.
    """CREATE TABLE IF NOT EXISTS llm_generations (
        generation_id VARCHAR(64) PRIMARY KEY,
        chart_hash VARCHAR(64) NOT NULL REFERENCES charts_cache(chart_hash) ON DELETE CASCADE,
        prompt_version VARCHAR(255) NOT NULL,
        model_name VARCHAR(128) NOT NULL,
        temperature NUMERIC(3, 2) NOT NULL,
        generated_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_llm_generations_chart_hash ON llm_generations (chart_hash)",
    # Existing databases kept the narrow column; widening is idempotent.
    "ALTER TABLE llm_generations ALTER COLUMN prompt_version TYPE VARCHAR(255)",
    # 3b. Cached LLM interpretations of transit forecasts: one row per
    #     (target chart, interval, local day, prompt version, model).
    """CREATE TABLE IF NOT EXISTS forecast_reports (
        report_id VARCHAR(64) PRIMARY KEY,
        chart_hash VARCHAR(64) NOT NULL,
        interval VARCHAR(16) NOT NULL,
        forecast_date DATE NOT NULL,
        prompt_version VARCHAR(255) NOT NULL,
        model_name VARCHAR(128) NOT NULL,
        temperature NUMERIC(3, 2) NOT NULL,
        generated_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_forecast_reports_chart "
    "ON forecast_reports (chart_hash, forecast_date)",
    # 3c. Cached LLM synastry readings: one row per (pair of charts, prompt
    #     version, model). Compatibility does not depend on the day, so the
    #     key carries no date.
    """CREATE TABLE IF NOT EXISTS synastry_reports (
        report_id VARCHAR(64) PRIMARY KEY,
        chart_hash_a VARCHAR(64) NOT NULL,
        chart_hash_b VARCHAR(64) NOT NULL,
        prompt_version VARCHAR(255) NOT NULL,
        model_name VARCHAR(128) NOT NULL,
        temperature NUMERIC(3, 2) NOT NULL,
        generated_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_synastry_reports_pair "
    "ON synastry_reports (chart_hash_a, chart_hash_b)",
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
    # --- Owner natal profile and premium contacts (forecast/synastry layer) --
    # The owner's natal data is persisted on the account so forecasts and
    # synastry are rebuilt from the stored profile only; the service layer
    # never accepts birth inputs from a request. The hash links the profile to
    # the same canonical identity as ``charts_cache``.
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_date DATE",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_time TIME",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_time_unknown BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_lat DOUBLE PRECISION",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_lon DOUBLE PRECISION",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_timezone VARCHAR(64)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_city TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS birth_chart_hash VARCHAR(64)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_premium BOOLEAN NOT NULL DEFAULT FALSE",
    # People whose charts the owner may calculate (premium feature). Rows are
    # always scoped by ``user_id``: the store resolves a contact only through
    # the pair (contact_id, user_id), so a foreign contact_id is unreachable.
    """CREATE TABLE IF NOT EXISTS user_contacts (
        contact_id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        relation VARCHAR(100),
        birth_date DATE,
        birth_time TIME,
        birth_time_unknown BOOLEAN NOT NULL DEFAULT FALSE,
        birth_lat DOUBLE PRECISION,
        birth_lon DOUBLE PRECISION,
        birth_timezone VARCHAR(64),
        birth_city TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_user_contacts_user_id ON user_contacts (user_id)",
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
