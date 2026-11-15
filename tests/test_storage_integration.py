"""Integration check of the storage layer against a real PostgreSQL.

Skipped automatically when ``DATABASE_URL`` is not configured. The test writes
one user, one cached chart and one history row under a reserved test Telegram
id and removes exactly those rows afterwards.
"""

import asyncio
import hashlib
import uuid
from datetime import date, datetime, time, timezone

import pytest

from storage.config import resolve_database_url
from storage.store import (
    CachedChart,
    ForecastReportRecord,
    GenerationRecord,
    PostgresChartStore,
    SynastryReportRecord,
)

try:
    DSN: str | None = resolve_database_url()
except RuntimeError:
    DSN = None

pytestmark = pytest.mark.skipif(not DSN, reason="DATABASE_URL is not configured")

TEST_TELEGRAM_ID = 999999999999
TEST_EMAIL = "storage-integration@example.invalid"
TEST_PROFILE = "Интеграционный тест"


async def _scenario() -> None:
    chart_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    user_id: int | None = None
    store = await PostgresChartStore.connect(max_size=2)
    try:
        user_id = await store.upsert_user(
            telegram_id=TEST_TELEGRAM_ID, first_name="Тест", display_name="Интеграция"
        )
        assert await store.find_user(telegram_id=TEST_TELEGRAM_ID) == user_id
        assert await store.upsert_user(telegram_id=TEST_TELEGRAM_ID, first_name="Тест") == user_id
        assert (
            await store.pool.fetchval(
                "SELECT display_name FROM users WHERE user_id = $1", user_id
            )
            == "Интеграция"
        )

        cached = CachedChart(
            chart_hash=chart_hash,
            input_params={"city": "Москва", "dt_utc": "1990-08-15T08:00:00+00:00"},
            calculated_chart={"planets": {"Sun": {"sign": "Leo", "house": 10}}, "cusps": [0.0] * 12},
            retrieved_chunks=[{"chunk_id": "node-chapter", "content": "Текст", "elements": []}],
        )
        assert await store.save_chart(cached) is True
        assert await store.save_chart(cached) is False  # second writer loses the race

        loaded = await store.get_chart(chart_hash)
        assert loaded is not None
        assert loaded.calculated_chart == cached.calculated_chart
        assert loaded.retrieved_chunks == cached.retrieved_chunks
        assert loaded.input_params["city"] == "Москва"

        request_id = await store.record_request(
            user_id=user_id, chart_hash=chart_hash, profile_name=TEST_PROFILE
        )

        generation_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        generation = GenerationRecord(
            generation_id=generation_id,
            chart_hash=chart_hash,
            prompt_version="v1.0",
            model_name="test-model",
            temperature=0.7,
            generated_text="Текст разбора",
        )
        assert await store.save_generation(generation) is True
        assert await store.save_generation(generation) is False
        assert await store.get_generation(generation_id) == "Текст разбора"
        assert await store.get_generation("0" * 64) is None

        await store.attach_generation(request_id, generation_id)

        history = await store.list_requests(user_id, limit=5)
        assert history[0].request_id == request_id
        assert history[0].profile_name == TEST_PROFILE
        assert history[0].input_params["city"] == "Москва"
        assert history[0].generation_id == generation_id

        # Cached LLM interpretations of forecasts (RAG + LLM layer).
        forecast_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        forecast_record = ForecastReportRecord(
            report_id=forecast_id,
            chart_hash=chart_hash,
            interval="week",
            forecast_date=date(2026, 1, 15),
            prompt_version="f1.0",
            model_name="test-model",
            temperature=0.7,
            generated_text="Прогноз периода",
        )
        assert await store.save_forecast_report(forecast_record) is True
        assert await store.save_forecast_report(forecast_record) is False
        assert await store.get_forecast_report(forecast_id) == "Прогноз периода"
        assert await store.get_forecast_report("0" * 64) is None

        # Cached LLM readings of synastry pairs.
        synastry_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        synastry_record = SynastryReportRecord(
            report_id=synastry_id,
            chart_hash_a=chart_hash,
            chart_hash_b="b" * 64,
            prompt_version="s1.0",
            model_name="test-model",
            temperature=0.7,
            generated_text="Разбор пары",
        )
        assert await store.save_synastry_report(synastry_record) is True
        assert await store.save_synastry_report(synastry_record) is False
        assert await store.get_synastry_report(synastry_id) == "Разбор пары"
        assert await store.get_synastry_report("0" * 64) is None

        assert await store.get_chart("0" * 64) is None
    finally:
        if user_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", user_id)
        await store.pool.execute(
            "DELETE FROM forecast_reports WHERE chart_hash = $1", chart_hash
        )
        await store.pool.execute(
            "DELETE FROM synastry_reports WHERE chart_hash_a = $1", chart_hash
        )
        await store.pool.execute("DELETE FROM charts_cache WHERE chart_hash = $1", chart_hash)
        await store.close()


def test_postgres_store_roundtrip():
    asyncio.run(_scenario())


async def _email_identity_scenario() -> None:
    user_id: int | None = None
    store = await PostgresChartStore.connect(max_size=2)
    try:
        user_id = await store.upsert_user(email=TEST_EMAIL, first_name="Веб")
        # Email is stored lowercased, so any spelling resolves to the same user.
        assert await store.find_user(email=TEST_EMAIL.upper()) == user_id
        assert await store.upsert_user(email=TEST_EMAIL, first_name="Веб") == user_id
    finally:
        if user_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", user_id)
        await store.close()


def test_postgres_store_resolves_user_by_email():
    asyncio.run(_email_identity_scenario())


async def _profile_contacts_scenario() -> None:
    from services.profiles import ProfileChangeLimitError, save_owner_profile
    from storage.birth import BirthData
    from storage.store import ContactInput, NatalProfile

    store = await PostgresChartStore.connect(max_size=2)
    user_id: int | None = None
    other_id: int | None = None
    invited_id: int | None = None
    try:
        for stale_id in (TEST_TELEGRAM_ID, TEST_TELEGRAM_ID + 1, TEST_TELEGRAM_ID + 2):
            await store.pool.execute("DELETE FROM users WHERE telegram_id = $1", stale_id)
        await store.pool.execute(
            "DELETE FROM feedback WHERE user_id = $1", TEST_TELEGRAM_ID
        )
        user_id = await store.upsert_user(telegram_id=TEST_TELEGRAM_ID, first_name="Тест")
        birth = BirthData(
            dt_utc=datetime(1990, 8, 15, 8, 0, tzinfo=timezone.utc),
            latitude=55.7558,
            longitude=37.6173,
            city="Москва, Россия",
            timezone="Europe/Moscow",
            datetime_local="1990-08-15T12:00:00",
        )
        assert await store.save_user_natal_profile(
            user_id,
            NatalProfile(
                birth_date=date(1990, 8, 15),
                birth_time=None,
                birth_time_unknown=True,
                birth_lat=birth.latitude,
                birth_lon=birth.longitude,
                birth_timezone="Europe/Moscow",
                birth_city=birth.city,
                birth_chart_hash="a" * 64,
            ),
        ) is True

        profile = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert profile is not None
        assert profile.birth_date == date(1990, 8, 15)
        assert profile.birth_time is None
        assert profile.birth_time_unknown is True
        assert abs(profile.birth_lat - 55.7558) < 1e-9
        assert profile.birth_timezone == "Europe/Moscow"
        assert profile.birth_city == "Москва, Россия"
        assert profile.birth_chart_hash == "a" * 64

        assert await store.set_user_premium(user_id, True) is True
        profile = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert profile is not None and profile.is_premium is True

        contact = await store.add_contact(
            user_id,
            ContactInput(
                name="Мария",
                relation="супруга",
                birth_date=date(1992, 3, 15),
                birth_time=time(14, 30),
                birth_time_unknown=False,
                birth_lat=55.79,
                birth_lon=49.11,
                birth_timezone="Europe/Moscow",
                birth_city="Казань",
            ),
        )
        unknown = await store.add_contact(
            user_id,
            ContactInput(
                name="Пётр",
                relation=None,
                birth_date=date(1985, 11, 2),
                birth_time=None,
                birth_time_unknown=True,
                birth_lat=59.94,
                birth_lon=30.31,
                birth_timezone="Europe/Moscow",
                birth_city="Санкт-Петербург",
            ),
        )
        assert unknown.birth_time is None and unknown.birth_time_unknown is True

        contacts = await store.list_contacts(user_id)
        assert [item.name for item in contacts] == ["Мария", "Пётр"]

        # A contact is reachable only through its owner's id.
        other_id = await store.upsert_user(telegram_id=TEST_TELEGRAM_ID + 1, first_name="Другой")
        assert await store.get_contact(contact.contact_id, user_id=other_id) is None
        assert await store.delete_contact(contact.contact_id, user_id=other_id) is False
        assert await store.delete_contact(contact.contact_id, user_id=user_id) is True
        remaining = await store.list_contacts(user_id)
        assert [item.name for item in remaining] == ["Пётр"]
        assert await store.get_contact(contact.contact_id, user_id=user_id) is None

        # Contacts are editable, and only by their owner.
        assert await store.update_contact(
            remaining[0].contact_id,
            user_id,
            ContactInput(
                name="Пётр Иванович",
                relation="коллега",
                birth_date=date(1985, 11, 2),
                birth_time=time(9, 0),
                birth_time_unknown=False,
                birth_lat=59.94,
                birth_lon=30.31,
                birth_timezone="Europe/Moscow",
                birth_city="Санкт-Петербург",
            ),
        ) is True
        edited = await store.get_contact(remaining[0].contact_id, user_id=user_id)
        assert edited is not None
        assert edited.name == "Пётр Иванович"
        assert edited.relation == "коллега"
        assert edited.birth_time == time(9, 0)
        assert await store.update_contact(
            remaining[0].contact_id,
            other_id,
            ContactInput(
                name="Чужой",
                birth_date=date(1985, 11, 2),
                birth_lat=1.0,
                birth_lon=1.0,
                birth_timezone="UTC",
            ),
        ) is False

        # The save path of the service layer (bot flow) writes the same columns;
        # identical data is reported as unchanged and spends no change.
        profile_before = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert profile_before is not None
        assert (
            await save_owner_profile(
                store, user_id, birth, time_unknown=False, profile=profile_before
            )
            == "saved"
        )
        refreshed = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert refreshed is not None
        assert refreshed.birth_time == time(12, 0)
        assert refreshed.birth_time_unknown is False
        assert len(refreshed.birth_chart_hash) == 64
        assert refreshed.profile_changes_count == 1  # differs from the direct save above

        assert (
            await save_owner_profile(
                store, user_id, birth, time_unknown=False, profile=refreshed
            )
            == "unchanged"
        )
        same = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert same is not None and same.profile_changes_count == 1

        changed = BirthData(
            dt_utc=datetime(1990, 8, 15, 9, 0, tzinfo=timezone.utc),
            latitude=55.7558,
            longitude=37.6173,
            city="Москва, Россия",
            timezone="Europe/Moscow",
            datetime_local="1990-08-15T13:00:00",
        )
        assert (
            await save_owner_profile(
                store, user_id, changed, time_unknown=False, profile=same
            )
            == "saved"
        )
        second = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert second is not None and second.profile_changes_count == 2

        # The SQL guard refuses changes beyond the monthly limit.
        await store.pool.execute(
            "UPDATE users SET profile_changes_count = 3, "
            "profile_changes_period = date_trunc('month', now() AT TIME ZONE 'UTC')::date "
            "WHERE user_id = $1",
            user_id,
        )
        with pytest.raises(ProfileChangeLimitError):
            await save_owner_profile(
                store,
                user_id,
                birth,
                time_unknown=False,
                profile=await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID),
            )
        blocked = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert blocked is not None and blocked.profile_changes_count == 3

        # A new month restarts the allowance.
        await store.pool.execute(
            "UPDATE users SET profile_changes_period = "
            "(date_trunc('month', now() AT TIME ZONE 'UTC') - interval '1 month')::date "
            "WHERE user_id = $1",
            user_id,
        )
        reset_profile = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert reset_profile is not None
        assert (
            await save_owner_profile(
                store, user_id, birth, time_unknown=False, profile=reset_profile
            )
            == "saved"
        )
        reset = await store.get_user_profile(telegram_id=TEST_TELEGRAM_ID)
        assert reset is not None and reset.profile_changes_count == 1

        # Paid premium extends by whole months and leaves an audit event.
        until_first = await store.grant_premium(
            user_id, 1, source="manual", reason="Интеграционный тест"
        )
        assert until_first is not None
        until_second = await store.grant_premium(
            user_id,
            2,
            source="referral",
            actor_user_id=other_id,
            reference="testcode1",
            reason="Реферальный бонус",
        )
        assert until_second is not None and until_second > until_first

        # Stars payments are idempotent and audited as telegram_stars.
        charge_id = f"test-charge-{uuid.uuid4()}"
        first_payment = await store.complete_payment(
            charge_id=charge_id,
            user_id=user_id,
            plan_key="3m",
            months=3,
            stars=250,
            reason="Тариф «3 месяца» (3m)",
        )
        assert first_payment is not None and first_payment > until_second
        repeat_payment = await store.complete_payment(
            charge_id=charge_id,
            user_id=user_id,
            plan_key="3m",
            months=3,
            stars=250,
        )
        assert repeat_payment is None

        # Referral code is allocated once and resolves back to the owner.
        await store.pool.execute(
            "UPDATE users SET referral_code = NULL WHERE user_id = $1", user_id
        )
        code = await store.ensure_referral_code(user_id, "testcode1")
        assert code == "testcode1"
        assert await store.ensure_referral_code(user_id, "othercode") is None
        assert await store.get_referral_code(user_id) == "testcode1"
        referrer = await store.find_user_by_referral_code("testcode1")
        assert referrer is not None and referrer[0] == user_id

        invited_id = await store.upsert_user(
            telegram_id=TEST_TELEGRAM_ID + 2, first_name="Гость"
        )
        assert (
            await store.create_referral(
                referrer_user_id=user_id, referee_user_id=invited_id, bonus_months=1
            )
            is True
        )
        assert (
            await store.create_referral(
                referrer_user_id=user_id, referee_user_id=invited_id, bonus_months=1
            )
            is False
        )
        stats = await store.referral_stats(user_id)
        assert stats == {"invited": 1, "bonus_months": 1}

        events = await store.list_premium_events(user_id, limit=10)
        assert [event.source for event in events] == [
            "telegram_stars",
            "referral",
            "manual",
        ]
        assert events[0].stars == 250 and events[0].months == 3
        assert events[0].premium_until_before is not None
        assert events[1].actor_user_id == other_id
        assert events[1].reference == "testcode1"

        # Feedback tickets: stored, forwarded and answered from the admin chat.
        ticket = await store.add_feedback(user_id, "tester", "Не приходит прогноз")
        assert ticket.status == "new"
        assert ticket.username == "tester"
        await store.attach_feedback_message(ticket.id, -1001234567890, 555)
        found = await store.find_feedback_by_admin_message(-1001234567890, 555)
        assert found is not None and found.id == ticket.id
        assert await store.mark_feedback_answered(ticket.id) is True
        assert await store.find_open_feedback(user_id) is None
        assert await store.mark_feedback_answered(999999999) is False
    finally:
        if user_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", user_id)
        if other_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", other_id)
        if invited_id is not None:
            await store.pool.execute("DELETE FROM users WHERE user_id = $1", invited_id)
        await store.pool.execute(
            "DELETE FROM feedback WHERE user_id = $1", TEST_TELEGRAM_ID
        )
        await store.close()


def test_postgres_store_natal_profile_and_contacts():
    asyncio.run(_profile_contacts_scenario())
