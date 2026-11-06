# Astrology RAG

Python-проект для поиска по книгам по астрологии. Исходные книги хранятся в `books/`. Основной конвейер: книги → чанки в PostgreSQL (Neon) → эмбеддинги в Qdrant → поиск и ответы по найденным фрагментам. Поверх этого работает мультистратегический ETL для A/B-сравнения способов нарезки.

## Структура

```text
books/                 Исходные книги
init_db.py             Создание схемы PostgreSQL
src/chunking/          Токенизация, загрузка FB2 и четыре стратегии чанкинга
src/indexing/          Создание коллекций Qdrant и батчевая загрузка
src/astronomy/         Эфемериды, дома, знаки и аспекты натальной карты
src/storage/           Кэш карт, пользователи и история запросов (PostgreSQL)
src/rag/               Эмбеддинги, поиск в Qdrant и генерация ответов
src/generator/         Промпты, LLM-синтез разбора карты и кэш генераций
src/bot.py             Telegram-бот для вопросов к базе знаний
scripts/               ETL, сравнение ретриверов и ручные проверки
```

FB2 можно загружать вместе с метаданными и структурными чанками командой:

```bash
uv run python parse_and_ingest.py
```

Скрипт читает `.fb2` и ZIP-архивы с FB2 из `books/vronsky`, создаёт или обновляет метаданные в PostgreSQL и пакетно записывает чанки. Размер пакета можно изменить через `--batch-size 250`; путь к книгам — через `--books-dir`.

## Настройка

Установите зависимости через `uv`:

```bash
uv sync
```

Создайте `.env` в корне проекта (шаблон — `.env.example`):

```dotenv
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=123456:telegram-token

# Эмбеддинги: OpenAI-совместимый /embeddings, модель BAAI/bge-m3 (1024)
EMBEDDING_BACKEND=api
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=
EMBEDDING_BATCH_SIZE=32

# LLM: чат-модель для разбора карты
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=
LLM_MODEL=gemini-flash-lite-latest
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=180
```

`EMBEDDING_MODEL` используется и для токенизации чанков, и для эмбеддингов. OpenRouter отдаёт `BAAI/bge-m3` через `/embeddings` (1024 измерения, векторы нормированы), поэтому `EMBEDDING_API_KEY` можно не задавать: ключ `OPENROUTER_API_KEY` подставляется автоматически, но только для эндпоинта `openrouter.ai` — чужому провайдеру он не уйдёт. Для другого OpenAI-совместимого провайдера задайте `EMBEDDING_BASE_URL` и `EMBEDDING_API_KEY` явно.

`LLM_API_KEY` тоже падает обратно в `OPENROUTER_API_KEY`. Локальная модель `sentence-transformers` больше не входит в обязательный рантайм: она доступна через extra `local-embeddings` (`uv sync --extra local-embeddings`), нужный для ETL и офлайн-индексации. Для контейнера достаточно `uv sync --no-dev`.

## PostgreSQL

Проверить соединение и создать таблицы можно командой:

```bash
uv run python init_db.py
```

Скрипт создаёт `schools`, `authors`, `books`, `book_authors`, `chunks`, `users`, `charts_cache`, `llm_generations`, `user_chart_requests` и индексы для `chunks.meta_tags`, `chunks.book_id` и ключей кэша карт. Он не загружает содержимое книг.

Сначала запустите `uv run python init_db.py`, затем `uv run python parse_and_ingest.py`.

## Мультистратегический ETL и A/B-тест чанкинга

Скрипт `scripts/build_strategy_collections.py` читает FB2/ZIP и сырые `.txt`/`.md` напрямую, режет каждый документ четырьмя способами и загружает результат в четыре независимые коллекции Qdrant Cloud:

| Стратегия | Коллекция | Правило нарезки |
|---|---|---|
| `fixed_500` | `vronsky_fixed_500` | 500 токенов, overlap 50, без структуры книги |
| `semantic_paragraph` | `vronsky_semantic_paragraph` | абзацы, слияние до ~300–400 символов |
| `structural_xml` | `vronsky_structural_xml` | один `<section>`/`<title>` = один чанк; мягкий сплит при >1500 символов |
| `parent_child` | `vronsky_parent_child` | child 100–150 токенов, в payload — `parent_id`, `parent_content` и `parent_token_count` |

Сначала безопасная проверка без сети и без загрузки модели:

```bash
uv run python scripts/build_strategy_collections.py --dry-run
```

Dry-run пишет JSONL с чанками и родительскими разделами в `data/strategy_chunks/`, а также `summary.json` со счётчиками по каждой стратегии. Это удобно для проверки корпуса и последующих экспериментов с RAGAS.

Загрузка в Qdrant Cloud:

```bash
uv run python scripts/build_strategy_collections.py --recreate
```

* параметры батча: `--batch-size 100` (точек на upsert), `--embed-batch-size 16` (текстов на forward pass);
* при сетевых ошибках батч повторяется с экспоненциальной задержкой (`--max-attempts 5`);
* `--skip-existing` позволяет продолжить прерванную загрузку, не считая эмбеддинги заново;
* `--only fixed_500,parent_child` — собрать только выбранные стратегии;
* `--limit-books 1` и `--max-chunks 200` — быстрый smoke-тест;
* `--recreate` удаляет целевые коллекции перед загрузкой. Без флага существующие коллекции сохраняются, а точки перезаписываются по детерминированным UUID.

После загрузки сравните выдачу на одном наборе запросов:

```bash
uv run python scripts/compare_retrievers.py
uv run python scripts/compare_retrievers.py --top-k 10 --queries-file data/eval_queries.txt
```

Отчёт сохраняется в `reports/retrieval_comparison.md` и `reports/retrieval_comparison.json`: top-1/top-k scores, latency, доля уникальных чанков и примеры выдачи по каждой коллекции. Cosine score зависит от размера чанка, поэтому отчёт — это сигнал для отбора, а не финальная оценка качества; итоговый выбор подтверждайте ручной разметкой или RAGAS.

## RAG

После загрузки и индексации чанков можно проверить поиск и генерацию ответа:

```bash
uv run scripts/index_qdrant.py
uv run scripts/query_rag.py "Что говорится о планетах?"
```

Индексатор берёт `clean_text` из `chunks`, считает эмбеддинги BGE-M3 через API и добавляет отсутствующие UUID в `astrology_chunks`. Для полной пересборки коллекции используется `uv run scripts/index_qdrant.py --recreate`. Это отдельная legacy-коллекция; четыре `vronsky_*` коллекции создаются мультистратегическим ETL и не мешают ей.

Для запуска Telegram-бота:

```bash
uv run src/bot.py
```

Текущая модель эмбеддингов — `BAAI/bge-m3` (1024 измерения), вызовы идут через OpenAI-совместимый `/embeddings`. Qdrant использует отдельную коллекцию `astrology_chunks`, чтобы не смешивать её с индексами исходного проекта.

## Астрономический калькулятор

`src/astronomy/calculator.py` — ядро натальной карты: положения планет, куспиды домов, знаки и мажорные аспекты. Калькулятор принимает только UTC и не занимается часовыми поясами — перевод локального времени и исторических переводов стрелок остаётся на уровне бэкенда.

```bash
uv run python scripts/natal_chart.py 1990-08-15T12:00:00+00:00 55.7558 37.6173 --house-system P
```

То же самое из кода (`src` должна быть в `PYTHONPATH` — так же поступают скрипты и pytest):

```python
from datetime import datetime, timezone

from astronomy.calculator import calculate_natal_chart

chart = calculate_natal_chart(
    datetime(1990, 8, 15, 12, 0, tzinfo=timezone.utc),
    lat=55.7558,
    lon=37.6173,
    house_system="P",  # 'P' — Плацидус, 'W' — whole sign, всего 15 систем
)
chart.planets["Sun"].longitude, chart.planets["Sun"].sign, chart.asc.house
chart.to_dict()  # JSON-совместимый слепок для промптов и Langfuse
```

Планета в `to_dict()` содержит ровно шесть полей: `longitude`, `sign`, `house`, `speed`, `is_retrograde`, `is_stationary`. Аспекты по умолчанию: соединение, секстиль, квадрат, трин и оппозиция с орбисами 8°/6°/7°/8°/8°; орбисы можно переопределить через `orbs`.

Пакет `pysweph` — community-форк `pyswisseph` с тем же модулем `swisseph`: у оригинала нет колёс под Python 3.12+, и на Windows он требует MSVC. Импорт и вызовы совместимы с `pyswisseph`, поэтому замена библиотеки сводится к правке зависимости в `pyproject.toml`.

## Структурный точный поиск (без векторного поиска)

`src/rag/structural_router.py` декомпозирует натальную карту (`chart.to_dict()`) на дискретные элементы — планета в знаке, планета в доме, мажорный аспект — и для каждого находит статьи по точным фильтрам payload (`planet`/`sign`/`house`/`aspect` + `school`/`language`/`book_id`). Векторный поиск не используется. Пары планет в аспектах приводятся к каноническому (лексикографическому) виду, поэтому запрос инвариантен к порядку планет.

Метаданные структурных чанков читаются только из заголовка раздела: `planet`/`planet_b`, `sign`/`sign_b`, `house`/`house_b`, `aspect`, `is_transit`. Второе тело пары и второй знак/дом сохраняются, поэтому узлы («Восходящий/Нисходящий Лунный Узел») и аспектные пары ищутся точно. Тип аспекта в корпусе не размечен: аспектные главы называются по паре, и роутер возвращает главу пары с `coverage="pair_chapter"`. Для пересборки payload после правок извлечения: `uv run python scripts/update_strategy_payloads.py --collections structural_xml`.

По умолчанию на элемент возвращается один лучший чанк (продакшен): `get_structural_interpretations(chart.to_dict(), metadata_filters)`. Для ручной проверки и оценки передайте `per_element=N` — роутер вернёт до N ранжированных кандидатов (совпадение по заголовку раздела ранжируется выше совпадения по тексту).

Ручная проверка кандидатов из командной строки:

```bash
uv run python scripts/structural_lookup.py --planet-sign Sun Leo --planet-house Saturn 2 --aspect Mercury Saturn trine --top 5
uv run python scripts/structural_lookup.py --utc 1990-08-15T12:00:00+00:00 --lat 55.7558 --lon 37.6173 --top 5
```

Отчёт со всеми кандидатами и точными фильтрами сохраняется в `reports/structural_lookup.md`; полный текст чанков — в JSON через `--json-out`.

## Персистентный слой карт: кэш и история запросов

`src/storage/` — бэкенд-слой между калькулятором, структурным поиском и клиентами. Он решает две задачи: не считать одну и ту же карту дважды и не терять историю запросов пользователя.

Таблицы создаёт `uv run python init_db.py`; все операторы идемпотентны, повторный запуск ничего не ломает.

| Таблица | Назначение |
|---|---|
| `users` | одна запись на человека: `telegram_id` для MVP, `email` — задел под сайт и приложение |
| `charts_cache` | одна запись на уникальные данные рождения: слепок карты и найденные чанки |
| `user_chart_requests` | история: кто, под каким профилем и когда запросил карту |

`chart_hash` — SHA-256 от канонического JSON: `dt_utc` в UTC, широта и долгота с округлением до 4 знаков, код системы домов и язык чанков. Округление склеивает микропогрешности геокодера, поэтому один город не распадается на несколько кэшей, а близнецы и повторные запросы получают одну строку `charts_cache`.

Пайплайн (`src/storage/pipeline.py`) работает по схеме cache-aside:

1. `users` — UPSERT по `telegram_id`/`email`, получение внутреннего `user_id`;
2. `chart_hash` — по каноническим данным рождения;
3. `SELECT` по `charts_cache`: попадание отдаёт готовый слепок и чанки, промах запускает pyswisseph и структурный поиск Qdrant;
4. промах сохраняется с `ON CONFLICT DO NOTHING`, поэтому гонка двух одинаковых запросов не перетирает данные;
5. в `user_chart_requests` добавляется запись с именем профиля (`Я`, `Мама`, …);
6. клиент получает карту и чанки.

`retrieved_chunks` — массив уникальных чанков, а не копия ответа на каждый элемент: текст раздела хранится один раз, а связь «элемент → раздел» лежит в `elements` вместе с рангом кандидата. Глава про узел, отвечающая сразу двум элементам, не дублируется; обратное преобразование — `element_chunk_index`.

Проверка на живых сервисах:

```bash
uv run python scripts/chart_request.py "1990-08-15T12:00:00" "Москва"
uv run python scripts/chart_request.py "1990-08-15T12:00:00" "Москва"
```

Первый запуск печатает `кэш: ПРОМАХ` и время расчёта, второй — `ПОПАДАНИЕ` и время только на чтение из PostgreSQL (проверка на Neon: 10,9 с → 1,5 с). Скрипт показывает карту, найденные разделы, готовый ответ для бота и историю пользователя; `--json-out` сохраняет полный отчёт, `--lat/--lon/--timezone` пропускают геокодинг, `--telegram-id` переключает профиль (по умолчанию `0` — служебный профиль CLI).

В боте:

```text
/start          приветствие и меню
🔮 Рассчитать карту   пошаговый ввод: имя → дата → время → место
🕘 История            прошлые запросы
❓ Помощь              что умеет бот
/chart 1990-08-15 12:00 Москва   быстрый расчёт одной командой
```

`/start` показывает приветствие и постоянное меню. Кнопка «Рассчитать карту» запускает мастер: имя (можно пропустить — оно нужно только для обращения), дата, время и город рождения. Время можно не указывать — тогда берётся 12:00 с предупреждением, что дома и ASC приблизительны. Перед расчётом бот показывает контрольную сводку с кнопками «Рассчитать» и «Изменить данные».

`/chart` с аргументами остаётся для быстрого расчёта: он определяет координаты через Nominatim, таймзону через `timezonefinder`, переводит локальное время в UTC по историческим правилам, отдаёт краткую карту и генерирует разбор через LLM (с проверкой кэша `llm_generations`). `/history` показывает последние запросы пользователя.

Тесты: `uv run pytest` — весь набор, сеть не требуется; `tests/test_storage_integration.py` проверяет PostgreSQL и включается только при заданном `DATABASE_URL`, удаляя созданные им строки.

Ограничения MVP:

* `user_chart_requests.chart_hash` объявлен с `ON DELETE CASCADE` (как в ТЗ): удаление строки кэша удалит и историю. Если история должна переживать очистку кэша, ограничение нужно заменить на `RESTRICT`;
* `llm_generations.chart_hash` тоже `ON DELETE CASCADE`: удаление строки кэша удалит и её разборы;
* TTL и вытеснения нет: `charts_cache` растёт вместе с числом уникальных карт (проверка на этой карте: около 100 КБ JSON и 37 КБ на диске после сжатия TOAST);
* смена состава полей `chart_hash` меняет все ключи, после неё кэш нужно перезаполнить.

## LLM-разбор натальной карты

`src/generator/` превращает расчёт и найденные чанки в связный разбор:

| Файл | Роль |
|---|---|
| `prompts.py` | версионируемый системный промпт (`PROMPT_VERSION = "v1.1"`) с правилами антигаллюцинации, структурой разбора и форматированием под Telegram |
| `synthesizer.py` | один вызов чат-модели через OpenAI-совместимый API; контекст ограничен (`max_chunk_chars`, `max_total_chars`), чтобы 35 разделов не вылетели за окно модели |
| `report.py` | `generate_generation_id` и `ChartReportService` — cache-aside поверх `llm_generations` |

`generation_id` — SHA-256 от `chart_hash`, версии промпта, имени модели и температуры (округлённой до 2 знаков). Одинаковый запрос не платит за токены дважды; смена промпта или модели даёт новый id и не ломает старые разборы. Каждая строка `user_chart_requests` хранит `generation_id`, поэтому историю можно воспроизвести.

По умолчанию используется `gemini-flash-lite-latest` через OpenAI-совместимый эндпоинт Google (`generativelanguage.googleapis.com/v1beta/openai/`); модель, температура и таймаут задаются в `.env` (`LLM_*`). `gemini-2.5-flash` новым ключам уже недоступен, а `gemini-3.8-flash` периодически отвечает 503 — lite-вариант стабильно работает и попадает в бесплатный лимит. Через `LLM_BASE_URL` можно вернуться на OpenRouter. Промпт лежит в коде, а не в переменных окружения: так он версионируется вместе с кэшем.

В боте `/chart` сначала отправляет краткую сводку карты, затем генерирует разбор с проверкой кэша и режет его на сообщения по 4096 символов. Модель просят оформлять текст под Telegram (`**жирный**`, эмодзи, без `###`), а бот превращает `**…**` и заголовки в HTML-теги `<b>` и отправляет с `parse_mode=HTML`; если Telegram не принял разметку, тот же текст уходит без неё. Если LLM недоступна, бот честно сообщает об этом и показывает превью найденных разделов.

Проверка на живой карте: flash-lite — около 18 секунд и 4,8 тыс. символов; pro-модель — около минуты и 9–10 тыс. символов; повторный запрос — 0,45 с без обращения к модели.

Ограничение: 35 разделов по ~1,2 КБ — это ~10 тысяч токенов входа; для моделей с маленьким окном контекст придётся урезать (`max_total_chars`).

## Docker и VPS

Образ содержит только рантайм бота: эмбеддинги и LLM вызываются по API, PyTorch в него не ставится (`uv sync --frozen --no-dev --no-install-project`).

Образ рассчитан на x86_64: `pysweph` публикует manylinux-колесо только под эту архитектуру.

```bash
docker compose build
docker compose up -d
docker compose logs -f bot
```

На VPS:

1. Скопируйте проект (без `.venv`, `books/`, `data/`).
2. Создайте `.env` из `.env.example` и заполните ключи; в образ он не попадает (`.dockerignore`).
3. Примените схему БД: `docker compose run --rm bot python init_db.py`.
4. Запустите бота: `docker compose up -d --build`.
5. Обновление: `git pull && docker compose up -d --build`.

PostgreSQL и Qdrant остаются внешними (Neon и Qdrant Cloud), поэтому рядом с ботом ничего поднимать не нужно. Состояние мастера хранится в памяти процесса: при рестарте контейнера незавершённый ввод сбрасывается.

## Разработка

```bash
uv run pytest
```
