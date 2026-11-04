# Astrology RAG

Python-проект для поиска по книгам по астрологии. Исходные книги хранятся в `books/`. Основной конвейер: книги → чанки в PostgreSQL (Neon) → эмбеддинги в Qdrant → поиск и ответы по найденным фрагментам. Поверх этого работает мультистратегический ETL для A/B-сравнения способов нарезки.

## Структура

```text
books/                 Исходные книги
init_db.py             Создание схемы PostgreSQL
src/chunking/          Токенизация, загрузка FB2 и четыре стратегии чанкинга
src/indexing/          Создание коллекций Qdrant и батчевая загрузка
src/rag/               Эмбеддинги, поиск в Qdrant и генерация ответов
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
QDRANT_API_KEY=your-api-key
OPENROUTER_API_KEY=your-openrouter-key
TELEGRAM_BOT_TOKEN=your-telegram-token
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=16
```

`EMBEDDING_MODEL` используется и для токенизации чанков, и для эмбеддингов. `EMBEDDING_DEVICE` принимает `auto`, `cpu` или `cuda`; по умолчанию выбирается CUDA, если PyTorch её видит. Не добавляйте `.env` с секретами в Git.

## PostgreSQL

Проверить соединение и создать таблицы можно командой:

```bash
uv run python init_db.py
```

Скрипт создаёт `schools`, `authors`, `books`, `book_authors`, `chunks` и индексы для `chunks.meta_tags` и `chunks.book_id`. Он не загружает содержимое книг.

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

Индексатор берёт `clean_text` из `chunks`, считает эмбеддинги BGE-M3 и добавляет отсутствующие UUID в `astrology_chunks`. Для полной пересборки коллекции используется `uv run scripts/index_qdrant.py --recreate`. Это отдельная legacy-коллекция; четыре `vronsky_*` коллекции создаются мультистратегическим ETL и не мешают ей.

Для запуска Telegram-бота:

```bash
uv run src/bot.py
```

Текущая модель эмбеддингов — `BAAI/bge-m3` (1024 измерения). Qdrant использует отдельную коллекцию `astrology_chunks`, чтобы не смешивать её с индексами исходного проекта.

## Разработка

```bash
uv run pytest
```
