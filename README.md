Russian version: [[вложенные файлы/READMERU|READMERU]]
# Astrology RAG

A Python project for searching and analyzing astrology books. Source books are stored in `books/`. The core pipeline flows from books to chunks in PostgreSQL (Neon), then embeddings in Qdrant, followed by search and synthesis over the retrieved snippets. A multi-strategy ETL is built on top for A/B testing different chunking methods.
## Structure

```Plaintext
books/                 Source books
init_db.py             PostgreSQL schema creation
src/chunking/          Tokenization, FB2 loading, and four chunking strategies
src/indexing/          Qdrant collection creation and batch ingestion
src/astronomy/         Ephemeris, houses, signs, and aspects for natal charts
src/storage/           Chart cache, users, and request history (PostgreSQL)
src/rag/               Embeddings, Qdrant search, and answer generation
src/generator/         Prompts, LLM synthesis of chart readings, and generation cache
src/bot.py             Telegram bot for querying the knowledge base
scripts/               ETL, retriever comparison, and manual checks
```

FB2 books can be loaded along with metadata and structural chunks using:

```Bash
uv run python parse_and_ingest.py
```

The script reads `.fb2` and ZIP archives containing FB2 files from `books/vronsky`, creates or updates metadata in PostgreSQL, and writes chunks in batches. The batch size can be modified via `--batch-size 250`, and the books directory via `--books-dir`.
## Setup

Install dependencies using `uv`:

```Bash
uv sync
```

Create a `.env` file in the project root (using `.env.example` as a template):

```.env
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=123456:telegram-token

# Embeddings: OpenAI-compatible /embeddings, model BAAI/bge-m3 (1024)
EMBEDDING_BACKEND=api
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=
EMBEDDING_BATCH_SIZE=32

# LLM: chat model for chart interpretation
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=
LLM_MODEL=gemini-flash-lite-latest
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=180
```

`EMBEDDING_MODEL` is used both for chunk tokenization and embeddings. OpenRouter serves `BAAI/bge-m3` via `/embeddings` (1024 dimensions, normalized vectors), so `EMBEDDING_API_KEY` can be left empty: the `OPENROUTER_API_KEY` key is supplied automatically, but only for the `openrouter.ai` endpoint—it is never sent to third-party providers. For another OpenAI-compatible provider, specify `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` explicitly.

`LLM_API_KEY` also falls back to `OPENROUTER_API_KEY`. The local `sentence-transformers` model is no longer part of the mandatory runtime: it is available via the `local-embeddings` extra (`uv sync --extra local-embeddings`), which is required for ETL and offline indexing. For containers, `uv sync --no-dev` is sufficient.
## PostgreSQL

Verify the connection and create tables using:

```Bash
uv run python init_db.py
```

The script creates `schools`, `authors`, `books`, `book_authors`, `chunks`, `users`, `charts_cache`, `llm_generations`, `user_chart_requests`, along with indexes for `chunks.meta_tags`, `chunks.book_id`, and chart cache keys. It does not load book contents.

Run `uv run python init_db.py` first, followed by `uv run python parse_and_ingest.py`.
## Multi-Strategy ETL & Chunking A/B Test

The script `scripts/build_strategy_collections.py` reads FB2/ZIP and raw `.txt`/`.md` files directly, splits each document using four different methods, and uploads the results to four independent Qdrant Cloud collections:

| **Strategy**         | **Collection**               | **Splitting Rule**                                                                             |
| -------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `fixed_500`          | `vronsky_fixed_500`          | 500 tokens, 50 overlap, ignoring book structure                                                |
| `semantic_paragraph` | `vronsky_semantic_paragraph` | Paragraphs, merging up to ~300–400 characters                                                  |
| `structural_xml`     | `vronsky_structural_xml`     | One `<section>`/`<title>` = one chunk; soft split if >1500 characters                          |
| `parent_child`       | `vronsky_parent_child`       | Child 100–150 tokens, payload includes `parent_id`, `parent_content`, and `parent_token_count` |

First, run a safe dry-run check without network calls or model loading:

```Bash
uv run python scripts/build_strategy_collections.py --dry-run
```

The dry-run outputs a JSONL file with chunks and parent sections to `data/strategy_chunks/`, as well as a `summary.json` with counters for each strategy. This is convenient for corpus verification and subsequent RAGAS experiments.

Upload to Qdrant Cloud:

```Bash
uv run python scripts/build_strategy_collections.py --recreate
```

- Batch parameters: `--batch-size 100` (points per upsert), `--embed-batch-size 16` (texts per forward pass);
- Network errors trigger retries with exponential backoff (`--max-attempts 5`);
- `--skip-existing` allows resuming an interrupted upload without recomputing embeddings;
- `--only fixed_500,parent_child` gathers only selected strategies;
- `--limit-books 1` and `--max-chunks 200` provide a quick smoke test;
- `--recreate` deletes target collections before uploading. Without this flag, existing collections are preserved and points are overwritten using deterministic UUIDs.

After uploading, compare retrieval outputs across a common query set:


```Bash
uv run python scripts/compare_retrievers.py
uv run python scripts/compare_retrievers.py --top-k 10 --queries-file data/eval_queries.txt
```

The report is saved to `reports/retrieval_comparison.md` and `reports/retrieval_comparison.json`: top-1/top-k scores, latency, unique chunk ratio, and retrieval examples per collection. Since cosine scores depend on chunk size, the report serves as a signal for selection rather than a final quality score; confirm final choices via manual evaluation or RAGAS.
## RAG

After uploading and indexing chunks, search and response generation can be tested with:

```Bash
uv run scripts/index_qdrant.py
uv run scripts/query_rag.py "What is said about the planets?"
```

The indexer takes `clean_text` from `chunks`, computes BGE-M3 embeddings via API, and adds missing UUIDs to `astrology_chunks`. To completely rebuild the collection, use `uv run scripts/index_qdrant.py --recreate`. This is a separate legacy collection; the four `vronsky_*` collections are created by the multi-strategy ETL and do not conflict with it.

To run the Telegram bot:

```Bash
uv run src/bot.py
```

The current embedding model is `BAAI/bge-m3` (1024 dimensions), called via an OpenAI-compatible `/embeddings` endpoint. Qdrant uses a separate `astrology_chunks` collection to keep it isolated from other project indexes.
## Astronomical Calculator

`src/astronomy/calculator.py` is the natal chart core: planetary positions, house cusps, signs, and major aspects. The calculator accepts UTC exclusively and does not handle time zones—converting local time and historical daylight saving changes is left to the backend layer.

```Bash
uv run python scripts/natal_chart.py 1990-08-15T12:00:00+00:00 55.7558 37.6173 --house-system P
```

The same can be done from code (`src` must be in `PYTHONPATH`—scripts and pytest handle this automatically):

```Python
from datetime import datetime, timezone

from astronomy.calculator import calculate_natal_chart

chart = calculate_natal_chart(
    datetime(1990, 8, 15, 12, 0, tzinfo=timezone.utc),
    lat=55.7558,
    lon=37.6173,
    house_system="P",  # 'P' — Placidus, 'W' — whole sign, 15 systems in total
)
chart.planets["Sun"].longitude, chart.planets["Sun"].sign, chart.asc.house
chart.to_dict()  # JSON-compatible snapshot for prompts and Langfuse
```

A planet in `to_dict()` contains exactly six fields: `longitude`, `sign`, `house`, `speed`, `is_retrograde`, `is_stationary`. Default aspects: conjunction, sextile, square, trine, and opposition with orbs of 8°/6°/7°/8°/8°; orbs can be overridden via `orbs`.

The `pysweph` package is a community fork of `pyswisseph` featuring the same `swisseph` module; the original lacks wheels for Python 3.12+ and requires MSVC on Windows. Import and function calls are compatible with `pyswisseph`, meaning library replacement reduces to editing the dependency in `pyproject.toml`.
## Structural Precise Search (Without Vector Search)

`src/rag/structural_router.py` decomposes a natal chart (`chart.to_dict()`) into discrete elements—planet in sign, planet in house, major aspect—and finds articles for each using precise payload filters (`planet`/`sign`/`house`/`aspect` + `school`/`language`/`book_id`). Vector search is not used. Planet pairs in aspects are normalized to a canonical (lexicographical) form, making queries invariant to planet order.

Structural chunk metadata is read solely from section headings: `planet`/`planet_b`, `sign`/`sign_b`, `house`/`house_b`, `aspect`, `is_transit`. The second body of a pair and the second sign/house are preserved, allowing precise searches for nodes ("Ascending/Descending Lunar Node") and aspect pairs. Aspect types are not explicitly labeled in the corpus: aspect chapters are named after the pair, and the router returns the pair chapter with `coverage="pair_chapter"`. To rebuild payloads after editing extraction logic: `uv run python scripts/update_strategy_payloads.py --collections structural_xml`.

By default, one best chunk is returned per element (production): `get_structural_interpretations(chart.to_dict(), metadata_filters)`. For manual inspection and evaluation, pass `per_element=N`—the router will return up to N ranked candidates (heading matches rank higher than text matches).

Manual inspection of candidates from the command line:

```Bash
uv run python scripts/structural_lookup.py --planet-sign Sun Leo --planet-house Saturn 2 --aspect Mercury Saturn trine --top 5
uv run python scripts/structural_lookup.py --utc 1990-08-15T12:00:00+00:00 --lat 55.7558 --lon 37.6173 --top 5
```

A report with all candidates and precise filters is saved to `reports/structural_lookup.md`; full chunk text can be output in JSON via `--json-out`.
## Persistent Chart Layer: Cache and Request History

`src/storage/` serves as the backend layer between the calculator, structural search, and clients. It fulfills two main requirements: avoiding redundant calculations for identical charts and preserving user request history.

Tables are created via `uv run python init_db.py`; all statements are idempotent, so repeated runs are safe.

|**Table**|**Purpose**|
|---|---|
|`users`|One record per person: `telegram_id` for MVP, `email` as a placeholder for website and app integration|
|`charts_cache`|One record per unique birth data set: chart snapshot and retrieved chunks|
|`user_chart_requests`|History: who requested a chart, under which profile, and when|

`chart_hash` is computed as an SHA-256 hash of canonical JSON: `dt_utc` in UTC, latitude and longitude rounded to 4 decimal places, house system code, and chunk language. Rounding glues together micro-discrepancies from geocoders so a single city doesn't fragment the cache, while twins and repeat requests share the same `charts_cache` row.

The pipeline (`src/storage/pipeline.py`) operates on a cache-aside pattern:

1. `users` — UPSERT by `telegram_id`/`email`, retrieving the internal `user_id`;
2. `chart_hash` — computed from canonical birth data;
3. `SELECT` from `charts_cache`: a hit returns the ready-made snapshot and chunks, a miss triggers pyswisseph calculations and Qdrant structural search;
4. a miss is saved with `ON CONFLICT DO NOTHING`, preventing race conditions when two identical requests arrive simultaneously;
5. a record with the profile name (`Me`, `Mom`, etc.) is added to `user_chart_requests`;
6. the client receives the chart and chunks.

`retrieved_chunks` is an array of unique chunks rather than a duplicate response copy for every element: the section text is stored once, and the "element → section" relation resides in `elements` alongside the candidate rank. A node chapter answering two elements simultaneously is not duplicated; the reverse transformation is handled by `element_chunk_index`.

Testing on live services:

```Bash
uv run python scripts/chart_request.py "1990-08-15T12:00:00" "Moscow"
uv run python scripts/chart_request.py "1990-08-15T12:00:00" "Moscow"
```

The first run prints `cache: MISS` and calculation time, while the second outputs `HIT` and time spent solely reading from PostgreSQL (Neon check: 10.9s → 1.5s). The script displays the chart, found sections, formatted bot response, and user history; `--json-out` saves the complete report, `--lat/--lon/--timezone` bypass geocoding, and `--telegram-id` switches profiles (default is `0`, a service CLI profile).

In the bot:

```Plaintext
/start          greeting and menu
🔮 Calculate Chart   step-by-step input: name → date → time → place
🕘 History            past requests
❓ Help              what the bot can do
/chart 1990-08-15 12:00 Moscow   quick calculation with a single command
```

`/start` displays a greeting and a persistent menu. The "Calculate Chart" button starts a wizard: name (optional, used only for addressing), date, time, and birthplace. Time can be skipped—in which case 12:00 is used with a warning that houses and ASC are approximate. Before calculation, the bot shows a summary with "Calculate" and "Edit Data" buttons.

`/chart` with arguments remains available for quick calculations: it resolves coordinates via Nominatim, time zone via `timezonefinder`, converts local time to UTC using historical rules, outputs a brief chart, and generates an interpretation via LLM (checking the `llm_generations` cache). `/history` displays the user's recent requests.

Tests: `uv run pytest` runs the full test suite without requiring network access; `tests/test_storage_integration.py` verifies PostgreSQL functionality and activates only when `DATABASE_URL` is set, cleaning up rows it creates.

MVP Limitations:

- `user_chart_requests.chart_hash` is defined with `ON DELETE CASCADE` (per specifications): deleting a cache row will also delete the history. If history must persist past cache cleanups, this constraint should be changed to `RESTRICT`;
- `llm_generations.chart_hash` also uses `ON DELETE CASCADE`: deleting a cache row removes its interpretations;
- No TTL or eviction policy: `charts_cache` grows with the number of unique charts (~100 KB JSON and 37 KB on disk after TOAST compression for the tested chart);
- Changing the structure of fields in `chart_hash` invalidates all keys, requiring the cache to be repopulated.
## LLM Natal Chart Interpretation

`src/generator/` transforms calculations and retrieved chunks into a cohesive interpretation:

|**File**|**Role**|
|---|---|
|`prompts.py`|Versioned system prompt (`PROMPT_VERSION = "v1.1"`) containing anti-hallucination rules, structure guidelines, and Telegram formatting|
|`synthesizer.py`|Single chat model call via an OpenAI-compatible API; context is bounded (`max_chunk_chars`, `max_total_chars`) to prevent 35 sections from exceeding the model's context window|
|`report.py`|`generate_generation_id` and `ChartReportService` implementing cache-aside over `llm_generations`|

`generation_id` is an SHA-256 hash of `chart_hash`, prompt version, model name, and temperature (rounded to 2 decimal places). Identical requests do not pay for tokens twice; changing the prompt or model produces a new ID without breaking past interpretations. Every `user_chart_requests` row stores a `generation_id`, allowing history to be reproduced.

By default, `gemini-flash-lite-latest` is used via Google's OpenAI-compatible endpoint (`[generativelanguage.googleapis.com/v1beta/openai/](https://generativelanguage.googleapis.com/v1beta/openai/)`); model, temperature, and timeout are configured in `.env` (`LLM_*`). `gemini-2.5-flash` is no longer available to new keys, and `gemini-3.8-flash` periodically returns 503 errors—the lite variant runs reliably and fits within free tier limits. You can switch back to OpenRouter via `LLM_BASE_URL`. Prompts reside in code rather than environment variables to version them alongside the cache.

In the bot, `/chart` first sends a brief chart summary, then generates an interpretation with cache checking and splits it into 4096-character messages. The model is instructed to format text for Telegram (`**bold**`, emojis, no `###`), and the bot converts `**…**` and headings into HTML `<b>` tags before sending with `parse_mode=HTML`; if Telegram rejects the markup, the text is sent plain. If the LLM is unavailable, the bot informs the user honestly and shows a preview of the retrieved sections.

Live chart benchmark: flash-lite takes ~18 seconds for ~4.8k characters; pro models take ~1 minute for 9–10k characters; repeated requests take 0.45s without calling the model.

Limitation: 35 sections of ~1.2 KB amount to ~10k input tokens; models with smaller context windows will require truncation (`max_total_chars`).
## Docker & VPS

The Docker image contains only the bot runtime: embeddings and LLM are called via API, and PyTorch is not installed (`uv sync --frozen --no-dev --no-install-project`).

The image targets x86_64: `pysweph` publishes manylinux wheels exclusively for this architecture.

```Bash
docker compose build
docker compose up -d
docker compose logs -f bot
```

On a VPS:
1. Copy the project files (excluding `.venv`, `books/`, `data/`).
2. Create `.env` from `.env.example` and populate keys; it is excluded from the image via `.dockerignore`.
3. Apply the database schema: `docker compose run --rm bot python init_db.py`.
4. Run the bot: `docker compose up -d --build`.
5. Update: `git pull && docker compose up -d --build`.

PostgreSQL and Qdrant remain external (Neon and Qdrant Cloud), so nothing else needs to be hosted alongside the bot. Wizard state is kept in process memory: restarting the container resets incomplete inputs.
## Development

```Bash
uv run pytest
```