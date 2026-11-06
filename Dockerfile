# syntax=docker/dockerfile:1
# Bot runtime only: cloud embeddings and LLM, no PyTorch.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer survives changes to src/.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY init_db.py ./

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1001 bot \
    && chown -R bot:bot /app
USER bot

CMD ["python", "src/bot.py"]
