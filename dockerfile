FROM python:3.13-slim-bookworm AS base
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

############################
# Builder (uv pre-installed)
############################
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv lock \
 && uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

############################
# Runner
############################
FROM base AS runner

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

RUN groupadd -r app \
 && useradd -m -r -g app -d /home/app app \
 && chown -R app:app /home/app /app

USER app
EXPOSE 8501


ENTRYPOINT ["tini", "--"]

# Streamlit UI
CMD ["streamlit", "run", "src/dotsOCR/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]