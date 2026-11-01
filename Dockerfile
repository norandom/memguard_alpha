# One versioned image template for Docker, Compose and Dagger.
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:0.9.7-python${PYTHON_VERSION}-bookworm-slim AS base

ARG PYTHON_VERSION
ENV UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    DISABLE_MKDOCS_2_WARNING=true

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app /outputs \
    && chown app:app /app /outputs
WORKDIR /app

FROM base AS dev-dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

FROM dev-dependencies AS dev
COPY --chown=app:app . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
USER app
CMD ["uv", "run", "--no-sync", "python", "examples/completion_workflow.py"]
