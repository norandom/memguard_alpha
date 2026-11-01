# One versioned image template for Docker, Compose and Dagger.
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:0.9.7-python${PYTHON_VERSION}-bookworm-slim AS base
