# syntax=docker/dockerfile:1

# ---- Estágio builder: instala dependências e o próprio projeto em um venv isolado ----
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Primeiro só as dependências (cache de camada não invalida a cada mudança de código-fonte)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Depois o código-fonte e a instalação (editable) do próprio pacote
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---- Estágio mlflow-server: reaproveita o venv do builder para rodar o servidor MLflow ----
FROM python:3.13-slim AS mlflow-server

RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY --from=builder --chown=app:app /app/.venv ./.venv
RUN mkdir -p /app/mlflow-data && chown app:app /app/mlflow-data

USER app

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////app/mlflow-data/mlflow.db", \
     "--artifacts-destination", "/app/mlflow-data/mlartifacts"]

# ---- Estágio runtime: imagem enxuta, só com o necessário para rodar a aplicação ----
FROM python:3.13-slim AS runtime

RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/

USER app

CMD ["python", "-m", "tech_challenge_recomendacao.treino.treinar"]
