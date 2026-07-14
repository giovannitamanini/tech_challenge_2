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
# dvc.yaml/dvc.lock/.dvc/config: permitem `dvc repro` (pipeline completo) dentro do
# container, não só o script de treino isolado — usado por `POST /treino` da API
# (`docker compose run --rm train dvc repro`, ver `api/servico_treino.py`).
COPY --chown=app:app dvc.yaml dvc.lock .dvcignore ./
COPY --chown=app:app .dvc/config ./.dvc/config

USER app

CMD ["python", "-m", "tech_challenge_recomendacao.treino.treinar"]

# ---- Estágio api: serve a API e dispara o treino num container irmão via Docker-fora-do-Docker ----
# `POST /treino` (api/servico_treino.py) roda `docker compose run --rm train dvc repro` a
# partir de dentro deste container — precisa do CLI do Docker/Compose aqui dentro e do socket
# do host montado em `/var/run/docker.sock` (ver docker-compose.yml, serviço `api`).
FROM docker:27-cli AS docker-cli

FROM runtime AS api

# Continua como root (não volta para `app`): este container já tem controle total do Docker
# do host via socket montado — rodar como não-root aqui não reduz o raio de explosão real,
# só complicaria a permissão de acesso ao socket. `train`/`mlflow` continuam não-root.
USER root

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
ADD https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 \
    /usr/local/libexec/docker/cli-plugins/docker-compose
RUN chmod +x /usr/local/libexec/docker/cli-plugins/docker-compose

EXPOSE 8000

CMD ["uvicorn", "tech_challenge_recomendacao.api.aplicacao:app", "--host", "0.0.0.0", "--port", "8000"]
