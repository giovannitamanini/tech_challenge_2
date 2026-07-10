# Tech Challenge — Sistema de Recomendação

Sistema de recomendação de produtos baseado no comportamento de navegação dos usuários, desenvolvido para o Tech Challenge da PosTech. O modelo central é uma rede neural (MLP/embedding-based) treinada com PyTorch, com pipeline de dados versionado via DVC e experimentos rastreados no MLflow.

> Este README cobre a configuração do ambiente de desenvolvimento. As instruções completas de uso (treino, pipeline DVC, Docker) serão adicionadas conforme as próximas etapas do projeto forem concluídas — ver o checklist completo em [`docs/TASKS.md`](docs/TASKS.md).

## Primeira vez rodando o projeto (resumo)

Sequência completa para quem acabou de clonar o repositório e quer chegar até a stack rodando em Docker. Cada passo é detalhado nas seções abaixo.

```bash
git clone <url-do-repositorio>
cd tech_challenge_2

uv sync
uv run pre-commit install
cp .env.example .env

# data/raw_data/ já vem do próprio git clone — não precisa de dvc pull

docker compose up --build mlflow -d  # sobe só o MLflow, para o `dvc repro` logar nele
uv run dvc repro                     # preprocess -> feature_eng -> train -> evaluate
uv run dvc metrics show
docker compose down

docker compose up --build            # stack completa: mlflow + train, já com os dados prontos
```

## Pré-requisitos

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) — gerenciador de dependências e ambiente do projeto

## Como obter o projeto

```bash
git clone <url-do-repositorio>
cd tech_challenge_2
```

## Instalação das dependências

```bash
uv sync
```

Esse comando cria o ambiente virtual em `.venv/` e instala todas as dependências de produção e de desenvolvimento, travadas em `uv.lock`. Deve funcionar de forma limpa em qualquer máquina, sem passos ocultos.

Em seguida, instale o git hook de linting (roda automaticamente a cada commit; precisa ser feito uma vez por clone, pois `.git/hooks/` não é versionado):

```bash
uv run pre-commit install
```

## Configuração

As configurações do projeto são lidas de um arquivo `.env` (nunca commitado). Copie o exemplo e ajuste se necessário — os valores padrão já funcionam para desenvolvimento local:

```bash
cp .env.example .env
```

## Linting

```bash
uv run ruff check .
```

Deve rodar sem nenhum erro. Para aplicar formatação automática:

```bash
uv run ruff format .
```

Para rodar manualmente todos os hooks configurados (equivalente ao que roda em cada commit):

```bash
uv run pre-commit run --all-files
```

## Validação do ambiente

```bash
uv run python scripts/validate_env.py
```

Verifica a versão do Python, o carregamento correto das configurações (`.env`/Pydantic Settings) e a existência dos diretórios de dados e modelos. Termina com código de saída `0` em caso de sucesso.

> **Nota:** o dataset bruto (`data/raw_data/`) vem commitado direto no git — nenhum passo extra é necessário. `data/processed_data/` só existe depois de rodar o pipeline (ver seção "Pipeline (DVC + MLflow)" abaixo) — até lá o script indicará essa pasta como não encontrada.

## Dados (DVC)

O dataset bruto (`data/raw_data/`) é a amostra **MovieLens ml-latest-small** (~3 MB) e é commitado direto no git — qualquer `git clone` já o traz, sem precisar de DVC.

O [DVC](https://dvc.org/) é usado para versionar os **artefatos gerados pelo pipeline** (`data/processed_data/`, `models/modelo_recomendador.pt`), não o dataset bruto:

```bash
uv run dvc pull
```

Baixa esses artefatos a partir do remote configurado em `.dvc/config` (`localremote`), se disponíveis lá.

> **Nota:** o remote configurado atualmente é uma pasta local (`../dvc-storage`, fora do repositório e fora do git), útil para desenvolvimento na mesma máquina onde os artefatos foram gerados. Ele **não é compartilhado automaticamente entre integrantes do grupo** — para obter `data/processed_data/` e `models/modelo_recomendador.pt` sem acesso a esse remote, basta rodar `uv run dvc repro`, que os regenera do zero a partir de `data/raw_data/` (já disponível via git).

## Pipeline (DVC + MLflow)

O pipeline (`dvc.yaml`) tem 4 stages, executáveis de ponta a ponta com um único comando:

```
preprocess → feature_eng → train → evaluate
```

- `preprocess` — limpa `data/raw_data/ratings.csv` (amostragem, deduplicação, filtro de frequência mínima) e salva `data/processed_data/interacoes.parquet`.
- `feature_eng` — codifica `userId`/`movieId` em índices contíguos e faz o split treino/teste.
- `train` — treina um modelo de recomendação (ver `src/tech_challenge_recomendacao/modelos/`) e loga parâmetros, métricas por época e o checkpoint como artefato no MLflow.
- `evaluate` — calcula RMSE/MAE do modelo treinado no conjunto de teste.

Os hiperparâmetros de cada stage ficam em [`configs/params.yaml`](configs/params.yaml).

O stage `train` precisa de um servidor MLflow acessível em `MLFLOW_TRACKING_URI` (`.env`). Forma mais simples: subir só o serviço `mlflow` do `docker-compose.yml` (`http://localhost:5000`, já compatível com o `MLFLOW_TRACKING_URI` padrão do `.env.example`):

```bash
docker compose up --build mlflow -d
```

Alternativa sem Docker — rodar o servidor diretamente na máquina:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Com o servidor no ar, em outro terminal:

```bash
uv run dvc repro
```

Reexecuta apenas os stages cujas dependências (código, dados ou `configs/params.yaml`) mudaram desde a última run. Para ver as métricas registradas:

```bash
uv run dvc metrics show
```

> **Nota:** o modelo atual (`FatoracaoMatricial`, uma fatoração matricial embedding-based) é um baseline simples, só para validar o pipeline ponta a ponta — a Etapa 4 treina e ajusta o modelo de fato, compara com baselines Scikit-Learn e registra o melhor no MLflow Model Registry.

## Docker

O `Dockerfile` é multi-stage, com três estágios: `builder` instala as dependências (e o próprio pacote) com `uv` em um venv isolado; `runtime` (estágio final/padrão) copia só esse venv, `src/` e `configs/` para uma imagem enxuta (`python:3.13-slim`, sem `uv`/cache de build), rodando como usuário não-root e executando o stage `train` por padrão; `mlflow-server` reaproveita o venv do `builder` (sem `src/`/`configs/`, que não são necessários) para rodar o servidor de tracking do MLflow.

### Uso via `docker-compose` (recomendado)

```bash
docker compose up --build
```

Sobe dois serviços: `mlflow` (servidor MLflow em `http://localhost:5000`, com backend SQLite e artefatos persistidos em `./mlflow-data/`, fora do git) e `train` (roda o stage `train`, aguardando o `mlflow` responder saudável antes de iniciar). `data/` e `models/` são montados como volumes a partir do host, então os resultados do treino ficam disponíveis fora do container. `MLFLOW_TRACKING_URI` é definido automaticamente para `http://mlflow:5000` (nome do serviço na rede interna do compose), sobrepondo o valor de `.env`.

> **Importante:** o serviço `train` executa só o stage `train` (`python -m tech_challenge_recomendacao.treino.treinar`), **não** o pipeline `dvc.yaml` inteiro. Ele espera que `data/processed_data/treino.parquet`, `data/processed_data/metadados_features.json` e (para o `evaluate`, fora do compose) `data/processed_data/teste.parquet` já existam. Rode `uv run dvc repro` antes de subir a stack completa (ver seção "Primeira vez rodando o projeto" e "Pipeline (DVC + MLflow)") — subir só o `docker compose up --build` num clone novo, sem isso, falha por faltar esses arquivos.

### Uso do `Dockerfile` isolado

```bash
docker build -t tech-challenge-recomendacao .
```

Por padrão builda o estágio `runtime` e roda o stage `train` (`python -m tech_challenge_recomendacao.treino.treinar`), lendo/gravando os diretórios de dados e modelos por volume:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  tech-challenge-recomendacao
```

Nesse modo (sem `docker-compose`), `MLFLOW_TRACKING_URI` no `.env` precisa apontar para um servidor MLflow acessível a partir do container — ex.: `http://host.docker.internal:5000` no Docker Desktop, com `uv run mlflow server` rodando no host. Como o MLflow bloqueia por padrão `Host` headers fora de localhost/IPs privados (proteção contra DNS rebinding), rodar o servidor manualmente assim exige liberar esse host explicitamente: `MLFLOW_SERVER_ALLOWED_HOSTS=localhost:5000,127.0.0.1:5000,host.docker.internal:5000 uv run mlflow server --host 0.0.0.0 --port 5000` (o serviço `mlflow` do `docker-compose.yml` já faz isso automaticamente).

## Estrutura do projeto

```
src/tech_challenge_recomendacao/
├── configuracoes.py       # configurações (.env via Pydantic Settings)
├── parametros.py          # parâmetros do pipeline (configs/params.yaml)
├── dados/                 # stages `preprocess` e `feature_eng`
├── modelos/               # modelos de recomendação + factory (fabrica.py)
├── treino/                # stage `train` (loop de treino + logging no MLflow)
└── avaliacao/              # stage `evaluate` (métricas no conjunto de teste)
tests/                            # testes automatizados
data/raw_data/                    # dataset bruto (MovieLens ml-latest-small, commitado direto no git)
data/processed_data/              # dados processados pelo pipeline (saída do DVC)
models/                           # modelos treinados (saída do DVC)
configs/                          # arquivos de configuração declarativos (params.yaml do DVC)
scripts/                          # scripts utilitários
docs/                             # documentação e checklist de tarefas do projeto
```
