# Sistema de Recomendação — Tech Challenge (Pós MLE FIAP)

Sistema de recomendação de filmes baseado no comportamento de usuários (dataset **MovieLens**),
desenvolvido para o Tech Challenge da PosTech. O modelo central é uma rede neural
embedding-based treinada com **PyTorch**, com pipeline de dados reprodutível via **DVC**,
experimentos rastreados no **MLflow**, uma **API FastAPI** para servir recomendações e
empacotamento via **Docker**.

> Progresso detalhado por etapa: [`docs/TASKS.md`](docs/TASKS.md). Estado atual: Etapas 1–3
> (estrutura, ambiente, containerização/versionamento) concluídas. Etapa 4: o modelo em
> produção já é a rede neural (`RedeNeural`, embeddings + MLP + vieses), treinada com early
> stopping e comparada com 6 baselines Scikit-Learn/estatísticos em 6 métricas (tabela
> completa em `models/comparacao_modelos.json`) — ver seção "Pipeline" abaixo. Ainda faltam:
> promoção do modelo no MLflow Model Registry (Staging → Production), Model Card/README
> finais de entrega e o vídeo STAR.

## Equipe

Giovanni de Aguirre Tamanini · Yan Levi Martins Meira · Cristiano Lima do Sacramento · Marcelo Souza

## Estrutura do projeto

```
src/tech_challenge_recomendacao/
  api/          # API FastAPI (rotas, schemas, serviço de recomendação)
  dados/        # pré-processamento e engenharia de features
  modelos/      # modelo(s) de recomendação (Factory em modelos/fabrica.py)
  treino/       # loop de treino, logging no MLflow
  avaliacao/    # avaliação do modelo treinado
  configuracoes.py  # única fonte de configuração do projeto (Pydantic Settings, lê .env)
tests/          # testes automatizados (pytest)
data/
  raw_data/         # dataset bruto (MovieLens), versionado via DVC
  processed_data/   # saída dos stages preprocess/feature_eng do pipeline DVC
models/         # modelo treinado (models/modelo_recomendador.pt), versionado via DVC
configs/        # parâmetros declarativos do pipeline (configs/params.yaml)
docs/           # TASKS.md, relatório técnico, model card, evidências, coleção Postman
scripts/        # scripts utilitários (ex.: validate_env.py)
dvc.yaml        # pipeline DVC: preprocess → feature_eng → train → evaluate
Dockerfile      # multi-stage: builder / mlflow-server / runtime
docker-compose.yml  # serviços mlflow + train
```

## Pré-requisitos

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) — gerenciador de dependências e ambiente do projeto
- [Docker](https://www.docker.com/) (opcional, para rodar via container/compose)
- Credenciais AWS (access key + secret key) com permissão no bucket S3 do remote DVC do
  projeto — peça ao dono do repositório para criar um IAM user para você (ver seção
  [Dados (DVC)](#dados-dvc) abaixo). **Nunca commite essas credenciais.**

## Instalação

```bash
git clone https://github.com/giovannitamanini/tech_challenge_2
cd tech_challenge_2

uv sync                        # cria .venv/ e instala todas as dependências (prod + dev)
uv run pre-commit install      # instala o git hook de lint (uma vez por clone)

cp .env.example .env           # os valores padrão já servem para desenvolvimento local

uv run python scripts/validate_env.py   # valida Python, .env e diretórios esperados
```

## Dados (DVC)

Os dados (`data/raw_data/`, `data/processed_data/`) e o modelo treinado (`models/`) não são
commitados diretamente no git — são versionados via [DVC](https://dvc.org/), com um remote
**S3** configurado em `.dvc/config` (`s3remote`, bucket compartilhado do projeto).

Cada pessoa do time precisa configurar sua **própria** credencial AWS localmente, gravada em
`.dvc/config.local` (arquivo já ignorado pelo git por padrão pelo próprio DVC — nunca vai para
o repositório):

```bash
uv run dvc remote modify --local s3remote access_key_id SUA_ACCESS_KEY_ID
uv run dvc remote modify --local s3remote secret_access_key SUA_SECRET_ACCESS_KEY
```

Depois disso:

```bash
uv run dvc pull    # baixa data/raw_data/, data/processed_data/ e models/ do S3
uv run dvc push    # envia dados/modelos versionados localmente para o S3
```

> **Se `dvc pull` reclamar de arquivos ausentes no cache e no remote:** significa que aquele
> artefato específico nunca chegou a ser enviado ao S3 por ninguém (comum logo após trocar de
> remote). Como `data/processed_data/` e `models/` são só saídas reproduzíveis do pipeline
> (não dados originais), a solução é rodar `dvc repro` (seção abaixo) para regerá-los
> localmente e então `dvc push`. Já `data/raw_data/` é o dado original — se ele estiver
> faltando, precisa vir de quem o versionou primeiro.

## Pipeline (DVC + MLflow)

O pipeline (`dvc.yaml`) tem 4 stages — `preprocess → feature_eng → train → evaluate` — cada um
executável de ponta a ponta via `dvc repro`. `feature_eng` faz um split **temporal por
usuário** (treino/validação/teste). O stage `train` treina a rede neural (`RedeNeural`) com
**early stopping** (monitorando RMSE de validação) e loga parâmetros/métricas/artefato no
MLflow a cada run. O stage `evaluate` compara o modelo treinado com 6 baselines Scikit-Learn/
estatísticos em 6 métricas, salvando a tabela comparativa completa em
`models/comparacao_modelos.json` (e as métricas do modelo campeão em
`data/processed_data/metricas_avaliacao.json`, consumidas pela API).

Precisa de um servidor MLflow acessível na URI configurada em `.env`
(`MLFLOW_TRACKING_URI`, padrão `http://localhost:5000`):

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db   # em um terminal separado

uv run dvc repro           # roda o pipeline completo
uv run dvc metrics show    # mostra as métricas de train/evaluate
```

Parâmetros do pipeline (tamanho da amostra, dimensão do embedding, épocas etc.) ficam em
[`configs/params.yaml`](configs/params.yaml), não hardcoded no código.

## API

Serve o modelo treinado (`models/modelo_recomendador.pt`) via HTTP:

```bash
uv run uvicorn tech_challenge_recomendacao.api.aplicacao:app --reload
```

Endpoints (ver também a coleção Postman em
[`docs/postman/api-recomendacao.postman_collection.json`](docs/postman/api-recomendacao.postman_collection.json)):

| Método | Rota | Descrição |
|---|---|---|
| GET | `/saude` | Health check da API |
| GET | `/modelo/info` | Metadados do modelo carregado e métricas do último `evaluate` |
| POST | `/previsoes` | Prevê a nota de um lote de pares (usuário, filme) |
| GET | `/recomendacoes/{usuario_id}` | Top-`k` filmes recomendados para um usuário |
| GET | `/filmes/{filme_id}/similares` | Top-`k` filmes similares a um filme (embeddings) |

## Docker

```bash
docker build -t tech-challenge-recomendacao .
docker run --rm --env-file .env -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  tech-challenge-recomendacao   # roda o stage `train` isolado (precisa de MLflow acessível)
```

Ou via `docker-compose.yml`, que sobe o MLflow e o treino juntos:

```bash
docker compose up --build
```

Serviços: `mlflow` (servidor de tracking em `http://localhost:5000`, backend SQLite em
`./mlflow-data/`) e `train` (stage `train`, aguarda o `mlflow` ficar saudável).

## Testes e lint

```bash
uv run pytest                    # suíte de testes (tests/)
uv run ruff check .              # lint — deve rodar sem erros
uv run ruff format .             # formatação
uv run pre-commit run --all-files   # roda os hooks configurados em todo o repositório
```

## Resultados exploratórios (notebook)

Antes do pipeline `src/` atual, Cristiano Sacramento conduziu uma análise exploratória em
notebook (`models/recomendacao_movielens.ipynb`, dataset MovieLens `ml-latest-small`,
100.836 interações), comparando uma rede neural embedding-based com seis baselines em seis
métricas. Resultado no conjunto de teste:

| Modelo | RMSE ↓ | MAE ↓ | NDCG@10 ↑ |
|---|---|---|---|
| **`NeuralRecommender`** (rede neural) | **0,8984** | **0,6911** | 0,0553 |
| SVD (sklearn) | 1,0282 | 0,8040 | **0,1700** |
| GBRT (sklearn) | 0,9377 | 0,7132 | 0,0036 |
| UserItemBias | 0,8996 | 0,6957 | 0,1196 |

**Achado central:** nenhum modelo vence em tudo — a rede neural lidera na predição de rating
(RMSE/MAE), enquanto as fatorações de matriz (SVD/NMF) lideram no ranking Top-10. O "melhor
modelo" depende do objetivo de negócio (regressão de nota vs. ordenação de lista); detalhes
completos no [relatório técnico](docs/RELATORIO.md). Esta análise virou o modelo em produção
da Etapa 4 (`modelos/rede_neural.py`) — o pipeline `src/`/DVC/MLflow agora usa o mesmo
dataset, o mesmo split temporal por usuário e a mesma arquitetura do notebook, então
`uv run dvc repro` deve reproduzir (mesma seed) números iguais ou muito próximos aos desta
tabela e do `docs/MODEL_CARD.md`.

## Documentação

- [`docs/TASKS.md`](docs/TASKS.md) — checklist completo do desafio, por etapa.
- [`docs/RELATORIO.md`](docs/RELATORIO.md) e [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) —
  relatório técnico e model card completos da análise exploratória em notebook acima.
- [`docs/evidencias/`](docs/evidencias/) — evidências de execução (instalação limpa,
  diagnóstico de baseline).

## Licença dos dados

Dataset MovieLens (GroupLens/University of Minnesota) — uso apenas para fins de
pesquisa/acadêmicos, sem uso comercial sem autorização do GroupLens Research Project, e com
citação obrigatória de Harper & Konstan (2015), *ACM TiiS*. Ver `data/raw_data/README.txt`
para o texto completo da licença.
