# CLAUDE.md

Este arquivo fornece orientações ao Claude Code (claude.ai/code) para trabalhar com o código deste repositório.

## Idioma do projeto

Tudo neste projeto deve ser feito em português do Brasil: mensagens de commit, comentários de código, docstrings, nomes de variáveis/funções/classes, documentação (README, Model Card etc.), mensagens de log e qualquer texto voltado ao usuário. A única exceção são palavras reservadas e sintaxe das linguagens de programação e bibliotecas (ex.: `class`, `def`, `import`, `return`, nomes de parâmetros exigidos por frameworks como `self`, `forward`, `__init__`), que naturalmente permanecem em inglês por serem parte da linguagem.

## Fluxo de trabalho com git

- Remote `origin`: https://github.com/giovannitamanini/tech_challenge_2 — `main` já está publicado lá e é a branch trunk.
- **A partir de agora, nenhum trabalho novo é commitado direto em `main`.** Cada unidade de trabalho (uma etapa, uma tarefa do `docs/TASKS.md`, uma correção) vai para uma branch própria, criada a partir do `main` atualizado.
  - Convenção de nome sugerida: `etapaN/descricao-curta` (ex.: `etapa3/dockerfile-multistage`).
- Fluxo por unidade de trabalho: criar a branch → commitar (sempre com aprovação explícita do usuário antes de cada commit, como já vínhamos fazendo) → `git push -u origin <branch>` → abrir Pull Request de `<branch>` para `main` no GitHub (a CLI `gh` não está instalada nesta máquina; usar a interface web do GitHub, ou o link "Compare & pull request" retornado pelo `git push`) → aguardar revisão/merge antes de seguir para a próxima branch.
- As Etapas 1 e 2 (commits até `921a804`) foram feitas direto em `main` antes desse fluxo ser definido — isso não precisa ser desfeito retroativamente. A regra vale para todo trabalho novo dali em diante.

## Status do projeto

Este repositório contém a lista de tarefas da atividade (`docs/TASKS.md`, PosTech "Tech Challenge") e o dataset bruto em `data/`. O repositório git local já foi inicializado, a estrutura de pastas base (`src/`, `tests/`, `models/`, `configs/`, além de `data/`) já foi criada, e a Etapa 2 (`pyproject.toml`/`uv.lock`, configurações via `.env`/Pydantic Settings, `scripts/validate_env.py`) está completa e verificada (ver `docs/evidencias/instalacao_limpa_etapa2.md`). Na Etapa 3, `dvc init` já foi rodado, `data/raw_data/` já é versionado via DVC (`data/raw_data.dvc`) com um remote **S3** configurado (`.dvc/config`, bucket compartilhado do projeto — ver seção "Dados (DVC)" no `README.md`), e o pipeline `dvc.yaml` com os 4 stages (`preprocess` → `feature_eng` → `train` → `evaluate`, em `src/tech_challenge_recomendacao/dados/`, `modelos/`, `treino/` e `avaliacao/`) já roda ponta a ponta via `dvc repro`, com os stages `train`/`evaluate` logando params/métricas/artefatos no MLflow. A Etapa 4 já está implementada: o modelo padrão agora é `modelos/rede_neural.py` (`RedeNeural` — embeddings de usuário/filme + MLP + vieses + sigmoide, porta fiel do `NeuralRecommender` do notebook de referência, `models/recomendacao_movielens.ipynb`), treinado com early stopping (`treino/treinar.py`, monitorando RMSE de validação) e comparado, no stage `evaluate`, com 6 baselines Scikit-Learn/estatísticos (`avaliacao/baselines_sklearn.py`: `GlobalMean`, `UserItemBias`, SVD, NMF, ItemKNN, GBRT) em 6 métricas (RMSE, MAE, Precision@10, Recall@10, NDCG@10, Coverage — `avaliacao/metricas_ranking.py`), com a tabela comparativa completa salva em `models/comparacao_modelos.json`. O modelo antigo (`modelos/fatoracao_matricial.py`, um produto interno de embeddings + viés) continua registrado na Factory (`modelos/fabrica.py`) como alternativa mais simples, mas não é mais o padrão. O MLflow Model Registry também já está implementado: o stage `train` (`treino/treinar.py`, `registrar_e_mover_para_staging`) loga o modelo de cada run como nova versão de `recomendador-movielens` (`mlflow.pytorch.log_model`, formato `pickle` — o `pt2`/traced-graph do PyTorch exigiria `input_example`, desnecessário aqui já que quem serve o modelo é o checkpoint customizado em `modelos/checkpoint.py`, não este artefato do MLflow) e move a versão para o estágio **Staging**. O stage `evaluate` (`avaliacao/avaliar.py`, `promover_melhor_modelo_para_producao`) decide a promoção **Staging → Production**: compara o RMSE de teste do candidato com o da versão hoje em Production (guardado na tag `rmse_teste` de cada versão do Registry) e só promove se empatar ou melhorar — sem nenhuma Production ainda, promove direto (bootstrap). Essa comparação é contra a Production atual, não contra os baselines de `comparacao_modelos.json` (que servem só ao relatório de métricas da Etapa 4) — testado com `dvc repro` ponta a ponta contra um MLflow local (`recomendador-movielens v1` promovido a Production com sucesso, RMSE de teste ≈ 0,9005). `docs/MODEL_CARD.md` e `docs/RELATORIO.md` já existem e cobrem performance/limitações/vieses — redigidos em cima da execução do notebook de referência, mas com números consistentes aos do pipeline `src/`/DVC. Ainda faltam da Etapa 4: o vídeo STAR e, opcionalmente, o deploy em nuvem.

**Nota de validação (2026-07-20):** o hook `.git/hooks/pre-commit` não estava instalado neste clone (`.git/hooks/` não é versionado — rodar `uv run pre-commit install`). `uv run ruff check .` sem escopo falha com 2 erros, mas só em `models/recomendacao_movielens.ipynb` (notebook de referência); `uv run ruff check src tests` passa limpo. Um audit por AST encontrou 5 funções com corpo (sem contar docstring) acima de 20 linhas — todas orquestradoras (`main()` de cada stage do pipeline, `treinar_com_early_stopping`, `avaliar_ranking`), 22–36 linhas — não bloqueante, mas fica registrado como possível limpeza futura. O `Dockerfile` multi-stage (`builder` com `uv sync`; `mlflow-server`, que reaproveita o venv do `builder` para rodar o servidor MLflow com backend SQLite; `runtime`, estágio final/padrão, enxuto, rodando como usuário não-root, CMD padrão executando o stage `train`) já foi criado, junto com o `.dockerignore`, e `docker build`/`docker run` (stage `train`, contra um `mlflow server` local) já foram testados com sucesso nesta máquina (Docker Desktop instalado). A imagem final do `runtime` ficou grande (~3GB) porque o `torch` do PyPI traz o stack CUDA completo mesmo sem uso de GPU — ainda não otimizado para CPU-only, decisão adiada deliberadamente. O `docker-compose.yml` (serviços `mlflow` + `train`, com healthcheck e `MLFLOW_SERVER_ALLOWED_HOSTS` configurado para o nome do serviço) já foi criado e testado com sucesso via `docker compose up --build`: o `mlflow` fica saudável, o `train` aguarda e roda até o fim (exit 0), e o artefato do modelo é logado corretamente no MLflow via proxy HTTP (`--artifacts-destination`, não `--default-artifact-root` puro — path de filesystem direto falha entre containers sem storage compartilhado). Além disso, a API ganhou `POST /treino`/`GET /treino/status/{execucao_id}` (`api/servico_treino.py`) para disparar o pipeline sob demanda. O `Dockerfile` ganhou um estágio `api` (Docker CLI + plugin `docker compose` instalados, roda como root deliberadamente — ver comentário no `Dockerfile`) e o `docker-compose.yml` ganhou um serviço `api`: ele dispara o serviço `train` como **container irmão** via socket do Docker do host montado (`/var/run/docker.sock`), rodando `docker compose run --rm train dvc repro` — API e treino ficam isolados um do outro, sem dependência de processo. Isso exige a variável `HOST_PROJECT_DIR` no `.env` (ver comentário em `.env.example`) para os volumes relativos do `train` resolverem contra o caminho real do host, não contra o container da `api` — no Docker Desktop (Windows/Mac) pode exigir o caminho traduzido que o daemon reconhece internamente, não o caminho bruto do Windows. **Atenção:** esta parte (estágio `api`, socket do Docker, `HOST_PROJECT_DIR`) foi implementada mas **não pôde ser testada** nesta máquina/sessão — o `docker`/`docker compose` CLI não estava acessível aqui (nem via bash nem PowerShell), apesar de o Docker Desktop ter sido usado com sucesso em sessões anteriores da Etapa 3. Rode `docker compose build api && docker compose up -d mlflow api` e `curl -X POST http://localhost:8000/treino` para validar antes de confiar nisso em produção.

Qualquer trabalho futuro aqui consiste em construir o projeto do zero conforme a especificação abaixo — não assuma que convenções ou comandos existem até que tenham sido de fato criados neste repositório.

`data/raw_data/` e `data/processed_data/` (dados reais, não os ponteiros `.dvc`) e qualquer arquivo `*.pdf` estão no `.gitignore` e não devem ser commitados diretamente no git.

## Lista de tarefas

Todas as tarefas do projeto, detalhadas e ordenadas por etapa, estão em [`docs/TASKS.md`](docs/TASKS.md). Consulte esse arquivo antes de iniciar qualquer trabalho de implementação e marque os itens (`- [ ]` → `- [x]`) conforme forem concluídos.

## O que está sendo construído

Um sistema de recomendação de produtos para uma empresa de e-commerce, baseado no comportamento de navegação dos usuários. O modelo central é uma rede neural (MLP ou embedding-based) treinada com PyTorch. Datasets sugeridos: Instacart Market Basket, RetailRocket ou MovieLens (ou qualquer dataset com ≥ 10.000 interações user-item).

Stack obrigatória:
- **PyTorch** — o modelo de recomendação em si
- **Scikit-Learn** — pré-processamento e modelos baseline para comparação
- **MLflow** — rastreamento de experimentos e Model Registry (promoção Staging → Production)
- **DVC** — versionamento de dados e pipeline reprodutível
- **Docker** — build multi-stage (estágio builder para dependências, estágio runtime para a aplicação), além de `docker-compose.yml` rodando um serviço de treino e um servidor MLflow

## Dataset

O dataset escolhido é o **MovieLens ml-latest-small** (GroupLens/University of Minnesota), o mesmo usado no notebook de referência (`models/recomendacao_movielens.ipynb`). `data/` está organizado em subpastas por estágio do pipeline:

- `data/raw_data/` — dados brutos originais e suas referências, tal como baixados, sem nenhuma transformação:
  - `movies.csv` — catálogo de filmes (id, título, gêneros) — 9.742 filmes
  - `ratings.csv` — 100.836 avaliações (5 estrelas, incrementos de meia estrela) de 610 usuários
  - `tags.csv` — 3.683 tags livres aplicadas pelos usuários
  - `links.csv` — mapeamento dos ids de filmes para IMDb/TMDb
  - `README.txt` — documentação original do dataset (licença de uso, citação obrigatória, descrição dos campos)
- `data/processed_data/` — saída dos estágios `preprocess`/`feature_eng` do pipeline DVC.

> **Nota histórica:** este projeto usou inicialmente o MovieLens **ml-32m** (~32 milhões de avaliações, ~870 MB), subamostrado no `preprocess`. Na Etapa 4, o dataset foi trocado para `ml-latest-small` para que o pipeline `src/`/DVC/MLflow reproduza exatamente o mesmo dataset, split e modelo já documentados no notebook de referência e no `docs/MODEL_CARD.md`/`docs/RELATORIO.md` — ambos ≥ 10.000 interações user-item, atendendo ao requisito do desafio.

Licença: uso apenas para fins de pesquisa/acadêmicos, sem uso comercial sem autorização do GroupLens Research Project, e com citação obrigatória do paper (Harper & Konstan, 2015) em qualquer publicação — ver `data/raw_data/README.txt` para o texto completo.

**Importante sobre versionamento:** `data/` é o local correto para os dados (consistente com a estrutura de projeto exigida no spec), mas esses arquivos **não devem ser commitados no git** — são versionados via DVC (`data/raw_data.dvc`, já rastreado desde a Etapa 3), com os dados reais entrando no `.gitignore` (via `data/.gitignore`, gerenciado pelo próprio DVC) e apenas os ponteiros `.dvc` indo para o git.

## Gerenciamento de dependências

O projeto usa **uv** (não Poetry) como gerenciador de pacotes e de ambiente — é a ferramenta atualmente mais recomendada no ecossistema Python (mesmo time do `ruff`, muito mais rápida que Poetry/pip, suporta nativamente `[dependency-groups]` do PEP 735). O briefing permite "Poetry ou uv"; uv foi a escolha feita aqui.

- `pyproject.toml` na raiz define o projeto como pacote instalável em layout `src/` (`src/tech_challenge_recomendacao/`), com build backend `uv_build`.
- `requires-python = ">=3.13"` — alinhado à versão usada no desenvolvimento (3.13.7).
- Dependências de produção em `[project.dependencies]`: `torch`, `scikit-learn`, `mlflow`, `dvc`, `pandas`, `numpy`, `pydantic`, `pydantic-settings`.
- Dependências de desenvolvimento em `[dependency-groups.dev]` (não `[project.optional-dependencies]` — grupos de dev não devem ser extras instaláveis publicamente): `pytest`, `pytest-cov`, `ruff`, `pre-commit`.
- `uv.lock` é o lock file — **sempre commitado**, deve ser regenerado (`uv lock`) sempre que dependências mudarem.
- Ao adicionar/remover dependências, prefira `uv add <pacote>` / `uv remove <pacote>` a editar o `pyproject.toml` manualmente, para manter `pyproject.toml` e `uv.lock` sincronizados.
- Configuração de `ruff` (`line-length = 100`, `target-version = "py313"`, `select = ["E", "F", "I"]`) e de `pytest` (`testpaths = ["tests"]`, coverage em `src/`) já estão em `pyproject.toml`.

## Configuração via `.env`

- `src/tech_challenge_recomendacao/configuracoes.py` define `Configuracoes` (Pydantic `BaseSettings`) — única fonte de configuração do projeto. Nenhum valor (caminhos, seed, URIs) deve ser hardcoded em outro lugar do código; sempre importar `configuracoes` desse módulo.
- `.env` (não commitado, listado no `.gitignore`) contém os valores reais lidos em runtime; `.env.example` (commitado) documenta todas as variáveis com valores de exemplo/padrão.
- Variáveis atuais: `SEMENTE_ALEATORIA`, `MLFLOW_TRACKING_URI`, `DIRETORIO_DADOS_BRUTOS`, `DIRETORIO_DADOS_PROCESSADOS`, `DIRETORIO_MODELOS`. Ao adicionar uma nova configuração, adicione o campo em `Configuracoes` **e** a variável correspondente em `.env.example`.
- `scripts/validate_env.py` valida a versão do Python, o carregamento das configurações e a existência dos diretórios esperados — rodar via `uv run python scripts/validate_env.py`.

## Linting e pre-commit

- `.pre-commit-config.yaml` configura o hook `ruff-check --fix` + `ruff-format` (via [ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit), `rev` travado em uma versão compatível com o `ruff` do `pyproject.toml`).
- O hook já está instalado neste clone em `.git/hooks/pre-commit` — roda automaticamente a cada `git commit`. **Atenção:** `.git/hooks/` não é versionado; qualquer clone novo precisa rodar `uv run pre-commit install` uma vez (isso deveria também entrar no `README.md` de setup na Etapa 4).
- Para rodar manualmente em todos os arquivos: `uv run pre-commit run --all-files`.

## Convenções obrigatórias do repositório

- `.dockerignore`, `.gitignore`, `.env.example` presentes; configurações externalizadas via `.env` + Pydantic Settings (sem configuração hardcoded).
- Histórico de commits semântico (ex.: Conventional Commits).
- Clean code: funções ≤ 20 linhas, nomes descritivos, SOLID, type hints em todas as funções públicas, docstrings no estilo Google.
- Pelo menos um design pattern GoF aplicado de forma significativa — ex.: **Factory** para construção de modelos, **Strategy** para preprocessors intercambiáveis, ou **Template Method** para loops de treino. Não force um pattern onde uma função simples resolveria melhor.
- `ruff` configurado e sem erros, com pre-commit hooks aplicando essa checagem.
- Seeds fixas em todo o código (split de dados, inicialização do modelo, treino) para reprodutibilidade.
- Estrutura de projeto esperada: `src/`, `tests/`, `data/`, `models/`, `configs/` (reflete a separação entre dados versionados via DVC, modelos treinados e código, conforme o pipeline abaixo).

## Arquitetura do pipeline (quando construído)

O pipeline DVC (`dvc.yaml`) deve ter ≥ 3 stages, esperados como:

```
preprocess → feature_eng → train → evaluate
```

Cada stage deve ser executável de ponta a ponta via `dvc repro`. Os dados são versionados com `dvc init` + um remote configurado (local ou S3), em vez de serem commitados diretamente no git.

O stage `train` registra params, métricas e artefatos no MLflow a cada run (≥ 3 runs rastreados esperados), e o melhor modelo é promovido através do MLflow Model Registry (Staging → Production). Os modelos baseline (Scikit-Learn) devem ser comparados com o MLP em PyTorch usando ≥ 4 métricas.

## Comandos

- `uv sync` — instala o projeto e todas as dependências (prod + dev) em `.venv/`; deve funcionar de forma limpa em um ambiente novo (critério explícito de avaliação).
- `uv run python scripts/validate_env.py` — valida versão do Python, carregamento do `.env`/Pydantic Settings e existência dos diretórios de dados/modelos.
- `uv run pytest` — roda a suíte de testes (`tests/`; cobre as funções puras de `dados/`, `modelos/` e `parametros.py`).
- `uv run ruff check .` — linting; deve estar sem erros.
- `uv run ruff format .` — formatação.
- `uv run pre-commit install` — instala o git hook local (necessário uma vez por clone; não é versionado).
- `uv run pre-commit run --all-files` — roda os hooks configurados (ruff check + format) em todo o repositório.
- `uv add <pacote>` / `uv add --dev <pacote>` — adiciona dependência de produção/dev e atualiza `pyproject.toml` + `uv.lock`.
- `uv run dvc pull` — baixa `data/raw_data/` do remote S3 configurado (`s3remote`, ver "Dados (DVC)" no `README.md`).
- `uv run dvc push` — envia dados/modelos rastreados pelo DVC para o remote S3.
- `uv run dvc repro` — executa o pipeline completo (`preprocess` → `feature_eng` → `train` → `evaluate`, `dvc.yaml`); precisa de um servidor MLflow acessível em `MLFLOW_TRACKING_URI` (`uv run mlflow server` localmente, ou o serviço `mlflow` do `docker-compose.yml`).
- `uv run dvc metrics show` — mostra as métricas dos stages `train`/`evaluate` (`data/processed_data/metricas_*.json`).
- `docker build -t tech-challenge-recomendacao .` — builda a imagem multi-stage (`Dockerfile`, estágio `runtime` por padrão); testado com sucesso.
- `docker run --rm --env-file .env -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" tech-challenge-recomendacao` — roda o stage `train` isolado no container (precisa de um MLflow acessível separadamente — ver seção "Docker" do `README.md`); testado com sucesso.
- `docker compose up --build` — sobe os serviços `mlflow` (servidor de tracking, `http://localhost:5000`, backend SQLite em `./mlflow-data/`) e `train` (stage `train`, aguardando o `mlflow` ficar saudável); testado com sucesso.

Atualize esta seção com os comandos reais conforme forem adicionados — não deixe esta seção desatualizada depois que a estrutura do projeto existir.

## Entregáveis a manter em mente

- Um Model Card documentando performance, limitações e vieses.
- Um README com instruções completas de instalação e uso.
- Um vídeo de 5 minutos no formato STAR faz parte da avaliação, mas está fora do escopo de alterações de código aqui.
- Deploy em nuvem (AWS/Azure/GCP) é opcional/bônus, não obrigatório.
