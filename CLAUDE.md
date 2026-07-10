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

Este repositório contém a lista de tarefas da atividade (`docs/TASKS.md`, PosTech "Tech Challenge") e o dataset bruto em `data/`. O repositório git local já foi inicializado, a estrutura de pastas base (`src/`, `tests/`, `models/`, `configs/`, além de `data/`) já foi criada, e a Etapa 2 (`pyproject.toml`/`uv.lock`, configurações via `.env`/Pydantic Settings, `scripts/validate_env.py`) está completa e verificada (ver `docs/evidencias/instalacao_limpa_etapa2.md`). Na Etapa 3, `dvc init` já foi rodado, `data/raw_data/` já é versionado via DVC (`data/raw_data.dvc`) com um remote local configurado (`.dvc/config`, pasta `dvc-storage/` fora do git — ver seção "Dados (DVC)" no `README.md`), e o pipeline `dvc.yaml` com os 4 stages (`preprocess` → `feature_eng` → `train` → `evaluate`, em `src/tech_challenge_recomendacao/dados/`, `modelos/`, `treino/` e `avaliacao/`) já roda ponta a ponta via `dvc repro`, com o stage `train` logando params/métricas/artefatos no MLflow. O modelo usado (`modelos/fatoracao_matricial.py`, uma fatoração matricial embedding-based, escolhido via Factory em `modelos/fabrica.py`) é um baseline simples só para provar o pipeline — a Etapa 4 substitui/expande por um modelo tunado, com early stopping e comparação com baselines Scikit-Learn. Dockerfile e `docker-compose.yml` ainda não existem. Qualquer trabalho futuro aqui consiste em construir o projeto do zero conforme a especificação abaixo — não assuma que convenções ou comandos existem até que tenham sido de fato criados neste repositório.

`data/raw_data/` e `data/processed_data/` (dados reais, não os ponteiros `.dvc`) e qualquer arquivo `*.pdf` estão no `.gitignore` e não devem ser commitados diretamente no git.

**Limitação conhecida:** o remote DVC configurado atualmente (`dvc-storage/`) é uma pasta local fora do repositório e do git — não é compartilhado automaticamente entre integrantes do grupo. Um clone novo do repositório terá o ponteiro `data/raw_data.dvc` mas não os dados reais em `data/raw_data/` até que alguém com acesso a esse remote (ou a um remote compartilhado reconfigurado) rode `dvc pull`. Depois disso, `dvc repro` recria `data/processed_data/` e `models/modelo_recomendador.pt` do zero. Isso é esperado no estágio atual do projeto — não é um bug a corrigir agora.

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

O dataset escolhido é o **MovieLens ml-32m** (GroupLens/University of Minnesota), um dataset de um serviço de recomendação de filmes. `data/` está organizado em subpastas por estágio do pipeline:

- `data/raw_data/` — dados brutos originais e suas referências, tal como baixados, sem nenhuma transformação:
  - `movies.csv` — catálogo de filmes (id, título, gêneros)
  - `ratings.csv` — ~32 milhões de avaliações (5 estrelas) de ~200 mil usuários; arquivo grande (~870 MB)
  - `tags.csv` — ~2 milhões de tags livres aplicadas pelos usuários
  - `links.csv` — mapeamento dos ids de filmes para IMDb/TMDb
  - `README.txt` — documentação original do dataset (licença de uso, citação obrigatória, descrição dos campos)
  - `checksums.txt` — checksums dos arquivos originais
- `data/processed_data/` — saída dos estágios de pré-processamento/feature engineering do pipeline DVC (`preprocess`, `feature_eng`); vazia até que esses stages sejam implementados.

Licença: uso apenas para fins de pesquisa/acadêmicos, sem uso comercial sem autorização do GroupLens Research Project, e com citação obrigatória do paper (Harper & Konstan, 2015) em qualquer publicação — ver `data/raw_data/README.txt` para o texto completo.

**Importante sobre versionamento:** `data/` é o local correto para os dados (consistente com a estrutura de projeto exigida no spec), mas dado o tamanho de `data/raw_data/ratings.csv` (~870 MB) esses arquivos **não devem ser commitados no git** — são versionados via DVC (`data/raw_data.dvc`, já rastreado desde a Etapa 3), com os dados reais entrando no `.gitignore` (via `data/.gitignore`, gerenciado pelo próprio DVC) e apenas os ponteiros `.dvc` indo para o git.

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
- `uv run dvc pull` — baixa `data/raw_data/` do remote local configurado (`dvc-storage/`, fora do git — ver "Dados (DVC)" no `README.md`).
- `uv run dvc push` — envia dados/modelos rastreados pelo DVC para o remote local.
- `uv run dvc repro` — executa o pipeline completo (`preprocess` → `feature_eng` → `train` → `evaluate`, `dvc.yaml`); precisa de um servidor MLflow acessível em `MLFLOW_TRACKING_URI` (`uv run mlflow server` localmente, ou o serviço do `docker-compose.yml` quando existir).
- `uv run dvc metrics show` — mostra as métricas dos stages `train`/`evaluate` (`data/processed_data/metricas_*.json`).
- `docker compose up` — ainda não configurado; vai subir o serviço de treino + servidor MLflow (Etapa 3).

Atualize esta seção com os comandos reais conforme forem adicionados — não deixe esta seção desatualizada depois que a estrutura do projeto existir.

## Entregáveis a manter em mente

- Um Model Card documentando performance, limitações e vieses.
- Um README com instruções completas de instalação e uso.
- Um vídeo de 5 minutos no formato STAR faz parte da avaliação, mas está fora do escopo de alterações de código aqui.
- Deploy em nuvem (AWS/Azure/GCP) é opcional/bônus, não obrigatório.
