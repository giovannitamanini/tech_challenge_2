# CLAUDE.md

Este arquivo fornece orientações ao Claude Code (claude.ai/code) para trabalhar com o código deste repositório.

## Idioma do projeto

Tudo neste projeto deve ser feito em português do Brasil: mensagens de commit, comentários de código, docstrings, nomes de variáveis/funções/classes, documentação (README, Model Card etc.), mensagens de log e qualquer texto voltado ao usuário. A única exceção são palavras reservadas e sintaxe das linguagens de programação e bibliotecas (ex.: `class`, `def`, `import`, `return`, nomes de parâmetros exigidos por frameworks como `self`, `forward`, `__init__`), que naturalmente permanecem em inglês por serem parte da linguagem.

## Status do projeto

Este repositório contém a lista de tarefas da atividade (`docs/TASKS.md`, PosTech "Tech Challenge") e o dataset bruto em `data/`. O repositório git local já foi inicializado, e a estrutura de pastas base (`src/`, `tests/`, `models/`, `configs/`, além de `data/`) já foi criada — ainda vazia, apenas com `.gitkeep`. Ainda não existe código-fonte, `pyproject.toml`, Dockerfile ou pipeline DVC. Qualquer trabalho futuro aqui consiste em construir o projeto do zero conforme a especificação abaixo — não assuma que convenções ou comandos existem até que tenham sido de fato criados neste repositório.

`data/` (dataset bruto) e qualquer arquivo `*.pdf` estão no `.gitignore` e não devem ser commitados.

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

**Importante sobre versionamento:** `data/` é o local correto para os dados (consistente com a estrutura de projeto exigida no spec), mas dado o tamanho de `data/raw_data/ratings.csv` (~870 MB) esses arquivos **não devem ser commitados no git** — devem ser versionados via DVC (`dvc add data/raw_data/...`) assim que o pipeline for inicializado, com `data/` entrando no `.gitignore` e apenas os ponteiros `.dvc` indo para o git.

## Convenções obrigatórias do repositório

- `pyproject.toml` gerenciado com Poetry (ou uv), com dependências de produção (pytorch, scikit-learn, mlflow, dvc) e de desenvolvimento (pytest, ruff) separadas. O lock file deve ser commitado.
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

## Comandos (a serem estabelecidos)

Ainda não existem comandos de build/lint/test pois nenhum código foi escrito. Assim que o `pyproject.toml` for criado, espera-se o fluxo padrão do Poetry:
- `poetry install` — deve funcionar de forma limpa em um ambiente novo (este é um critério explícito de avaliação)
- `poetry run pytest` — suíte de testes (criar testes por módulo em `tests/`)
- `poetry run ruff check .` — linting, deve estar sem erros
- `dvc repro` — executa o pipeline completo de dados/treino
- `docker compose up` — sobe o serviço de treino + servidor MLflow

Atualize esta seção com os comandos reais conforme forem adicionados — não deixe esta seção desatualizada depois que a estrutura do projeto existir.

## Entregáveis a manter em mente

- Um Model Card documentando performance, limitações e vieses.
- Um README com instruções completas de instalação e uso.
- Um vídeo de 5 minutos no formato STAR faz parte da avaliação, mas está fora do escopo de alterações de código aqui.
- Deploy em nuvem (AWS/Azure/GCP) é opcional/bônus, não obrigatório.
