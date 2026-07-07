# TASKS.md

Lista detalhada de tarefas do projeto, referente à atividade PosTech "Tech Challenge". Ver também as diretrizes gerais em [`CLAUDE.md`](../CLAUDE.md).

## Contexto do problema

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários. O modelo central é uma rede neural (MLP ou embedding-based) treinada com PyTorch, com pipeline completo containerizado em Docker, dados versionados com DVC, experimentos rastreados no MLflow e código seguindo padrões profissionais de clean code.

- **Atividade em grupo · Obrigatória · Avaliada** (vale 90% da nota de todas as disciplinas da fase).
- **Entrega obrigatória:** Repositório GitHub + Vídeo de 5 minutos (método STAR).
- **Entrega opcional (bônus):** Deploy em ambiente de produção em nuvem (AWS, Azure ou GCP).
- **Dataset:** já definido neste projeto como MovieLens ml-32m (ver seção "Dataset" no `CLAUDE.md`), que atende ao requisito de ≥ 10.000 interações user-item.

As tarefas abaixo estão organizadas nas 4 etapas de desenvolvimento definidas no briefing, na ordem em que devem ser executadas. Requisitos que são transversais (valem para o projeto inteiro, não uma etapa específica) estão marcados como tal.

---

## Etapa 1 — Clean Code e Estrutura

**Foco:** projeto limpo com padrões de engenharia desde o início.

- [x] Definir estrutura de projeto com as pastas `src/`, `tests/`, `data/`, `models/`, `configs/`.
- [ ] Aplicar naming conventions descritivas e princípios SOLID desde a primeira linha de código.
- [ ] Implementar pelo menos 1 design pattern GoF de forma significativa:
  - **Factory** para criação/instanciação dos modelos, e/ou
  - **Strategy** para preprocessors intercambiáveis, e/ou
  - **Template Method** para os loops de treino.
- [ ] Adicionar type hints em todas as funções públicas.
- [ ] Adicionar docstrings no estilo Google em todas as funções públicas.
- [ ] Configurar `ruff` rodando sem erros.
- [ ] Configurar pre-commit hooks que executam o `ruff` automaticamente.

**Entregável da etapa:** repositório base com estrutura limpa e linting passando.

---

## Etapa 2 — Ambiente e Dependências

**Foco:** reprodutibilidade garantida com gerenciamento moderno de dependências.

- [x] Configurar `pyproject.toml` com Poetry (ou uv):
  - dependências de produção: `pytorch`, `scikit-learn`, `mlflow`, `dvc` (entre outras necessárias);
  - dependências de desenvolvimento: `pytest`, `ruff` (entre outras necessárias).
- [x] Gerar o lock file e commitá-lo no repositório.
- [x] Externalizar todas as configurações para `.env`, lidas via Pydantic Settings (nada de configuração hardcoded no código).
- [x] Criar `.env.example` com todas as variáveis documentadas (sem valores sensíveis).
- [x] Criar script de validação de ambiente em `scripts/validate_env.py`.
- [ ] Verificar que o projeto instala de forma limpa em um ambiente novo (do zero).

**Entregável da etapa:** projeto instalável do zero com `poetry install`.

---

## Etapa 3 — Containerização e Versionamento

**Foco:** Docker + DVC + MLflow integrados em um pipeline reprodutível.

- [ ] Criar `Dockerfile` multi-stage:
  - estágio `builder` — instala dependências;
  - estágio `runtime` — copia apenas o necessário para rodar a aplicação (imagem otimizada/enxuta).
- [ ] Criar `docker-compose.yml` com:
  - serviço de treino do modelo;
  - serviço de servidor MLflow.
- [ ] Rodar `dvc init` no repositório.
- [ ] Versionar o dataset (`data/`) com DVC (`dvc add`).
- [ ] Configurar um remote do DVC (local ou S3).
- [ ] Criar o pipeline DVC em `dvc.yaml` com no mínimo 3 stages, seguindo o fluxo:
  1. `preprocess`
  2. `feature_eng`
  3. `train`
  4. `evaluate`
- [ ] Garantir que cada stage é executável de ponta a ponta via `dvc repro`.
- [ ] Instrumentar o stage `train` para logar no MLflow, a cada run: parâmetros, métricas e artefatos.

**Entregável da etapa:** pipeline reprodutível via `dvc repro` + Docker funcional.

---

## Etapa 4 — Rede Neural, Registry e Entrega

**Foco:** modelo neural treinado, registrado e documentado.

- [ ] Treinar um modelo MLP ou embedding-based com PyTorch para a tarefa de recomendação.
- [ ] Aplicar early stopping no treino.
- [ ] Implementar e treinar baselines com Scikit-Learn para comparação.
- [ ] Comparar o modelo PyTorch com os baselines usando no mínimo 4 métricas.
- [ ] Rastrear no mínimo 3 runs de experimentos no MLflow.
- [ ] Registrar o melhor modelo no MLflow Model Registry.
- [ ] Promover o modelo registrado pelo fluxo Staging → Production.
- [ ] Escrever um Model Card cobrindo: performance, limitações e possíveis vieses do modelo.
- [ ] Finalizar o `README.md` do projeto com instruções completas de instalação e uso.
- [ ] Gravar o vídeo de até 5 minutos no formato STAR (ver seção abaixo).
- [ ] *(Opcional / bônus)* Fazer o deploy do container em nuvem (AWS, Azure ou GCP), acessível via URL pública.

**Entregável da etapa:** repositório final + modelo no Model Registry + vídeo STAR.

---

## Requisitos transversais (valem para o projeto inteiro)

Estes itens não pertencem a uma única etapa — devem ser mantidos válidos do início ao fim do projeto:

- [ ] `.dockerignore` configurado.
- [ ] `.gitignore` configurado (já iniciado neste repositório).
- [ ] `.env.example` configurado (ver Etapa 2).
- [ ] Histórico de commits semântico (ex.: Conventional Commits) do início ao fim do projeto.
- [ ] Funções com no máximo 20 linhas.
- [ ] Seeds fixadas em todos os pontos relevantes (split de dados, inicialização de modelo, treino) para garantir reprodutibilidade.
- [ ] Lock file sempre atualizado e commitado.

## Vídeo STAR (5 minutos, obrigatório)

O vídeo deve cobrir os 4 elementos do método STAR:

- [ ] **Situation:** problema de negócio e contexto do dataset.
- [ ] **Task:** objetivos técnicos e restrições.
- [ ] **Action:** decisões de arquitetura, modelo, versionamento e containerização.
- [ ] **Result:** resultados obtidos, trade-offs e lições aprendidas.

Critério de avaliação do vídeo: clareza, cobertura dos 4 elementos, duração ≤ 5 minutos.

## Bibliotecas requeridas

- **PyTorch** — rede neural para o modelo de recomendação.
- **Scikit-Learn** — pré-processamento e baselines.
- **MLflow** — tracking de experimentos e Model Registry.
- **DVC** — versionamento de dados e pipeline reprodutível.

## Critérios de avaliação (pesos)

| Critério | Peso | Descrição |
|---|---|---|
| Clean code e estrutura | 15% | SOLID, naming, type hints, design patterns, linting |
| Reprodutibilidade | 15% | Poetry, lock file, `.env`, instalação limpa |
| Docker | 15% | Multi-stage, imagem otimizada, compose funcional |
| DVC + Pipeline | 15% | Dataset versionado, pipeline ≥ 3 stages, `dvc repro` funcional |
| Rede neural (PyTorch) | 15% | MLP funcional, early stopping, comparação com baselines |
| MLflow + Registry | 10% | ≥ 3 runs rastreados, modelo promovido a Production |
| Vídeo STAR | 10% | Clareza, cobertura dos 4 elementos, ≤ 5 min |
| Bônus: deploy em nuvem | 5% | Container acessível via URL pública |

## Entregáveis finais

- [ ] Repositório GitHub completo e organizado.
- [ ] Modelo registrado e promovido no MLflow Model Registry.
- [ ] Vídeo STAR de até 5 minutos.
- [ ] *(Opcional)* Deploy em nuvem acessível via URL pública.
