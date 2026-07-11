# 🎬 Sistema de Recomendação Neural — MovieLens

> **Tech Challenge · Fase 02 — Pós MLE FIAP**
> Recomendação de filmes com **rede neural (PyTorch)**, comparada a **baselines Scikit-Learn**, com avaliação profunda em 6 métricas.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![Lint](https://img.shields.io/badge/ruff-passing-brightgreen)
![Status](https://img.shields.io/badge/notebook-executado%20sem%20erros-success)

---

## 📌 Sobre

Protótipo do **modelo central** do desafio: um recomendador **embedding-based** treinado com
PyTorch sobre o dataset **MovieLens `ml-latest-small`** (100.836 interações usuário–item).
O notebook cobre EDA, pré-processamento reprodutível, treino com *early stopping*,
**comparação com 6 baselines** e interpretação — servindo de base para a modularização em
`src/` com MLflow + DVC + Docker.

**Mapeamento de domínio:** usuário ↔ cliente · filme ↔ produto · rating ↔ sinal de interação.

## 📊 Resultados (conjunto de teste)

| Modelo | RMSE ↓ | MAE ↓ | NDCG@10 ↑ |
|---|---|---|---|
| **`NeuralRecommender`** (rede neural) | **0,8984** | **0,6911** | 0,0553 |
| SVD (sklearn) | 1,0282 | 0,8040 | **0,1700** |
| GBRT (sklearn) | 0,9377 | 0,7132 | 0,0036 |
| UserItemBias | 0,8996 | 0,6957 | 0,1196 |

> **Achado central:** *nenhum modelo vence em tudo.* A rede neural lidera na **predição de
> rating** (RMSE/MAE); as **fatorações de matriz** lideram no **ranking Top-10**. O "melhor
> modelo" depende do objetivo de negócio — detalhes no [relatório](RELATORIO.md).

## 🗂️ Estrutura

```
.
├── recomendacao_movielens.ipynb   # notebook principal (EDA + modelos + avaliação)
├── RELATORIO.md                   # relatório técnico profundo
├── MODEL_CARD.md                  # model card (performance, limitações, vieses)
├── README.md                      # este arquivo
└── ml-latest-small/               # dataset (ratings, movies, tags, links)
```

## 🚀 Como executar

**Requisitos:** Python 3.12 · numpy · pandas · scikit-learn · torch (CPU) · matplotlib.

```bash
# 1) Ambiente
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn torch matplotlib jupyter

# 2) Dataset (se ainda não extraído)
unzip ml-latest-small.zip -d .

# 3) Executar de ponta a ponta
jupyter nbconvert --to notebook --execute --inplace recomendacao_movielens.ipynb
#   ou abrir no Jupyter/VSCode e usar "Run All"
```

Execução completa: **~25 s em CPU**, determinística (`SEED = 42`), **`ruff` sem erros**.

> ⚠️ Rode as células **de cima para baixo** (Run All) — elas compartilham estado.

## 🧠 Modelo & baselines

- **`NeuralRecommender`** — embeddings de usuário/item → **MLP** `[64, 32, 16]` + termos de
  viés → saída reescalada para `[0,5; 5,0]`. 347.807 parâmetros, Adam + early stopping.
- **Baselines (4 sklearn + 2 referências):** `TruncatedSVD`, `NMF`, `ItemKNN`
  (`NearestNeighbors`), `HistGradientBoostingRegressor`, `UserItemBias`, `GlobalMean`.
- **Design patterns:** Template Method (`MatrixRecommender`), Factory (`build_baselines`),
  Strategy (avaliador de ranking).
- **Métricas (6):** RMSE, MAE, Precision@10, Recall@10, NDCG@10, Cobertura.

## 📚 Documentação

- 📄 **[RELATORIO.md](RELATORIO.md)** — relatório técnico completo (EDA, metodologia, resultados, aderência aos requisitos).
- 🧾 **[MODEL_CARD.md](MODEL_CARD.md)** — uso pretendido, limitações, vieses e considerações éticas.

## 🛣️ Roadmap (MLOps)

Migrar o notebook para `src/` + testes · **MLflow** (tracking + Model Registry) ·
**DVC** (pipeline `preprocess → train → evaluate`) · **Docker** multi-stage ·
**Poetry** + `.env` · melhoria do ranking com feedback implícito (BPR).

## 👤 Autoria

**Cristiano Sacramento** — membro da equipe · Tech Challenge Fase 02, Pós MLE FIAP.

---

<sub>Dataset: MovieLens (GroupLens) — uso não comercial. Harper & Konstan, 2015, ACM TiiS. https://doi.org/10.1145/2827872</sub>
