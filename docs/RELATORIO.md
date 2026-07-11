# Relatório Técnico — Sistema de Recomendação Neural (MovieLens)

**Tech Challenge · Fase 02 — Pós MLE FIAP**
**Criado pelo membro da equipe Cristiano Sacramento.**
Notebook: [`recomendacao_movielens.ipynb`](recomendacao_movielens.ipynb) · Dataset: MovieLens `ml-latest-small`

---

## 1. Sumário executivo

Este relatório documenta o **protótipo analítico do modelo central** do Tech Challenge: um
sistema de recomendação baseado em **rede neural com embeddings (PyTorch)**, comparado com
**seis baselines** (quatro deles Scikit-Learn) sobre **seis métricas**.

**Principais resultados:**

- Dataset validado: **100.836 interações** usuário–item (requisito ≥ 10.000 atendido com folga).
- A **rede neural** obtém a **melhor predição de rating** (RMSE **0,898** / MAE **0,691**),
  à frente de todos os baselines.
- Em **ranking Top-10**, as **fatorações latentes** (`SVD`, `NMF`) vencem
  (NDCG@10 ≈ 0,16–0,17), superando a rede — um *trade-off* real e esperado.
- **Conclusão central:** *"melhor modelo" depende do objetivo de negócio* — estimar a nota
  (regressão) é diferente de ordenar uma lista (ranking). Reportar as duas famílias de
  métrica é o que torna a avaliação honesta.

O notebook é **100% reprodutível** (seeds fixados, código *device-agnostic*), passa no
linter **`ruff`** sem erros (padrão PEP8, linha ≤ 88) e executa de ponta a ponta em ~25 s
em CPU, sem erros.

---

## 2. Contexto e objetivo

O enunciado descreve uma empresa de e-commerce que precisa recomendar produtos a partir do
comportamento dos usuários, tendo como modelo central uma **rede neural (MLP/embedding-based)
em PyTorch**. O enunciado aceita explicitamente **MovieLens** como dataset de recomendação
(ou qualquer base com ≥ 10.000 interações user-item).

**Mapeamento de domínio adotado:**

| E-commerce (enunciado) | MovieLens (este trabalho) |
|---|---|
| Cliente | Usuário (`userId`) |
| Produto | Filme (`movieId`) |
| Sinal de interação/preferência | Rating (0,5–5,0 estrelas) |

A mesma arquitetura de embeddings + MLP vale para sinais de navegação/compra.

---

## 3. Dataset — `ml-latest-small` (GroupLens)

| Propriedade | Valor |
|---|---|
| Interações (ratings) | **100.836** |
| Usuários únicos | **610** |
| Filmes únicos | **9.724** |
| Tags | 3.683 |
| Escala de rating | 0,5 – 5,0 (incrementos de meia estrela) |
| Densidade da matriz | **1,70 %** (esparsidade **98,30 %**) |
| Período | 29/03/1996 – 24/09/2018 |

Cada usuário avaliou **no mínimo 20 filmes**. A validação é feita **programaticamente** no
notebook (`assert n_inter >= 10_000`), garantindo que os dados estão dentro da especificação
antes de qualquer treino.

**Arquivos usados:** `ratings.csv` (interações), `movies.csv` (títulos/gêneros).
`tags.csv` e `links.csv` ficam disponíveis para extensões futuras.

---

## 4. Análise Exploratória (EDA)

Quatro perguntas guiaram a EDA, cada uma com impacto direto em decisões de modelagem.

### 4.1 Distribuição dos ratings
- Rating **médio 3,502**, **mediana 3,5**. Distribuição concentrada em 3–4 estrelas, com
  leve assimetria positiva. → Justifica prever no intervalo `[0,5; 5,0]` e usar **RMSE/MAE**.

### 4.2 Atividade e popularidade (long-tail)
- Ratings por **usuário**: min 20 · mediana 70 · máx 2.698.
- Ratings por **filme**: min 1 · mediana 3 · máx 329.
- **Os 20 % de filmes mais populares concentram 77,0 % de todas as interações.**
  → *Long-tail* acentuado: explica por que baselines de popularidade são competitivos e por
  que **cobertura de catálogo** é uma métrica relevante.

### 4.3 Esparsidade
- Matriz 610 × 9.724 com **98,30 % de células vazias**. → Motiva **embeddings** +
  regularização em vez de métodos densos.

### 4.4 Deriva temporal e gêneros
- Volume de avaliações varia bastante ao longo dos anos. → Justifica **split temporal**
  (prever o futuro) em vez de aleatório.
- Gêneros dominantes: Drama e Comédia. → Contexto para interpretar os embeddings (§7).

*(Gráficos correspondentes estão embutidos no notebook, seções 3 e 9.)*

---

## 5. Metodologia

### 5.1 Pré-processamento e split
- **Encoding contíguo** de `userId`/`movieId` para índices `0..N-1` (necessário às
  `nn.Embedding`).
- **Split temporal por usuário**: para cada usuário, ordena por tempo e reserva os **15 %
  finais** para teste e os **15 %** anteriores para validação. Isso **evita vazamento**
  (não prevê passado com futuro) e **garante que todo usuário aparece no treino**.
- Itens presentes apenas em val/test (nunca vistos no treino) são **removidos** — o modelo
  não pode ter embedding para um item desconhecido.

| Partição | Interações |
|---|---|
| Treino | 70.614 |
| Validação | 13.925 |
| Teste | 13.603 |

Média global (treino) = **3,522**.

### 5.2 Modelos avaliados

**Rede neural — `NeuralRecommender` (PyTorch)**
Arquitetura embedding-based, para cada par (usuário, item):
1. `nn.Embedding` de usuário e de item (dim 32);
2. concatenação → **MLP** `[64, 32, 16]` com ReLU + Dropout (0,2);
3. soma de **viés de usuário + viés de item + viés global**;
4. **sigmoide reescalada** para `[0,5; 5,0]` (só prevê ratings válidos).

- Parâmetros treináveis: **347.807**.
- Otimizador Adam (lr 1e-3, weight decay 1e-5), `MSELoss`, batch 1024.
- **Early stopping** (paciência 4) monitorando RMSE de validação.

**Baselines** — cobrindo as três grandes famílias de RecSys clássico:

| Baseline | Família | Biblioteca |
|---|---|---|
| `GlobalMean` | referência trivial (piso) | numpy |
| `UserItemBias` | modelo de viés (μ + bᵤ + bᵢ) | pandas/numpy |
| `SVD` | fatoração de matriz latente | **`sklearn.decomposition.TruncatedSVD`** |
| `NMF` | fatoração não-negativa | **`sklearn.decomposition.NMF`** |
| `ItemKNN` | CF por vizinhança item-item (cosseno) | **`sklearn.neighbors.NearestNeighbors`** |
| `GBRT` | ML supervisionado sobre features | **`sklearn.ensemble.HistGradientBoostingRegressor`** |

### 5.3 Métricas (6)
- **Predição:** RMSE, MAE (via `sklearn.metrics`).
- **Ranking Top-10:** Precision@10, Recall@10, NDCG@10 e **Cobertura de catálogo**.
  Relevância = rating de teste ≥ 4,0; ranking calculado sobre todo o catálogo, excluindo
  itens já vistos no treino.

### 5.4 Padrões de projeto (clean code)
- **Template Method** — base `MatrixRecommender` implementa `predict`/`score` uma vez;
  cada baseline só preenche a matriz `recon`.
- **Factory** — `build_baselines()` instancia e treina todos os modelos.
- **Strategy** — o avaliador de ranking recebe a *função de score*, servindo igualmente a
  baselines e à rede neural.

### 5.5 Reprodutibilidade
- Seeds fixados em Python, NumPy e PyTorch (`SEED = 42`).
- Código *device-agnostic* (CPU/GPU via `DEVICE`).
- Hiperparâmetros centralizados em um `dataclass Config` (pronto para virar Pydantic
  Settings + `.env`).
- Linter **`ruff`** sem erros (PEP8, linha ≤ 88).

---

## 6. Resultados

Tabela comparativa completa (conjunto de **teste**), gerada pelo notebook:

| Modelo | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
|---|---|---|---|---|---|---|
| GlobalMean | 1,0546 | 0,8275 | — | — | — | — |
| UserItemBias | 0,8996 | 0,6957 | 0,0303 | 0,0356 | 0,1196 | 0,0051 |
| **SVD (sklearn)** | 1,0282 | 0,8040 | 0,0487 | 0,0565 | **0,1700** | 0,0528 |
| NMF (sklearn) | 3,0310 | 2,8371 | **0,0491** | **0,0691** | 0,1616 | **0,0535** |
| ItemKNN (sklearn) | 1,0032 | 0,7713 | 0,0003 | 0,0001 | 0,0016 | 0,0187 |
| GBRT (sklearn) | 0,9377 | 0,7132 | 0,0007 | 0,0001 | 0,0036 | 0,0198 |
| **NeuralRecommender** | **0,8984** | **0,6911** | 0,0079 | 0,0087 | 0,0553 | 0,0031 |

*(Treino da rede: early stopping na época 9, melhor val RMSE 0,8782.)*

### Leitura crítica — nenhum modelo vence em tudo

- **Melhor predição de rating (RMSE/MAE):** `NeuralRecommender`, empatado com `UserItemBias`
  e seguido pelo `GBRT`. Modelos otimizados para *erro de nota* dominam aqui.
- **Melhor ranking Top-10:** as **fatorações latentes** (`SVD`, `NMF`) — bem à frente da
  rede (NDCG 0,17 vs 0,055).
- **`ItemKNN` e `GBRT`:** boa nota pontual, mas ranking fraco — ordenam por sinais de
  popularidade/desvio que discriminam mal os itens *não vistos* de cada usuário.
- **`NMF` com RMSE ≈ 3,0 (valor didático):** aplicada à matriz esparsa com ausências
  tratadas como zero, ela subestima a *escala absoluta* (péssimo RMSE) mas preserva a
  *ordem* (ótimo ranking). É o caso clássico de por que **avaliar em uma métrica só engana**.

**Interpretação de negócio:** para *prever a nota* que um usuário daria, a rede neural é a
melhor escolha. Para *montar uma lista ranqueada* de recomendações, uma fatoração de matriz
é hoje mais eficaz. A rede, dominada pelos termos de viés, tende a recomendar clássicos de
alto viés para todos (ver §7), o que reduz seu desempenho em ranking.

---

## 7. Interpretabilidade

### 7.1 Viés de item aprendido (melhores × piores)
O viés de item da rede aproxima a "qualidade geral" corrigida por popularidade:

- **Maior viés (melhores):** Goodfellas · Star Wars V (Empire Strikes Back) · LOTR: The Two
  Towers · Shawshank Redemption · Monty Python and the Holy Grail · Forrest Gump · Pulp
  Fiction · Schindler's List · Braveheart · Star Wars IV.
- **Menor viés (piores):** Johnny Mnemonic · Planet of the Apes (2001) · Nine Months ·
  Charlie's Angels · Mortal Kombat · City Slickers II · The Flintstones · Coneheads · Wild
  Wild West · Batman & Robin.

O resultado é coerente com o senso comum crítico, indicando que os embeddings capturaram
sinal real de qualidade.

### 7.2 PCA dos embeddings
Projeção 2D dos embeddings de filmes (apenas itens com ≥ 30 ratings): os **2 primeiros
componentes explicam 31,2 %** da variância, revelando eixos latentes de gosto.

### 7.3 Exemplo de recomendação Top-N
Para o usuário 609 (favoritos: *Fight Club*, *Inglourious Basterds*, *Bourne Ultimatum*,
*Mulholland Drive*, *Drive*), a rede recomenda clássicos aclamados como *Double Indemnity*,
*A Streetcar Named Desire*, *Ran*, *High Noon* e *Seven Samurai* — ilustrando tanto a
capacidade de sugerir itens de alta qualidade quanto o viés para clássicos consagrados.

---

## 8. Model Card (resumo)

| Item | Descrição |
|---|---|
| **Uso pretendido** | Prever rating e gerar recomendações Top-N de filmes; protótipo educacional. |
| **Dados de treino** | MovieLens `ml-latest-small` (610 usuários, 9.724 filmes, 100.836 ratings). |
| **Métrica principal** | RMSE = 0,898 (teste). |
| **Limitações** | (1) *Cold-start*: não recomenda a usuários/itens fora do treino. (2) Domínio de nicho (filmes, usuários que avaliam ≥ 20). (3) Fraco em ranking Top-K vs. fatoração de matriz. |
| **Vieses conhecidos** | **Viés de popularidade** (77 % das interações em 20 % dos filmes) e **viés para clássicos** — a rede tende a recomendar títulos de alto viés a todos, reduzindo diversidade (cobertura baixa). |
| **Mitigações futuras** | Treino com feedback implícito + perda de ranking (BPR); regularização mais forte dos termos de viés; re-ranking por diversidade. |

---

## 9. Aderência aos requisitos do Tech Challenge

| Requisito | Situação neste notebook |
|---|---|
| Rede neural PyTorch (MLP/embedding-based) | ✅ `NeuralRecommender` (embeddings + MLP + viés) |
| Early stopping | ✅ na rede (val RMSE) e no `GBRT` |
| Comparação com baselines Scikit-Learn | ✅ **4 modelos sklearn** (SVD, NMF, ItemKNN, GBRT) + 2 referências |
| ≥ 4 métricas | ✅ **6** métricas (RMSE, MAE, P@10, R@10, NDCG@10, Cobertura) |
| Dataset ≥ 10.000 interações | ✅ 100.836 (validado por `assert`) |
| Seeds fixados / device-agnostic | ✅ `set_seed` + `DEVICE` |
| Clean code (funções curtas, type hints, docstrings) | ✅ `ruff` limpo (PEP8, ≤ 88) |
| Design patterns | ✅ Template Method + Factory + Strategy |

> **Escopo:** o notebook cobre as etapas de **dados e modelo**. Os itens de **infraestrutura**
> (MLflow, DVC, Docker, Poetry) são de repositório e estão no roadmap abaixo.

---