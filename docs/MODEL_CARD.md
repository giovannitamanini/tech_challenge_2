# Model Card — `NeuralRecommender`

Sistema de recomendação de filmes baseado em rede neural com embeddings.
Documento no padrão *Model Cards for Model Reporting* (Mitchell et al., 2019).

> **Criado pelo membro da equipe Cristiano Sacramento** · Tech Challenge Fase 02 — Pós MLE FIAP.

---

## 1. Detalhes do modelo

| Campo | Valor |
|---|---|
| **Nome** | `NeuralRecommender` |
| **Versão** | 1.0 |
| **Data** | Julho/2026 |
| **Tipo** | Filtragem colaborativa neural (embeddings + MLP + termos de viés) |
| **Framework** | PyTorch 2.11 (CPU/GPU — *device-agnostic*) |
| **Tarefa** | Predição de rating (regressão) e geração de recomendações Top-N |
| **Autoria** | Cristiano Sacramento (equipe do Tech Challenge) |
| **Licença do código** | Uso educacional (Tech Challenge FIAP) |
| **Licença dos dados** | MovieLens/GroupLens — **uso não comercial** sem autorização prévia |

### Arquitetura
Para cada par (usuário, item):
1. `nn.Embedding` de usuário e de item (dimensão **32**);
2. concatenação dos dois vetores → **MLP** `[64, 32, 16]` com ReLU + Dropout (0,2);
3. soma de **viés de usuário + viés de item + viés global**;
4. **sigmoide reescalada** para o intervalo `[0,5; 5,0]` (só produz ratings válidos).

- **Parâmetros treináveis:** 347.807.
- **Otimização:** Adam (lr = 1e-3, weight decay = 1e-5), perda `MSELoss`, batch 1024.
- **Regularização:** dropout 0,2 + weight decay + **early stopping** (paciência 4).
- **Reprodutibilidade:** seeds fixados (`SEED = 42`) em Python/NumPy/PyTorch.

---

## 2. Uso pretendido

**Casos de uso primários**
- Prever a nota (0,5–5,0) que um usuário daria a um filme ainda não avaliado.
- Gerar listas de recomendação **Top-N** de filmes por usuário.

**Usuários primários**
- Equipe do Tech Challenge (contexto educacional/MLOps); serve de protótipo para um
  motor de recomendação de e-commerce (usuário↔cliente, filme↔produto).

**Usos fora de escopo (não recomendados)**
- Decisões de alto risco (crédito, saúde, contratação) — não aplicável.
- Produção sem revalidação, monitoramento e retreinamento periódico.
- Domínios diferentes de filmes sem novo treino.
- Recomendação para **usuários ou itens não vistos no treino** (ver *cold-start*, §7).
- Uso **comercial** dos dados MovieLens sem autorização do GroupLens.

---

## 3. Fatores

- **Fatores relevantes:** nível de atividade do usuário (nº de avaliações),
  popularidade do item, época da avaliação (deriva temporal).
- **Fatores de avaliação:** desempenho medido de forma agregada no conjunto de teste.
- **Grupos não avaliados:** o dataset MovieLens é **anonimizado e sem atributos
  demográficos** (idade, gênero, região). Portanto, **não é possível auditar
  justiça (fairness) por grupo demográfico** — uma limitação relevante deste modelo.

---

## 4. Métricas

**Predição de rating**
- **RMSE** (erro quadrático médio) — métrica principal.
- **MAE** (erro absoluto médio).

**Ranking Top-10** (relevância = rating de teste ≥ 4,0)
- **Precision@10**, **Recall@10**, **NDCG@10** e **Cobertura de catálogo**.

Ranking calculado sobre todo o catálogo, excluindo itens já vistos no treino.
Não há limiar de decisão em produção definido (protótipo).

---

## 5. Dados

| | Descrição |
|---|---|
| **Dataset** | MovieLens `ml-latest-small` (GroupLens, 2018) |
| **Volume** | 100.836 ratings · 610 usuários · 9.724 filmes |
| **Período** | 29/03/1996 – 24/09/2018 |
| **Seleção** | Usuários escolhidos ao acaso, **cada um com ≥ 20 avaliações** |
| **Pré-processamento** | IDs recodificados para índices contíguos; sem imputação |
| **Partição** | **Split temporal por usuário** — 15 % finais p/ teste, 15 % anteriores p/ validação |

**Tamanho das partições:** Treino 70.614 · Validação 13.925 · Teste 13.603.
Itens presentes apenas em val/test (nunca vistos no treino) foram removidos para evitar
vazamento. Média global de rating (treino) = **3,522**.

---

## 6. Análise quantitativa (resultados)

Conjunto de **teste**. Melhor valor por métrica em **negrito**.

| Modelo | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
|---|---|---|---|---|---|---|
| GlobalMean | 1,0546 | 0,8275 | — | — | — | — |
| UserItemBias | 0,8996 | 0,6957 | 0,0303 | 0,0356 | 0,1196 | 0,0051 |
| SVD (sklearn) | 1,0282 | 0,8040 | 0,0487 | 0,0565 | **0,1700** | 0,0528 |
| NMF (sklearn) | 3,0310 | 2,8371 | **0,0491** | **0,0691** | 0,1616 | **0,0535** |
| ItemKNN (sklearn) | 1,0032 | 0,7713 | 0,0003 | 0,0001 | 0,0016 | 0,0187 |
| GBRT (sklearn) | 0,9377 | 0,7132 | 0,0007 | 0,0001 | 0,0036 | 0,0198 |
| **`NeuralRecommender`** | **0,8984** | **0,6911** | 0,0079 | 0,0087 | 0,0553 | 0,0031 |

**Desempenho do modelo:** RMSE **0,8984** / MAE **0,6911** no teste (melhor entre todos os
modelos avaliados). Treino encerrado por early stopping na **época 9** (melhor RMSE de
validação = 0,8782).

**Leitura crítica (honestidade obrigatória):**
- Na **predição de rating**, o `NeuralRecommender` é o melhor modelo.
- No **ranking Top-10**, ele é **superado** pelas fatorações de matriz (`SVD`, `NMF`):
  NDCG@10 ≈ 0,17 contra 0,055 da rede. **Para a tarefa de ordenar recomendações, uma
  fatoração de matriz é atualmente mais eficaz** — dominado por termos de viés, o modelo
  tende a recomendar clássicos populares a todos, reduzindo personalização e diversidade.

---

## 7. Limitações

1. **Cold-start:** não recomenda a usuários ou itens ausentes do treino (embeddings
   inexistentes).
2. **Ranking fraco vs. fatoração de matriz:** ver §6 — a rede perde em Precision/Recall/NDCG.
3. **Baixa cobertura/diversidade:** recomenda majoritariamente títulos consagrados
   (Coverage@10 ≈ 0,3 %), reforçando *blockbusters*.
4. **Domínio de nicho:** filmes, usuários engajados (≥ 20 avaliações); não generaliza para
   outros catálogos sem retreino.
5. **Sem sinal contextual:** ignora gêneros, tags, texto e tempo na predição (usa apenas o
   par usuário–item).
6. **Dataset de desenvolvimento:** o `ml-latest-small` é explicitamente rotulado pelo
   GroupLens como *não* adequado para resultados de pesquisa publicáveis.

---

## 8. Vieses conhecidos

- **Viés de popularidade:** 20 % dos filmes concentram **77 %** das interações. O modelo
  aprende e **amplifica** esse desequilíbrio (recomenda o que já é popular).
- **Viés para clássicos:** os itens de maior viés aprendido são filmes aclamados antigos
  (ex.: *Shawshank Redemption*, *Star Wars*, *Pulp Fiction*), sugeridos a perfis variados.
- **Viés de seleção:** amostra restrita a usuários com ≥ 20 avaliações — não representa
  usuários novos/pouco ativos (justamente os de maior risco de cold-start).
- **Fairness não auditável:** ausência de atributos demográficos impede avaliar
  disparidades por grupo (idade, gênero, região).

---

## 9. Considerações éticas

- **Privacidade:** dados anonimizados (apenas IDs); **sem PII**. Ainda assim, embeddings de
  usuário podem, em tese, correlacionar-se a padrões sensíveis de consumo.
- **Bolha de filtragem:** recomendar sempre o popular pode reduzir a diversidade de
  descoberta do usuário. Mitigação: re-ranking por diversidade/novidade.
- **Uso responsável dos dados:** respeitar a licença GroupLens (**não comercial** sem
  autorização) e citar a fonte em publicações.

---

## 10. Recomendações e próximos passos

- **Para ranking em produção:** preferir uma fatoração de matriz **ou** retreinar a rede com
  **feedback implícito** e perda de ranking (**BPR** / *sampled-softmax*) em vez de MSE.
- **Reduzir o viés de popularidade:** regularizar mais os termos de viés; aplicar
  re-ranking por cobertura/diversidade; penalizar itens de cauda curta.
- **Mitigar cold-start:** incorporar *features* de conteúdo (gêneros, tags) para itens/
  usuários novos (modelo híbrido).
- **Governança MLOps:** versionar dados (DVC), rastrear experimentos e métricas (MLflow,
  ≥ 3 runs), promover a melhor versão no Model Registry e **monitorar drift** em produção.

---

## 11. Pegada computacional

- Treinado em **CPU** (torch 2.11+cpu); execução completa do pipeline em **~25 s**.
- 347.807 parâmetros; custo de treino e inferência desprezível — adequado a hardware comum.

---

## 12. Como reproduzir / citar

**Reproduzir:** executar `recomendacao_movielens.ipynb` de ponta a ponta (menu *Run All*)
com `pandas`, `torch`, `scikit-learn` e `matplotlib` instalados e a pasta
`ml-latest-small/` no mesmo diretório. Resultado determinístico (`SEED = 42`).

**Citação do dataset:**
> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and
> Context.* ACM TiiS 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

---

*Model Card gerado a partir da execução determinística do notebook; todos os números
refletem a saída real das células.*
