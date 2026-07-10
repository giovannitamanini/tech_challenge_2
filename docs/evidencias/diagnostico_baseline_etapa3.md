# Evidência: diagnóstico do gap treino/teste do baseline (fim da Etapa 3)

Registro da investigação do resultado ruim do baseline atual (`FatoracaoMatricial`) antes de iniciar a Etapa 4, para descartar bug de pipeline (vazamento de dados, mismatch de índices) versus simples subtreinamento.

- **Commit testado:** `edc3ad5b282d0576dd59102ff66219207085a9f8`
- **Data:** 2026-07-10
- **Métricas de partida:** `data/processed_data/metricas_treino.json` (`perda_treino_mse: 4.919`) e `data/processed_data/metricas_avaliacao.json` (`rmse: 3.939`, `mae: 3.101`), geradas por `dvc repro` com os parâmetros atuais de `configs/params.yaml` (`dimensao_embedding: 16`, `taxa_aprendizado: 0.01`, `epocas: 5`, `tamanho_lote: 1024`).

## Motivação

Notas do MovieLens vão de 0.5 a 5.0. Um RMSE de teste de 3.94 é quase do tamanho da própria escala — bom demais para ignorar, ruim demais para seguir sem checar se não é bug. Em especial, o RMSE de teste (3.94) ficou bem pior que a perda de treino (`√4.919 ≈ 2.22`), um gap que poderia indicar vazamento de dados ou mismatch de índices entre treino e teste, não apenas underfitting comum.

## Hipóteses descartadas

**1. Cold start (usuário/filme no teste nunca visto no treino):**

```python
usuarios so no teste (cold start): 5 / 6370
filmes so no teste (cold start): 50 / 3890
linhas teste afetadas por usuario novo: 17 / 11625
linhas teste afetadas por filme novo: 76 / 11625
```

Menos de 1% das linhas de teste são afetadas — não explica um RMSE tão alto.

**2. Mismatch de índices entre treino/teste:** descartado por leitura de código. Em `engenharia_features.py` (`codificar_ids`), o `LabelEncoder` roda sobre o dataset **completo** (`avaliacoes_codificadas`), antes do `train_test_split` (`dividir_treino_teste`) — os índices de usuário/filme são consistentes entre os dois splits por construção.

**3. Bug de serialização do checkpoint:** descartado por leitura de código. `checkpoint.py` salva `state_dict` + metadados de arquitetura e reconstrói o modelo com `criar_modelo(...)` antes de carregar os pesos — sem inconsistência entre o que foi treinado e o que é avaliado.

## Causa real: subtreinamento, não bug

Previsões do modelo no conjunto de teste, comparadas à escala real (0.5–5):

```
previsões: min -15.03, max 16.22, média 2.10, std 3.60
reais:     min   0.50, max  5.00, média 3.38, std 1.03
```

Histórico de perda por época, extraído do MLflow (`perda_treino_mse`, run mais recente do experimento `recomendacao-movielens`):

```
época 1: perda 27.72
época 2: perda 17.07
época 3: perda 10.97
época 4: perda  7.26
época 5: perda  4.92
```

A perda ainda caia de forma acentuada (quase pela metade a cada época) quando o treino parou — `epocas: 5` (`configs/params.yaml`) encerrou o treino cedo demais. Como `nn.Embedding` inicializa os pesos com `N(0,1)` e a dimensão do embedding é 16, o produto interno usuário×filme começa com variância alta (daí previsões chegando a ±15/16); 5 épocas não foram suficientes para essa variância convergir para a escala real das notas.

Confirmação por comparação com baseline trivial (prever sempre a média do treino, ~3.38, para toda linha de teste):

```
RMSE baseline (média do treino): 1.0285
RMSE modelo atual:               3.9391
```

O baseline burro bate o modelo atual — evidência adicional de que o modelo não teve treino suficiente, e não que haja um bug de dados.

## Conclusão

Não há bug de pipeline a corrigir antes da Etapa 4. O resultado ruim do baseline é esperado e já está documentado como tal (`README.md`, `docs/TASKS.md`) — o baseline (`FatoracaoMatricial`) existe só para provar o pipeline ponta a ponta, não para ter bom desempenho. Fica registrado como item concreto para a Etapa 4 (além do que já está no `docs/TASKS.md`):

- Aumentar `epocas` e/ou adicionar early stopping (já previsto).
- Considerar inicializar os embeddings com variância menor (ex.: `nn.init.normal_(..., std=0.01)`) para convergir mais rápido, dado o comportamento observado aqui.
