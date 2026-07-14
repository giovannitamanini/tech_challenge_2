"""Métricas de ranking Top-K (Precision, Recall, NDCG, Coverage) — Strategy.

Porta fiel de `ranking_metrics`/`evaluate_ranking` do notebook de referência
(`models/recomendacao_movielens.ipynb`): o avaliador recebe uma *função de score*
(Strategy) — o mesmo avaliador serve tanto para os baselines Scikit-Learn quanto para a
rede neural, sem precisar conhecer a implementação de cada modelo.
"""

import math
from typing import Callable

import numpy as np
import pandas as pd


def ganho_acumulado_descontado(relevancias: list[int], k: int) -> float:
    """Discounted Cumulative Gain (DCG) das primeiras `k` posições.

    Args:
        relevancias: Relevância (0 ou 1) de cada item recomendado, na ordem do ranking.
        k: Nº de posições consideradas.

    Returns:
        DCG@k.
    """
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevancias[:k]))


def metricas_top_k(
    recomendados: list[int], relevantes: set[int], k: int
) -> tuple[float, float, float]:
    """Calcula Precision@k, Recall@k e NDCG@k para um único usuário.

    Args:
        recomendados: Filmes recomendados, ordenados do mais para o menos relevante.
        relevantes: Filmes relevantes (rating de teste acima do limiar) para o usuário.
        k: Tamanho do corte do ranking.

    Returns:
        Tupla `(precision@k, recall@k, ndcg@k)`.
    """
    topo = recomendados[:k]
    acertos = [1 if filme in relevantes else 0 for filme in topo]
    precisao = sum(acertos) / k
    recall = sum(acertos) / len(relevantes) if relevantes else 0.0
    ideal = ganho_acumulado_descontado(sorted(acertos, reverse=True), k)
    ndcg = ganho_acumulado_descontado(acertos, k) / ideal if ideal > 0 else 0.0
    return precisao, recall, ndcg


def construir_historico_usuarios(
    treino: pd.DataFrame, teste: pd.DataFrame, relevancia_minima: float
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Monta, por usuário, os filmes já vistos no treino e os relevantes no teste.

    Args:
        treino: Split de treino (`usuario_idx`, `filme_idx`, `rating`).
        teste: Split de teste (`usuario_idx`, `filme_idx`, `rating`).
        relevancia_minima: Rating mínimo (no teste) para um filme contar como relevante.

    Returns:
        Tupla `(vistos_no_treino, relevantes_no_teste)`, ambos `usuario_idx -> set[filme_idx]`.
    """
    vistos_no_treino = treino.groupby("usuario_idx")["filme_idx"].apply(set).to_dict()
    relevantes = teste[teste["rating"] >= relevancia_minima]
    relevantes_no_teste = relevantes.groupby("usuario_idx")["filme_idx"].apply(set).to_dict()
    return vistos_no_treino, relevantes_no_teste


def avaliar_ranking(
    funcao_score: Callable[[int], np.ndarray],
    vistos_no_treino: dict[int, set[int]],
    relevantes_no_teste: dict[int, set[int]],
    n_filmes: int,
    k: int,
) -> dict[str, float]:
    """Calcula Precision/Recall/NDCG@k médios e a cobertura de catálogo, dado `funcao_score`.

    Para cada usuário com filmes relevantes no teste, pontua todo o catálogo, remove os
    filmes já vistos no treino e compara o Top-k com os filmes relevantes.

    Args:
        funcao_score: Função `usuario_idx -> vetor de notas para todo o catálogo`
            (Strategy — serve tanto para baselines quanto para a rede neural).
        vistos_no_treino: Filmes já avaliados no treino, por usuário.
        relevantes_no_teste: Filmes relevantes (rating alto) no teste, por usuário.
        n_filmes: Nº total de filmes no catálogo.
        k: Tamanho do corte do ranking (ex.: `10` para Precision@10).

    Returns:
        Dicionário com `Precision@k`, `Recall@k`, `NDCG@k` e `Coverage`.
    """
    todos_os_filmes = np.arange(n_filmes)
    precisoes, recalls, ndcgs, recomendados_no_total = [], [], [], set()

    for usuario_idx, relevantes in relevantes_no_teste.items():
        if not relevantes:
            continue
        notas = funcao_score(usuario_idx).copy()
        notas[list(vistos_no_treino.get(usuario_idx, set()))] = -np.inf
        topo = todos_os_filmes[np.argpartition(-notas, k)[:k]]
        topo = topo[np.argsort(-notas[topo])].tolist()

        precisao, recall, ndcg = metricas_top_k(topo, relevantes, k)
        precisoes.append(precisao)
        recalls.append(recall)
        ndcgs.append(ndcg)
        recomendados_no_total.update(topo)

    return {
        f"Precision@{k}": float(np.mean(precisoes)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "Coverage": len(recomendados_no_total) / n_filmes,
    }
