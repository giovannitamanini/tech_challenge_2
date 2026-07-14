"""Testes das métricas de ranking (`avaliacao/metricas_ranking.py`)."""

import numpy as np
import pandas as pd

from tech_challenge_recomendacao.avaliacao.metricas_ranking import (
    avaliar_ranking,
    construir_historico_usuarios,
    ganho_acumulado_descontado,
    metricas_top_k,
)


def test_ganho_acumulado_descontado_favorece_acertos_no_topo() -> None:
    """Um acerto na 1ª posição deve valer mais do que o mesmo acerto mais abaixo."""
    dcg_no_topo = ganho_acumulado_descontado([1, 0, 0], k=3)
    dcg_no_fim = ganho_acumulado_descontado([0, 0, 1], k=3)

    assert dcg_no_topo > dcg_no_fim


def test_metricas_top_k_com_acerto_perfeito() -> None:
    """Se todos os relevantes estão no topo do ranking, NDCG deve ser 1.0."""
    precisao, recall, ndcg = metricas_top_k(
        recomendados=[10, 20, 30], relevantes={10, 20, 30}, k=3
    )

    assert precisao == 1.0
    assert recall == 1.0
    assert ndcg == 1.0


def test_metricas_top_k_sem_nenhum_acerto() -> None:
    """Sem nenhum item relevante no topo, precisão/recall/NDCG devem ser 0."""
    precisao, recall, ndcg = metricas_top_k(recomendados=[1, 2, 3], relevantes={99}, k=3)

    assert precisao == 0.0
    assert recall == 0.0
    assert ndcg == 0.0


def test_construir_historico_usuarios_separa_vistos_e_relevantes() -> None:
    """Deve separar os filmes vistos no treino dos relevantes (rating alto) no teste."""
    treino = pd.DataFrame(
        {"usuario_idx": [0, 0, 1], "filme_idx": [0, 1, 2], "rating": [5.0, 3.0, 4.0]}
    )
    teste = pd.DataFrame(
        {"usuario_idx": [0, 0, 1], "filme_idx": [2, 3, 4], "rating": [5.0, 2.0, 4.0]}
    )

    vistos_no_treino, relevantes_no_teste = construir_historico_usuarios(
        treino, teste, relevancia_minima=4.0
    )

    assert vistos_no_treino[0] == {0, 1}
    assert relevantes_no_teste[0] == {2}  # rating 5.0 é relevante; rating 2.0 não
    assert relevantes_no_teste[1] == {4}


def test_avaliar_ranking_recomenda_o_item_de_maior_score_e_exclui_vistos() -> None:
    """O avaliador deve excluir itens vistos no treino e recompensar quem acerta o topo."""
    n_filmes = 4

    def funcao_score(usuario_idx: int) -> np.ndarray:
        # O filme 3 tem sempre o maior score, para qualquer usuário.
        return np.array([0.1, 0.2, 0.3, 0.9])

    resultado = avaliar_ranking(
        funcao_score,
        vistos_no_treino={0: {3}},  # usuário 0 já viu o filme 3 (o de maior score)
        relevantes_no_teste={0: {2}},
        n_filmes=n_filmes,
        k=2,
    )

    assert resultado["Precision@2"] == 0.5  # filme 2 (relevante) entra no top-2 restante
    assert resultado["Recall@2"] == 1.0
    assert 0.0 <= resultado["Coverage"] <= 1.0
