"""Testes das funções puras do stage `feature_eng`."""

import pandas as pd

from tech_challenge_recomendacao.dados.engenharia_features import codificar_ids


def test_codificar_ids_gera_indices_contiguos_0_based() -> None:
    """Ids arbitrários de usuário/filme devem virar índices 0..n-1."""
    avaliacoes = pd.DataFrame(
        {
            "userId": [100, 100, 200],
            "movieId": [7, 9, 7],
            "rating": [5.0, 4.0, 3.0],
        }
    )

    codificadas = codificar_ids(avaliacoes)

    assert set(codificadas["usuario_idx"]) == {0, 1}
    assert set(codificadas["filme_idx"]) == {0, 1}
    assert codificadas["rating"].tolist() == [5.0, 4.0, 3.0]
