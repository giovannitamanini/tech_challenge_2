"""Testes das funções puras do stage `feature_eng`."""

import pandas as pd

from tech_challenge_recomendacao.dados.engenharia_features import (
    ajustar_codificadores,
    codificar_ids,
)


def _avaliacoes_de_exemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [100, 100, 200],
            "movieId": [7, 9, 7],
            "rating": [5.0, 4.0, 3.0],
        }
    )


def test_codificar_ids_gera_indices_contiguos_0_based() -> None:
    """Ids arbitrários de usuário/filme devem virar índices 0..n-1."""
    avaliacoes = _avaliacoes_de_exemplo()
    codificador_usuarios, codificador_filmes = ajustar_codificadores(avaliacoes)

    codificadas = codificar_ids(avaliacoes, codificador_usuarios, codificador_filmes)

    assert set(codificadas["usuario_idx"]) == {0, 1}
    assert set(codificadas["filme_idx"]) == {0, 1}
    assert codificadas["rating"].tolist() == [5.0, 4.0, 3.0]


def test_ajustar_codificadores_e_reprodutivel_por_ordem_dos_ids() -> None:
    """O índice atribuído a cada id deve ser estável (ordenação crescente do `LabelEncoder`)."""
    avaliacoes = _avaliacoes_de_exemplo()

    codificador_usuarios, codificador_filmes = ajustar_codificadores(avaliacoes)

    assert list(codificador_usuarios.classes_) == [100, 200]
    assert list(codificador_filmes.classes_) == [7, 9]
