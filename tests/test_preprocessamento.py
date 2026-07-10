"""Testes das funções puras do stage `preprocess`."""

import pandas as pd

from tech_challenge_recomendacao.dados.preprocessamento import (
    aplicar_amostragem,
    filtrar_por_frequencia_minima,
)


def _avaliacoes_de_exemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 3],
            "movieId": [10, 20, 30, 10, 10],
            "rating": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


def test_aplicar_amostragem_mantem_tudo_quando_tamanho_e_nulo() -> None:
    """`tamanho_amostra=None` não deve alterar o número de linhas."""
    avaliacoes = _avaliacoes_de_exemplo()

    resultado = aplicar_amostragem(avaliacoes, tamanho_amostra=None, semente=42)

    assert len(resultado) == len(avaliacoes)


def test_aplicar_amostragem_reduz_e_e_reprodutivel() -> None:
    """A mesma semente deve gerar sempre a mesma amostra."""
    avaliacoes = _avaliacoes_de_exemplo()

    amostra_1 = aplicar_amostragem(avaliacoes, tamanho_amostra=2, semente=42)
    amostra_2 = aplicar_amostragem(avaliacoes, tamanho_amostra=2, semente=42)

    assert len(amostra_1) == 2
    assert amostra_1.index.tolist() == amostra_2.index.tolist()


def test_filtrar_por_frequencia_minima_remove_usuarios_e_filmes_raros() -> None:
    """Usuários/filmes abaixo do mínimo devem ser removidos."""
    avaliacoes = _avaliacoes_de_exemplo()

    resultado = filtrar_por_frequencia_minima(avaliacoes, min_por_usuario=2, min_por_filme=2)

    # Só o usuário 1 tem >= 2 avaliações; só o filme 10 aparece >= 2 vezes.
    assert set(resultado["userId"]) == {1}
    assert set(resultado["movieId"]) == {10}
