"""Testes das funções puras do stage `feature_eng`."""

import pandas as pd

from tech_challenge_recomendacao.dados.engenharia_features import (
    ajustar_codificadores,
    codificar_ids,
    dividir_temporal_por_usuario,
    remover_itens_nao_vistos_no_treino,
)


def _avaliacoes_de_exemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [100, 100, 200],
            "movieId": [7, 9, 7],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [1000, 2000, 3000],
        }
    )


def _avaliacoes_codificadas_de_exemplo() -> pd.DataFrame:
    """Um usuário (0) com 10 avaliações em ordem temporal crescente, filme_idx 0..9."""
    return pd.DataFrame(
        {
            "usuario_idx": [0] * 10,
            "filme_idx": list(range(10)),
            "rating": [3.0] * 10,
            "timestamp": list(range(10)),
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


def test_dividir_temporal_por_usuario_respeita_a_ordem_e_as_fracoes() -> None:
    """Os últimos registros (por timestamp) de cada usuário devem ir para teste/validação."""
    avaliacoes = _avaliacoes_codificadas_de_exemplo()

    treino, validacao, teste = dividir_temporal_por_usuario(
        avaliacoes, fracao_validacao=0.2, fracao_teste=0.2
    )

    # 10 avaliações, 20%/20% => 2 em teste, 2 em validação, 6 em treino.
    assert len(teste) == 2
    assert len(validacao) == 2
    assert len(treino) == 6
    # Teste deve conter os timestamps mais recentes; treino, os mais antigos.
    assert teste["timestamp"].min() > validacao["timestamp"].max()
    assert validacao["timestamp"].min() > treino["timestamp"].max()


def test_dividir_temporal_mantem_usuario_no_treino_com_poucas_avaliacoes() -> None:
    """Mesmo um usuário com poucas avaliações deve ficar representado no treino."""
    avaliacoes = pd.DataFrame(
        {
            "usuario_idx": [0, 0, 0],
            "filme_idx": [0, 1, 2],
            "rating": [3.0, 3.0, 3.0],
            "timestamp": [1, 2, 3],
        }
    )

    treino, validacao, teste = dividir_temporal_por_usuario(
        avaliacoes, fracao_validacao=0.15, fracao_teste=0.15
    )

    assert len(treino) == 1
    assert len(validacao) == 1
    assert len(teste) == 1


def test_remover_itens_nao_vistos_no_treino_filtra_vazamento() -> None:
    """Filmes que só aparecem em validação/teste (nunca no treino) devem ser removidos."""
    treino = pd.DataFrame({"filme_idx": [0, 1, 2]})
    avaliacoes = pd.DataFrame({"filme_idx": [1, 2, 3, 4]})

    resultado = remover_itens_nao_vistos_no_treino(treino, avaliacoes)

    assert set(resultado["filme_idx"]) == {1, 2}
