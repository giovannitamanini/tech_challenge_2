"""Testes da factory de modelos e dos modelos `FatoracaoMatricial`/`RedeNeural`."""

import pytest
import torch

from tech_challenge_recomendacao.modelos.capacidades import ExpoeEmbeddingsItem
from tech_challenge_recomendacao.modelos.fabrica import criar_modelo
from tech_challenge_recomendacao.modelos.fatoracao_matricial import FatoracaoMatricial
from tech_challenge_recomendacao.modelos.rede_neural import FAIXA_RATING, RedeNeural


def test_criar_modelo_instancia_fatoracao_matricial() -> None:
    """A factory deve devolver o tipo concreto correto para uma chave conhecida."""
    modelo = criar_modelo("fatoracao_matricial", n_usuarios=10, n_filmes=5, dimensao_embedding=4)

    assert isinstance(modelo, FatoracaoMatricial)


def test_criar_modelo_instancia_rede_neural_com_hiperparametros_extras() -> None:
    """A factory deve repassar hiperparâmetros extras (camadas ocultas, dropout) ao modelo."""
    modelo = criar_modelo(
        "rede_neural",
        n_usuarios=10,
        n_filmes=5,
        dimensao_embedding=4,
        camadas_ocultas=(8, 4),
        dropout=0.1,
    )

    assert isinstance(modelo, RedeNeural)


def test_criar_modelo_com_tipo_desconhecido_leva_erro() -> None:
    """Um `tipo_modelo` não registrado deve falhar de forma explícita."""
    with pytest.raises(ValueError, match="desconhecido"):
        criar_modelo("modelo_inexistente", n_usuarios=10, n_filmes=5, dimensao_embedding=4)


def test_fatoracao_matricial_forward_produz_uma_nota_por_par() -> None:
    """A previsão deve ter uma nota escalar para cada par (usuário, filme) do lote."""
    modelo = FatoracaoMatricial(n_usuarios=10, n_filmes=5, dimensao_embedding=4)
    usuario_idx = torch.tensor([0, 1, 2])
    filme_idx = torch.tensor([0, 1, 2])

    previsoes = modelo(usuario_idx, filme_idx)

    assert previsoes.shape == (3,)


def test_fatoracao_matricial_expoe_embeddings_de_item() -> None:
    """A fatoração matricial deve implementar a capacidade `ExpoeEmbeddingsItem`."""
    modelo = FatoracaoMatricial(n_usuarios=10, n_filmes=5, dimensao_embedding=4)

    assert isinstance(modelo, ExpoeEmbeddingsItem)
    assert modelo.embeddings_item().shape == (5, 4)


def test_rede_neural_forward_produz_uma_nota_por_par_dentro_da_faixa() -> None:
    """A previsão deve ter uma nota escalar por par, sempre dentro de `FAIXA_RATING`."""
    modelo = RedeNeural(
        n_usuarios=10, n_filmes=5, dimensao_embedding=4, camadas_ocultas=(8, 4), dropout=0.0
    )
    modelo.eval()
    usuario_idx = torch.tensor([0, 1, 2])
    filme_idx = torch.tensor([0, 1, 2])

    previsoes = modelo(usuario_idx, filme_idx)

    assert previsoes.shape == (3,)
    assert torch.all(previsoes >= FAIXA_RATING[0])
    assert torch.all(previsoes <= FAIXA_RATING[1])


def test_rede_neural_expoe_embeddings_de_item() -> None:
    """A rede neural deve implementar a capacidade `ExpoeEmbeddingsItem`."""
    modelo = RedeNeural(n_usuarios=10, n_filmes=5, dimensao_embedding=4)

    assert isinstance(modelo, ExpoeEmbeddingsItem)
    assert modelo.embeddings_item().shape == (5, 4)
