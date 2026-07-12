"""Testes da factory de modelos e do modelo de fatoração matricial."""

import pytest
import torch

from tech_challenge_recomendacao.modelos.capacidades import ExpoeEmbeddingsItem
from tech_challenge_recomendacao.modelos.fabrica import criar_modelo
from tech_challenge_recomendacao.modelos.fatoracao_matricial import FatoracaoMatricial


def test_criar_modelo_instancia_fatoracao_matricial() -> None:
    """A factory deve devolver o tipo concreto correto para uma chave conhecida."""
    modelo = criar_modelo("fatoracao_matricial", n_usuarios=10, n_filmes=5, dimensao_embedding=4)

    assert isinstance(modelo, FatoracaoMatricial)


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
