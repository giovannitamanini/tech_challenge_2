"""Testes de `ServicoRecomendacao`, isolados de FastAPI e do disco (fixtures em `conftest.py`)."""

import pytest

from tech_challenge_recomendacao.api.erros import (
    FilmeNaoEncontradoErro,
    RecursoNaoSuportadoErro,
    UsuarioNaoEncontradoErro,
)
from tech_challenge_recomendacao.api.servico_recomendacao import ServicoRecomendacao


def test_prever_lote_devolve_uma_nota_por_par(servico_recomendacao: ServicoRecomendacao) -> None:
    """Um lote de pares conhecidos deve devolver uma nota prevista para cada par."""
    notas = servico_recomendacao.prever_lote([(10, 100), (20, 200)])

    assert len(notas) == 2
    assert all(isinstance(nota, float) for nota in notas)


def test_prever_lote_com_usuario_desconhecido_leva_erro(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `usuario_id` fora do mapeamento deve levantar `UsuarioNaoEncontradoErro`."""
    with pytest.raises(UsuarioNaoEncontradoErro):
        servico_recomendacao.prever_lote([(999, 100)])


def test_prever_lote_com_filme_desconhecido_leva_erro(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `filme_id` fora do mapeamento deve levantar `FilmeNaoEncontradoErro`."""
    with pytest.raises(FilmeNaoEncontradoErro):
        servico_recomendacao.prever_lote([(10, 999)])


def test_recomendar_exclui_filmes_ja_avaliados_no_treino(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """O filme já visto pelo usuário (filme_id 100) não deve aparecer nas recomendações."""
    recomendacoes = servico_recomendacao.recomendar(usuario_id=10, k=10)

    filme_ids = {filme_id for filme_id, _, _ in recomendacoes}
    assert filme_ids == {200, 300, 400}


def test_recomendar_traz_o_titulo_do_catalogo(servico_recomendacao: ServicoRecomendacao) -> None:
    """Filme com título no catálogo deve vir com `titulo` preenchido; sem, com `None`."""
    recomendacoes = servico_recomendacao.recomendar(usuario_id=10, k=10)

    titulos_por_filme = {filme_id: titulo for filme_id, titulo, _ in recomendacoes}
    assert titulos_por_filme[200] == "Filme B"
    assert titulos_por_filme[400] is None  # 400 não está no catálogo sintético


def test_recomendar_respeita_k(servico_recomendacao: ServicoRecomendacao) -> None:
    """Deve devolver no máximo `k` recomendações."""
    recomendacoes = servico_recomendacao.recomendar(usuario_id=20, k=2)

    assert len(recomendacoes) == 2


def test_recomendar_com_usuario_desconhecido_leva_erro(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `usuario_id` fora do mapeamento deve levantar `UsuarioNaoEncontradoErro`."""
    with pytest.raises(UsuarioNaoEncontradoErro):
        servico_recomendacao.recomendar(usuario_id=999, k=5)


def test_filmes_similares_exclui_o_proprio_filme(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """O filme de referência não deve aparecer na própria lista de similares."""
    similares = servico_recomendacao.filmes_similares(filme_id=100, k=10)

    filme_ids = {filme_id for filme_id, _, _ in similares}
    assert filme_ids == {200, 300, 400}


def test_filmes_similares_com_filme_desconhecido_leva_erro(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `filme_id` fora do mapeamento deve levantar `FilmeNaoEncontradoErro`."""
    with pytest.raises(FilmeNaoEncontradoErro):
        servico_recomendacao.filmes_similares(filme_id=999, k=5)


def test_filmes_similares_com_modelo_sem_suporte_leva_erro(
    servico_sem_embeddings_item: ServicoRecomendacao,
) -> None:
    """Um modelo sem `ExpoeEmbeddingsItem` deve levantar `RecursoNaoSuportadoErro`."""
    with pytest.raises(RecursoNaoSuportadoErro):
        servico_sem_embeddings_item.filmes_similares(filme_id=100, k=5)


def test_info_modelo_traz_os_metadados_do_servico(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """`info_modelo` deve refletir os metadados passados na construção do serviço."""
    info = servico_recomendacao.info_modelo()

    assert info["tipo_modelo"] == "fatoracao_matricial"
    assert info["n_usuarios"] == 3
    assert info["n_filmes"] == 4
    assert info["metricas_avaliacao"] == {"rmse": 0.9, "mae": 0.7}
