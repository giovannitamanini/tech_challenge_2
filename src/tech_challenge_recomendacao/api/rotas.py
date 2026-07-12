"""Rotas HTTP da API de recomendação.

Cada rota só traduz entre o schema HTTP e o `ServicoRecomendacao` — erros de domínio
(usuário/filme desconhecido, capacidade não suportada) propagam para os
`exception_handler` registrados em `api/aplicacao.py`, sem `try/except` aqui.
"""

from fastapi import APIRouter, Depends

from tech_challenge_recomendacao.api.dependencias import obter_servico_recomendacao
from tech_challenge_recomendacao.api.esquemas import (
    FilmeSimilarItem,
    PrevisaoItem,
    RecomendacaoItem,
    RequisicaoPrevisoes,
    RespostaFilmesSimilares,
    RespostaModeloInfo,
    RespostaPrevisoes,
    RespostaRecomendacoes,
    RespostaSaude,
)
from tech_challenge_recomendacao.api.servico_recomendacao import ServicoRecomendacao

roteador = APIRouter()


@roteador.get("/saude", response_model=RespostaSaude)
def consultar_saude() -> RespostaSaude:
    """Verifica se a API está no ar."""
    return RespostaSaude()


@roteador.get("/modelo/info", response_model=RespostaModeloInfo)
def consultar_info_modelo(
    servico: ServicoRecomendacao = Depends(obter_servico_recomendacao),
) -> RespostaModeloInfo:
    """Metadados do modelo carregado e métricas do último `evaluate`."""
    return RespostaModeloInfo(**servico.info_modelo())


@roteador.post("/previsoes", response_model=RespostaPrevisoes)
def prever(
    requisicao: RequisicaoPrevisoes,
    servico: ServicoRecomendacao = Depends(obter_servico_recomendacao),
) -> RespostaPrevisoes:
    """Prevê a nota de um lote de pares (usuário, filme)."""
    pares = [(par.usuario_id, par.filme_id) for par in requisicao.pares]
    notas = servico.prever_lote(pares)
    previsoes = [
        PrevisaoItem(usuario_id=u, filme_id=f, nota_prevista=nota)
        for (u, f), nota in zip(pares, notas, strict=True)
    ]
    return RespostaPrevisoes(previsoes=previsoes)


@roteador.get("/recomendacoes/{usuario_id}", response_model=RespostaRecomendacoes)
def recomendar(
    usuario_id: int,
    k: int = 10,
    servico: ServicoRecomendacao = Depends(obter_servico_recomendacao),
) -> RespostaRecomendacoes:
    """Recomenda os `k` filmes com maior nota prevista para o usuário."""
    recomendacoes = [
        RecomendacaoItem(filme_id=filme_id, nota_prevista=nota)
        for filme_id, nota in servico.recomendar(usuario_id, k)
    ]
    return RespostaRecomendacoes(usuario_id=usuario_id, recomendacoes=recomendacoes)


@roteador.get("/filmes/{filme_id}/similares", response_model=RespostaFilmesSimilares)
def buscar_filmes_similares(
    filme_id: int,
    k: int = 10,
    servico: ServicoRecomendacao = Depends(obter_servico_recomendacao),
) -> RespostaFilmesSimilares:
    """Busca os `k` filmes mais similares a um filme, via embeddings de item."""
    similares = [
        FilmeSimilarItem(filme_id=fid, similaridade=sim)
        for fid, sim in servico.filmes_similares(filme_id, k)
    ]
    return RespostaFilmesSimilares(filme_id=filme_id, similares=similares)
