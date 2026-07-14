"""Rotas HTTP da API de recomendação.

Cada rota só traduz entre o schema HTTP e o `ServicoRecomendacao` — erros de domínio
(usuário/filme desconhecido, capacidade não suportada) propagam para os
`exception_handler` registrados em `api/aplicacao.py`, sem `try/except` aqui.
"""

from fastapi import APIRouter, Depends

from tech_challenge_recomendacao.api.dependencias import (
    obter_servico_recomendacao,
    obter_servico_treino,
)
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
    RespostaStatusTreino,
    RespostaTreinoIniciado,
)
from tech_challenge_recomendacao.api.servico_recomendacao import ServicoRecomendacao
from tech_challenge_recomendacao.api.servico_treino import ServicoTreino

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
        RecomendacaoItem(filme_id=filme_id, titulo=titulo, nota_prevista=nota)
        for filme_id, titulo, nota in servico.recomendar(usuario_id, k)
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
        FilmeSimilarItem(filme_id=fid, titulo=titulo, similaridade=sim)
        for fid, titulo, sim in servico.filmes_similares(filme_id, k)
    ]
    return RespostaFilmesSimilares(filme_id=filme_id, similares=similares)


@roteador.post("/treino", response_model=RespostaTreinoIniciado, status_code=202)
def iniciar_treino(
    servico: ServicoTreino = Depends(obter_servico_treino),
) -> RespostaTreinoIniciado:
    """Dispara o pipeline de treino (`dvc repro`) em background, um de cada vez."""
    execucao = servico.iniciar_treino()
    return RespostaTreinoIniciado(
        execucao_id=execucao.execucao_id, status="em_execucao", iniciado_em=execucao.iniciado_em
    )


@roteador.get("/treino/status/{execucao_id}", response_model=RespostaStatusTreino)
def consultar_status_treino(
    execucao_id: str, servico: ServicoTreino = Depends(obter_servico_treino)
) -> RespostaStatusTreino:
    """Consulta o status (e as métricas, se concluída) de uma execução de treino."""
    return RespostaStatusTreino(**servico.consultar_status(execucao_id))
