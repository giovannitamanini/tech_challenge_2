"""Factory da aplicação FastAPI: registra as rotas e o tratamento centralizado de erros.

`app`, definido no fim deste módulo, é o ponto de entrada do servidor ASGI:
`uv run uvicorn tech_challenge_recomendacao.api.aplicacao:app --reload`.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tech_challenge_recomendacao.api.erros import (
    ExecucaoTreinoNaoEncontradaErro,
    FilmeNaoEncontradoErro,
    RecursoNaoSuportadoErro,
    TreinoJaEmAndamentoErro,
    UsuarioNaoEncontradoErro,
)
from tech_challenge_recomendacao.api.rotas import roteador

TITULO = "API de Recomendação — Tech Challenge"


def criar_aplicacao() -> FastAPI:
    """Monta a aplicação FastAPI: rotas + tradução de erros de domínio para HTTP.

    Returns:
        Aplicação FastAPI pronta para servir (`uvicorn ...:app`) ou testar (`TestClient`).
    """
    app = FastAPI(title=TITULO)
    app.include_router(roteador)
    _registrar_tratadores_de_erro(app)
    return app


def _registrar_tratadores_de_erro(app: FastAPI) -> None:
    """Traduz cada erro de domínio para uma resposta HTTP, num único lugar.

    Mantém as rotas (`api/rotas.py`) livres de `try/except`: um novo erro de domínio só
    exige um novo `exception_handler` aqui, não tocar nas rotas existentes.
    """

    @app.exception_handler(UsuarioNaoEncontradoErro)
    @app.exception_handler(FilmeNaoEncontradoErro)
    @app.exception_handler(ExecucaoTreinoNaoEncontradaErro)
    async def _tratar_recurso_nao_encontrado(request: Request, erro: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detalhe": str(erro)})

    @app.exception_handler(RecursoNaoSuportadoErro)
    async def _tratar_recurso_nao_suportado(request: Request, erro: Exception) -> JSONResponse:
        return JSONResponse(status_code=501, content={"detalhe": str(erro)})

    @app.exception_handler(TreinoJaEmAndamentoErro)
    async def _tratar_treino_em_andamento(request: Request, erro: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detalhe": str(erro)})


app = criar_aplicacao()
