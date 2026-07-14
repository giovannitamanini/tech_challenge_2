"""Injeção de dependência da API (FastAPI `Depends`)."""

from functools import lru_cache

from tech_challenge_recomendacao.api.servico_recomendacao import (
    ServicoRecomendacao,
    carregar_servico_recomendacao,
)
from tech_challenge_recomendacao.api.servico_treino import ServicoTreino


@lru_cache
def obter_servico_recomendacao() -> ServicoRecomendacao:
    """Devolve o `ServicoRecomendacao` do processo, carregando-o na primeira chamada.

    O `lru_cache` garante que o modelo e os artefatos do pipeline sejam lidos do disco
    uma única vez por processo, não a cada requisição. Em teste, esta função é
    substituída via `app.dependency_overrides` por um serviço construído em memória.

    Returns:
        Instância única (por processo) de `ServicoRecomendacao`.
    """
    return carregar_servico_recomendacao()


@lru_cache
def obter_servico_treino() -> ServicoTreino:
    """Devolve o `ServicoTreino` do processo, criando-o na primeira chamada.

    O `lru_cache` garante que o estado da execução em andamento (o subprocesso do
    `dvc repro`) seja compartilhado entre requisições do mesmo processo — sem isso, cada
    requisição criaria um `ServicoTreino` novo e "esqueceria" a execução anterior.

    Returns:
        Instância única (por processo) de `ServicoTreino`.
    """
    return ServicoTreino()
