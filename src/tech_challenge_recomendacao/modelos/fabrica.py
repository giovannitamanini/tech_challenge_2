"""Factory (GoF) para instanciar modelos de recomendação a partir de um nome.

Centraliza a construção dos modelos para que o stage `train` (e a Etapa 4, ao
adicionar novas arquiteturas para comparação) não precise conhecer as classes
concretas — apenas o nome configurado em `configs/params.yaml` (`treino.tipo_modelo`).
"""

from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.fatoracao_matricial import FatoracaoMatricial
from tech_challenge_recomendacao.modelos.rede_neural import RedeNeural

_MODELOS_DISPONIVEIS: dict[str, type[ModeloRecomendador]] = {
    "fatoracao_matricial": FatoracaoMatricial,
    "rede_neural": RedeNeural,
}


def criar_modelo(
    tipo_modelo: str,
    n_usuarios: int,
    n_filmes: int,
    dimensao_embedding: int,
    **hiperparametros_extra: object,
) -> ModeloRecomendador:
    """Instancia um modelo de recomendação a partir do nome configurado.

    Args:
        tipo_modelo: Chave em `_MODELOS_DISPONIVEIS` (ex.: `"rede_neural"`).
        n_usuarios: Nº de usuários distintos.
        n_filmes: Nº de filmes distintos.
        dimensao_embedding: Dimensão do espaço latente de usuários e filmes.
        **hiperparametros_extra: Hiperparâmetros específicos do modelo (ex.: `camadas_ocultas`
            e `dropout` para `"rede_neural"`), repassados diretamente ao construtor.

    Returns:
        Instância do modelo correspondente, ainda não treinada.

    Raises:
        ValueError: Se `tipo_modelo` não estiver em `_MODELOS_DISPONIVEIS`.
    """
    classe_modelo = _MODELOS_DISPONIVEIS.get(tipo_modelo)
    if classe_modelo is None:
        disponiveis = ", ".join(sorted(_MODELOS_DISPONIVEIS))
        raise ValueError(f"Modelo '{tipo_modelo}' desconhecido. Disponíveis: {disponiveis}.")
    return classe_modelo(n_usuarios, n_filmes, dimensao_embedding, **hiperparametros_extra)
