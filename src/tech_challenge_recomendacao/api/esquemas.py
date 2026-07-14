"""DTOs Pydantic de request/response da API de recomendação."""

from datetime import datetime

from pydantic import BaseModel, Field


class RespostaSaude(BaseModel):
    """Resposta de `GET /saude`."""

    status: str = "ok"


class RespostaModeloInfo(BaseModel):
    """Resposta de `GET /modelo/info`."""

    tipo_modelo: str
    dimensao_embedding: int
    n_usuarios: int
    n_filmes: int
    metricas_avaliacao: dict[str, float] | None = Field(
        default=None, description="RMSE/MAE do último stage `evaluate`, se disponível."
    )
    versao_registrada: str | None = Field(
        default=None, description="Versão/estágio no MLflow Model Registry, se houver."
    )


class ParPrevisao(BaseModel):
    """Um par (usuário, filme) a prever."""

    usuario_id: int
    filme_id: int


class RequisicaoPrevisoes(BaseModel):
    """Corpo de `POST /previsoes`: lote de pares a prever."""

    pares: list[ParPrevisao]


class PrevisaoItem(BaseModel):
    """Previsão de nota para um par (usuário, filme)."""

    usuario_id: int
    filme_id: int
    nota_prevista: float


class RespostaPrevisoes(BaseModel):
    """Resposta de `POST /previsoes`."""

    previsoes: list[PrevisaoItem]


class RecomendacaoItem(BaseModel):
    """Um filme recomendado, com a nota prevista."""

    filme_id: int
    titulo: str | None = Field(default=None, description="Título do filme, se conhecido.")
    nota_prevista: float


class RespostaRecomendacoes(BaseModel):
    """Resposta de `GET /recomendacoes/{usuario_id}`."""

    usuario_id: int
    recomendacoes: list[RecomendacaoItem]


class FilmeSimilarItem(BaseModel):
    """Um filme similar, com o grau de similaridade."""

    filme_id: int
    titulo: str | None = Field(default=None, description="Título do filme, se conhecido.")
    similaridade: float


class RespostaFilmesSimilares(BaseModel):
    """Resposta de `GET /filmes/{filme_id}/similares`."""

    filme_id: int
    similares: list[FilmeSimilarItem]


class RespostaTreinoIniciado(BaseModel):
    """Resposta de `POST /treino`."""

    execucao_id: str = Field(description="Id da execução (pid do subprocesso do `dvc repro`).")
    status: str
    iniciado_em: datetime


class RespostaStatusTreino(BaseModel):
    """Resposta de `GET /treino/status/{execucao_id}`."""

    execucao_id: str
    status: str = Field(description="`em_execucao`, `concluido` ou `falhou`.")
    iniciado_em: datetime
    codigo_saida: int | None = Field(
        default=None, description="Código de saída do subprocesso, se já tiver terminado."
    )
    metricas_treino: dict[str, float | int] | None = Field(
        default=None, description="Conteúdo de `metricas_treino.json`, se a execução concluiu."
    )
    metricas_avaliacao: dict[str, float] | None = Field(
        default=None, description="Conteúdo de `metricas_avaliacao.json`, se a execução concluiu."
    )
