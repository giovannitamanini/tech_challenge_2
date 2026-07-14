"""Carregamento tipado dos parâmetros do pipeline DVC (`configs/params.yaml`)."""

from pathlib import Path

import yaml
from pydantic import BaseModel

CAMINHO_PARAMETROS_PADRAO = Path("configs/params.yaml")


class ParametrosPreprocessamento(BaseModel):
    """Parâmetros do stage `preprocess`."""

    tamanho_amostra: int | None
    min_avaliacoes_por_usuario: int
    min_avaliacoes_por_filme: int


class ParametrosEngenhariaFeatures(BaseModel):
    """Parâmetros do stage `feature_eng`."""

    fracao_validacao: float
    fracao_teste: float


class ParametrosTreino(BaseModel):
    """Parâmetros do stage `train`."""

    tipo_modelo: str
    dimensao_embedding: int
    camadas_ocultas: list[int]
    dropout: float
    taxa_aprendizado: float
    decaimento_peso: float
    epocas: int
    paciencia: int
    tamanho_lote: int


class ParametrosAvaliacao(BaseModel):
    """Parâmetros do stage `evaluate`."""

    relevancia_minima: float
    top_k: int


class Parametros(BaseModel):
    """Parâmetros de todos os stages do pipeline DVC."""

    preprocessamento: ParametrosPreprocessamento
    engenharia_features: ParametrosEngenhariaFeatures
    treino: ParametrosTreino
    avaliacao: ParametrosAvaliacao


def carregar_parametros(caminho: Path = CAMINHO_PARAMETROS_PADRAO) -> Parametros:
    """Carrega e valida os parâmetros do pipeline a partir do YAML.

    Args:
        caminho: Caminho do arquivo YAML de parâmetros.

    Returns:
        Parâmetros de todos os stages, validados e tipados.
    """
    conteudo = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    return Parametros.model_validate(conteudo)
