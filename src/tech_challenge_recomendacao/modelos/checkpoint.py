"""Serialização de checkpoints: state_dict do modelo + metadados para reconstruí-lo."""

from pathlib import Path
from typing import TypedDict

import torch

from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.fabrica import criar_modelo


class MetadadosCheckpoint(TypedDict):
    """Metadados necessários para reconstruir a arquitetura antes de carregar os pesos."""

    tipo_modelo: str
    n_usuarios: int
    n_filmes: int
    dimensao_embedding: int


def salvar_checkpoint(
    modelo: ModeloRecomendador, metadados: MetadadosCheckpoint, caminho: Path
) -> None:
    """Salva os pesos do modelo e os metadados necessários para recarregá-lo.

    Args:
        modelo: Modelo treinado.
        metadados: Metadados de arquitetura (tipo, dimensões).
        caminho: Caminho do arquivo de checkpoint (`.pt`).
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": modelo.state_dict(), "metadados": dict(metadados)}, caminho)


def carregar_checkpoint(caminho: Path) -> ModeloRecomendador:
    """Reconstrói e carrega um modelo salvo por `salvar_checkpoint`.

    Args:
        caminho: Caminho do arquivo de checkpoint (`.pt`).

    Returns:
        Modelo em modo de avaliação (`eval()`), pronto para inferência.
    """
    checkpoint = torch.load(caminho, weights_only=False)
    metadados: MetadadosCheckpoint = checkpoint["metadados"]
    modelo = criar_modelo(
        metadados["tipo_modelo"],
        metadados["n_usuarios"],
        metadados["n_filmes"],
        metadados["dimensao_embedding"],
    )
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    return modelo
