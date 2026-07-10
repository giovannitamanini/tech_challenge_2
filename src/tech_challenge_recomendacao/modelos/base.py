"""Interface comum a todos os modelos de recomendação do projeto."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class ModeloRecomendador(nn.Module, ABC):
    """Contrato que todo modelo de recomendação (embedding-based ou MLP) deve seguir."""

    @abstractmethod
    def forward(self, usuario_idx: torch.Tensor, filme_idx: torch.Tensor) -> torch.Tensor:
        """Prevê a nota de cada par (usuário, filme).

        Args:
            usuario_idx: Índices de usuário, shape `(lote,)`.
            filme_idx: Índices de filme, shape `(lote,)`.

        Returns:
            Notas previstas, shape `(lote,)`.
        """
        raise NotImplementedError
