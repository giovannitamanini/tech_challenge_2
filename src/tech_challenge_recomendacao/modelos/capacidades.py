"""Capacidades opcionais que um `ModeloRecomendador` pode implementar.

Separadas do contrato principal (`ModeloRecomendador`) porque nem toda arquitetura de
modelo tem uma tabela de embeddings de item — exigir isso no contrato base forçaria
implementações futuras sem essa representação latente a implementar algo sem sentido
para elas (violaria o Interface Segregation Principle). Consumidores que precisam dessa
capacidade (ex.: a API, para o endpoint de filmes similares) checam via `isinstance`.
"""

from abc import ABC, abstractmethod

import torch


class ExpoeEmbeddingsItem(ABC):
    """Capacidade opcional de modelos que mantêm uma tabela de embeddings de item."""

    @abstractmethod
    def embeddings_item(self) -> torch.Tensor:
        """Devolve a tabela de embeddings de item.

        Returns:
            Tensor de embeddings, shape `(n_filmes, dimensao_embedding)`.
        """
        raise NotImplementedError
