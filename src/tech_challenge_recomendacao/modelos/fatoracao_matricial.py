"""Modelo embedding-based de fatoração matricial para previsão de notas."""

import torch
from torch import nn

from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.capacidades import ExpoeEmbeddingsItem


class FatoracaoMatricial(ModeloRecomendador, ExpoeEmbeddingsItem):
    """Prevê a nota como o produto interno de embeddings de usuário e filme, mais vieses.

    Attributes:
        embedding_usuario: Embedding latente de cada usuário.
        embedding_filme: Embedding latente de cada filme.
        vies_usuario: Viés escalar por usuário.
        vies_filme: Viés escalar por filme.
        vies_global: Viés global, aprendido, somado a toda previsão.
    """

    def __init__(self, n_usuarios: int, n_filmes: int, dimensao_embedding: int) -> None:
        """Inicializa as tabelas de embedding e os vieses.

        Args:
            n_usuarios: Nº de usuários distintos (tamanho da tabela de embedding).
            n_filmes: Nº de filmes distintos (tamanho da tabela de embedding).
            dimensao_embedding: Dimensão do espaço latente de usuários e filmes.
        """
        super().__init__()
        self.embedding_usuario = nn.Embedding(n_usuarios, dimensao_embedding)
        self.embedding_filme = nn.Embedding(n_filmes, dimensao_embedding)
        self.vies_usuario = nn.Embedding(n_usuarios, 1)
        self.vies_filme = nn.Embedding(n_filmes, 1)
        self.vies_global = nn.Parameter(torch.zeros(1))

    def forward(self, usuario_idx: torch.Tensor, filme_idx: torch.Tensor) -> torch.Tensor:
        """Prevê a nota de cada par (usuário, filme) via produto interno + vieses.

        Args:
            usuario_idx: Índices de usuário, shape `(lote,)`.
            filme_idx: Índices de filme, shape `(lote,)`.

        Returns:
            Notas previstas, shape `(lote,)`.
        """
        produto_interno = (
            self.embedding_usuario(usuario_idx) * self.embedding_filme(filme_idx)
        ).sum(dim=1)
        vieses = self.vies_usuario(usuario_idx).squeeze(1) + self.vies_filme(filme_idx).squeeze(1)
        return produto_interno + vieses + self.vies_global

    def embeddings_item(self) -> torch.Tensor:
        """Devolve a tabela de embeddings de filme.

        Returns:
            Tensor de embeddings, shape `(n_filmes, dimensao_embedding)`.
        """
        return self.embedding_filme.weight
