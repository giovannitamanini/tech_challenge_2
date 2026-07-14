"""Rede neural embedding-based (MLP + vieses) para previsão de notas.

Porta fiel do `NeuralRecommender` do notebook de referência
(`models/recomendacao_movielens.ipynb`): combina fatoração (embeddings) com a
não-linearidade de um MLP, no estilo *Neural Collaborative Filtering*.
"""

import torch
from torch import nn

from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.capacidades import ExpoeEmbeddingsItem

FAIXA_RATING = (0.5, 5.0)


class RedeNeural(ModeloRecomendador, ExpoeEmbeddingsItem):
    """Prevê a nota via embeddings de usuário/filme concatenados, passados por um MLP.

    Para cada par (usuário, filme): os embeddings são concatenados e passam por um MLP
    (camadas ocultas + ReLU + dropout); soma-se o viés de usuário e de filme; uma
    sigmoide reescala a saída para `FAIXA_RATING`, garantindo que o modelo só produza
    notas válidas.

    Attributes:
        embedding_usuario: Embedding latente de cada usuário.
        embedding_filme: Embedding latente de cada filme.
        vies_usuario: Viés escalar por usuário.
        vies_filme: Viés escalar por filme.
        mlp: Rede totalmente conectada sobre os embeddings concatenados.
    """

    def __init__(
        self,
        n_usuarios: int,
        n_filmes: int,
        dimensao_embedding: int,
        camadas_ocultas: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        """Inicializa as tabelas de embedding, os vieses e o MLP.

        Args:
            n_usuarios: Nº de usuários distintos (tamanho da tabela de embedding).
            n_filmes: Nº de filmes distintos (tamanho da tabela de embedding).
            dimensao_embedding: Dimensão do espaço latente de usuários e filmes.
            camadas_ocultas: Nº de neurônios de cada camada oculta do MLP.
            dropout: Probabilidade de dropout aplicada após cada camada oculta.
        """
        super().__init__()
        self.embedding_usuario = nn.Embedding(n_usuarios, dimensao_embedding)
        self.embedding_filme = nn.Embedding(n_filmes, dimensao_embedding)
        self.vies_usuario = nn.Embedding(n_usuarios, 1)
        self.vies_filme = nn.Embedding(n_filmes, 1)
        self.mlp = self._construir_mlp(dimensao_embedding, camadas_ocultas, dropout)
        self._inicializar_pesos()

    def _construir_mlp(
        self, dimensao_embedding: int, camadas_ocultas: tuple[int, ...], dropout: float
    ) -> nn.Sequential:
        """Monta o MLP: `Linear -> ReLU -> Dropout` por camada oculta, mais a saída escalar."""
        camadas: list[nn.Module] = []
        dimensao_entrada = dimensao_embedding * 2
        for dimensao_saida in camadas_ocultas:
            camadas += [nn.Linear(dimensao_entrada, dimensao_saida), nn.ReLU(), nn.Dropout(dropout)]
            dimensao_entrada = dimensao_saida
        camadas.append(nn.Linear(dimensao_entrada, 1))
        return nn.Sequential(*camadas)

    def _inicializar_pesos(self) -> None:
        """Inicializa embeddings com desvio padrão pequeno e vieses em zero."""
        for embedding in (self.embedding_usuario, self.embedding_filme):
            nn.init.normal_(embedding.weight, std=0.05)
        nn.init.zeros_(self.vies_usuario.weight)
        nn.init.zeros_(self.vies_filme.weight)

    def forward(self, usuario_idx: torch.Tensor, filme_idx: torch.Tensor) -> torch.Tensor:
        """Prevê a nota de cada par (usuário, filme) via embeddings + MLP + vieses.

        Args:
            usuario_idx: Índices de usuário, shape `(lote,)`.
            filme_idx: Índices de filme, shape `(lote,)`.

        Returns:
            Notas previstas em `FAIXA_RATING`, shape `(lote,)`.
        """
        embeddings = torch.cat(
            [self.embedding_usuario(usuario_idx), self.embedding_filme(filme_idx)], dim=1
        )
        saida = self.mlp(embeddings).squeeze(1)
        saida = (
            saida
            + self.vies_usuario(usuario_idx).squeeze(1)
            + self.vies_filme(filme_idx).squeeze(1)
        )
        minimo, maximo = FAIXA_RATING
        return minimo + (maximo - minimo) * torch.sigmoid(saida)

    def embeddings_item(self) -> torch.Tensor:
        """Devolve a tabela de embeddings de filme (antes do MLP).

        Returns:
            Tensor de embeddings, shape `(n_filmes, dimensao_embedding)`.
        """
        return self.embedding_filme.weight
