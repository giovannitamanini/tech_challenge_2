"""Baselines Scikit-Learn para comparação com a rede neural (Template Method + Factory).

Porta fiel dos baselines do notebook de referência
(`models/recomendacao_movielens.ipynb`): `RecomendadorMatricial` implementa uma única vez
`prever`/`pontuar` (Template Method); cada baseline só preenche a matriz `reconstrucao`.
`construir_baselines` (Factory) instancia e treina todos.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

FAIXA_RATING = (0.5, 5.0)


def _montar_matriz_usuario_filme(
    df: pd.DataFrame, valores: np.ndarray, n_usuarios: int, n_filmes: int
) -> csr_matrix:
    """Monta a matriz esparsa usuários×filmes a partir de `(usuario_idx, filme_idx, valores)`."""
    linhas, colunas = df["usuario_idx"].to_numpy(), df["filme_idx"].to_numpy()
    return csr_matrix((valores, (linhas, colunas)), shape=(n_usuarios, n_filmes))


class RecomendadorMatricial:
    """Template Method: subclasses só preenchem `reconstrucao` (nota usuário×filme)."""

    reconstrucao: np.ndarray

    def prever(self, usuarios: np.ndarray, filmes: np.ndarray) -> np.ndarray:
        """Prevê a nota de cada par `(usuario_idx, filme_idx)`, recortada em `FAIXA_RATING`."""
        return np.clip(self.reconstrucao[usuarios, filmes], *FAIXA_RATING).astype(np.float32)

    def pontuar(self, usuario_idx: int) -> np.ndarray:
        """Devolve o vetor de notas previstas para um usuário, contra todo o catálogo."""
        return self.reconstrucao[usuario_idx]


class RecomendadorMediaGlobal:
    """Baseline trivial (piso): prevê sempre a média global de rating do treino."""

    def __init__(self, media_global: float) -> None:
        self.media_global = media_global

    def prever(self, usuarios: np.ndarray, filmes: np.ndarray) -> np.ndarray:
        """Prevê a média global para todos os pares, ignorando usuário/filme."""
        return np.full(len(usuarios), self.media_global, dtype=np.float32)


class RecomendadorViesUsuarioItem(RecomendadorMatricial):
    """Viés clássico de RecSys: média global + viés de usuário + viés de filme."""

    def __init__(self, media_global: float, regularizacao: float = 10.0) -> None:
        self.media_global = media_global
        self.regularizacao = regularizacao

    def treinar(
        self, treino: pd.DataFrame, n_usuarios: int, n_filmes: int
    ) -> "RecomendadorViesUsuarioItem":
        """Ajusta os vieses de usuário/filme (regularizados pela contagem de avaliações)."""
        desvio = treino["rating"] - self.media_global
        por_filme = desvio.groupby(treino["filme_idx"])
        vies_filme = por_filme.sum() / (por_filme.count() + self.regularizacao)
        residuo = desvio - treino["filme_idx"].map(vies_filme)
        por_usuario = residuo.groupby(treino["usuario_idx"])
        vies_usuario = por_usuario.sum() / (por_usuario.count() + self.regularizacao)

        vu = vies_usuario.reindex(range(n_usuarios)).fillna(0.0).to_numpy()
        vf = vies_filme.reindex(range(n_filmes)).fillna(0.0).to_numpy()
        self.reconstrucao = self.media_global + vu[:, None] + vf[None, :]
        return self


class RecomendadorSVD(RecomendadorMatricial):
    """Fatoração de matriz latente via `TruncatedSVD` (Scikit-Learn) sobre resíduos."""

    def __init__(self, media_global: float, n_componentes: int = 32, semente: int = 42) -> None:
        self.media_global = media_global
        self.svd = TruncatedSVD(n_components=n_componentes, random_state=semente)

    def treinar(
        self, treino: pd.DataFrame, n_usuarios: int, n_filmes: int
    ) -> "RecomendadorSVD":
        """Ajusta o SVD sobre a matriz de resíduos (rating - média global)."""
        residuo = (treino["rating"] - self.media_global).to_numpy(np.float32)
        matriz = _montar_matriz_usuario_filme(treino, residuo, n_usuarios, n_filmes)
        embutido = self.svd.fit_transform(matriz)
        self.reconstrucao = self.media_global + embutido @ self.svd.components_
        return self


class RecomendadorNMF(RecomendadorMatricial):
    """Fatoração não-negativa (`NMF`, Scikit-Learn) sobre a matriz de ratings."""

    def __init__(self, n_componentes: int = 32, semente: int = 42) -> None:
        self.nmf = NMF(
            n_components=n_componentes, init="nndsvda", max_iter=600, random_state=semente
        )

    def treinar(self, treino: pd.DataFrame, n_usuarios: int, n_filmes: int) -> "RecomendadorNMF":
        """Ajusta o NMF diretamente sobre os ratings (não-negativos)."""
        valores = treino["rating"].to_numpy(np.float32)
        matriz = _montar_matriz_usuario_filme(treino, valores, n_usuarios, n_filmes)
        pesos = self.nmf.fit_transform(matriz)
        self.reconstrucao = pesos @ self.nmf.components_
        return self


def _similaridade_item_item(filme_usuario: csr_matrix, k: int) -> csr_matrix:
    """Similaridade cosseno esparsa filme-filme, mantendo os `k` vizinhos mais próximos."""
    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(filme_usuario)
    distancias, indices = knn.kneighbors(filme_usuario)
    n = filme_usuario.shape[0]
    linhas = np.repeat(np.arange(n), k + 1)
    sim = csr_matrix((1.0 - distancias.ravel(), (linhas, indices.ravel())), shape=(n, n))
    sim.setdiag(0.0)
    sim.eliminate_zeros()
    norma = np.abs(sim).sum(axis=1).A.ravel()
    norma[norma == 0] = 1.0
    return sim.multiply(1.0 / norma[:, None]).tocsr()


class RecomendadorItemKNN(RecomendadorMatricial):
    """Filtragem colaborativa baseada em memória: kNN item-item com similaridade cosseno."""

    def __init__(self, k: int = 40) -> None:
        self.k = k

    def treinar(
        self, treino: pd.DataFrame, n_usuarios: int, n_filmes: int, media_global: float
    ) -> "RecomendadorItemKNN":
        """Centra os ratings pela média de cada filme e reconstrói via vizinhança item-item."""
        media_filme = (
            treino.groupby("filme_idx")["rating"].mean().reindex(range(n_filmes))
            .fillna(media_global).to_numpy()
        )
        desvio = treino["rating"].to_numpy(np.float32) - media_filme[treino["filme_idx"].to_numpy()]
        centrada = _montar_matriz_usuario_filme(
            treino, desvio.astype(np.float32), n_usuarios, n_filmes
        )
        similaridade = _similaridade_item_item(centrada.T.tocsr(), self.k)
        self.reconstrucao = media_filme[None, :] + (centrada @ similaridade.T).toarray()
        return self


def _agregado_por_coluna(
    treino: pd.DataFrame, coluna: str, como: str, tamanho: int, preenchimento: float
) -> np.ndarray:
    """Agregado (`mean`/`size`) de `rating` por `coluna`, reindexado ao catálogo completo."""
    agrupado = treino.groupby(coluna)["rating"]
    serie = agrupado.size() if como == "size" else agrupado.agg(como)
    return serie.reindex(range(tamanho)).fillna(preenchimento).to_numpy(dtype=np.float32)


class RecomendadorGBRT(RecomendadorMatricial):
    """Gradient boosting (`HistGradientBoostingRegressor`) sobre features tabulares simples."""

    def __init__(self, semente: int = 42) -> None:
        self.modelo = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=semente
        )

    def treinar(
        self, treino: pd.DataFrame, n_usuarios: int, n_filmes: int, media_global: float
    ) -> "RecomendadorGBRT":
        """Ajusta o GBRT sobre features de média/contagem por usuário e por filme."""
        self._media_usuario = _agregado_por_coluna(
            treino, "usuario_idx", "mean", n_usuarios, media_global
        )
        self._contagem_usuario = _agregado_por_coluna(
            treino, "usuario_idx", "size", n_usuarios, 0.0
        )
        self._media_filme = _agregado_por_coluna(
            treino, "filme_idx", "mean", n_filmes, media_global
        )
        self._contagem_filme = _agregado_por_coluna(treino, "filme_idx", "size", n_filmes, 0.0)

        features = self._montar_features(
            treino["usuario_idx"].to_numpy(), treino["filme_idx"].to_numpy()
        )
        self.modelo.fit(features, treino["rating"].to_numpy())
        self.reconstrucao = self._reconstrucao_completa(n_usuarios, n_filmes)
        return self

    def _montar_features(self, usuarios: np.ndarray, filmes: np.ndarray) -> np.ndarray:
        """Monta a matriz de features (médias/contagens de usuário e filme) para um lote."""
        return np.column_stack(
            [
                self._media_usuario[usuarios],
                self._contagem_usuario[usuarios],
                self._media_filme[filmes],
                self._contagem_filme[filmes],
            ]
        )

    def _reconstrucao_completa(self, n_usuarios: int, n_filmes: int) -> np.ndarray:
        """Prevê a nota de todo o catálogo, para todos os usuários (matriz densa)."""
        filmes = np.arange(n_filmes)
        reconstrucao = np.empty((n_usuarios, n_filmes), dtype=np.float32)
        for usuario_idx in range(n_usuarios):
            usuarios = np.full(n_filmes, usuario_idx)
            reconstrucao[usuario_idx] = self.modelo.predict(self._montar_features(usuarios, filmes))
        return reconstrucao


def construir_baselines(
    treino: pd.DataFrame, n_usuarios: int, n_filmes: int, media_global: float, semente: int = 42
) -> dict[str, object]:
    """Factory (GoF): instancia e treina todos os baselines de referência.

    Args:
        treino: Split de treino (`usuario_idx`, `filme_idx`, `rating`).
        n_usuarios: Nº de usuários distintos.
        n_filmes: Nº de filmes distintos.
        media_global: Média de rating do treino.
        semente: Semente aleatória, para reprodutibilidade dos baselines estocásticos.

    Returns:
        Mapa `nome do baseline -> instância já treinada`.
    """
    return {
        "GlobalMean": RecomendadorMediaGlobal(media_global),
        "UserItemBias": RecomendadorViesUsuarioItem(media_global).treinar(
            treino, n_usuarios, n_filmes
        ),
        "SVD (sklearn)": RecomendadorSVD(media_global, semente=semente).treinar(
            treino, n_usuarios, n_filmes
        ),
        "NMF (sklearn)": RecomendadorNMF(semente=semente).treinar(treino, n_usuarios, n_filmes),
        "ItemKNN (sklearn)": RecomendadorItemKNN().treinar(
            treino, n_usuarios, n_filmes, media_global
        ),
        "GBRT (sklearn)": RecomendadorGBRT(semente=semente).treinar(
            treino, n_usuarios, n_filmes, media_global
        ),
    }
