"""Testes dos baselines Scikit-Learn (`avaliacao/baselines_sklearn.py`)."""

import numpy as np
import pandas as pd

from tech_challenge_recomendacao.avaliacao.baselines_sklearn import (
    RecomendadorGBRT,
    RecomendadorItemKNN,
    RecomendadorMediaGlobal,
    RecomendadorNMF,
    RecomendadorSVD,
    RecomendadorViesUsuarioItem,
    construir_baselines,
)

N_USUARIOS, N_FILMES = 3, 4


def _treino_de_exemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "usuario_idx": [0, 0, 1, 1, 2, 2],
            "filme_idx": [0, 1, 1, 2, 2, 3],
            "rating": [5.0, 4.0, 3.0, 2.0, 4.0, 5.0],
        }
    )


def test_recomendador_media_global_ignora_usuario_e_filme() -> None:
    """`RecomendadorMediaGlobal` deve prever sempre a mesma média, para qualquer par."""
    modelo = RecomendadorMediaGlobal(media_global=3.5)

    previsoes = modelo.prever(np.array([0, 1]), np.array([0, 2]))

    assert previsoes.tolist() == [3.5, 3.5]


def test_recomendador_vies_usuario_item_produz_reconstrucao_completa() -> None:
    """`RecomendadorViesUsuarioItem` deve gerar uma reconstrução `n_usuarios x n_filmes`."""
    treino = _treino_de_exemplo()
    modelo = RecomendadorViesUsuarioItem(media_global=3.5).treinar(treino, N_USUARIOS, N_FILMES)

    assert modelo.reconstrucao.shape == (N_USUARIOS, N_FILMES)
    previsoes = modelo.prever(np.array([0]), np.array([1]))
    assert 0.5 <= previsoes[0] <= 5.0


def test_recomendador_svd_produz_reconstrucao_completa() -> None:
    """`RecomendadorSVD` (TruncatedSVD) deve gerar uma reconstrução completa e prever notas."""
    treino = _treino_de_exemplo()
    modelo = RecomendadorSVD(media_global=3.5, n_componentes=2).treinar(
        treino, N_USUARIOS, N_FILMES
    )

    assert modelo.reconstrucao.shape == (N_USUARIOS, N_FILMES)
    assert modelo.pontuar(0).shape == (N_FILMES,)


def test_recomendador_nmf_produz_reconstrucao_nao_negativa() -> None:
    """`RecomendadorNMF` deve gerar uma reconstrução completa (NMF não produz negativos)."""
    treino = _treino_de_exemplo()
    modelo = RecomendadorNMF(n_componentes=2).treinar(treino, N_USUARIOS, N_FILMES)

    assert modelo.reconstrucao.shape == (N_USUARIOS, N_FILMES)
    assert np.all(modelo.reconstrucao >= 0)


def test_recomendador_item_knn_produz_reconstrucao_completa() -> None:
    """`RecomendadorItemKNN` deve gerar uma reconstrução completa com poucos vizinhos."""
    treino = _treino_de_exemplo()
    modelo = RecomendadorItemKNN(k=2).treinar(treino, N_USUARIOS, N_FILMES, media_global=3.5)

    assert modelo.reconstrucao.shape == (N_USUARIOS, N_FILMES)


def test_recomendador_gbrt_produz_reconstrucao_completa() -> None:
    """`RecomendadorGBRT` deve gerar uma reconstrução completa a partir de features tabulares."""
    treino = _treino_de_exemplo()
    modelo = RecomendadorGBRT().treinar(treino, N_USUARIOS, N_FILMES, media_global=3.5)

    assert modelo.reconstrucao.shape == (N_USUARIOS, N_FILMES)


def test_construir_baselines_treina_e_devolve_todos_os_baselines_esperados() -> None:
    """A factory deve treinar e devolver os 6 baselines de referência, prontos para prever."""
    rng = np.random.default_rng(42)
    # >= 32 usuários e filmes: os baselines SVD/NMF usam 32 componentes por padrão
    # (`n_componentes <= min(n_usuarios, n_filmes)` é exigido pelo Scikit-Learn).
    n_usuarios, n_filmes = 40, 50
    # Garante ao menos uma avaliação por filme, para os baselines baseados em similaridade.
    usuarios = np.concatenate(
        [rng.integers(0, n_usuarios, size=n_filmes), rng.integers(0, n_usuarios, size=200)]
    )
    filmes = np.concatenate([np.arange(n_filmes), rng.integers(0, n_filmes, size=200)])
    ratings = rng.uniform(0.5, 5.0, size=len(usuarios))
    treino = pd.DataFrame({"usuario_idx": usuarios, "filme_idx": filmes, "rating": ratings})
    media_global = float(treino["rating"].mean())

    baselines = construir_baselines(treino, n_usuarios, n_filmes, media_global)

    assert set(baselines) == {
        "GlobalMean",
        "UserItemBias",
        "SVD (sklearn)",
        "NMF (sklearn)",
        "ItemKNN (sklearn)",
        "GBRT (sklearn)",
    }
    for modelo in baselines.values():
        previsoes = modelo.prever(np.array([0, 1]), np.array([0, 1]))
        assert previsoes.shape == (2,)
