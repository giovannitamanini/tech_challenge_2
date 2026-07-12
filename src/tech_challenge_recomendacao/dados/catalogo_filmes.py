"""Persistência do catálogo de títulos de filmes (`filme_id` real -> título).

`movies.csv` (dados brutos) nunca é lido pelos estágios `preprocess`/`train` — só
`ratings.csv` importa pro treino do modelo. Esse módulo extrai só o título de cada
filme e o persiste como saída do stage `feature_eng`, para que consumidores do modelo
treinado (ex.: a API) exibam o nome do filme sem precisar ler `data/raw_data/` em tempo
de servir requisições.
"""

import json
from pathlib import Path

import pandas as pd


def extrair_titulos(caminho_movies: Path) -> dict[int, str]:
    """Extrai o mapeamento `filme_id -> título` de `movies.csv`.

    Args:
        caminho_movies: Caminho do arquivo `movies.csv` bruto.

    Returns:
        Mapa `filme_id -> título`.
    """
    filmes = pd.read_csv(caminho_movies, usecols=["movieId", "title"])
    return dict(zip(filmes["movieId"].tolist(), filmes["title"].tolist(), strict=True))


def salvar_catalogo_filmes(titulos: dict[int, str], caminho: Path) -> None:
    """Salva o catálogo de títulos como JSON.

    Args:
        titulos: Mapa `filme_id -> título` a persistir.
        caminho: Caminho do arquivo de saída (`.json`).
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(titulos, indent=2, ensure_ascii=False), encoding="utf-8")


def carregar_catalogo_filmes(caminho: Path) -> dict[int, str]:
    """Carrega o catálogo de títulos salvo por `salvar_catalogo_filmes`.

    Args:
        caminho: Caminho do arquivo de catálogo (`.json`).

    Returns:
        Mapa `filme_id -> título`, com chaves `int`.
    """
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    return {int(filme_id): titulo for filme_id, titulo in bruto.items()}
