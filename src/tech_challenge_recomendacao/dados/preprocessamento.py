"""Stage `preprocess` do pipeline DVC: limpeza das avaliações brutas do MovieLens."""

import sys
from pathlib import Path

import pandas as pd

from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.parametros import ParametrosPreprocessamento, carregar_parametros

sys.stdout.reconfigure(encoding="utf-8")

COLUNAS_AVALIACOES = ["userId", "movieId", "rating"]


def carregar_avaliacoes_brutas(caminho_ratings: Path) -> pd.DataFrame:
    """Lê `ratings.csv`, descartando a coluna de timestamp (não usada no modelo).

    Args:
        caminho_ratings: Caminho do arquivo `ratings.csv` bruto.

    Returns:
        DataFrame com as colunas `userId`, `movieId` e `rating`.
    """
    tipos = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    return pd.read_csv(caminho_ratings, usecols=COLUNAS_AVALIACOES, dtype=tipos)


def aplicar_amostragem(
    avaliacoes: pd.DataFrame, tamanho_amostra: int | None, semente: int
) -> pd.DataFrame:
    """Reduz o dataset a uma amostra aleatória reprodutível, se configurado.

    Args:
        avaliacoes: Avaliações completas.
        tamanho_amostra: Nº de linhas desejado, ou `None` para manter todas.
        semente: Semente aleatória para reprodutibilidade da amostragem.

    Returns:
        Avaliações amostradas (ou as originais, se `tamanho_amostra` for `None`).
    """
    if tamanho_amostra is None or tamanho_amostra >= len(avaliacoes):
        return avaliacoes
    return avaliacoes.sample(n=tamanho_amostra, random_state=semente)


def filtrar_por_frequencia_minima(
    avaliacoes: pd.DataFrame, min_por_usuario: int, min_por_filme: int
) -> pd.DataFrame:
    """Remove usuários e filmes com poucas avaliações (filtro em uma única passada).

    Args:
        avaliacoes: Avaliações de entrada.
        min_por_usuario: Nº mínimo de avaliações por usuário.
        min_por_filme: Nº mínimo de avaliações por filme.

    Returns:
        Avaliações filtradas. Como o filtro é aplicado uma única vez (não
        iterativamente), alguns usuários/filmes podem ficar levemente abaixo
        do mínimo após a filtragem cruzada — aceitável para este pipeline.
    """
    contagem_usuarios = avaliacoes["userId"].value_counts()
    contagem_filmes = avaliacoes["movieId"].value_counts()
    usuarios_validos = contagem_usuarios[contagem_usuarios >= min_por_usuario].index
    filmes_validos = contagem_filmes[contagem_filmes >= min_por_filme].index
    filtro = avaliacoes["userId"].isin(usuarios_validos) & avaliacoes["movieId"].isin(
        filmes_validos
    )
    return avaliacoes.loc[filtro]


def executar_preprocessamento(parametros: ParametrosPreprocessamento) -> pd.DataFrame:
    """Orquestra a leitura, amostragem, deduplicação e filtragem das avaliações.

    Args:
        parametros: Parâmetros do stage `preprocess` (`configs/params.yaml`).

    Returns:
        Avaliações limpas, prontas para a engenharia de features.
    """
    caminho_ratings = Path(configuracoes.diretorio_dados_brutos) / "ratings.csv"
    avaliacoes = carregar_avaliacoes_brutas(caminho_ratings)
    avaliacoes = aplicar_amostragem(
        avaliacoes, parametros.tamanho_amostra, configuracoes.semente_aleatoria
    )
    avaliacoes = avaliacoes.drop_duplicates(subset=["userId", "movieId"], keep="last")
    return filtrar_por_frequencia_minima(
        avaliacoes, parametros.min_avaliacoes_por_usuario, parametros.min_avaliacoes_por_filme
    )


def main() -> None:
    """Executa o stage `preprocess` e salva o resultado em `data/processed_data/`."""
    parametros = carregar_parametros().preprocessamento
    avaliacoes_limpas = executar_preprocessamento(parametros)

    diretorio_saida = Path(configuracoes.diretorio_dados_processados)
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    caminho_saida = diretorio_saida / "interacoes.parquet"
    avaliacoes_limpas.to_parquet(caminho_saida, index=False)

    print(
        f"[preprocess] {len(avaliacoes_limpas)} avaliações salvas em '{caminho_saida}' "
        f"({avaliacoes_limpas['userId'].nunique()} usuários, "
        f"{avaliacoes_limpas['movieId'].nunique()} filmes)."
    )


if __name__ == "__main__":
    main()
