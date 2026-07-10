"""Stage `feature_eng` do pipeline DVC: encoding de ids e split treino/teste."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.parametros import ParametrosEngenhariaFeatures, carregar_parametros

sys.stdout.reconfigure(encoding="utf-8")

COLUNAS_PROCESSADAS = ["usuario_idx", "filme_idx", "rating"]


def codificar_ids(avaliacoes: pd.DataFrame) -> pd.DataFrame:
    """Converte `userId`/`movieId` em índices contíguos 0-based (para embeddings).

    Args:
        avaliacoes: Avaliações limpas, saída do stage `preprocess`.

    Returns:
        DataFrame com as colunas `usuario_idx`, `filme_idx` e `rating`.
    """
    codificador_usuarios = LabelEncoder()
    codificador_filmes = LabelEncoder()
    return pd.DataFrame(
        {
            "usuario_idx": codificador_usuarios.fit_transform(avaliacoes["userId"]),
            "filme_idx": codificador_filmes.fit_transform(avaliacoes["movieId"]),
            "rating": avaliacoes["rating"].to_numpy(),
        }
    )


def dividir_treino_teste(
    avaliacoes_codificadas: pd.DataFrame, parametros: ParametrosEngenhariaFeatures, semente: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide as interações codificadas em treino e teste.

    Args:
        avaliacoes_codificadas: Avaliações já com `usuario_idx`/`filme_idx`.
        parametros: Parâmetros do stage `feature_eng`.
        semente: Semente aleatória para reprodutibilidade do split.

    Returns:
        Tupla `(treino, teste)`.
    """
    return train_test_split(
        avaliacoes_codificadas,
        test_size=parametros.proporcao_teste,
        random_state=semente,
    )


def main() -> None:
    """Executa o stage `feature_eng` e salva treino/teste/metadados em `data/processed_data/`."""
    parametros = carregar_parametros().engenharia_features
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)

    avaliacoes = pd.read_parquet(diretorio_dados / "interacoes.parquet")
    avaliacoes_codificadas = codificar_ids(avaliacoes)
    treino, teste = dividir_treino_teste(
        avaliacoes_codificadas, parametros, configuracoes.semente_aleatoria
    )

    treino[COLUNAS_PROCESSADAS].to_parquet(diretorio_dados / "treino.parquet", index=False)
    teste[COLUNAS_PROCESSADAS].to_parquet(diretorio_dados / "teste.parquet", index=False)

    metadados = {
        "n_usuarios": int(avaliacoes_codificadas["usuario_idx"].nunique()),
        "n_filmes": int(avaliacoes_codificadas["filme_idx"].nunique()),
    }
    (diretorio_dados / "metadados_features.json").write_text(
        json.dumps(metadados, indent=2), encoding="utf-8"
    )

    print(
        f"[feature_eng] treino={len(treino)} teste={len(teste)} "
        f"usuários={metadados['n_usuarios']} filmes={metadados['n_filmes']}."
    )


if __name__ == "__main__":
    main()
