"""Stage `feature_eng` do pipeline DVC: encoding de ids e split temporal por usuário."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.dados.catalogo_filmes import (
    extrair_titulos,
    salvar_catalogo_filmes,
)
from tech_challenge_recomendacao.dados.mapeamento_ids import extrair_mapeamento, salvar_mapeamento
from tech_challenge_recomendacao.parametros import carregar_parametros

sys.stdout.reconfigure(encoding="utf-8")

COLUNAS_PROCESSADAS = ["usuario_idx", "filme_idx", "rating"]


def ajustar_codificadores(avaliacoes: pd.DataFrame) -> tuple[LabelEncoder, LabelEncoder]:
    """Ajusta um `LabelEncoder` de usuários e outro de filmes a partir das avaliações.

    Args:
        avaliacoes: Avaliações limpas, saída do stage `preprocess`.

    Returns:
        Tupla `(codificador_usuarios, codificador_filmes)`, já ajustados (`fit`).
    """
    codificador_usuarios = LabelEncoder().fit(avaliacoes["userId"])
    codificador_filmes = LabelEncoder().fit(avaliacoes["movieId"])
    return codificador_usuarios, codificador_filmes


def codificar_ids(
    avaliacoes: pd.DataFrame,
    codificador_usuarios: LabelEncoder,
    codificador_filmes: LabelEncoder,
) -> pd.DataFrame:
    """Converte `userId`/`movieId` em índices contíguos 0-based (para embeddings).

    Args:
        avaliacoes: Avaliações limpas, saída do stage `preprocess`.
        codificador_usuarios: `LabelEncoder` de usuários já ajustado.
        codificador_filmes: `LabelEncoder` de filmes já ajustado.

    Returns:
        DataFrame com as colunas `usuario_idx`, `filme_idx`, `rating` e `timestamp`.
    """
    return pd.DataFrame(
        {
            "usuario_idx": codificador_usuarios.transform(avaliacoes["userId"]),
            "filme_idx": codificador_filmes.transform(avaliacoes["movieId"]),
            "rating": avaliacoes["rating"].to_numpy(),
            "timestamp": avaliacoes["timestamp"].to_numpy(),
        }
    )


def dividir_temporal_por_usuario(
    avaliacoes_codificadas: pd.DataFrame, fracao_validacao: float, fracao_teste: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide as interações em treino/validação/teste, por ordem temporal de cada usuário.

    Para cada usuário, as avaliações são ordenadas por `timestamp`; as últimas
    `fracao_teste` viram teste, a fração anterior a essas (`fracao_validacao`) vira
    validação, e o restante (o passado de cada usuário) vira treino. Isso simula prever o
    futuro de cada usuário e garante que todo usuário apareça no treino.

    Args:
        avaliacoes_codificadas: Avaliações com `usuario_idx`/`filme_idx`/`timestamp`.
        fracao_validacao: Fração (por usuário) reservada para validação.
        fracao_teste: Fração (por usuário) reservada para teste.

    Returns:
        Tupla `(treino, validacao, teste)`.
    """
    ordenadas = avaliacoes_codificadas.sort_values(["usuario_idx", "timestamp"])
    treino, validacao, teste = [], [], []
    for _, grupo in ordenadas.groupby("usuario_idx", sort=False):
        n = len(grupo)
        n_teste = max(1, round(n * fracao_teste))
        n_validacao = max(1, round(n * fracao_validacao))
        teste.append(grupo.iloc[n - n_teste :])
        validacao.append(grupo.iloc[n - n_teste - n_validacao : n - n_teste])
        treino.append(grupo.iloc[: n - n_teste - n_validacao])
    return pd.concat(treino), pd.concat(validacao), pd.concat(teste)


def remover_itens_nao_vistos_no_treino(
    treino: pd.DataFrame, avaliacoes: pd.DataFrame
) -> pd.DataFrame:
    """Remove de `avaliacoes` os filmes que não aparecem no treino (evita vazamento).

    O modelo não pode ter embedding para um filme que nunca viu no treino; sem esse
    filtro, validação/teste conteriam filmes cujo índice o modelo jamais aprendeu.

    Args:
        treino: Split de treino já separado.
        avaliacoes: Split de validação ou teste a filtrar.

    Returns:
        `avaliacoes` restrito aos filmes vistos no treino.
    """
    filmes_vistos = set(treino["filme_idx"].unique())
    return avaliacoes[avaliacoes["filme_idx"].isin(filmes_vistos)]


def main() -> None:
    """Executa o stage `feature_eng` e salva treino/validação/teste/metadados."""
    parametros = carregar_parametros().engenharia_features
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)

    avaliacoes = pd.read_parquet(diretorio_dados / "interacoes.parquet")
    codificador_usuarios, codificador_filmes = ajustar_codificadores(avaliacoes)
    avaliacoes_codificadas = codificar_ids(avaliacoes, codificador_usuarios, codificador_filmes)

    treino, validacao, teste = dividir_temporal_por_usuario(
        avaliacoes_codificadas, parametros.fracao_validacao, parametros.fracao_teste
    )
    validacao = remover_itens_nao_vistos_no_treino(treino, validacao)
    teste = remover_itens_nao_vistos_no_treino(treino, teste)

    treino[COLUNAS_PROCESSADAS].to_parquet(diretorio_dados / "treino.parquet", index=False)
    validacao[COLUNAS_PROCESSADAS].to_parquet(diretorio_dados / "validacao.parquet", index=False)
    teste[COLUNAS_PROCESSADAS].to_parquet(diretorio_dados / "teste.parquet", index=False)

    metadados = {
        "n_usuarios": int(avaliacoes_codificadas["usuario_idx"].nunique()),
        "n_filmes": int(avaliacoes_codificadas["filme_idx"].nunique()),
    }
    (diretorio_dados / "metadados_features.json").write_text(
        json.dumps(metadados, indent=2), encoding="utf-8"
    )
    salvar_mapeamento(
        extrair_mapeamento(codificador_usuarios, codificador_filmes),
        diretorio_dados / "mapeamento_ids.json",
    )
    caminho_movies = Path(configuracoes.diretorio_dados_brutos) / "movies.csv"
    caminho_catalogo = diretorio_dados / "catalogo_filmes.json"
    salvar_catalogo_filmes(extrair_titulos(caminho_movies), caminho_catalogo)

    print(
        f"[feature_eng] treino={len(treino)} validação={len(validacao)} teste={len(teste)} "
        f"usuários={metadados['n_usuarios']} filmes={metadados['n_filmes']}."
    )


if __name__ == "__main__":
    main()
