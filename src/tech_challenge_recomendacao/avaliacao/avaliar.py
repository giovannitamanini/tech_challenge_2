"""Stage `evaluate` do pipeline DVC: compara o modelo treinado com baselines Scikit-Learn.

Calcula, para o modelo campeão (checkpoint treinado) e para os baselines de
`avaliacao/baselines_sklearn.py`, as mesmas 6 métricas do notebook de referência (RMSE,
MAE, Precision@k, Recall@k, NDCG@k, Coverage), e salva a tabela comparativa completa em
`models/comparacao_modelos.json`.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tech_challenge_recomendacao.avaliacao.baselines_sklearn import (
    RecomendadorMatricial,
    construir_baselines,
)
from tech_challenge_recomendacao.avaliacao.metricas_ranking import (
    avaliar_ranking,
    construir_historico_usuarios,
)
from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.checkpoint import carregar_checkpoint
from tech_challenge_recomendacao.parametros import carregar_parametros

sys.stdout.reconfigure(encoding="utf-8")

NOME_EXPERIMENTO = "recomendacao-movielens"
NOME_MODELO_REGISTRADO = "recomendador-movielens"
CAMINHO_MODELO = Path("models/modelo_recomendador.pt")
CAMINHO_METRICAS = Path("data/processed_data/metricas_avaliacao.json")
CAMINHO_COMPARACAO = Path("models/comparacao_modelos.json")

FuncaoScore = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class ContextoAvaliacao:
    """Agrupa tudo que `avaliar_modelo` precisa além das previsões/score do modelo."""

    teste: pd.DataFrame
    vistos_no_treino: dict[int, set[int]]
    relevantes_no_teste: dict[int, set[int]]
    n_filmes: int
    top_k: int


def prever_teste(modelo: ModeloRecomendador, teste: pd.DataFrame) -> np.ndarray:
    """Gera as previsões do modelo (checkpoint carregado) para todo o conjunto de teste.

    Args:
        modelo: Modelo já treinado, em modo de avaliação.
        teste: Split de teste (`usuario_idx`, `filme_idx`, `rating`).

    Returns:
        Notas previstas, como array numpy.
    """
    with torch.no_grad():
        usuario_idx = torch.tensor(teste["usuario_idx"].to_numpy(), dtype=torch.long)
        filme_idx = torch.tensor(teste["filme_idx"].to_numpy(), dtype=torch.long)
        return modelo(usuario_idx, filme_idx).numpy()


def score_modelo_pytorch(modelo: ModeloRecomendador, n_filmes: int) -> FuncaoScore:
    """Constrói a função de score (Strategy) de um `ModeloRecomendador` já treinado.

    Args:
        modelo: Modelo já treinado, em modo de avaliação.
        n_filmes: Nº total de filmes no catálogo.

    Returns:
        Função `usuario_idx -> vetor de notas para todo o catálogo`.
    """

    def _score(usuario_idx: int) -> np.ndarray:
        with torch.no_grad():
            usuarios = torch.full((n_filmes,), usuario_idx, dtype=torch.long)
            filmes = torch.arange(n_filmes, dtype=torch.long)
            return modelo(usuarios, filmes).numpy()

    return _score


def calcular_rmse_mae(reais: np.ndarray, previsoes: np.ndarray) -> dict[str, float]:
    """Calcula RMSE e MAE entre as notas reais e as previstas.

    Args:
        reais: Notas reais do conjunto de teste.
        previsoes: Notas previstas pelo modelo.

    Returns:
        Dicionário com as chaves `RMSE` e `MAE`.
    """
    return {
        "RMSE": float(mean_squared_error(reais, previsoes) ** 0.5),
        "MAE": float(mean_absolute_error(reais, previsoes)),
    }


def avaliar_modelo(
    previsoes_teste: np.ndarray, funcao_score: FuncaoScore | None, contexto: ContextoAvaliacao
) -> dict[str, float]:
    """Calcula RMSE/MAE e, se o modelo suportar ranking, também Precision/Recall/NDCG/Coverage.

    Args:
        previsoes_teste: Notas previstas para o conjunto de teste.
        funcao_score: Função de score (Strategy) para métricas de ranking, ou `None` se o
            modelo não suportar pontuar todo o catálogo (ex.: `GlobalMean`).
        contexto: Dados e parâmetros compartilhados por todas as avaliações.

    Returns:
        Dicionário com as métricas calculadas.
    """
    metricas = calcular_rmse_mae(contexto.teste["rating"].to_numpy(), previsoes_teste)
    if funcao_score is not None:
        metricas.update(
            avaliar_ranking(
                funcao_score,
                contexto.vistos_no_treino,
                contexto.relevantes_no_teste,
                contexto.n_filmes,
                contexto.top_k,
            )
        )
    return metricas


def avaliar_baselines(
    treino: pd.DataFrame,
    n_usuarios: int,
    n_filmes: int,
    media_global: float,
    contexto: ContextoAvaliacao,
) -> dict[str, dict[str, float]]:
    """Treina e avalia todos os baselines Scikit-Learn, com as mesmas métricas do modelo.

    Args:
        treino: Split de treino, usado para ajustar os baselines.
        n_usuarios: Nº de usuários distintos.
        n_filmes: Nº de filmes distintos.
        media_global: Média de rating do treino.
        contexto: Dados e parâmetros compartilhados por todas as avaliações.

    Returns:
        Mapa `nome do baseline -> métricas`.
    """
    baselines = construir_baselines(treino, n_usuarios, n_filmes, media_global)
    usuarios_teste = contexto.teste["usuario_idx"].to_numpy()
    filmes_teste = contexto.teste["filme_idx"].to_numpy()

    resultado = {}
    for nome, baseline in baselines.items():
        previsoes = baseline.prever(usuarios_teste, filmes_teste)
        funcao_score = baseline.pontuar if isinstance(baseline, RecomendadorMatricial) else None
        resultado[nome] = avaliar_modelo(previsoes, funcao_score, contexto)
    return resultado


def salvar_metricas_campeao(comparacao: dict[str, dict[str, float]], nome_modelo: str) -> None:
    """Salva `data/processed_data/metricas_avaliacao.json` (RMSE/MAE flat do modelo campeão).

    Formato mantido estável (não a tabela comparativa completa) porque é consumido pela
    API (`ServicoRecomendacao`/`RespostaModeloInfo`) como `dict[str, float]`.

    Args:
        comparacao: Tabela comparativa completa (todos os modelos).
        nome_modelo: Chave do modelo campeão (checkpoint carregado) em `comparacao`.
    """
    metricas_campeao = {
        "rmse": comparacao[nome_modelo]["RMSE"],
        "mae": comparacao[nome_modelo]["MAE"],
    }
    CAMINHO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_METRICAS.write_text(json.dumps(metricas_campeao, indent=2), encoding="utf-8")


def salvar_e_logar_comparacao(comparacao: dict[str, dict[str, float]], nome_modelo: str) -> None:
    """Salva a tabela comparativa completa em `models/` e a loga como artefato do MLflow.

    Args:
        comparacao: Tabela comparativa completa (todos os modelos).
        nome_modelo: Chave do modelo campeão, para logar suas métricas como métricas do run.
    """
    CAMINHO_COMPARACAO.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_COMPARACAO.write_text(json.dumps(comparacao, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(configuracoes.mlflow_tracking_uri)
    mlflow.set_experiment(NOME_EXPERIMENTO)
    with mlflow.start_run():
        for metrica, valor in comparacao[nome_modelo].items():
            # Nomes de métrica do MLflow não aceitam "@" (usado em "Precision@10" etc.).
            nome_metrica = metrica.replace("@", "_em_")
            mlflow.log_metric(f"teste_{nome_metrica}", valor)
        mlflow.log_artifact(str(CAMINHO_COMPARACAO))


def promover_melhor_modelo_para_producao(rmse_candidato: float) -> None:
    """Promove a versão mais recente do Model Registry (Staging) para Production.

    Compara o RMSE de teste do candidato com o da versão hoje em Production (guardado
    na tag `rmse_teste` de cada versão): só promove se o candidato empatar ou melhorar
    esse RMSE. Sem nenhuma versão em Production ainda, o candidato é promovido direto
    (bootstrap do Registry). A comparação é contra a Production atual — não contra os
    baselines de `comparacao_modelos.json`, que servem para o relatório de métricas
    (§ Etapa 4), não como critério de promoção do Registry.

    Args:
        rmse_candidato: RMSE de teste do modelo campeão nesta run do `evaluate`.
    """
    cliente = MlflowClient()
    candidatas = cliente.get_latest_versions(NOME_MODELO_REGISTRADO, stages=["Staging"])
    if not candidatas:
        print(f"[evaluate] Nenhuma versão em Staging para '{NOME_MODELO_REGISTRADO}'.")
        return
    versao_candidata = max(candidatas, key=lambda v: int(v.version))
    cliente.set_model_version_tag(
        NOME_MODELO_REGISTRADO, versao_candidata.version, "rmse_teste", str(rmse_candidato)
    )

    versoes_producao = cliente.get_latest_versions(NOME_MODELO_REGISTRADO, stages=["Production"])
    if versoes_producao:
        rmse_producao = float(versoes_producao[0].tags.get("rmse_teste", "inf"))
        if rmse_candidato > rmse_producao:
            print(
                f"[evaluate] Candidato v{versao_candidata.version} (RMSE={rmse_candidato:.4f}) "
                f"não supera a Production atual (RMSE={rmse_producao:.4f}) — mantido em Staging."
            )
            return

    cliente.transition_model_version_stage(
        name=NOME_MODELO_REGISTRADO,
        version=versao_candidata.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"[evaluate] {NOME_MODELO_REGISTRADO} v{versao_candidata.version} promovido a "
        f"Production (RMSE={rmse_candidato:.4f})."
    )


def main() -> None:
    """Executa o stage `evaluate`: compara o modelo campeão com os baselines Scikit-Learn."""
    parametros = carregar_parametros()
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)
    treino = pd.read_parquet(diretorio_dados / "treino.parquet")
    teste = pd.read_parquet(diretorio_dados / "teste.parquet")
    metadados = json.loads(
        (diretorio_dados / "metadados_features.json").read_text(encoding="utf-8")
    )
    n_usuarios, n_filmes = metadados["n_usuarios"], metadados["n_filmes"]
    media_global = float(treino["rating"].mean())

    vistos_no_treino, relevantes_no_teste = construir_historico_usuarios(
        treino, teste, parametros.avaliacao.relevancia_minima
    )
    contexto = ContextoAvaliacao(
        teste, vistos_no_treino, relevantes_no_teste, n_filmes, parametros.avaliacao.top_k
    )

    modelo, metadados_checkpoint = carregar_checkpoint(CAMINHO_MODELO)
    nome_modelo = metadados_checkpoint["tipo_modelo"]
    previsoes_modelo = prever_teste(modelo, teste)
    score_modelo = score_modelo_pytorch(modelo, n_filmes)

    comparacao = {nome_modelo: avaliar_modelo(previsoes_modelo, score_modelo, contexto)}
    comparacao.update(avaliar_baselines(treino, n_usuarios, n_filmes, media_global, contexto))

    salvar_metricas_campeao(comparacao, nome_modelo)
    salvar_e_logar_comparacao(comparacao, nome_modelo)
    promover_melhor_modelo_para_producao(comparacao[nome_modelo]["RMSE"])

    metricas_campeao = comparacao[nome_modelo]
    print(
        f"[evaluate] {nome_modelo}: RMSE={metricas_campeao['RMSE']:.4f} "
        f"MAE={metricas_campeao['MAE']:.4f}"
    )


if __name__ == "__main__":
    main()
