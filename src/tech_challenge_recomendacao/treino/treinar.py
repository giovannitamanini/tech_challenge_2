"""Stage `train` do pipeline DVC: treina o modelo e loga params/métricas/artefatos no MLflow."""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.checkpoint import MetadadosCheckpoint, salvar_checkpoint
from tech_challenge_recomendacao.modelos.fabrica import criar_modelo
from tech_challenge_recomendacao.parametros import ParametrosTreino, carregar_parametros

sys.stdout.reconfigure(encoding="utf-8")

NOME_EXPERIMENTO = "recomendacao-movielens"
CAMINHO_MODELO = Path("models/modelo_recomendador.pt")
CAMINHO_METRICAS = Path("data/processed_data/metricas_treino.json")


def montar_dataloader(caminho_treino: Path, tamanho_lote: int) -> DataLoader:
    """Carrega o split de treino e o envolve em um `DataLoader` com embaralhamento.

    Args:
        caminho_treino: Caminho do parquet de treino (`usuario_idx`, `filme_idx`, `rating`).
        tamanho_lote: Tamanho do lote (batch size).

    Returns:
        `DataLoader` pronto para o loop de treino.
    """
    dados = pd.read_parquet(caminho_treino)
    dataset = TensorDataset(
        torch.tensor(dados["usuario_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(dados["filme_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(dados["rating"].to_numpy(), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=tamanho_lote, shuffle=True)


def treinar_uma_epoca(
    modelo: ModeloRecomendador,
    dataloader: DataLoader,
    otimizador: Optimizer,
    funcao_perda: nn.Module,
) -> float:
    """Roda uma época de treino e retorna a perda média (MSE) dos lotes.

    Args:
        modelo: Modelo em treinamento.
        dataloader: Lotes de treino.
        otimizador: Otimizador já associado aos parâmetros do modelo.
        funcao_perda: Função de perda (MSE, para regressão da nota).

    Returns:
        Perda média (MSE) da época.
    """
    modelo.train()
    perda_acumulada, n_lotes = 0.0, 0
    for usuario_idx, filme_idx, rating in dataloader:
        otimizador.zero_grad()
        perda = funcao_perda(modelo(usuario_idx, filme_idx), rating)
        perda.backward()
        otimizador.step()
        perda_acumulada += perda.item()
        n_lotes += 1
    return perda_acumulada / n_lotes


def montar_parametros_mlflow(parametros: ParametrosTreino, metadados: dict[str, int]) -> dict:
    """Monta o dicionário de hiperparâmetros a logar no MLflow.

    Args:
        parametros: Parâmetros do stage `train`.
        metadados: Metadados de `feature_eng` (`n_usuarios`, `n_filmes`).

    Returns:
        Dicionário pronto para `mlflow.log_params`.
    """
    return {
        "tipo_modelo": parametros.tipo_modelo,
        "dimensao_embedding": parametros.dimensao_embedding,
        "taxa_aprendizado": parametros.taxa_aprendizado,
        "epocas": parametros.epocas,
        "tamanho_lote": parametros.tamanho_lote,
        "semente_aleatoria": configuracoes.semente_aleatoria,
        **metadados,
    }


def executar_treino(
    modelo: ModeloRecomendador, dataloader: DataLoader, parametros: ParametrosTreino
) -> float:
    """Roda todas as épocas de treino, logando a perda de cada uma no MLflow.

    Args:
        modelo: Modelo a treinar.
        dataloader: Lotes de treino.
        parametros: Parâmetros do stage `train` (épocas, taxa de aprendizado).

    Returns:
        Perda (MSE) da última época.
    """
    otimizador = torch.optim.Adam(modelo.parameters(), lr=parametros.taxa_aprendizado)
    funcao_perda = nn.MSELoss()
    perda_final = 0.0
    for epoca in range(1, parametros.epocas + 1):
        perda_final = treinar_uma_epoca(modelo, dataloader, otimizador, funcao_perda)
        mlflow.log_metric("perda_treino_mse", perda_final, step=epoca)
        print(f"[train] época {epoca}/{parametros.epocas} - perda MSE: {perda_final:.4f}")
    return perda_final


def carregar_metadados_features(diretorio_dados: Path) -> dict[str, int]:
    """Carrega `n_usuarios`/`n_filmes` gerados pelo stage `feature_eng`.

    Args:
        diretorio_dados: Diretório de dados processados.

    Returns:
        Metadados com `n_usuarios` e `n_filmes`.
    """
    caminho = diretorio_dados / "metadados_features.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_e_logar_checkpoint(
    modelo: ModeloRecomendador, parametros: ParametrosTreino, metadados: dict[str, int]
) -> None:
    """Salva o checkpoint local (saída do DVC) e o loga como artefato do MLflow.

    Args:
        modelo: Modelo treinado.
        parametros: Parâmetros do stage `train`.
        metadados: Metadados de `feature_eng`.
    """
    metadados_checkpoint: MetadadosCheckpoint = {
        "tipo_modelo": parametros.tipo_modelo,
        "n_usuarios": metadados["n_usuarios"],
        "n_filmes": metadados["n_filmes"],
        "dimensao_embedding": parametros.dimensao_embedding,
    }
    salvar_checkpoint(modelo, metadados_checkpoint, CAMINHO_MODELO)
    mlflow.log_artifact(str(CAMINHO_MODELO))


def registrar_run_mlflow(
    modelo: ModeloRecomendador,
    dataloader: DataLoader,
    parametros: ParametrosTreino,
    metadados: dict[str, int],
) -> float:
    """Abre um run do MLflow, treina o modelo e loga params/métricas/artefatos.

    Args:
        modelo: Modelo a treinar.
        dataloader: Lotes de treino.
        parametros: Parâmetros do stage `train`.
        metadados: Metadados de `feature_eng`.

    Returns:
        Perda (MSE) da última época.
    """
    mlflow.set_tracking_uri(configuracoes.mlflow_tracking_uri)
    mlflow.set_experiment(NOME_EXPERIMENTO)
    with mlflow.start_run():
        mlflow.log_params(montar_parametros_mlflow(parametros, metadados))
        perda_final = executar_treino(modelo, dataloader, parametros)
        salvar_e_logar_checkpoint(modelo, parametros, metadados)
    return perda_final


def main() -> None:
    """Executa o stage `train`: treina o modelo e loga params/métricas/artefatos no MLflow."""
    torch.manual_seed(configuracoes.semente_aleatoria)
    parametros = carregar_parametros().treino
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)
    metadados = carregar_metadados_features(diretorio_dados)

    dataloader = montar_dataloader(diretorio_dados / "treino.parquet", parametros.tamanho_lote)
    modelo = criar_modelo(
        parametros.tipo_modelo,
        metadados["n_usuarios"],
        metadados["n_filmes"],
        parametros.dimensao_embedding,
    )

    perda_final = registrar_run_mlflow(modelo, dataloader, parametros, metadados)

    CAMINHO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_METRICAS.write_text(
        json.dumps({"perda_treino_mse": perda_final}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
