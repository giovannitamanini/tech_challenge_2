"""Stage `train` do pipeline DVC: treina o modelo (com early stopping) e loga no MLflow."""

import copy
import json
import math
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
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
NOME_MODELO_REGISTRADO = "recomendador-movielens"
CAMINHO_MODELO = Path("models/modelo_recomendador.pt")
CAMINHO_METRICAS = Path("data/processed_data/metricas_treino.json")


def montar_dataloader(caminho_parquet: Path, tamanho_lote: int, embaralhar: bool) -> DataLoader:
    """Carrega um split (`usuario_idx`, `filme_idx`, `rating`) e o envolve em um `DataLoader`.

    Args:
        caminho_parquet: Caminho do parquet do split.
        tamanho_lote: Tamanho do lote (batch size).
        embaralhar: Se `True`, embaralha os lotes a cada época (uso: treino).

    Returns:
        `DataLoader` pronto para o loop de treino/avaliação.
    """
    dados = pd.read_parquet(caminho_parquet)
    dataset = TensorDataset(
        torch.tensor(dados["usuario_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(dados["filme_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(dados["rating"].to_numpy(), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=tamanho_lote, shuffle=embaralhar)


def rodar_epoca(
    modelo: ModeloRecomendador,
    dataloader: DataLoader,
    funcao_perda: nn.Module,
    otimizador: Optimizer | None = None,
) -> float:
    """Roda uma época (treina se `otimizador` for informado) e retorna o RMSE.

    Args:
        modelo: Modelo em treino ou avaliação.
        dataloader: Lotes da época (treino ou validação).
        funcao_perda: Função de perda (MSE, para regressão da nota).
        otimizador: Otimizador já associado aos parâmetros do modelo, ou `None` para
            só avaliar (época de validação, sem gradiente).

    Returns:
        RMSE médio da época.
    """
    treinando = otimizador is not None
    modelo.train(treinando)
    erro_quadratico_acumulado, n = 0.0, 0
    with torch.set_grad_enabled(treinando):
        for usuario_idx, filme_idx, rating in dataloader:
            previsoes = modelo(usuario_idx, filme_idx)
            perda = funcao_perda(previsoes, rating)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            erro_quadratico_acumulado += perda.item() * len(rating)
            n += len(rating)
    return math.sqrt(erro_quadratico_acumulado / n)


class _EstadoEarlyStopping:
    """Rastreia a melhor época (menor RMSE de validação) e decide quando parar."""

    def __init__(self, paciencia: int) -> None:
        self._paciencia = paciencia
        self.melhor_rmse_validacao = float("inf")
        self.melhor_estado: dict[str, torch.Tensor] | None = None
        self._epocas_sem_melhora = 0

    def atualizar(self, rmse_validacao: float, modelo: ModeloRecomendador) -> bool:
        """Registra o RMSE da época e devolve `True` se o treino deve parar."""
        if rmse_validacao < self.melhor_rmse_validacao - 1e-4:
            self.melhor_rmse_validacao = rmse_validacao
            self.melhor_estado = copy.deepcopy(modelo.state_dict())
            self._epocas_sem_melhora = 0
        else:
            self._epocas_sem_melhora += 1
        return self._epocas_sem_melhora >= self._paciencia


def _rodar_epoca_de_treino(
    modelo: ModeloRecomendador,
    dataloader_treino: DataLoader,
    dataloader_validacao: DataLoader,
    otimizador: Optimizer,
    funcao_perda: nn.Module,
    epoca: int,
    epocas_max: int,
) -> float:
    """Roda uma época de treino+validação, loga no MLflow e devolve o RMSE de validação."""
    rmse_treino = rodar_epoca(modelo, dataloader_treino, funcao_perda, otimizador)
    rmse_validacao = rodar_epoca(modelo, dataloader_validacao, funcao_perda)
    mlflow.log_metric("rmse_treino", rmse_treino, step=epoca)
    mlflow.log_metric("rmse_validacao", rmse_validacao, step=epoca)
    print(
        f"[train] época {epoca}/{epocas_max} - "
        f"RMSE treino: {rmse_treino:.4f} - RMSE validação: {rmse_validacao:.4f}"
    )
    return rmse_validacao


def treinar_com_early_stopping(
    modelo: ModeloRecomendador,
    dataloader_treino: DataLoader,
    dataloader_validacao: DataLoader,
    parametros: ParametrosTreino,
) -> dict[str, float | int]:
    """Treina o modelo monitorando o RMSE de validação, com early stopping.

    Ao final, o modelo fica carregado com os pesos da melhor época (menor RMSE de
    validação), não necessariamente os da última — porta `train_model` do notebook.

    Args:
        modelo: Modelo a treinar.
        dataloader_treino: Lotes de treino, embaralhados a cada época.
        dataloader_validacao: Lotes de validação, usados para o critério de parada.
        parametros: Parâmetros do stage `train` (épocas, paciência, taxa de aprendizado).

    Returns:
        Métricas finais: `rmse_treino`, `melhor_rmse_validacao` e `epocas_executadas`.
    """
    funcao_perda = nn.MSELoss()
    otimizador = torch.optim.Adam(
        modelo.parameters(), lr=parametros.taxa_aprendizado, weight_decay=parametros.decaimento_peso
    )
    estado = _EstadoEarlyStopping(parametros.paciencia)

    for epoca in range(1, parametros.epocas + 1):
        rmse_validacao = _rodar_epoca_de_treino(
            modelo,
            dataloader_treino,
            dataloader_validacao,
            otimizador,
            funcao_perda,
            epoca,
            parametros.epocas,
        )
        if estado.atualizar(rmse_validacao, modelo):
            print(
                f"[train] early stopping na época {epoca} "
                f"(melhor RMSE validação: {estado.melhor_rmse_validacao:.4f})."
            )
            break

    modelo.load_state_dict(estado.melhor_estado)
    return {
        "rmse_treino": rodar_epoca(modelo, dataloader_treino, funcao_perda),
        "melhor_rmse_validacao": estado.melhor_rmse_validacao,
        "epocas_executadas": epoca,
    }


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
        "camadas_ocultas": parametros.camadas_ocultas,
        "dropout": parametros.dropout,
        "taxa_aprendizado": parametros.taxa_aprendizado,
        "decaimento_peso": parametros.decaimento_peso,
        "epocas_max": parametros.epocas,
        "paciencia": parametros.paciencia,
        "tamanho_lote": parametros.tamanho_lote,
        "semente_aleatoria": configuracoes.semente_aleatoria,
        **metadados,
    }


def carregar_metadados_features(diretorio_dados: Path) -> dict[str, int]:
    """Carrega `n_usuarios`/`n_filmes` gerados pelo stage `feature_eng`.

    Args:
        diretorio_dados: Diretório de dados processados.

    Returns:
        Metadados com `n_usuarios` e `n_filmes`.
    """
    caminho = diretorio_dados / "metadados_features.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


def hiperparametros_extra_do_modelo(parametros: ParametrosTreino) -> dict[str, object]:
    """Monta os hiperparâmetros específicos do `tipo_modelo`, para a Factory e o checkpoint.

    Args:
        parametros: Parâmetros do stage `train`.

    Returns:
        Dicionário de hiperparâmetros extras (vazio para modelos sem parâmetros próprios).
    """
    if parametros.tipo_modelo == "rede_neural":
        return {"camadas_ocultas": tuple(parametros.camadas_ocultas), "dropout": parametros.dropout}
    return {}


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
        "hiperparametros_extra": hiperparametros_extra_do_modelo(parametros),
    }
    salvar_checkpoint(modelo, metadados_checkpoint, CAMINHO_MODELO)
    mlflow.log_artifact(str(CAMINHO_MODELO))


def registrar_e_mover_para_staging(modelo: ModeloRecomendador) -> str:
    """Registra o modelo treinado no MLflow Model Registry e o move para o estágio Staging.

    A promoção Staging → Production é decidida depois, pelo stage `evaluate`
    (`avaliacao/avaliar.py`), que compara o RMSE desta versão com os baselines.

    Args:
        modelo: Modelo já treinado (pesos da melhor época).

    Returns:
        Número da versão recém-criada no Model Registry.
    """
    info_modelo = mlflow.pytorch.log_model(
        modelo,
        name="modelo_pytorch",
        registered_model_name=NOME_MODELO_REGISTRADO,
        serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
    )
    versao = info_modelo.registered_model_version
    MlflowClient().transition_model_version_stage(
        name=NOME_MODELO_REGISTRADO, version=versao, stage="Staging"
    )
    return versao


def registrar_run_mlflow(
    modelo: ModeloRecomendador,
    dataloader_treino: DataLoader,
    dataloader_validacao: DataLoader,
    parametros: ParametrosTreino,
    metadados: dict[str, int],
) -> dict[str, float | int]:
    """Abre um run do MLflow, treina o modelo (com early stopping) e loga tudo.

    Args:
        modelo: Modelo a treinar.
        dataloader_treino: Lotes de treino.
        dataloader_validacao: Lotes de validação.
        parametros: Parâmetros do stage `train`.
        metadados: Metadados de `feature_eng`.

    Returns:
        Métricas finais do treino (`rmse_treino`, `melhor_rmse_validacao`, `epocas_executadas`).
    """
    mlflow.set_tracking_uri(configuracoes.mlflow_tracking_uri)
    mlflow.set_experiment(NOME_EXPERIMENTO)
    with mlflow.start_run():
        mlflow.log_params(montar_parametros_mlflow(parametros, metadados))
        metricas = treinar_com_early_stopping(
            modelo, dataloader_treino, dataloader_validacao, parametros
        )
        salvar_e_logar_checkpoint(modelo, parametros, metadados)
        registrar_e_mover_para_staging(modelo)
    return metricas


def main() -> None:
    """Executa o stage `train`: treina o modelo (com early stopping) e loga no MLflow."""
    torch.manual_seed(configuracoes.semente_aleatoria)
    parametros = carregar_parametros().treino
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)
    metadados = carregar_metadados_features(diretorio_dados)

    dataloader_treino = montar_dataloader(
        diretorio_dados / "treino.parquet", parametros.tamanho_lote, embaralhar=True
    )
    dataloader_validacao = montar_dataloader(
        diretorio_dados / "validacao.parquet", parametros.tamanho_lote, embaralhar=False
    )
    modelo = criar_modelo(
        parametros.tipo_modelo,
        metadados["n_usuarios"],
        metadados["n_filmes"],
        parametros.dimensao_embedding,
        **hiperparametros_extra_do_modelo(parametros),
    )

    metricas = registrar_run_mlflow(
        modelo, dataloader_treino, dataloader_validacao, parametros, metadados
    )

    CAMINHO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_METRICAS.write_text(json.dumps(metricas, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
