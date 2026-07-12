"""Stage `evaluate` do pipeline DVC: métricas do modelo treinado no conjunto de teste."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.checkpoint import carregar_checkpoint

sys.stdout.reconfigure(encoding="utf-8")

CAMINHO_MODELO = Path("models/modelo_recomendador.pt")
CAMINHO_METRICAS = Path("data/processed_data/metricas_avaliacao.json")


def prever_teste(modelo: ModeloRecomendador, teste: pd.DataFrame) -> np.ndarray:
    """Gera as previsões do modelo para todo o conjunto de teste.

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


def calcular_metricas(reais: np.ndarray, previsoes: np.ndarray) -> dict[str, float]:
    """Calcula RMSE e MAE entre as notas reais e as previstas.

    Args:
        reais: Notas reais do conjunto de teste.
        previsoes: Notas previstas pelo modelo.

    Returns:
        Dicionário com as chaves `rmse` e `mae`.
    """
    return {
        "rmse": float(mean_squared_error(reais, previsoes) ** 0.5),
        "mae": float(mean_absolute_error(reais, previsoes)),
    }


def main() -> None:
    """Executa o stage `evaluate` e salva as métricas em `data/processed_data/`."""
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)
    teste = pd.read_parquet(diretorio_dados / "teste.parquet")

    modelo, _ = carregar_checkpoint(CAMINHO_MODELO)
    previsoes = prever_teste(modelo, teste)
    metricas = calcular_metricas(teste["rating"].to_numpy(), previsoes)

    CAMINHO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_METRICAS.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    print(f"[evaluate] RMSE={metricas['rmse']:.4f} MAE={metricas['mae']:.4f}")


if __name__ == "__main__":
    main()
