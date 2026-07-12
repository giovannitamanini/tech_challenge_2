"""Fixtures de teste da API: um `ServicoRecomendacao` pequeno, todo em memória."""

import pytest
import torch

from tech_challenge_recomendacao.api.servico_recomendacao import ServicoRecomendacao
from tech_challenge_recomendacao.dados.mapeamento_ids import MapeamentoIds
from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.fatoracao_matricial import FatoracaoMatricial

N_USUARIOS = 3
N_FILMES = 4
DIMENSAO_EMBEDDING = 2
SEMENTE = 42


class _ModeloSemEmbeddingsItem(ModeloRecomendador):
    """Modelo mínimo que não implementa `ExpoeEmbeddingsItem`, para testar o erro 501."""

    def forward(self, usuario_idx: torch.Tensor, filme_idx: torch.Tensor) -> torch.Tensor:
        return torch.zeros(usuario_idx.shape[0])


@pytest.fixture(autouse=True)
def _sem_chamada_real_ao_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita que os testes dependam de um servidor MLflow real (`/modelo/info` é best-effort)."""
    monkeypatch.setattr(
        "tech_challenge_recomendacao.api.servico_recomendacao._obter_versao_registrada",
        lambda: None,
    )


@pytest.fixture
def mapeamento_exemplo() -> MapeamentoIds:
    """Mapeamento sintético: usuários {10, 20, 30} e filmes {100, 200, 300, 400}."""
    return {
        "usuario_id_para_idx": {10: 0, 20: 1, 30: 2},
        "filme_id_para_idx": {100: 0, 200: 1, 300: 2, 400: 3},
    }


@pytest.fixture
def servico_recomendacao(mapeamento_exemplo: MapeamentoIds) -> ServicoRecomendacao:
    """`ServicoRecomendacao` com um modelo pequeno, todo em memória (sem tocar disco)."""
    torch.manual_seed(SEMENTE)
    modelo = FatoracaoMatricial(
        n_usuarios=N_USUARIOS, n_filmes=N_FILMES, dimensao_embedding=DIMENSAO_EMBEDDING
    )
    modelo.eval()
    return ServicoRecomendacao(
        modelo=modelo,
        mapeamento=mapeamento_exemplo,
        itens_vistos_por_usuario={0: {0}},  # usuario_id 10 já avaliou o filme_id 100
        tipo_modelo="fatoracao_matricial",
        dimensao_embedding=DIMENSAO_EMBEDDING,
        n_usuarios=N_USUARIOS,
        n_filmes=N_FILMES,
        metricas_avaliacao={"rmse": 0.9, "mae": 0.7},
    )


@pytest.fixture
def servico_sem_embeddings_item(mapeamento_exemplo: MapeamentoIds) -> ServicoRecomendacao:
    """`ServicoRecomendacao` cujo modelo não implementa `ExpoeEmbeddingsItem`."""
    return ServicoRecomendacao(
        modelo=_ModeloSemEmbeddingsItem(),
        mapeamento=mapeamento_exemplo,
        itens_vistos_por_usuario={},
        tipo_modelo="modelo_sem_embeddings",
        dimensao_embedding=DIMENSAO_EMBEDDING,
        n_usuarios=N_USUARIOS,
        n_filmes=N_FILMES,
        metricas_avaliacao=None,
    )
