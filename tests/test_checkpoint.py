"""Testes da serialização de checkpoints (`modelos/checkpoint.py`)."""

from pathlib import Path

from tech_challenge_recomendacao.modelos.checkpoint import carregar_checkpoint, salvar_checkpoint
from tech_challenge_recomendacao.modelos.fatoracao_matricial import FatoracaoMatricial
from tech_challenge_recomendacao.modelos.rede_neural import RedeNeural

METADADOS_DE_EXEMPLO = {
    "tipo_modelo": "fatoracao_matricial",
    "n_usuarios": 3,
    "n_filmes": 4,
    "dimensao_embedding": 2,
    "hiperparametros_extra": {},
}


def test_carregar_checkpoint_reconstroi_o_modelo_e_devolve_os_metadados(tmp_path: Path) -> None:
    """O modelo recarregado deve ter a mesma arquitetura do original, com os metadados salvos."""
    modelo = FatoracaoMatricial(n_usuarios=3, n_filmes=4, dimensao_embedding=2)
    caminho = tmp_path / "modelo.pt"

    salvar_checkpoint(modelo, METADADOS_DE_EXEMPLO, caminho)
    recarregado, metadados = carregar_checkpoint(caminho)

    assert isinstance(recarregado, FatoracaoMatricial)
    assert recarregado.embeddings_item().shape == modelo.embeddings_item().shape
    assert not recarregado.training  # carregado em modo eval()
    assert metadados == METADADOS_DE_EXEMPLO


def test_carregar_checkpoint_reconstroi_rede_neural_com_hiperparametros_extras(
    tmp_path: Path,
) -> None:
    """Hiperparâmetros extras (camadas ocultas, dropout) devem ser preservados no checkpoint."""
    modelo = RedeNeural(
        n_usuarios=3, n_filmes=4, dimensao_embedding=2, camadas_ocultas=(8, 4), dropout=0.1
    )
    metadados_rede_neural = {
        "tipo_modelo": "rede_neural",
        "n_usuarios": 3,
        "n_filmes": 4,
        "dimensao_embedding": 2,
        "hiperparametros_extra": {"camadas_ocultas": (8, 4), "dropout": 0.1},
    }
    caminho = tmp_path / "modelo.pt"

    salvar_checkpoint(modelo, metadados_rede_neural, caminho)
    recarregado, metadados = carregar_checkpoint(caminho)

    assert isinstance(recarregado, RedeNeural)
    assert recarregado.embeddings_item().shape == modelo.embeddings_item().shape
    assert metadados == metadados_rede_neural
