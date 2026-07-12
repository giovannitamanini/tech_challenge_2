"""Testes da serialização de checkpoints (`modelos/checkpoint.py`)."""

from pathlib import Path

from tech_challenge_recomendacao.modelos.checkpoint import carregar_checkpoint, salvar_checkpoint
from tech_challenge_recomendacao.modelos.fatoracao_matricial import FatoracaoMatricial

METADADOS_DE_EXEMPLO = {
    "tipo_modelo": "fatoracao_matricial",
    "n_usuarios": 3,
    "n_filmes": 4,
    "dimensao_embedding": 2,
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
