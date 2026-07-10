"""Testes de `parametros.carregar_parametros`."""

from pathlib import Path

from tech_challenge_recomendacao.parametros import carregar_parametros


def test_carrega_parametros_reais_do_pipeline() -> None:
    """O `configs/params.yaml` real do projeto deve validar sem erros."""
    parametros = carregar_parametros(Path("configs/params.yaml"))

    assert parametros.preprocessamento.min_avaliacoes_por_usuario > 0
    assert 0 < parametros.engenharia_features.proporcao_teste < 1
    assert parametros.treino.tipo_modelo == "fatoracao_matricial"


def test_tamanho_amostra_aceita_nulo(tmp_path: Path) -> None:
    """`tamanho_amostra: null` deve ser aceito (usa o dataset completo)."""
    caminho = tmp_path / "params.yaml"
    caminho.write_text(
        """
preprocessamento:
  tamanho_amostra: null
  min_avaliacoes_por_usuario: 1
  min_avaliacoes_por_filme: 1
engenharia_features:
  proporcao_teste: 0.2
treino:
  tipo_modelo: fatoracao_matricial
  dimensao_embedding: 8
  taxa_aprendizado: 0.01
  epocas: 1
  tamanho_lote: 32
""",
        encoding="utf-8",
    )

    parametros = carregar_parametros(caminho)

    assert parametros.preprocessamento.tamanho_amostra is None
