"""Testes da persistência do mapeamento de ids (`dados/mapeamento_ids.py`)."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tech_challenge_recomendacao.dados.mapeamento_ids import (
    carregar_mapeamento,
    extrair_mapeamento,
    salvar_mapeamento,
)


def _codificadores_de_exemplo() -> tuple[LabelEncoder, LabelEncoder]:
    codificador_usuarios = LabelEncoder().fit(pd.Series([100, 200, 300]))
    codificador_filmes = LabelEncoder().fit(pd.Series([7, 9]))
    return codificador_usuarios, codificador_filmes


def test_extrair_mapeamento_associa_id_real_ao_indice_do_codificador() -> None:
    """Cada id real deve apontar para a posição correspondente em `classes_`."""
    codificador_usuarios, codificador_filmes = _codificadores_de_exemplo()

    mapeamento = extrair_mapeamento(codificador_usuarios, codificador_filmes)

    assert mapeamento["usuario_id_para_idx"] == {100: 0, 200: 1, 300: 2}
    assert mapeamento["filme_id_para_idx"] == {7: 0, 9: 1}


def test_salvar_e_carregar_mapeamento_faz_round_trip(tmp_path: Path) -> None:
    """Salvar e recarregar deve devolver o mesmo mapeamento, com chaves `int`."""
    codificador_usuarios, codificador_filmes = _codificadores_de_exemplo()
    mapeamento = extrair_mapeamento(codificador_usuarios, codificador_filmes)
    caminho = tmp_path / "mapeamento_ids.json"

    salvar_mapeamento(mapeamento, caminho)
    recarregado = carregar_mapeamento(caminho)

    assert recarregado == mapeamento
    assert all(isinstance(k, int) for k in recarregado["usuario_id_para_idx"])
