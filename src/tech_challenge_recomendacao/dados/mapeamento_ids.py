"""Persistência do mapeamento entre ids reais do dataset e índices internos dos embeddings.

O stage `feature_eng` converte `userId`/`movieId` em índices contíguos 0-based via
`LabelEncoder` (ver `engenharia_features.py`), necessários para indexar as tabelas de
embedding do modelo. Esse módulo salva e recarrega esse mapeamento, para que qualquer
consumidor externo do modelo treinado (ex.: a API) traduza ids reais em índices sem
precisar reajustar o `LabelEncoder`.
"""

import json
from pathlib import Path
from typing import TypedDict

from sklearn.preprocessing import LabelEncoder


class MapeamentoIds(TypedDict):
    """Mapeamento persistido entre ids reais do dataset e índices internos dos embeddings."""

    usuario_id_para_idx: dict[int, int]
    filme_id_para_idx: dict[int, int]


def extrair_mapeamento(
    codificador_usuarios: LabelEncoder, codificador_filmes: LabelEncoder
) -> MapeamentoIds:
    """Extrai o mapeamento id-real -> índice a partir de codificadores já ajustados.

    Args:
        codificador_usuarios: `LabelEncoder` de usuários já ajustado (`fit`).
        codificador_filmes: `LabelEncoder` de filmes já ajustado (`fit`).

    Returns:
        Mapeamento `usuario_id_para_idx`/`filme_id_para_idx`.
    """
    return {
        "usuario_id_para_idx": {
            int(id_real): idx for idx, id_real in enumerate(codificador_usuarios.classes_)
        },
        "filme_id_para_idx": {
            int(id_real): idx for idx, id_real in enumerate(codificador_filmes.classes_)
        },
    }


def salvar_mapeamento(mapeamento: MapeamentoIds, caminho: Path) -> None:
    """Salva o mapeamento de ids como JSON.

    Args:
        mapeamento: Mapeamento a persistir.
        caminho: Caminho do arquivo de saída (`.json`).
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(mapeamento, indent=2), encoding="utf-8")


def carregar_mapeamento(caminho: Path) -> MapeamentoIds:
    """Carrega o mapeamento de ids salvo por `salvar_mapeamento`.

    As chaves de um objeto JSON são sempre strings; aqui elas são convertidas de volta
    para `int`, restaurando o tipo original dos ids do dataset.

    Args:
        caminho: Caminho do arquivo de mapeamento (`.json`).

    Returns:
        Mapeamento `usuario_id_para_idx`/`filme_id_para_idx`, com chaves `int`.
    """
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    return {
        "usuario_id_para_idx": {int(k): v for k, v in bruto["usuario_id_para_idx"].items()},
        "filme_id_para_idx": {int(k): v for k, v in bruto["filme_id_para_idx"].items()},
    }
