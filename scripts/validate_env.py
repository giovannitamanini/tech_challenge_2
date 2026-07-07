"""Script de validação do ambiente de desenvolvimento."""

import sys
from pathlib import Path

from tech_challenge_recomendacao.configuracoes import configuracoes

sys.stdout.reconfigure(encoding="utf-8")

PYTHON_MINIMO = (3, 13)


def validar_versao_python() -> bool:
    """Verifica se a versão do Python instalada atende ao mínimo exigido.

    Returns:
        True se a versão for compatível, False caso contrário.
    """
    versao_atual = sys.version_info[:2]
    if versao_atual < PYTHON_MINIMO:
        print(
            f"[ERRO] Python {'.'.join(map(str, PYTHON_MINIMO))}+ é exigido, "
            f"encontrado {'.'.join(map(str, versao_atual))}."
        )
        return False
    print(f"[OK] Python {'.'.join(map(str, versao_atual))}")
    return True


def validar_configuracoes() -> bool:
    """Verifica se as configurações do `.env` foram carregadas corretamente.

    Returns:
        True se as configurações carregarem sem erro, False caso contrário.
    """
    try:
        print(
            f"[OK] Configurações carregadas: "
            f"semente_aleatoria={configuracoes.semente_aleatoria}, "
            f"mlflow_tracking_uri={configuracoes.mlflow_tracking_uri}"
        )
        return True
    except Exception as erro:  # noqa: BLE001
        print(f"[ERRO] Falha ao carregar configurações: {erro}")
        return False


def validar_diretorio(caminho: str, descricao: str) -> bool:
    """Verifica se um diretório esperado existe no sistema de arquivos.

    Args:
        caminho: Caminho do diretório a ser validado.
        descricao: Rótulo legível usado nas mensagens de saída.

    Returns:
        True se o diretório existir, False caso contrário.
    """
    if not Path(caminho).is_dir():
        print(f"[ERRO] {descricao} não encontrado em '{caminho}'.")
        return False
    print(f"[OK] {descricao} encontrado em '{caminho}'.")
    return True


def main() -> None:
    """Executa todas as validações de ambiente e encerra com o código de saída apropriado."""
    resultados = [
        validar_versao_python(),
        validar_configuracoes(),
        validar_diretorio(configuracoes.diretorio_dados_brutos, "Diretório de dados brutos"),
        validar_diretorio(
            configuracoes.diretorio_dados_processados, "Diretório de dados processados"
        ),
        validar_diretorio(configuracoes.diretorio_modelos, "Diretório de modelos"),
    ]

    if all(resultados):
        print("\nAmbiente validado com sucesso.")
        sys.exit(0)
    print("\nFalha na validação do ambiente.")
    sys.exit(1)


if __name__ == "__main__":
    main()
