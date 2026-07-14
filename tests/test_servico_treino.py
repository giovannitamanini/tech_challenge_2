"""Testes de `ServicoTreino` (`api/servico_treino.py`).

Usa subprocessos reais e curtos (`sys.executable -c "..."`) no lugar de `dvc repro`, para
testar o comportamento de verdade (pid, `poll()`, código de saída) sem depender do pipeline.
"""

import json
import sys
import time
from pathlib import Path

from tech_challenge_recomendacao.api.erros import (
    ExecucaoTreinoNaoEncontradaErro,
    TreinoJaEmAndamentoErro,
)
from tech_challenge_recomendacao.api.servico_treino import ServicoTreino

COMANDO_DEMORADO = [sys.executable, "-c", "import time; time.sleep(1)"]
COMANDO_RAPIDO_SUCESSO = [sys.executable, "-c", "pass"]
COMANDO_RAPIDO_FALHA = [sys.executable, "-c", "import sys; sys.exit(1)"]


def _servico(tmp_path: Path, comando: list[str]) -> ServicoTreino:
    return ServicoTreino(
        comando=comando,
        caminho_log=tmp_path / "log.txt",
        caminho_metricas_treino=tmp_path / "metricas_treino.json",
        caminho_metricas_avaliacao=tmp_path / "metricas_avaliacao.json",
    )


def _aguardar_conclusao(servico: ServicoTreino, execucao_id: str, tentativas: int = 20) -> dict:
    for _ in range(tentativas):
        status = servico.consultar_status(execucao_id)
        if status["status"] != "em_execucao":
            return status
        time.sleep(0.1)
    raise TimeoutError("Subprocesso de teste não terminou a tempo.")


def test_iniciar_treino_devolve_execucao_em_andamento(tmp_path: Path) -> None:
    """Logo após iniciar, o status deve ser `em_execucao` (o subprocesso ainda não terminou)."""
    servico = _servico(tmp_path, COMANDO_DEMORADO)

    execucao = servico.iniciar_treino()
    status = servico.consultar_status(execucao.execucao_id)

    assert status["status"] == "em_execucao"
    assert status["codigo_saida"] is None
    execucao.processo.wait()  # limpa o processo antes do teste terminar


def test_iniciar_treino_com_outro_em_andamento_leva_erro(tmp_path: Path) -> None:
    """Um segundo `iniciar_treino` deve falhar enquanto o primeiro ainda está rodando."""
    servico = _servico(tmp_path, COMANDO_DEMORADO)
    execucao = servico.iniciar_treino()

    try:
        try:
            servico.iniciar_treino()
            raise AssertionError("Deveria ter levantado TreinoJaEmAndamentoErro.")
        except TreinoJaEmAndamentoErro:
            pass
    finally:
        execucao.processo.wait()


def test_consultar_status_de_execucao_desconhecida_leva_erro(tmp_path: Path) -> None:
    """Um `execucao_id` que nunca foi iniciado deve levantar `ExecucaoTreinoNaoEncontradaErro`."""
    servico = _servico(tmp_path, COMANDO_RAPIDO_SUCESSO)

    try:
        servico.consultar_status("id-que-nao-existe")
        raise AssertionError("Deveria ter levantado ExecucaoTreinoNaoEncontradaErro.")
    except ExecucaoTreinoNaoEncontradaErro:
        pass


def test_execucao_concluida_com_sucesso_le_as_metricas(tmp_path: Path) -> None:
    """Ao concluir com código 0, o status deve ser `concluido` e as métricas devem ser lidas."""
    (tmp_path / "metricas_treino.json").write_text(
        json.dumps({"rmse_treino": 0.5}), encoding="utf-8"
    )
    (tmp_path / "metricas_avaliacao.json").write_text(
        json.dumps({"rmse": 0.6, "mae": 0.4}), encoding="utf-8"
    )
    servico = _servico(tmp_path, COMANDO_RAPIDO_SUCESSO)

    execucao = servico.iniciar_treino()
    status = _aguardar_conclusao(servico, execucao.execucao_id)

    assert status["status"] == "concluido"
    assert status["codigo_saida"] == 0
    assert status["metricas_treino"] == {"rmse_treino": 0.5}
    assert status["metricas_avaliacao"] == {"rmse": 0.6, "mae": 0.4}


def test_execucao_com_falha_nao_le_metricas(tmp_path: Path) -> None:
    """Se o subprocesso falhar (código != 0), o status deve ser `falhou` e sem métricas."""
    (tmp_path / "metricas_treino.json").write_text(json.dumps({"rmse_treino": 0.5}))
    servico = _servico(tmp_path, COMANDO_RAPIDO_FALHA)

    execucao = servico.iniciar_treino()
    status = _aguardar_conclusao(servico, execucao.execucao_id)

    assert status["status"] == "falhou"
    assert status["codigo_saida"] == 1
    assert status["metricas_treino"] is None


def test_pode_iniciar_novo_treino_apos_o_anterior_concluir(tmp_path: Path) -> None:
    """Depois que a execução anterior termina, um novo `iniciar_treino` deve ser aceito."""
    servico = _servico(tmp_path, COMANDO_RAPIDO_SUCESSO)
    primeira_execucao = servico.iniciar_treino()
    _aguardar_conclusao(servico, primeira_execucao.execucao_id)

    segunda_execucao = servico.iniciar_treino()

    assert segunda_execucao.execucao_id != primeira_execucao.execucao_id
    segunda_execucao.processo.wait()
