"""Testes das rotas de treino sob demanda (`POST /treino`, `GET /treino/status/{id}`)."""

import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from tech_challenge_recomendacao.api.aplicacao import criar_aplicacao
from tech_challenge_recomendacao.api.dependencias import obter_servico_treino
from tech_challenge_recomendacao.api.servico_treino import ServicoTreino

COMANDO_DEMORADO = [sys.executable, "-c", "import time; time.sleep(1)"]
COMANDO_RAPIDO_SUCESSO = [sys.executable, "-c", "pass"]


def _cliente(servico: ServicoTreino) -> TestClient:
    """Monta um `TestClient` com o `ServicoTreino` em memória injetado no lugar do real."""
    app = criar_aplicacao()
    app.dependency_overrides[obter_servico_treino] = lambda: servico
    return TestClient(app)


def _servico(tmp_path: Path, comando: list[str]) -> ServicoTreino:
    return ServicoTreino(
        comando=comando,
        caminho_log=tmp_path / "log.txt",
        caminho_metricas_treino=tmp_path / "metricas_treino.json",
        caminho_metricas_avaliacao=tmp_path / "metricas_avaliacao.json",
    )


def test_post_treino_responde_202_com_execucao_em_andamento(tmp_path: Path) -> None:
    """`POST /treino` deve responder 202 e devolver o id da execução recém-iniciada."""
    servico = _servico(tmp_path, COMANDO_DEMORADO)

    resposta = _cliente(servico).post("/treino")

    corpo = resposta.json()
    assert resposta.status_code == 202
    assert corpo["status"] == "em_execucao"
    assert corpo["execucao_id"]
    servico._execucao_atual.processo.wait()  # limpa o processo antes do teste terminar


def test_post_treino_com_outro_em_andamento_responde_409(tmp_path: Path) -> None:
    """Um segundo `POST /treino` deve responder 409 enquanto o primeiro ainda roda."""
    servico = _servico(tmp_path, COMANDO_DEMORADO)
    cliente = _cliente(servico)
    cliente.post("/treino")

    resposta = cliente.post("/treino")

    assert resposta.status_code == 409
    servico._execucao_atual.processo.wait()


def test_status_de_execucao_desconhecida_responde_404(tmp_path: Path) -> None:
    """`GET /treino/status/{id}` com um id desconhecido deve responder 404."""
    servico = _servico(tmp_path, COMANDO_RAPIDO_SUCESSO)

    resposta = _cliente(servico).get("/treino/status/id-que-nao-existe")

    assert resposta.status_code == 404


def test_status_reflete_conclusao_da_execucao(tmp_path: Path) -> None:
    """Depois que o subprocesso termina, o status consultado deve virar `concluido`."""
    servico = _servico(tmp_path, COMANDO_RAPIDO_SUCESSO)
    cliente = _cliente(servico)
    execucao_id = cliente.post("/treino").json()["execucao_id"]

    for _ in range(20):
        resposta = cliente.get(f"/treino/status/{execucao_id}")
        if resposta.json()["status"] != "em_execucao":
            break
        time.sleep(0.1)

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "concluido"
    assert resposta.json()["codigo_saida"] == 0
