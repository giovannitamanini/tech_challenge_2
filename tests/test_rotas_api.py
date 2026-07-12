"""Testes das rotas HTTP da API (`api/rotas.py`), via `TestClient` — sem servidor real."""

from fastapi.testclient import TestClient

from tech_challenge_recomendacao.api.aplicacao import criar_aplicacao
from tech_challenge_recomendacao.api.dependencias import obter_servico_recomendacao
from tech_challenge_recomendacao.api.servico_recomendacao import ServicoRecomendacao


def _cliente(servico: ServicoRecomendacao) -> TestClient:
    """Monta um `TestClient` com o serviço em memória injetado no lugar do real."""
    app = criar_aplicacao()
    app.dependency_overrides[obter_servico_recomendacao] = lambda: servico
    return TestClient(app)


def test_saude_responde_200(servico_recomendacao: ServicoRecomendacao) -> None:
    """`GET /saude` deve sempre responder 200."""
    resposta = _cliente(servico_recomendacao).get("/saude")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_modelo_info_responde_com_os_metadados(servico_recomendacao: ServicoRecomendacao) -> None:
    """`GET /modelo/info` deve devolver os metadados do serviço injetado."""
    resposta = _cliente(servico_recomendacao).get("/modelo/info")

    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["tipo_modelo"] == "fatoracao_matricial"
    assert corpo["n_usuarios"] == 3
    assert corpo["metricas_avaliacao"] == {"rmse": 0.9, "mae": 0.7}


def test_previsoes_responde_uma_nota_por_par(servico_recomendacao: ServicoRecomendacao) -> None:
    """`POST /previsoes` deve devolver uma previsão para cada par do lote."""
    corpo = {"pares": [{"usuario_id": 10, "filme_id": 100}, {"usuario_id": 20, "filme_id": 200}]}

    resposta = _cliente(servico_recomendacao).post("/previsoes", json=corpo)

    assert resposta.status_code == 200
    assert len(resposta.json()["previsoes"]) == 2


def test_previsoes_com_usuario_desconhecido_responde_404(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `usuario_id` desconhecido em `/previsoes` deve responder 404."""
    corpo = {"pares": [{"usuario_id": 999, "filme_id": 100}]}

    resposta = _cliente(servico_recomendacao).post("/previsoes", json=corpo)

    assert resposta.status_code == 404


def test_recomendacoes_exclui_filme_ja_visto(servico_recomendacao: ServicoRecomendacao) -> None:
    """`GET /recomendacoes/{usuario_id}` não deve trazer o filme já avaliado no treino."""
    resposta = _cliente(servico_recomendacao).get("/recomendacoes/10", params={"k": 10})

    corpo = resposta.json()
    filme_ids = {item["filme_id"] for item in corpo["recomendacoes"]}
    assert resposta.status_code == 200
    assert 100 not in filme_ids


def test_recomendacoes_traz_o_titulo_do_filme(servico_recomendacao: ServicoRecomendacao) -> None:
    """`GET /recomendacoes/{usuario_id}` deve trazer o título de cada filme recomendado."""
    resposta = _cliente(servico_recomendacao).get("/recomendacoes/10", params={"k": 10})

    corpo = resposta.json()
    titulos_por_filme = {item["filme_id"]: item["titulo"] for item in corpo["recomendacoes"]}
    assert titulos_por_filme[200] == "Filme B"


def test_recomendacoes_com_usuario_desconhecido_responde_404(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `usuario_id` desconhecido em `/recomendacoes` deve responder 404."""
    resposta = _cliente(servico_recomendacao).get("/recomendacoes/999")

    assert resposta.status_code == 404


def test_filmes_similares_exclui_o_proprio_filme(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """`GET /filmes/{filme_id}/similares` não deve trazer o próprio filme na lista."""
    resposta = _cliente(servico_recomendacao).get("/filmes/100/similares", params={"k": 10})

    corpo = resposta.json()
    filme_ids = {item["filme_id"] for item in corpo["similares"]}
    assert resposta.status_code == 200
    assert 100 not in filme_ids


def test_filmes_similares_traz_o_titulo_do_filme(servico_recomendacao: ServicoRecomendacao) -> None:
    """`GET /filmes/{filme_id}/similares` deve trazer o título de cada filme similar."""
    resposta = _cliente(servico_recomendacao).get("/filmes/100/similares", params={"k": 10})

    corpo = resposta.json()
    titulos_por_filme = {item["filme_id"]: item["titulo"] for item in corpo["similares"]}
    assert titulos_por_filme[200] == "Filme B"


def test_filmes_similares_com_filme_desconhecido_responde_404(
    servico_recomendacao: ServicoRecomendacao,
) -> None:
    """Um `filme_id` desconhecido em `/filmes/.../similares` deve responder 404."""
    resposta = _cliente(servico_recomendacao).get("/filmes/999/similares")

    assert resposta.status_code == 404


def test_filmes_similares_com_modelo_sem_suporte_responde_501(
    servico_sem_embeddings_item: ServicoRecomendacao,
) -> None:
    """Um modelo sem `ExpoeEmbeddingsItem` deve responder 501 em `/filmes/.../similares`."""
    resposta = _cliente(servico_sem_embeddings_item).get("/filmes/100/similares")

    assert resposta.status_code == 501
