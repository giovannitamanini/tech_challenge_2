"""Lógica de negócio da API de recomendação.

`ServicoRecomendacao` carrega o modelo treinado e os artefatos do pipeline (mapeamento
de ids, itens já vistos no treino, métricas de avaliação) uma única vez por processo e
serve previsões, recomendações e busca de filmes similares — diferente dos scripts de
stage do pipeline, que rodam uma vez e terminam, este serviço vive pelo tempo de vida do
processo da API (ver `api/dependencias.py`).
"""

import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from tech_challenge_recomendacao.api.erros import (
    FilmeNaoEncontradoErro,
    RecursoNaoSuportadoErro,
    UsuarioNaoEncontradoErro,
)
from tech_challenge_recomendacao.configuracoes import configuracoes
from tech_challenge_recomendacao.dados.catalogo_filmes import carregar_catalogo_filmes
from tech_challenge_recomendacao.dados.mapeamento_ids import MapeamentoIds, carregar_mapeamento
from tech_challenge_recomendacao.modelos.base import ModeloRecomendador
from tech_challenge_recomendacao.modelos.capacidades import ExpoeEmbeddingsItem
from tech_challenge_recomendacao.modelos.checkpoint import carregar_checkpoint

NOME_MODELO_REGISTRO = "recomendacao-movielens"
# Sem servidor MLflow no ar, o cliente por padrão tenta de novo várias vezes com backoff
# crescente (~5 tentativas) antes de desistir — inaceitável para uma consulta best-effort
# num endpoint HTTP. Uma única tentativa com timeout curto garante falha rápida.
TIMEOUT_MLFLOW_REGISTRO_SEGUNDOS = "2"
MAX_TENTATIVAS_MLFLOW_REGISTRO = "1"


class ServicoRecomendacao:
    """Serve previsões, recomendações e busca de filmes similares a partir do modelo treinado."""

    def __init__(
        self,
        modelo: ModeloRecomendador,
        mapeamento: MapeamentoIds,
        itens_vistos_por_usuario: dict[int, set[int]],
        tipo_modelo: str,
        dimensao_embedding: int,
        n_usuarios: int,
        n_filmes: int,
        metricas_avaliacao: dict[str, float] | None,
        titulos_filmes: dict[int, str],
    ) -> None:
        self._modelo = modelo
        self._usuario_id_para_idx = mapeamento["usuario_id_para_idx"]
        self._filme_id_para_idx = mapeamento["filme_id_para_idx"]
        self._filme_idx_para_id = {idx: id_ for id_, idx in self._filme_id_para_idx.items()}
        self._itens_vistos_por_usuario = itens_vistos_por_usuario
        self._titulos_filmes = titulos_filmes
        self.tipo_modelo = tipo_modelo
        self.dimensao_embedding = dimensao_embedding
        self.n_usuarios = n_usuarios
        self.n_filmes = n_filmes
        self.metricas_avaliacao = metricas_avaliacao

    def _idx_usuario(self, usuario_id: int) -> int:
        """Traduz um `usuario_id` real para o índice interno do embedding."""
        idx = self._usuario_id_para_idx.get(usuario_id)
        if idx is None:
            raise UsuarioNaoEncontradoErro(usuario_id)
        return idx

    def _idx_filme(self, filme_id: int) -> int:
        """Traduz um `filme_id` real para o índice interno do embedding."""
        idx = self._filme_id_para_idx.get(filme_id)
        if idx is None:
            raise FilmeNaoEncontradoErro(filme_id)
        return idx

    def _titulo_filme(self, filme_id: int) -> str | None:
        """Título do filme, se conhecido pelo catálogo (`data/raw_data/movies.csv`)."""
        return self._titulos_filmes.get(filme_id)

    def prever_lote(self, pares: list[tuple[int, int]]) -> list[float]:
        """Prevê a nota de cada par (usuario_id, filme_id).

        Args:
            pares: Lista de pares de ids reais `(usuario_id, filme_id)`.

        Returns:
            Nota prevista para cada par, na mesma ordem de `pares`.

        Raises:
            UsuarioNaoEncontradoErro: Se algum `usuario_id` for desconhecido.
            FilmeNaoEncontradoErro: Se algum `filme_id` for desconhecido.
        """
        usuarios_idx = [self._idx_usuario(u) for u, _ in pares]
        filmes_idx = [self._idx_filme(f) for _, f in pares]
        with torch.no_grad():
            usuarios = torch.tensor(usuarios_idx, dtype=torch.long)
            filmes = torch.tensor(filmes_idx, dtype=torch.long)
            notas = self._modelo(usuarios, filmes)
        return notas.tolist()

    def recomendar(self, usuario_id: int, k: int) -> list[tuple[int, str | None, float]]:
        """Recomenda os `k` filmes com maior nota prevista para o usuário.

        Filmes já avaliados no treino são excluídos das recomendações.

        Args:
            usuario_id: Id real do usuário.
            k: Número de filmes a recomendar.

        Returns:
            Lista de `(filme_id, titulo, nota_prevista)`, da maior para a menor nota.

        Raises:
            UsuarioNaoEncontradoErro: Se `usuario_id` for desconhecido.
        """
        usuario_idx = self._idx_usuario(usuario_id)
        with torch.no_grad():
            usuarios = torch.full((self.n_filmes,), usuario_idx, dtype=torch.long)
            filmes = torch.arange(self.n_filmes, dtype=torch.long)
            notas = self._modelo(usuarios, filmes)
        vistos = self._itens_vistos_por_usuario.get(usuario_idx, set())
        for filme_idx in vistos:
            notas[filme_idx] = -torch.inf
        k_efetivo = min(k, self.n_filmes - len(vistos))
        top_k = torch.topk(notas, k=k_efetivo).indices.tolist()

        resultado = []
        for idx in top_k:
            filme_id = self._filme_idx_para_id[idx]
            resultado.append((filme_id, self._titulo_filme(filme_id), notas[idx].item()))
        return resultado

    def filmes_similares(self, filme_id: int, k: int) -> list[tuple[int, str | None, float]]:
        """Busca os `k` filmes mais similares, via cosseno entre embeddings de item.

        Args:
            filme_id: Id real do filme de referência.
            k: Número de filmes similares a devolver.

        Returns:
            Lista de `(filme_id, titulo, similaridade)`, da mais para a menos similar.

        Raises:
            FilmeNaoEncontradoErro: Se `filme_id` for desconhecido.
            RecursoNaoSuportadoErro: Se o modelo carregado não expuser embeddings de item.
        """
        if not isinstance(self._modelo, ExpoeEmbeddingsItem):
            raise RecursoNaoSuportadoErro(
                f"Modelo '{self.tipo_modelo}' não expõe embeddings de item."
            )
        filme_idx_referencia = self._idx_filme(filme_id)
        with torch.no_grad():
            embeddings = self._modelo.embeddings_item()
            similaridades = torch.nn.functional.cosine_similarity(
                embeddings[filme_idx_referencia].unsqueeze(0), embeddings
            )
        similaridades[filme_idx_referencia] = -torch.inf
        top_k = torch.topk(similaridades, k=min(k, self.n_filmes - 1)).indices.tolist()

        resultado = []
        for idx in top_k:
            filme_id_similar = self._filme_idx_para_id[idx]
            titulo = self._titulo_filme(filme_id_similar)
            resultado.append((filme_id_similar, titulo, similaridades[idx].item()))
        return resultado

    def info_modelo(self) -> dict[str, object]:
        """Metadados do modelo carregado, para o endpoint `/modelo/info`."""
        return {
            "tipo_modelo": self.tipo_modelo,
            "dimensao_embedding": self.dimensao_embedding,
            "n_usuarios": self.n_usuarios,
            "n_filmes": self.n_filmes,
            "metricas_avaliacao": self.metricas_avaliacao,
            "versao_registrada": _obter_versao_registrada(),
        }


def _obter_versao_registrada() -> str | None:
    """Consulta a versão mais recente do modelo no MLflow Model Registry, se houver.

    Best-effort: o pipeline ainda não registra/promove o modelo (tarefa futura, ver
    `CLAUDE.md`) e o servidor de tracking pode estar fora do ar. Timeout curto e uma
    única tentativa evitam que a ausência de um servidor MLflow trave o endpoint
    `/modelo/info`; em qualquer falha, devolve `None` em vez de derrubar a requisição.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", TIMEOUT_MLFLOW_REGISTRO_SEGUNDOS)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", MAX_TENTATIVAS_MLFLOW_REGISTRO)
    try:
        mlflow.set_tracking_uri(configuracoes.mlflow_tracking_uri)
        versoes = MlflowClient().get_latest_versions(NOME_MODELO_REGISTRO)
        if not versoes:
            return None
        versao = versoes[0]
        return f"v{versao.version} ({versao.current_stage})"
    except Exception:
        return None


def _carregar_itens_vistos(caminho_treino: Path) -> dict[int, set[int]]:
    """Carrega, por usuário, o conjunto de filmes já avaliados no treino.

    Args:
        caminho_treino: Caminho do parquet de treino (`usuario_idx`, `filme_idx`, `rating`).

    Returns:
        Mapa `usuario_idx -> conjunto de filme_idx` já avaliados.
    """
    treino = pd.read_parquet(caminho_treino)
    return treino.groupby("usuario_idx")["filme_idx"].apply(set).to_dict()


def _carregar_metricas_avaliacao(caminho: Path) -> dict[str, float] | None:
    """Carrega as métricas do stage `evaluate`, se o arquivo já existir.

    Args:
        caminho: Caminho de `metricas_avaliacao.json`.

    Returns:
        Métricas (`rmse`, `mae`), ou `None` se o stage `evaluate` ainda não rodou.
    """
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def _carregar_titulos_filmes(caminho: Path) -> dict[int, str]:
    """Carrega o catálogo de títulos, se o `feature_eng` já tiver gerado esse artefato.

    Args:
        caminho: Caminho de `catalogo_filmes.json`.

    Returns:
        Mapa `filme_id -> título`, ou vazio se o artefato ainda não existir (pipelines
        executados antes desta funcionalidade existir) — filmes ficam sem título, mas a
        API continua funcionando normalmente.
    """
    if not caminho.exists():
        return {}
    return carregar_catalogo_filmes(caminho)


def carregar_servico_recomendacao() -> ServicoRecomendacao:
    """Constrói o `ServicoRecomendacao` a partir dos artefatos do pipeline em disco.

    O checkpoint (não `configs/params.yaml`) é a fonte de verdade de `tipo_modelo`/
    `dimensao_embedding`/`n_usuarios`/`n_filmes`: é o que descreve o modelo realmente
    treinado e carregado, e não pode divergir dele — diferente dos parâmetros de
    configuração, que podem mudar sem que o modelo em disco seja retreinado.

    Returns:
        Serviço pronto para uso, com o modelo e os artefatos carregados em memória.
    """
    diretorio_dados = Path(configuracoes.diretorio_dados_processados)
    caminho_checkpoint = Path(configuracoes.diretorio_modelos) / "modelo_recomendador.pt"

    modelo, metadados_checkpoint = carregar_checkpoint(caminho_checkpoint)
    metricas_avaliacao = _carregar_metricas_avaliacao(diretorio_dados / "metricas_avaliacao.json")

    return ServicoRecomendacao(
        modelo=modelo,
        mapeamento=carregar_mapeamento(diretorio_dados / "mapeamento_ids.json"),
        itens_vistos_por_usuario=_carregar_itens_vistos(diretorio_dados / "treino.parquet"),
        tipo_modelo=metadados_checkpoint["tipo_modelo"],
        dimensao_embedding=metadados_checkpoint["dimensao_embedding"],
        n_usuarios=metadados_checkpoint["n_usuarios"],
        n_filmes=metadados_checkpoint["n_filmes"],
        metricas_avaliacao=metricas_avaliacao,
        titulos_filmes=_carregar_titulos_filmes(diretorio_dados / "catalogo_filmes.json"),
    )
