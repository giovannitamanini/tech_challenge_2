"""Lógica de negócio do treino sob demanda: dispara o pipeline em um container Docker.

Diferente do `ServicoRecomendacao` (que só lê artefatos já treinados), este serviço inicia e
acompanha uma execução do pipeline de treino via subprocesso, para que o retreino não bloqueie
a thread da API nem dependa de infraestrutura externa (fila de tarefas) — adequado a um treino
de cada vez, disparado manualmente via `POST /treino`.

O comando padrão sobe o serviço `train` do `docker-compose.yml` (mesma imagem/volumes/rede
testados na Etapa 3), sobrescrevendo seu `CMD` para rodar `dvc repro` (pipeline completo:
`preprocess → feature_eng → train → evaluate`) em vez de só o script de treino isolado —
mantém o treino sob demanda isolado num container, coerente com o resto do projeto. Requer
Docker (e o daemon) disponíveis onde a API estiver rodando.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tech_challenge_recomendacao.api.erros import (
    ExecucaoTreinoNaoEncontradaErro,
    TreinoJaEmAndamentoErro,
)

COMANDO_TREINO_PADRAO = ["docker", "compose", "run", "--rm", "train", "dvc", "repro"]
CAMINHO_LOG_PADRAO = Path("data/processed_data/ultima_execucao_treino.log")
CAMINHO_METRICAS_TREINO_PADRAO = Path("data/processed_data/metricas_treino.json")
CAMINHO_METRICAS_AVALIACAO_PADRAO = Path("data/processed_data/metricas_avaliacao.json")


@dataclass
class ExecucaoTreino:
    """Estado de uma execução do pipeline de treino disparada via `POST /treino`."""

    execucao_id: str
    processo: subprocess.Popen
    iniciado_em: datetime


class ServicoTreino:
    """Dispara e acompanha execuções do pipeline de treino (`dvc repro`), uma por vez.

    Mantém em memória apenas a execução mais recente: como só um treino roda por vez, não
    há necessidade de um registro histórico completo para este escopo.
    """

    def __init__(
        self,
        comando: list[str] | None = None,
        caminho_log: Path = CAMINHO_LOG_PADRAO,
        caminho_metricas_treino: Path = CAMINHO_METRICAS_TREINO_PADRAO,
        caminho_metricas_avaliacao: Path = CAMINHO_METRICAS_AVALIACAO_PADRAO,
    ) -> None:
        self._comando = comando if comando is not None else COMANDO_TREINO_PADRAO
        self._caminho_log = caminho_log
        self._caminho_metricas_treino = caminho_metricas_treino
        self._caminho_metricas_avaliacao = caminho_metricas_avaliacao
        self._execucao_atual: ExecucaoTreino | None = None

    def iniciar_treino(self) -> ExecucaoTreino:
        """Dispara o pipeline de treino em um subprocesso, se nenhum outro estiver rodando.

        Returns:
            A execução recém-iniciada (`execucao_id` é o pid do subprocesso).

        Raises:
            TreinoJaEmAndamentoErro: Se já existir uma execução em andamento.
        """
        if self._execucao_atual is not None and self._em_andamento(self._execucao_atual):
            raise TreinoJaEmAndamentoErro(self._execucao_atual.execucao_id)

        processo = self._disparar_subprocesso()
        self._execucao_atual = ExecucaoTreino(
            execucao_id=str(processo.pid),
            processo=processo,
            iniciado_em=datetime.now(timezone.utc),
        )
        return self._execucao_atual

    def _disparar_subprocesso(self) -> subprocess.Popen:
        """Abre o arquivo de log e dispara o subprocesso (fecha o handle do lado do pai)."""
        self._caminho_log.parent.mkdir(parents=True, exist_ok=True)
        with self._caminho_log.open("w", encoding="utf-8") as arquivo_log:
            return subprocess.Popen(
                self._comando, cwd=Path.cwd(), stdout=arquivo_log, stderr=subprocess.STDOUT
            )

    def consultar_status(self, execucao_id: str) -> dict[str, object]:
        """Consulta o status da execução mais recente, se o id corresponder a ela.

        Args:
            execucao_id: Id devolvido por `POST /treino`.

        Returns:
            Dicionário pronto para a resposta de `GET /treino/status/{execucao_id}`.

        Raises:
            ExecucaoTreinoNaoEncontradaErro: Se `execucao_id` não for a execução conhecida.
        """
        execucao = self._execucao_atual
        if execucao is None or execucao.execucao_id != execucao_id:
            raise ExecucaoTreinoNaoEncontradaErro(execucao_id)

        codigo_saida = execucao.processo.poll()
        status = self._status_textual(codigo_saida)
        return {
            "execucao_id": execucao.execucao_id,
            "status": status,
            "iniciado_em": execucao.iniciado_em,
            "codigo_saida": codigo_saida,
            "metricas_treino": self._ler_json_se_concluido(status, self._caminho_metricas_treino),
            "metricas_avaliacao": self._ler_json_se_concluido(
                status, self._caminho_metricas_avaliacao
            ),
        }

    def _em_andamento(self, execucao: ExecucaoTreino) -> bool:
        """`True` se o subprocesso da execução ainda não terminou."""
        return execucao.processo.poll() is None

    def _status_textual(self, codigo_saida: int | None) -> str:
        """Traduz o código de saída do subprocesso para um status textual."""
        if codigo_saida is None:
            return "em_execucao"
        return "concluido" if codigo_saida == 0 else "falhou"

    def _ler_json_se_concluido(self, status: str, caminho: Path) -> dict | None:
        """Lê um artefato JSON do pipeline, só quando a execução terminou com sucesso."""
        if status != "concluido" or not caminho.exists():
            return None
        return json.loads(caminho.read_text(encoding="utf-8"))
