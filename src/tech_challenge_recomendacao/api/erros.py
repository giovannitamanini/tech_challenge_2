"""Erros de domínio da API de recomendação.

Independentes do `fastapi`: `ServicoRecomendacao` levanta essas exceções sem conhecer
HTTP, e a camada de aplicação (`api/aplicacao.py`) as traduz para respostas HTTP em um
único lugar (`@app.exception_handler`), mantendo as rotas livres de `try/except`.
"""


class UsuarioNaoEncontradoErro(Exception):
    """Levantado quando um `usuario_id` não existe no mapeamento de ids conhecido."""

    def __init__(self, usuario_id: int) -> None:
        self.usuario_id = usuario_id
        super().__init__(f"Usuário '{usuario_id}' não encontrado.")


class FilmeNaoEncontradoErro(Exception):
    """Levantado quando um `filme_id` não existe no mapeamento de ids conhecido."""

    def __init__(self, filme_id: int) -> None:
        self.filme_id = filme_id
        super().__init__(f"Filme '{filme_id}' não encontrado.")


class RecursoNaoSuportadoErro(Exception):
    """Levantado quando o modelo carregado não suporta a operação pedida."""


class TreinoJaEmAndamentoErro(Exception):
    """Levantado ao tentar iniciar um treino enquanto outro já está em execução."""

    def __init__(self, execucao_id: str) -> None:
        self.execucao_id = execucao_id
        super().__init__(f"Já existe um treino em andamento (execução '{execucao_id}').")


class ExecucaoTreinoNaoEncontradaErro(Exception):
    """Levantado quando `execucao_id` não corresponde a nenhuma execução conhecida."""

    def __init__(self, execucao_id: str) -> None:
        self.execucao_id = execucao_id
        super().__init__(f"Execução de treino '{execucao_id}' não encontrada.")
