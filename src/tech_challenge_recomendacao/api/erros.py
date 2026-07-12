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
