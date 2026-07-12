"""Testes da persistência do catálogo de títulos (`dados/catalogo_filmes.py`)."""

from pathlib import Path

from tech_challenge_recomendacao.dados.catalogo_filmes import (
    carregar_catalogo_filmes,
    extrair_titulos,
    salvar_catalogo_filmes,
)


def test_extrair_titulos_le_movieid_e_title(tmp_path: Path) -> None:
    """Deve extrair só as colunas `movieId`/`title`, ignorando `genres`."""
    caminho = tmp_path / "movies.csv"
    caminho.write_text(
        "movieId,title,genres\n1,Filme Um,Ação\n2,Filme Dois,Comédia|Drama\n", encoding="utf-8"
    )

    titulos = extrair_titulos(caminho)

    assert titulos == {1: "Filme Um", 2: "Filme Dois"}


def test_salvar_e_carregar_catalogo_faz_round_trip(tmp_path: Path) -> None:
    """Salvar e recarregar deve devolver o mesmo catálogo, com chaves `int`."""
    titulos = {1: "Filme Um", 2: "Filme Dois"}
    caminho = tmp_path / "catalogo_filmes.json"

    salvar_catalogo_filmes(titulos, caminho)
    recarregado = carregar_catalogo_filmes(caminho)

    assert recarregado == titulos
    assert all(isinstance(k, int) for k in recarregado)


def test_salvar_catalogo_preserva_acentuacao(tmp_path: Path) -> None:
    """Títulos com acentuação devem ser preservados legíveis no JSON salvo (não escapados)."""
    caminho = tmp_path / "catalogo_filmes.json"

    salvar_catalogo_filmes({1: "Amélie Poulain"}, caminho)

    assert "Amélie Poulain" in caminho.read_text(encoding="utf-8")
