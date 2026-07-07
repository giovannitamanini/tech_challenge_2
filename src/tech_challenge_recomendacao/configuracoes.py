"""Configurações do projeto, carregadas a partir de variáveis de ambiente."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracoes(BaseSettings):
    """Configurações centrais do projeto, lidas do arquivo `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    semente_aleatoria: int = 42
    mlflow_tracking_uri: str = "http://localhost:5000"
    diretorio_dados_brutos: str = "data/raw_data"
    diretorio_dados_processados: str = "data/processed_data"
    diretorio_modelos: str = "models"


configuracoes = Configuracoes()
