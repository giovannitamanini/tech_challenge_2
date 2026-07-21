"""Configurações do projeto, carregadas a partir de variáveis de ambiente."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracoes(BaseSettings):
    """Configurações centrais do projeto, lidas do arquivo `.env`."""

    # extra="ignore": `.env` tem variáveis que não passam por `Configuracoes` (ex.:
    # `HOST_PROJECT_DIR`, usada só pela substituição de variáveis do próprio docker
    # compose — ver comentário em `.env.example`), então não devem quebrar a validação.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    semente_aleatoria: int = 42
    mlflow_tracking_uri: str = "http://localhost:5000"
    diretorio_dados_brutos: str = "data/raw_data"
    diretorio_dados_processados: str = "data/processed_data"
    diretorio_modelos: str = "models"


configuracoes = Configuracoes()
