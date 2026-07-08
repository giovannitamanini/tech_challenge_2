# Tech Challenge — Sistema de Recomendação

Sistema de recomendação de produtos baseado no comportamento de navegação dos usuários, desenvolvido para o Tech Challenge da PosTech. O modelo central é uma rede neural (MLP/embedding-based) treinada com PyTorch, com pipeline de dados versionado via DVC e experimentos rastreados no MLflow.

> Este README cobre a configuração do ambiente de desenvolvimento. As instruções completas de uso (treino, pipeline DVC, Docker) serão adicionadas conforme as próximas etapas do projeto forem concluídas — ver o checklist completo em [`docs/TASKS.md`](docs/TASKS.md).

## Pré-requisitos

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) — gerenciador de dependências e ambiente do projeto

## Como obter o projeto

```bash
git clone <url-do-repositorio>
cd tech_challenge_2
```

## Instalação das dependências

```bash
uv sync
```

Esse comando cria o ambiente virtual em `.venv/` e instala todas as dependências de produção e de desenvolvimento, travadas em `uv.lock`. Deve funcionar de forma limpa em qualquer máquina, sem passos ocultos.

Em seguida, instale o git hook de linting (roda automaticamente a cada commit; precisa ser feito uma vez por clone, pois `.git/hooks/` não é versionado):

```bash
uv run pre-commit install
```

## Configuração

As configurações do projeto são lidas de um arquivo `.env` (nunca commitado). Copie o exemplo e ajuste se necessário — os valores padrão já funcionam para desenvolvimento local:

```bash
cp .env.example .env
```

## Linting

```bash
uv run ruff check .
```

Deve rodar sem nenhum erro. Para aplicar formatação automática:

```bash
uv run ruff format .
```

Para rodar manualmente todos os hooks configurados (equivalente ao que roda em cada commit):

```bash
uv run pre-commit run --all-files
```

## Validação do ambiente

```bash
uv run python scripts/validate_env.py
```

Verifica a versão do Python, o carregamento correto das configurações (`.env`/Pydantic Settings) e a existência dos diretórios de dados e modelos. Termina com código de saída `0` em caso de sucesso.

> **Nota:** como o dataset (`data/`) é versionado via DVC e ainda não via git, um clone novo do repositório não terá `data/raw_data/` nem `data/processed_data/` até que o pipeline DVC seja configurado (Etapa 3) — nesse caso o script indicará essas pastas como não encontradas. Isso é esperado no estágio atual do projeto.

## Estrutura do projeto

```
src/tech_challenge_recomendacao/  # código-fonte do pacote
tests/                            # testes automatizados
data/raw_data/                    # dataset bruto (MovieLens ml-32m, versionado via DVC)
data/processed_data/              # dados processados pelo pipeline
models/                           # modelos treinados
configs/                          # arquivos de configuração declarativos (ex.: DVC params.yaml)
scripts/                          # scripts utilitários
docs/                             # documentação e checklist de tarefas do projeto
```
