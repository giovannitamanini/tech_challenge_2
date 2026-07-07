# Evidência: instalação limpa em ambiente novo (Etapa 2)

Registro da verificação do item "Verificar que o projeto instala de forma limpa em um ambiente novo (do zero)" do `docs/TASKS.md` (Etapa 2).

## Metodologia

Em vez de depender de um colega rodar isso manualmente, simulei o cenário equivalente: um `git clone` do repositório local para um diretório temporário fora do projeto, contendo **apenas o que está commitado** (nada de `.venv/`, `.env`, `data/` ou qualquer arquivo ignorado pelo git). Isso reproduz exatamente o que qualquer pessoa obteria ao clonar o repositório do zero.

- **Commit testado:** `a450320d18ea2b4712e51cf1087a92da16507fac`
- **Data:** 2026-07-07
- **Ferramentas:** `uv 0.10.6`, `Python 3.13.7` (Windows)

## Passo 1 — `uv sync` em clone limpo

```
$ uv sync
Using CPython 3.13.7 interpreter at: ...\Python313\python.exe
Creating virtual environment at: .venv
Resolved 195 packages in 1ms
Installed 175 packages in 17.80s
 + tech-challenge-recomendacao==0.1.0 (pacote local, editável)
 + torch==2.12.1
 + mlflow==3.14.0
 + dvc==3.67.1
 + scikit-learn==1.9.0
 + pydantic-settings==2.14.2
 ... (177 pacotes no total)

real    0m17.978s
EXIT: 0
```

**Resultado: ✅ sucesso.** `pyproject.toml` + `uv.lock` reproduzem o ambiente completo (produção + dev) sem nenhuma dependência externa "invisível" — tudo que é necessário está declarado e travado no lock file.

## Passo 2 — `scripts/validate_env.py` no clone limpo

Testado em duas condições: sem `.env` (só defaults do `Configuracoes`) e copiando `.env.example` → `.env` (fluxo documentado). Resultado idêntico nos dois casos:

```
[OK] Python 3.13
[OK] Configurações carregadas: semente_aleatoria=42, mlflow_tracking_uri=http://localhost:5000
[ERRO] Diretório de dados brutos não encontrado em 'data/raw_data'.
[ERRO] Diretório de dados processados não encontrado em 'data/processed_data'.
[OK] Diretório de modelos encontrado em 'models'.

Falha na validação do ambiente.
EXIT: 1
```

**Resultado: ⚠️ falha esperada, não é um bug.** `data/` está inteiramente no `.gitignore` (o dataset é grande e será versionado via DVC, não via git — ver `CLAUDE.md`). Um clone novo, hoje, legitimamente não tem `data/raw_data/` nem `data/processed_data/`, porque o pipeline DVC (`dvc init` + remote + `dvc pull`) ainda não foi configurado — isso é o próprio objetivo da Etapa 3. `models/` passou porque ainda é rastreado por um `.gitkeep`.

## Conclusão

O critério de avaliação "Reprodutibilidade" (instalação limpa via `uv sync`) está **satisfeito**: qualquer pessoa que clonar o repositório consegue reproduzir o ambiente de dependências exatamente, do zero, sem passos ocultos. A checagem de diretórios de dados em `validate_env.py` só vai passar de ponta a ponta depois que a Etapa 3 (DVC) disponibilizar o dataset via `dvc pull` — isso fica registrado como trabalho pendente, não como falha da Etapa 2.
