# wokibi-eval

Repositório de **execução** para avaliação Wokibi (métricas de ranking, UMAP, juiz clínico, jobs batch). Não é publicado no TestPyPI — `poetry install` instala o pacote local **`wokibi_eval`** em modo editável (como libs irmãs no venv).

**Dependências publicadas:** `wokibi-ai`, `wokibi-data`, `cliid` (via lock).

## Estrutura

| Caminho | Conteúdo |
|---------|----------|
| `src/wokibi_eval/evaluation/` | Ranking, UMAP, `ClinicalEvaluator` |
| `src/wokibi_eval/recommender/` | `eval_recommender` |
| `src/wokibi_eval/prompts/` | YAML do juiz clínico |
| `pipelines/clinical/` | Job batch `detalhe` vs `detalhe_{model}` |
| `results/clinical/` | CSVs de eval (`<dataset>/<data_version>/detalhe_*_eval.csv`) |
| `notebooks/recommendation/` | Exploração offline (UMAP, MAP/recall) |
| `notebooks/clinical/` | Tutorial do job `detalhe` vs `detalhe_{model}` |
| `tests/` | `pytest` |

Imports: `from wokibi_eval.evaluation...`, `from wokibi_eval.recommender...` (kernel / `poetry run` = venv deste repo).

## Desenvolvimento

```bash
cd wokibi-eval
cp .env.example .env   # preencha Mongo / GOOGLE_API_KEY (ver .env.example)
poetry install
./scripts/use-local-wokibi.sh link all   # opcional: cliid, wokibi-data, wokibi-ai editáveis
poetry run pytest tests/ -q
```

Clínico (spaCy, sentence-transformers, scikit-learn):

```bash
poetry install --with clinical
python -m spacy download pt_core_news_lg
```

## Job batch (detalhe)

```bash
poetry run python pipelines/clinical/events_detalhe_eval.py \
  --dataset=nefrologia \
  --model_name=gpt-4o-mini \
  --verbose=True \
  --break_on=False \
  --max_events=100
```

`--max_events` limita quantos eventos do Mongo são processados nesta run (use com `--break_on=False`; omita para varrer todos os elegíveis).

Saída (no repo, versionada por `data_version` do parceiro em `wokibi-data`):

`results/clinical/<dataset>/<data_version>/detalhe_<modelo_sanitizado>_eval.csv`

Ex.: `results/clinical/nefrologia/2026-07-05/detalhe_gpt_4o_mini_eval.csv`. Override opcional da base: `WOKIBI_EVAL_OUTPUT_ROOT` no `.env`.

## Notebooks

```bash
poetry install --with dev
poetry install --with clinical   # notebook clinical/1.0-events-detalhe-eval.ipynb
poetry run python -m spacy download pt_core_news_lg
poetry run jupyter lab
```

Use o **kernel do `.venv`** deste projeto (não é preciso cwd na raiz). Índice e passo a passo: [notebooks/README.md](notebooks/README.md) (recommendation + clinical).

## Licença

Wokibi
