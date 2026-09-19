# Changelog

## Unreleased

### Added

- Job clínico grava `detalhe_<modelo>_cohort.jsonl` (pares congelados) e `detalhe_<modelo>_experiment.yaml` (manifesto) junto ao CSV de métricas; sync preenche cohort a partir do CSV sem reavaliar.
- Job clínico grava `detalhe_<modelo>_README.md` com link para `detalhe_<modelo>_summary.png` (figura adicionada manualmente).

### Changed

- Dependências: removidos pins diretos de `numpy` e `pandas` (via `wokibi-ai` / `wokibi-data`).
- Job clínico grava CSVs em `results/clinical/<dataset>/<data_version>/` no repo (não usa mais `CLOUD_PATH`/datalake `gold/`).
- Pacote local **`wokibi_eval`** em `src/` (editable via `poetry install`); `pipelines/` e `notebooks/` permanecem na raiz do repo.
- Imports: `wokibi_eval.evaluation.*`, `wokibi_eval.recommender.*` (sem `sys.path` nos notebooks com kernel do venv).

### Breaking Changes (histórico)

- Repositório de **execução**: não publica no TestPyPI.
- CLI `wokibi-events-detalhe-eval` removido; use `python pipelines/clinical/events_detalhe_eval.py`.
