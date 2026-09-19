# Changelog

## Unreleased

### Added

- Job clínico grava `detalhe_<modelo>_cohort.jsonl` (pares congelados) e `detalhe_<modelo>_experiment.yaml` (manifesto) junto ao CSV de métricas; sync preenche cohort a partir do CSV sem reavaliar.
- Job clínico grava `README.md` por pasta; figura única `experiment_summary.png` (todos os modelos no mesmo painel).
- Script `scripts/migrate_clinical_folder_readme.py` para consolidar `detalhe_*_README.md` legados.

### Changed

- Dependências: removidos pins diretos de `numpy` e `pandas` (via `wokibi-ai` / `wokibi-data`).
- Job clínico grava CSVs em `results/clinical/<dataset>/<data_version>/` no repo (não usa mais `CLOUD_PATH`/datalake `gold/`).
- Pacote local **`wokibi_eval`** em `src/` (editable via `poetry install`); `pipelines/` e `notebooks/` permanecem na raiz do repo.
- Imports: `wokibi_eval.evaluation.*`, `wokibi_eval.recommender.*` (sem `sys.path` nos notebooks com kernel do venv).

### Breaking Changes (histórico)

- Repositório de **execução**: não publica no TestPyPI.
- CLI `wokibi-events-detalhe-eval` removido; use `python pipelines/clinical/events_detalhe_eval.py`.
