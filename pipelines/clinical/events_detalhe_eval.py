"""CLI entrypoint for clinical detalhe evaluation."""

import logging

import fire

from wokibi_eval.clinical.events_detalhe_eval import main

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
)

if __name__ == "__main__":
    fire.Fire(main)

"""
Execução manual (cwd = raiz wokibi-eval)

Pipeline CLI (equivalente ao notebook notebooks/clinical/1.0-events-detalhe-eval.ipynb):

poetry run python pipelines/clinical/events_detalhe_eval.py \
  --dataset=nefrologia \
  --tipo_evento='evolução clínica (internação)' \
  --model_name=gpt-oss-120b,gpt-4o-mini,gemini-2.5-flash \
  --judge_model=gemini-flash-lite-latest \
  --verbose=True \
  --break_on=False \
  --max_events=33

Smoke (1 evento):

poetry run python pipelines/clinical/events_detalhe_eval.py \
  --dataset=nefrologia --model_name=gpt-4o-mini

Sem juiz LLM (só métricas locais):

poetry run python pipelines/clinical/events_detalhe_eval.py \
  --dataset=nefrologia --model_name=gpt-4o-mini --enable_judge=False

Referência:
  model_name: gpt-4o-mini; gpt-oss-120b
  tipo_evento: evolução clínica (internação); anamnese (internação); evolução clínica (ambulatório)
  Saída: results/clinical/<dataset>/<data_version>/detalhe_<modelo>_eval.csv,
         detalhe_<modelo>_cohort.jsonl, detalhe_<modelo>_experiment.yaml,
         detalhe_<modelo>_README.md (ref.: detalhe_<modelo>_summary.png manual)

Notebook interativo: poetry run jupyter lab → notebooks/clinical/1.0-events-detalhe-eval.ipynb
"""
