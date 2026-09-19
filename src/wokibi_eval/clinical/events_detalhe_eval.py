"""Evaluate generated detalhe_{model} fields against the original detalhe."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from wokibi_ai.llm import LLM
from wokibi_ai.llm_fields import detalhe_field_name
from wokibi_data.beans.mongo.event import EventBean
from wokibi_data.config import LLM_MODEL, partner_parameters, partners_datasets

from wokibi_eval.clinical.experiment_record import (
    append_cohort_record,
    build_experiment_readme_markdown,
    build_manifest,
    load_cohort_event_ids,
    write_experiment_manifest,
    write_experiment_readme,
)
from wokibi_eval.evaluation.clinical_evaluator import ClinicalEvaluator
from wokibi_eval.paths import (
    clinical_detalhe_cohort_path,
    clinical_detalhe_eval_csv_path,
    clinical_detalhe_eval_dir,
    clinical_detalhe_experiment_path,
    clinical_detalhe_readme_path,
    clinical_detalhe_summary_image_path,
)

logger = logging.getLogger(__name__)


def _normalize_model_names(model_name: str | list | tuple) -> list[str]:
    items = (
        model_name
        if isinstance(model_name, (list, tuple))
        else str(model_name).split(",")
    )
    return [m.strip() for m in items if str(m).strip()]


class EventsDetalheEvaluator:
    """Evaluate existing detalhe_{model} pairs and append rows under results/clinical/."""

    def __init__(
        self,
        dataset: str,
        model_names: list[str],
        enable_judge: bool = True,
        judge_model: str | None = None,
    ):
        if not model_names:
            raise ValueError("model_names must contain at least one model")

        self.dataset = dataset
        self.model_names = model_names
        self.enable_judge = enable_judge
        resolved_judge = judge_model or LLM_MODEL
        self.judge_model = resolved_judge
        self.detalhe_fields = {
            name: detalhe_field_name(name) for name in model_names
        }
        self.event_bean = EventBean(dataset)

        judge_llm = None
        if enable_judge:
            judge_llm = LLM(resolved_judge)
            judge_llm.start_llm()
        self.evaluator = ClinicalEvaluator(
            judge_llm=judge_llm,
            enable_judge=enable_judge,
        )

        self.data_version = partner_parameters[dataset]["data_version"]
        eval_root = clinical_detalhe_eval_dir(dataset, self.data_version)
        self.eval_dir = str(eval_root)
        self.eval_paths = {
            name: str(clinical_detalhe_eval_csv_path(dataset, self.data_version, name))
            for name in model_names
        }
        self.cohort_paths = {
            name: str(clinical_detalhe_cohort_path(dataset, self.data_version, name))
            for name in model_names
        }
        self.experiment_paths = {
            name: str(
                clinical_detalhe_experiment_path(dataset, self.data_version, name)
            )
            for name in model_names
        }
        self._seen_ids = {
            name: self._load_seen_event_ids(path)
            for name, path in self.eval_paths.items()
        }
        self._seen_cohort_ids = {
            name: load_cohort_event_ids(path)
            for name, path in self.cohort_paths.items()
        }
        self._new_eval_rows: dict[str, int] = {name: 0 for name in model_names}
        self._run_started_at: str | None = None
        self._run_params: dict[str, object] = {}
        logger.info(
            "Evaluating %s for dataset=%s (output=%s, judge=%s, already_done=%s)",
            list(self.detalhe_fields.values()),
            dataset,
            self.eval_dir,
            self.judge_model if enable_judge else None,
            {name: len(ids) for name, ids in self._seen_ids.items()},
        )

    def _load_seen_event_ids(self, eval_path: str) -> set:
        if not os.path.exists(eval_path):
            return set()
        df = pd.read_csv(eval_path, usecols=["event_id"])
        return set(df["event_id"].astype(str))

    def _append_evaluation_row(
        self,
        model_name: str,
        event_id,
        detalhe: str,
        processed: str,
        tipo_evento: str | None = None,
    ) -> None:
        """Evaluate pair and append one CSV row under results/clinical/."""
        row = self.evaluator.evaluate(detalhe, processed)
        evaluated_at = datetime.now(timezone.utc).isoformat()
        row.update({
            "event_id": event_id,
            "dataset": self.dataset,
            "model_name": model_name,
            "detalhe_field": self.detalhe_fields[model_name],
            "judge_model": self.judge_model,
            "tipo_evento": tipo_evento,
            "evaluated_at": evaluated_at,
            "eval_schema_version": 2,
        })

        eval_path = self.eval_paths[model_name]
        os.makedirs(self.eval_dir, exist_ok=True)
        write_header = not os.path.exists(eval_path)
        pd.DataFrame([row]).to_csv(
            eval_path, mode="a", header=write_header, index=False
        )
        self._new_eval_rows[model_name] += 1

        event_id_str = str(event_id)
        if event_id_str not in self._seen_cohort_ids[model_name]:
            self._append_cohort_row(
                model_name,
                event_id_str,
                detalhe,
                processed,
                tipo_evento=tipo_evento,
                evaluated_at=evaluated_at,
            )

    def _append_cohort_row(
        self,
        model_name: str,
        event_id: str,
        detalhe: str,
        processed: str,
        *,
        tipo_evento: str | None,
        evaluated_at: str | None,
    ) -> None:
        append_cohort_record(
            self.cohort_paths[model_name],
            {
                "event_id": event_id,
                "tipo_evento": tipo_evento,
                "detalhe": detalhe,
                "processed": processed,
                "model_name": model_name,
                "detalhe_field": self.detalhe_fields[model_name],
                "evaluated_at": evaluated_at,
            },
        )
        self._seen_cohort_ids[model_name].add(event_id)

    def _sync_cohort_from_eval_csv(self, model_name: str, verbose: bool = False) -> int:
        """Backfill cohort JSONL rows for event_ids already in the eval CSV."""
        eval_path = self.eval_paths[model_name]
        if not os.path.exists(eval_path):
            return 0

        df = pd.read_csv(eval_path)
        if "event_id" not in df.columns:
            return 0

        field = self.detalhe_fields[model_name]
        added = 0
        for _, csv_row in df.iterrows():
            event_id = str(csv_row["event_id"])
            if event_id in self._seen_cohort_ids[model_name]:
                continue

            event = self.event_bean.getEventById(self.dataset, event_id)
            if not event:
                if verbose:
                    logger.warning(
                        "Cohort sync: event %s not found in Mongo", event_id
                    )
                continue

            event_dict = event.to_dict()
            detalhe = event_dict.get("detalhe")
            processed = event_dict.get(field)
            if not detalhe or not processed:
                if verbose:
                    logger.warning(
                        "Cohort sync: empty detalhe or %s for event %s",
                        field,
                        event_id,
                    )
                continue

            tipo = csv_row.get("tipo_evento")
            if pd.isna(tipo):
                tipo = event_dict.get("tipo_evento")
            tipo_evento = str(tipo) if tipo is not None and not pd.isna(tipo) else None

            evaluated_at = csv_row.get("evaluated_at")
            if pd.isna(evaluated_at):
                evaluated_at = None
            else:
                evaluated_at = str(evaluated_at)

            self._append_cohort_row(
                model_name,
                event_id,
                detalhe,
                processed,
                tipo_evento=tipo_evento,
                evaluated_at=evaluated_at,
            )
            added += 1
        return added

    def _write_experiment_manifest(
        self,
        model_name: str,
        finished_at: str,
    ) -> None:
        eval_path = Path(self.eval_paths[model_name])
        cohort_path = Path(self.cohort_paths[model_name])
        experiment_path = Path(self.experiment_paths[model_name])
        readme_path = clinical_detalhe_readme_path(
            self.dataset, self.data_version, model_name
        )
        summary_image_path = clinical_detalhe_summary_image_path(
            self.dataset, self.data_version, model_name
        )

        readme_md = build_experiment_readme_markdown(
            dataset=self.dataset,
            data_version=self.data_version,
            model_name=model_name,
            experiment_yaml_name=experiment_path.name,
            eval_csv_name=eval_path.name,
            cohort_jsonl_name=cohort_path.name,
            summary_image_name=summary_image_path.name,
        )
        write_experiment_readme(readme_path, readme_md)

        latest_run = {
            **self._run_params,
            "started_at": self._run_started_at,
            "finished_at": finished_at,
            "new_eval_rows": self._new_eval_rows[model_name],
            "cohort_row_count": len(self._seen_cohort_ids[model_name]),
        }
        manifest = build_manifest(
            dataset=self.dataset,
            data_version=self.data_version,
            model_name=model_name,
            detalhe_field=self.detalhe_fields[model_name],
            eval_csv_name=eval_path.name,
            cohort_jsonl_name=cohort_path.name,
            experiment_yaml_name=experiment_path.name,
            readme_md_name=readme_path.name,
            summary_image_name=summary_image_path.name,
            enable_judge=self.enable_judge,
            judge_model=self.judge_model if self.enable_judge else None,
            latest_run=latest_run,
        )
        write_experiment_manifest(experiment_path, manifest)

    def run(
        self,
        verbose: bool = False,
        break_on: bool = True,
        tipo_evento: str | list[str] | None = None,
        max_events: int | None = None,
    ) -> None:
        """Evaluate events that already have every requested detalhe_{model}."""
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be >= 1 when set")

        self._run_started_at = datetime.now(timezone.utc).isoformat()
        self._run_params = {
            "tipo_evento": tipo_evento,
            "max_events": max_events,
            "break_on": break_on,
        }
        for name in self.model_names:
            self._new_eval_rows[name] = 0

        fields = list(self.detalhe_fields.values())
        logger.info(
            "Evaluating %s for eligible aligned events (max_events=%s).",
            fields,
            max_events,
        )

        skip = 0
        events_seen = 0
        done = False
        while True:
            batch_size = 1 if break_on else 100
            if max_events is not None:
                remaining = max_events - events_seen
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)

            events = self.event_bean.getEventsByDetalheField(
                self.dataset,
                fields,
                missing=False,
                limit=batch_size,
                skip=skip,
                tipo_evento=tipo_evento,
            )
            if not events:
                break

            for event in tqdm(events):
                if max_events is not None and events_seen >= max_events:
                    done = True
                    break

                event_dict = event.to_dict()
                event_id = str(event_dict.get("id"))
                detalhe = event_dict.get("detalhe")
                event_tipo = (
                    tipo_evento if isinstance(tipo_evento, str) else None
                ) or event_dict.get("tipo_evento")

                for model_name in self.model_names:
                    field = self.detalhe_fields[model_name]
                    if event_id in self._seen_ids[model_name]:
                        if verbose:
                            logger.info(
                                "Skipping already evaluated event: %s (%s)",
                                event_id,
                                field,
                            )
                        continue

                    processed = event_dict.get(field)
                    if not detalhe or not processed:
                        if verbose:
                            logger.warning(
                                "Skipping event %s: empty detalhe or %s",
                                event_id,
                                field,
                            )
                        continue

                    if verbose:
                        logger.info("Evaluating event: %s (%s)", event_id, field)

                    self._append_evaluation_row(
                        model_name,
                        event_id,
                        detalhe,
                        processed,
                        tipo_evento=event_tipo,
                    )
                    self._seen_ids[model_name].add(event_id)

                events_seen += 1

            skip += len(events)
            if break_on or done:
                break

        finished_at = datetime.now(timezone.utc).isoformat()
        for model_name in self.model_names:
            synced = self._sync_cohort_from_eval_csv(model_name, verbose=verbose)
            if synced and verbose:
                logger.info(
                    "Cohort sync: appended %s rows for model=%s",
                    synced,
                    model_name,
                )
            self._write_experiment_manifest(model_name, finished_at)


def main(
    dataset: str,
    model_name: str | list | tuple,
    verbose: bool = False,
    break_on: bool = True,
    tipo_evento: str | list[str] | None = None,
    enable_judge: bool = True,
    judge_model: str | None = None,
    max_events: int | None = None,
):
    """Avalia pares detalhe vs detalhe_{model} e grava CSV em results/clinical/.

    Pré-requisitos: Mongo com extra_variables.detalhe e detalhe_{model} preenchidos
    para cada model_name (cohort alinhado quando há vários modelos).

    Args:
        dataset: Parceiro/coleção (ex. nefrologia, uti). Em código, ``dataset=""``
            processa todos os ``partners_datasets``.
        model_name: Um ou mais IDs de modelo. Aceita string única, vários separados
            por vírgula, ou lista/tupla (API Python).
        verbose: Logs por evento (skip, warning, eval).
        break_on: Se True, um batch só (1 evento com break_on padrão) e encerra.
            Se False, pagina em lotes de 100 até acabar ou atingir max_events.
        tipo_evento: Filtro opcional no Mongo. Omitir = todos os tipos elegíveis.
            String = tipo exato; lista = qualquer um dos tipos ($in).
        enable_judge: Se True, usa LLM juiz. Se False, só métricas locais.
        judge_model: ID do modelo juiz (``wokibi_ai.llm.LLM``). Omitir = ``LLM_MODEL``
            do wokibi-data.
        max_events: Máximo de eventos visitados nesta run (1 por documento).
            None = sem teto. Use com break_on=False para amostra limitada.

    CLI (cwd = raiz wokibi-eval):

        # Smoke: 1 evento
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=nefrologia --model_name=gpt-4o-mini

        # Amostra de 100 eventos
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=uti --model_name=gpt-oss-120b \\
          --break_on=False --max_events=100

        # Vários modelos (mesmo evento precisa ter todos os detalhe_{model})
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=nefrologia \\
          --model_name=gpt-4o-mini,gpt-oss-120b \\
          --break_on=False --max_events=50

        # Filtro por tipo (aspas no shell se houver espaços)
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=uti \\
          --tipo_evento='evolução clínica (internação)' \\
          --model_name=gpt-oss-120b --break_on=False

        # Vários tipos (JSON no Fire)
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=uti \\
          --tipo_evento='["anamnese (internação)","evolução clínica (internação)"]' \\
          --model_name=gpt-oss-120b --break_on=False

        # Sem juiz LLM
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=nefrologia --model_name=gpt-4o-mini --enable_judge=False

        # Juiz com modelo explícito
        poetry run python pipelines/clinical/events_detalhe_eval.py \\
          --dataset=uti --model_name=gpt-oss-120b \\
          --judge_model=gemini-2.5-flash --break_on=True

    Python::

        main(dataset="nefrologia", model_name="gpt-4o-mini", break_on=True)
        main(
            dataset="uti",
            model_name=["gpt-4o-mini", "gpt-oss-120b"],
            tipo_evento=["evolução clínica (internação)", "anamnese (internação)"],
            break_on=False,
            max_events=100,
            verbose=True,
        )

    Saída em ``results/clinical/<dataset>/<data_version>/``:

    - ``detalhe_<modelo>_eval.csv`` (métricas)
    - ``detalhe_<modelo>_cohort.jsonl`` (textos congelados por evento)
    - ``detalhe_<modelo>_experiment.yaml`` (manifesto da run)
    - ``detalhe_<modelo>_README.md`` (índice; referencia ``detalhe_<modelo>_summary.png``)

    ``data_version`` = ``partner_parameters[dataset]`` no wokibi-data.
    """
    model_names = _normalize_model_names(model_name)

    def process_dataset(ds_name: str):
        logger.info(
            "Evaluating %s for dataset: %s",
            [detalhe_field_name(name) for name in model_names],
            ds_name,
        )
        EventsDetalheEvaluator(
            ds_name,
            model_names,
            enable_judge=enable_judge,
            judge_model=judge_model,
        ).run(
            verbose=verbose,
            break_on=break_on,
            tipo_evento=tipo_evento,
            max_events=max_events,
        )

    if bool(dataset):
        process_dataset(dataset)
    else:
        for ds_name in partners_datasets.keys():
            process_dataset(ds_name)
