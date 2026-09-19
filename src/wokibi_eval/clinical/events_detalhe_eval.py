"""Evaluate generated detalhe_{model} fields against the original detalhe."""

import logging
import os
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm
from wokibi_ai.llm import LLM
from wokibi_ai.llm_fields import detalhe_field_name, sanitize_model_name
from wokibi_data.beans.mongo.event import EventBean
from wokibi_data.config import LLM_MODEL, partner_parameters, partners_datasets

from wokibi_eval.evaluation.clinical_evaluator import ClinicalEvaluator
from wokibi_eval.paths import clinical_detalhe_eval_dir

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

        version = partner_parameters[dataset]["data_version"]
        self.eval_dir = str(clinical_detalhe_eval_dir(dataset, version))
        self.eval_paths = {
            name: os.path.join(
                self.eval_dir,
                f"detalhe_{sanitize_model_name(name)}_eval.csv",
            )
            for name in model_names
        }
        self._seen_ids = {
            name: self._load_seen_event_ids(path)
            for name, path in self.eval_paths.items()
        }
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
        row.update({
            "event_id": event_id,
            "dataset": self.dataset,
            "model_name": model_name,
            "detalhe_field": self.detalhe_fields[model_name],
            "judge_model": self.judge_model,
            "tipo_evento": tipo_evento,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "eval_schema_version": 2,
        })

        eval_path = self.eval_paths[model_name]
        os.makedirs(self.eval_dir, exist_ok=True)
        write_header = not os.path.exists(eval_path)
        pd.DataFrame([row]).to_csv(
            eval_path, mode="a", header=write_header, index=False
        )

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

    Saída: ``results/clinical/<dataset>/<data_version>/detalhe_<modelo>_eval.csv``
    (``data_version`` = ``partner_parameters[dataset]`` no wokibi-data).
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
