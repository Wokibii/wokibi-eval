"""Frozen cohort JSONL and experiment YAML for clinical detalhe evaluations."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_SCHEMA_VERSION = 1
EVAL_SCHEMA_VERSION = 2


def load_cohort_event_ids(jsonl_path: str | Path) -> set[str]:
    """Return event_ids already present in a cohort JSONL file."""
    path = Path(jsonl_path)
    if not path.is_file():
        return set()

    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("event_id") is not None:
                ids.add(str(data["event_id"]))
    return ids


def append_cohort_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a single line to the cohort file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def write_experiment_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write or overwrite the experiment manifest YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            manifest,
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def _package_version() -> str:
    try:
        return version("wokibi-eval")
    except PackageNotFoundError:
        return "unknown"


def build_experiment_readme_markdown(
    *,
    dataset: str,
    data_version: str,
    model_name: str,
    experiment_yaml_name: str,
    eval_csv_name: str,
    cohort_jsonl_name: str,
    summary_image_name: str,
) -> str:
    """Build markdown for the per-model experiment README."""
    return "\n".join([
        f"# Experimento clinical — {dataset} / {data_version} / {model_name}",
        "",
        "## Artefatos",
        "",
        f"- Manifesto: [{experiment_yaml_name}](./{experiment_yaml_name})",
        f"- Métricas: [{eval_csv_name}](./{eval_csv_name})",
        f"- Cohort: [{cohort_jsonl_name}](./{cohort_jsonl_name})",
        "",
        "## Resumo visual",
        "",
        f"![Resumo do experimento](./{summary_image_name})",
        "",
        f"_Coloque `{summary_image_name}` nesta pasta para exibir os gráficos._",
        "",
    ])


def write_experiment_readme(path: str | Path, markdown: str) -> None:
    """Write or overwrite the per-model experiment README."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")


def build_manifest(
    *,
    dataset: str,
    data_version: str,
    model_name: str,
    detalhe_field: str,
    eval_csv_name: str,
    cohort_jsonl_name: str,
    experiment_yaml_name: str,
    readme_md_name: str,
    summary_image_name: str,
    enable_judge: bool,
    judge_model: str | None,
    latest_run: dict[str, Any],
) -> dict[str, Any]:
    """Build the experiment manifest dict."""
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "dataset": dataset,
        "data_version": data_version,
        "model_name": model_name,
        "detalhe_field": detalhe_field,
        "artifacts": {
            "eval_csv": eval_csv_name,
            "cohort_jsonl": cohort_jsonl_name,
            "experiment_yaml": experiment_yaml_name,
            "readme_md": readme_md_name,
            "summary_image": summary_image_name,
        },
        "evaluator": {
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "enable_judge": enable_judge,
            "judge_model": judge_model,
        },
        "latest_run": latest_run,
        "package": {
            "wokibi_eval": _package_version(),
        },
    }
