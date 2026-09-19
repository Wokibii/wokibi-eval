"""Frozen cohort JSONL and experiment YAML for clinical detalhe evaluations."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from wokibi_eval.paths import (
    CLINICAL_FOLDER_README_NAME,
    CLINICAL_FOLDER_SUMMARY_NAME,
)

EXPERIMENT_SCHEMA_VERSION = 1
EVAL_SCHEMA_VERSION = 2
_EVAL_CSV_GLOB = "detalhe_*_eval.csv"
_LEGACY_README_GLOB = "detalhe_*_README.md"
_EXPERIMENT_YAML_GLOB = "detalhe_*_experiment.yaml"


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


def _model_name_from_experiment_yaml(yaml_path: Path) -> str | None:
    if not yaml_path.is_file():
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if isinstance(data, dict) and data.get("model_name"):
        return str(data["model_name"])
    return None


def discover_eval_models(eval_dir: str | Path) -> list[dict[str, str]]:
    """List models and artifact filenames from detalhe_*_eval.csv in a folder."""
    root = Path(eval_dir)
    models: list[dict[str, str]] = []
    for csv_path in sorted(root.glob(_EVAL_CSV_GLOB)):
        if not csv_path.is_file():
            continue
        base = csv_path.name[: -len("_eval.csv")]
        experiment_yaml = f"{base}_experiment.yaml"
        yaml_path = root / experiment_yaml
        model_name = _model_name_from_experiment_yaml(yaml_path)
        if not model_name:
            model_name = base.removeprefix("detalhe_")
        models.append({
            "model_name": model_name,
            "eval_csv": csv_path.name,
            "cohort_jsonl": f"{base}_cohort.jsonl",
            "experiment_yaml": experiment_yaml,
        })
    return models


def build_folder_experiment_readme_markdown(
    *,
    dataset: str,
    data_version: str,
    models: list[dict[str, str]],
    summary_image_name: str = CLINICAL_FOLDER_SUMMARY_NAME,
) -> str:
    """Build markdown for the shared experiment README (all models)."""
    lines = [
        f"# Experimento clinical — {dataset} / {data_version}",
        "",
        "Comparação de sumarizações `detalhe` vs `detalhe_{model}` por modelo.",
        "",
        "## Resumo visual",
        "",
        f"![Resumo do experimento](./{summary_image_name})",
        "",
        f"_Coloque `{summary_image_name}` nesta pasta para exibir os gráficos._",
        "",
    ]
    if not models:
        lines.extend([
            "_Nenhum `detalhe_*_eval.csv` encontrado nesta pasta._",
            "",
        ])
        return "\n".join(lines)

    lines.append("## Modelos")
    lines.append("")
    for entry in models:
        model_name = entry["model_name"]
        lines.extend([
            f"### {model_name}",
            "",
            f"- Manifesto: [{entry['experiment_yaml']}](./{entry['experiment_yaml']})",
            f"- Métricas: [{entry['eval_csv']}](./{entry['eval_csv']})",
            f"- Cohort: [{entry['cohort_jsonl']}](./{entry['cohort_jsonl']})",
            "",
        ])
    return "\n".join(lines)


def write_experiment_readme(path: str | Path, markdown: str) -> None:
    """Write or overwrite an experiment README file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")


def write_folder_experiment_readme(
    eval_dir: str | Path,
    dataset: str,
    data_version: str,
) -> Path:
    """Discover models in eval_dir and write the shared README.md."""
    models = discover_eval_models(eval_dir)
    markdown = build_folder_experiment_readme_markdown(
        dataset=dataset,
        data_version=data_version,
        models=models,
    )
    readme_path = Path(eval_dir) / CLINICAL_FOLDER_README_NAME
    write_experiment_readme(readme_path, markdown)
    return readme_path


def update_experiment_yaml_artifact_refs(yaml_path: Path) -> bool:
    """Set shared readme_md and summary_image; return True if file was updated."""
    if not yaml_path.is_file():
        return False
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        data["artifacts"] = artifacts
    changed = False
    if artifacts.get("readme_md") != CLINICAL_FOLDER_README_NAME:
        artifacts["readme_md"] = CLINICAL_FOLDER_README_NAME
        changed = True
    if artifacts.get("summary_image") != CLINICAL_FOLDER_SUMMARY_NAME:
        artifacts["summary_image"] = CLINICAL_FOLDER_SUMMARY_NAME
        changed = True
    if not changed:
        return False
    write_experiment_manifest(yaml_path, data)
    return True


def migrate_eval_folder(
    eval_dir: str | Path,
    dataset: str,
    data_version: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Regenerate folder README, fix YAML refs, remove legacy per-model READMEs."""
    root = Path(eval_dir)
    result: dict[str, Any] = {
        "path": str(root),
        "dataset": dataset,
        "data_version": data_version,
        "readme_written": False,
        "yamls_updated": 0,
        "legacy_readmes_removed": [],
    }
    if not list(root.glob(_EVAL_CSV_GLOB)):
        return result

    if dry_run:
        result["readme_written"] = True
    else:
        write_folder_experiment_readme(root, dataset, data_version)
        result["readme_written"] = True

    for yaml_path in sorted(root.glob(_EXPERIMENT_YAML_GLOB)):
        if dry_run:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                artifacts = data.get("artifacts") or {}
                if (
                    artifacts.get("readme_md") != CLINICAL_FOLDER_README_NAME
                    or artifacts.get("summary_image") != CLINICAL_FOLDER_SUMMARY_NAME
                ):
                    result["yamls_updated"] += 1
        elif update_experiment_yaml_artifact_refs(yaml_path):
            result["yamls_updated"] += 1

    for legacy in sorted(root.glob(_LEGACY_README_GLOB)):
        result["legacy_readmes_removed"].append(legacy.name)
        if not dry_run:
            legacy.unlink()

    return result


def build_manifest(
    *,
    dataset: str,
    data_version: str,
    model_name: str,
    detalhe_field: str,
    eval_csv_name: str,
    cohort_jsonl_name: str,
    experiment_yaml_name: str,
    enable_judge: bool,
    judge_model: str | None,
    latest_run: dict[str, Any],
    readme_md_name: str = CLINICAL_FOLDER_README_NAME,
    summary_image_name: str = CLINICAL_FOLDER_SUMMARY_NAME,
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
