"""Filesystem paths for wokibi-eval jobs (repo-local artifacts)."""

import os
from pathlib import Path

from wokibi_ai.llm_fields import sanitize_model_name

_REPO_ROOT = Path(__file__).resolve().parents[2]

CLINICAL_FOLDER_README_NAME = "README.md"
CLINICAL_FOLDER_SUMMARY_NAME = "experiment_summary.png"


def repo_root() -> Path:
    """Root of the wokibi-eval repository."""
    return _REPO_ROOT


def clinical_detalhe_eval_dir(dataset: str, data_version: str) -> Path:
    """
    Directory for detalhe vs detalhe_{model} evaluation CSVs.

    Default: ``<repo>/results/clinical/<dataset>/<data_version>/``.
    Override base with env ``WOKIBI_EVAL_OUTPUT_ROOT`` (dataset/version appended).
    """
    base = os.getenv("WOKIBI_EVAL_OUTPUT_ROOT")
    root = Path(base) if base else _REPO_ROOT / "results" / "clinical"
    return root / dataset / data_version


def _detalhe_model_basename(model_name: str) -> str:
    return f"detalhe_{sanitize_model_name(model_name)}"


def clinical_detalhe_eval_csv_path(
    dataset: str, data_version: str, model_name: str
) -> Path:
    """Path to metrics CSV for a detalhe vs detalhe_{model} evaluation."""
    return clinical_detalhe_eval_dir(dataset, data_version) / (
        f"{_detalhe_model_basename(model_name)}_eval.csv"
    )


def clinical_detalhe_cohort_path(
    dataset: str, data_version: str, model_name: str
) -> Path:
    """Path to frozen cohort JSONL (original + processed texts per event)."""
    return clinical_detalhe_eval_dir(dataset, data_version) / (
        f"{_detalhe_model_basename(model_name)}_cohort.jsonl"
    )


def clinical_detalhe_experiment_path(
    dataset: str, data_version: str, model_name: str
) -> Path:
    """Path to experiment manifest YAML for a clinical eval run."""
    return clinical_detalhe_eval_dir(dataset, data_version) / (
        f"{_detalhe_model_basename(model_name)}_experiment.yaml"
    )


def clinical_detalhe_folder_summary_image_path(
    dataset: str, data_version: str
) -> Path:
    """Path to shared summary figure PNG for all models (added manually)."""
    return clinical_detalhe_eval_dir(dataset, data_version) / (
        CLINICAL_FOLDER_SUMMARY_NAME
    )


def clinical_detalhe_folder_readme_path(dataset: str, data_version: str) -> Path:
    """Path to shared experiment README for all models in a data_version folder."""
    return clinical_detalhe_eval_dir(dataset, data_version) / (
        CLINICAL_FOLDER_README_NAME
    )
