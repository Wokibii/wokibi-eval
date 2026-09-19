"""Filesystem paths for wokibi-eval jobs (repo-local artifacts)."""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
