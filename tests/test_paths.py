"""Tests for repo-local artifact paths."""

from wokibi_eval.paths import (
    CLINICAL_FOLDER_README_NAME,
    clinical_detalhe_cohort_path,
    clinical_detalhe_eval_csv_path,
    clinical_detalhe_eval_dir,
    clinical_detalhe_experiment_path,
    clinical_detalhe_folder_readme_path,
    CLINICAL_FOLDER_SUMMARY_NAME,
    clinical_detalhe_folder_summary_image_path,
    repo_root,
)


def test_clinical_detalhe_eval_dir_default():
    path = clinical_detalhe_eval_dir("nefrologia", "2026-01-01")
    assert path == repo_root() / "results" / "clinical" / "nefrologia" / "2026-01-01"


def test_clinical_detalhe_eval_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WOKIBI_EVAL_OUTPUT_ROOT", str(tmp_path / "custom"))
    path = clinical_detalhe_eval_dir("uti", "2026-08-30")
    assert path == tmp_path / "custom" / "uti" / "2026-08-30"


def test_clinical_detalhe_artifact_paths_default():
    base = clinical_detalhe_eval_dir("uti", "2026-08-30")
    model = "gpt-oss-120b"
    assert clinical_detalhe_eval_csv_path("uti", "2026-08-30", model) == (
        base / "detalhe_gpt_oss_120b_eval.csv"
    )
    assert clinical_detalhe_cohort_path("uti", "2026-08-30", model) == (
        base / "detalhe_gpt_oss_120b_cohort.jsonl"
    )
    assert clinical_detalhe_experiment_path("uti", "2026-08-30", model) == (
        base / "detalhe_gpt_oss_120b_experiment.yaml"
    )
    assert clinical_detalhe_folder_readme_path("uti", "2026-08-30") == (
        base / CLINICAL_FOLDER_README_NAME
    )
    assert clinical_detalhe_folder_summary_image_path("uti", "2026-08-30") == (
        base / CLINICAL_FOLDER_SUMMARY_NAME
    )


def test_clinical_detalhe_artifact_paths_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WOKIBI_EVAL_OUTPUT_ROOT", str(tmp_path / "out"))
    model = "gpt-4o-mini"
    assert clinical_detalhe_cohort_path("nefrologia", "2026-01-01", model) == (
        tmp_path / "out" / "nefrologia" / "2026-01-01" / "detalhe_gpt_4o_mini_cohort.jsonl"
    )
