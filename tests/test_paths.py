"""Tests for repo-local artifact paths."""

from wokibi_eval.paths import clinical_detalhe_eval_dir, repo_root


def test_clinical_detalhe_eval_dir_default():
    path = clinical_detalhe_eval_dir("nefrologia", "2026-01-01")
    assert path == repo_root() / "results" / "clinical" / "nefrologia" / "2026-01-01"


def test_clinical_detalhe_eval_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WOKIBI_EVAL_OUTPUT_ROOT", str(tmp_path / "custom"))
    path = clinical_detalhe_eval_dir("uti", "2026-08-30")
    assert path == tmp_path / "custom" / "uti" / "2026-08-30"
