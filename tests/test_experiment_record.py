"""Tests for clinical experiment cohort JSONL and manifest YAML."""

import json

import yaml

from wokibi_eval.clinical.experiment_record import (
    append_cohort_record,
    build_folder_experiment_readme_markdown,
    build_manifest,
    discover_eval_models,
    load_cohort_event_ids,
    migrate_eval_folder,
    write_experiment_manifest,
    write_experiment_readme,
    write_folder_experiment_readme,
)
from wokibi_eval.paths import CLINICAL_FOLDER_README_NAME, CLINICAL_FOLDER_SUMMARY_NAME


def test_append_cohort_and_load_ids(tmp_path):
    path = tmp_path / "cohort.jsonl"
    append_cohort_record(
        path,
        {
            "event_id": "1",
            "detalhe": "original",
            "processed": "resumo",
        },
    )
    append_cohort_record(
        path,
        {
            "event_id": "2",
            "detalhe": "b",
            "processed": "c",
        },
    )

    assert load_cohort_event_ids(path) == {"1", "2"}

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["detalhe"] == "original"


def test_write_experiment_manifest_round_trip(tmp_path):
    path = tmp_path / "experiment.yaml"
    manifest = build_manifest(
        dataset="uti",
        data_version="2026-08-30",
        model_name="gpt-oss-120b",
        detalhe_field="detalhe_gpt_oss_120b",
        eval_csv_name="detalhe_gpt_oss_120b_eval.csv",
        cohort_jsonl_name="detalhe_gpt_oss_120b_cohort.jsonl",
        experiment_yaml_name="detalhe_gpt_oss_120b_experiment.yaml",
        enable_judge=True,
        judge_model="gemini-2.5-flash",
        latest_run={
            "break_on": True,
            "cohort_row_count": 3,
            "new_eval_rows": 1,
        },
    )
    write_experiment_manifest(path, manifest)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["dataset"] == "uti"
    assert loaded["artifacts"]["cohort_jsonl"] == "detalhe_gpt_oss_120b_cohort.jsonl"
    assert loaded["artifacts"]["readme_md"] == CLINICAL_FOLDER_README_NAME
    assert loaded["artifacts"]["summary_image"] == CLINICAL_FOLDER_SUMMARY_NAME
    assert loaded["evaluator"]["judge_model"] == "gemini-2.5-flash"
    assert loaded["latest_run"]["cohort_row_count"] == 3
    assert "wokibi_eval" in loaded["package"]


def test_discover_eval_models_reads_model_name_from_yaml(tmp_path):
    (tmp_path / "detalhe_gpt_oss_120b_eval.csv").write_text("x", encoding="utf-8")
    yaml_path = tmp_path / "detalhe_gpt_oss_120b_experiment.yaml"
    write_experiment_manifest(
        yaml_path,
        {"model_name": "gpt-oss-120b", "artifacts": {}},
    )
    models = discover_eval_models(tmp_path)
    assert len(models) == 1
    assert models[0]["model_name"] == "gpt-oss-120b"
    assert "summary_image" not in models[0]


def test_build_folder_experiment_readme_markdown_two_models():
    md = build_folder_experiment_readme_markdown(
        dataset="nefrologia",
        data_version="2026-08-30",
        models=[
            {
                "model_name": "gpt-4o-mini",
                "eval_csv": "detalhe_gpt_4o_mini_eval.csv",
                "cohort_jsonl": "detalhe_gpt_4o_mini_cohort.jsonl",
                "experiment_yaml": "detalhe_gpt_4o_mini_experiment.yaml",
            },
            {
                "model_name": "gpt-oss-120b",
                "eval_csv": "detalhe_gpt_oss_120b_eval.csv",
                "cohort_jsonl": "detalhe_gpt_oss_120b_cohort.jsonl",
                "experiment_yaml": "detalhe_gpt_oss_120b_experiment.yaml",
            },
        ],
    )
    assert "### gpt-4o-mini" in md
    assert "### gpt-oss-120b" in md
    assert f"./{CLINICAL_FOLDER_SUMMARY_NAME}" in md
    assert md.count(f"./{CLINICAL_FOLDER_SUMMARY_NAME}") == 1


def test_write_folder_experiment_readme(tmp_path):
    (tmp_path / "detalhe_a_eval.csv").write_text("", encoding="utf-8")
    path = write_folder_experiment_readme(tmp_path, "uti", "2026-01-01")
    assert path.name == CLINICAL_FOLDER_README_NAME
    assert "uti / 2026-01-01" in path.read_text(encoding="utf-8")


def test_migrate_eval_folder_dry_run_keeps_legacy(tmp_path):
    (tmp_path / "detalhe_a_eval.csv").write_text("", encoding="utf-8")
    legacy = tmp_path / "detalhe_a_README.md"
    legacy.write_text("old", encoding="utf-8")
    yaml_path = tmp_path / "detalhe_a_experiment.yaml"
    write_experiment_manifest(
        yaml_path,
        {"artifacts": {"readme_md": "detalhe_a_README.md"}},
    )

    result = migrate_eval_folder(tmp_path, "uti", "v1", dry_run=True)
    assert result["readme_written"] is True
    assert legacy.exists()
    assert not (tmp_path / CLINICAL_FOLDER_README_NAME).exists()


def test_migrate_eval_folder_writes_and_removes_legacy(tmp_path):
    (tmp_path / "detalhe_a_eval.csv").write_text("", encoding="utf-8")
    legacy = tmp_path / "detalhe_a_README.md"
    legacy.write_text("old", encoding="utf-8")
    yaml_path = tmp_path / "detalhe_a_experiment.yaml"
    write_experiment_manifest(
        yaml_path,
        {"artifacts": {"readme_md": "detalhe_a_README.md"}},
    )

    migrate_eval_folder(tmp_path, "uti", "v1", dry_run=False)
    assert (tmp_path / CLINICAL_FOLDER_README_NAME).exists()
    assert not legacy.exists()
    updated = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert updated["artifacts"]["readme_md"] == CLINICAL_FOLDER_README_NAME
    assert updated["artifacts"]["summary_image"] == CLINICAL_FOLDER_SUMMARY_NAME


def test_write_experiment_readme(tmp_path):
    path = tmp_path / "readme.md"
    write_experiment_readme(path, "# test\n")
    assert path.read_text(encoding="utf-8") == "# test\n"
