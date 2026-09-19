"""Tests for clinical experiment cohort JSONL and manifest YAML."""

import json

import yaml

from wokibi_eval.clinical.experiment_record import (
    append_cohort_record,
    build_experiment_readme_markdown,
    build_manifest,
    load_cohort_event_ids,
    write_experiment_manifest,
    write_experiment_readme,
)


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
        readme_md_name="detalhe_gpt_oss_120b_README.md",
        summary_image_name="detalhe_gpt_oss_120b_summary.png",
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
    assert loaded["artifacts"]["readme_md"] == "detalhe_gpt_oss_120b_README.md"
    assert loaded["artifacts"]["summary_image"] == "detalhe_gpt_oss_120b_summary.png"
    assert loaded["evaluator"]["judge_model"] == "gemini-2.5-flash"
    assert loaded["latest_run"]["cohort_row_count"] == 3
    assert "wokibi_eval" in loaded["package"]


def test_build_experiment_readme_markdown_links_and_image():
    md = build_experiment_readme_markdown(
        dataset="uti",
        data_version="2026-08-30",
        model_name="gpt-oss-120b",
        experiment_yaml_name="detalhe_gpt_oss_120b_experiment.yaml",
        eval_csv_name="detalhe_gpt_oss_120b_eval.csv",
        cohort_jsonl_name="detalhe_gpt_oss_120b_cohort.jsonl",
        summary_image_name="detalhe_gpt_oss_120b_summary.png",
    )
    assert "./detalhe_gpt_oss_120b_summary.png" in md
    assert "./detalhe_gpt_oss_120b_eval.csv" in md
    assert "./detalhe_gpt_oss_120b_cohort.jsonl" in md
    assert "./detalhe_gpt_oss_120b_experiment.yaml" in md


def test_write_experiment_readme(tmp_path):
    path = tmp_path / "readme.md"
    write_experiment_readme(path, "# test\n")
    assert path.read_text(encoding="utf-8") == "# test\n"
