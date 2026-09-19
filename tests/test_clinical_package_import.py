"""Ensure clinical submodules load without optional clinical (spacy) dependencies."""

import importlib


def test_import_experiment_record_without_loading_evaluator():
    import wokibi_eval.clinical.experiment_record as experiment_record

    assert experiment_record.discover_eval_models is not None
    # events_detalhe_eval (and ClinicalEvaluator) must not load via package __init__
    clinical_pkg = importlib.import_module("wokibi_eval.clinical")
    assert "EventsDetalheEvaluator" not in clinical_pkg.__dict__
