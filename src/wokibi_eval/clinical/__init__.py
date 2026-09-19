"""Clinical evaluation jobs (detalhe vs detalhe_{model})."""

from typing import Any

__all__ = ["EventsDetalheEvaluator", "main"]


def __getattr__(name: str) -> Any:
    """Lazy imports so lightweight modules (e.g. experiment_record) avoid spacy."""
    if name == "EventsDetalheEvaluator":
        from wokibi_eval.clinical.events_detalhe_eval import EventsDetalheEvaluator

        return EventsDetalheEvaluator
    if name == "main":
        from wokibi_eval.clinical.events_detalhe_eval import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
