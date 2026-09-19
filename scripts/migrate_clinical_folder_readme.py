"""Migrate per-model clinical READMEs to a single folder README.md."""

from __future__ import annotations

import logging
from pathlib import Path

import fire

from wokibi_eval.clinical.experiment_record import migrate_eval_folder
from wokibi_eval.paths import repo_root

logger = logging.getLogger(__name__)


def migrate_all(
    root: str = "results/clinical",
    dry_run: bool = False,
) -> list[dict]:
    """Regenerate README.md and remove detalhe_*_README.md under each eval folder."""
    base = Path(root)
    if not base.is_absolute():
        base = repo_root() / base

    if not base.is_dir():
        raise FileNotFoundError(f"Clinical results root not found: {base}")

    results: list[dict] = []
    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for version_dir in sorted(dataset_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            if not list(version_dir.glob("detalhe_*_eval.csv")):
                continue
            summary = migrate_eval_folder(
                version_dir,
                dataset_dir.name,
                version_dir.name,
                dry_run=dry_run,
            )
            results.append(summary)
            logger.info(
                "%s readme=%s yamls_updated=%s legacy_removed=%s",
                version_dir,
                summary["readme_written"],
                summary["yamls_updated"],
                summary["legacy_readmes_removed"],
            )
    return results


def main(root: str = "results/clinical", dry_run: bool = False) -> None:
    migrate_all(root=root, dry_run=dry_run)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    fire.Fire(main)
