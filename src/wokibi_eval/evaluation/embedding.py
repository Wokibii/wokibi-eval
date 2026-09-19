"""UMAP projection and scatter plots for embedding vectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import umap


def get_umap(
    embeddings: np.ndarray | list[list[float]],
    labels: np.ndarray | list[Any],
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project high-dimensional embeddings to 2D with UMAP."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(matrix)
    return pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "label": list(labels)},
    )


def plot_umap(
    df: pd.DataFrame,
    *,
    with_labels: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 8),
):
    """Scatter plot of a UMAP dataframe produced by :func:`get_umap`."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        ax.scatter(
            subset["x"],
            subset["y"],
            label=str(label),
            alpha=0.7,
            s=20,
        )
    if with_labels:
        ax.legend()
    if title:
        ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()
    return fig
