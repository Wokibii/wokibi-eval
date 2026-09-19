import numpy as np

from wokibi_eval.evaluation.embedding import get_umap


def test_get_umap_returns_xy_label_columns():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(30, 8))
    labels = ["a"] * 15 + ["b"] * 15
    df = get_umap(embeddings, labels, n_neighbors=5, min_dist=0.1, random_state=0)
    assert list(df.columns) == ["x", "y", "label"]
    assert len(df) == 30
    assert set(df["label"]) == {"a", "b"}
