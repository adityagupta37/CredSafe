import numpy as np

from credsafe.fairness.metrics import group_metrics


def test_group_metrics_shape():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.8, 0.3, 0.7])
    g = np.array(["A", "A", "B", "B"])
    out = group_metrics(y, s, g, threshold=0.5)
    assert "A" in out and "B" in out
