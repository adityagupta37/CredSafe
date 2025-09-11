from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def selection_rate(decisions: np.ndarray) -> float:
    return float(np.mean(decisions))


def ks_stat(y_true: np.ndarray, scores: np.ndarray) -> float:
    # Simple KS: max difference between CDFs of good/bad scores
    from scipy.stats import ks_2samp

    good = scores[y_true == 0]
    bad = scores[y_true == 1]
    if good.size == 0 or bad.size == 0:
        return 0.0
    return float(ks_2samp(good, bad).statistic)


def group_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    group: np.ndarray,
    threshold: float,
    groups: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    labels = groups or np.unique(group)
    for g in labels:
        mask = group == g
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        sc = scores[mask]
        dec = (sc < threshold).astype(int)
        try:
            auc = roc_auc_score(yt, sc)
        except Exception:
            auc = float("nan")
        sr = selection_rate(dec)
        out[str(g)] = {"auc": float(auc), "selection_rate": float(sr)}
    # parity deltas
    if out:
        srs = [m["selection_rate"] for m in out.values()]
        out["_parity"] = {"selection_rate_delta_max": float(max(srs) - min(srs))}
    return out

