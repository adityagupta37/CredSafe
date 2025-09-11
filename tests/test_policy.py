from credsafe.policy.cutoff_profit import ProfitPolicy, select_threshold_by_profit
import numpy as np


def test_select_threshold_runs():
    y = np.array([0, 1, 0, 1, 0, 0, 1])
    pd = np.linspace(0.1, 0.9, len(y))
    out = select_threshold_by_profit(y, pd, ProfitPolicy())
    assert 0.0 < out["threshold"] < 1.0

