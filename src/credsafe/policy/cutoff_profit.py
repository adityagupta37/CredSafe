from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProfitPolicy:
    lgd: float = 0.45
    ead: float = 1.0
    annual_yield: float = 0.24
    servicing_cost: float = 0.02
    threshold_grid_size: int = 200


def expected_profit(
    y_true: np.ndarray, pd: np.ndarray, threshold: float, cfg: ProfitPolicy
) -> tuple[float, dict[str, float]]:
    approve = pd < threshold
    approval_rate = approve.mean() if approve.size else 0.0

    # Expected Loss and Yield are conditional on approval
    pd_approved = pd[approve]
    el = (pd_approved * cfg.lgd * cfg.ead).mean() if pd_approved.size else 0.0
    yield_rate = cfg.annual_yield
    cost = cfg.servicing_cost
    profit_per_loan = yield_rate - el - cost
    expected_profit_value = approval_rate * profit_per_loan
    return expected_profit_value, {
        "approval_rate": float(approval_rate),
        "expected_loss": float(el),
        "profit_per_loan": float(profit_per_loan),
    }


def select_threshold_by_profit(
    y_true: np.ndarray,
    pd: np.ndarray,
    cfg: ProfitPolicy | None = None,
) -> dict[str, float]:
    cfg = cfg or ProfitPolicy()
    grid = np.linspace(0.01, 0.99, cfg.threshold_grid_size)
    best = {"threshold": 0.5, "expected_profit": -np.inf}
    for t in grid:
        ep, details = expected_profit(y_true, pd, t, cfg)
        if ep > best["expected_profit"]:
            best = {
                "threshold": float(t),
                "expected_profit": float(ep),
                **details,
            }
    return best
