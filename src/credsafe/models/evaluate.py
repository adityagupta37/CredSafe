from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from credsafe.fairness.metrics import group_metrics, ks_stat
from credsafe.policy.cutoff_profit import ProfitPolicy, select_threshold_by_profit
from credsafe.utils.io import save_json


def main() -> None:
    dcfg = OmegaConf.load("configs/dataset.yaml").dataset
    pcfg = OmegaConf.load("configs/policy.yaml").policy
    mcfg = OmegaConf.load("configs/model_pd.yaml")

    df = pd.read_parquet(Path(dcfg.processed_dir) / "dataset.parquet")
    y = df[dcfg.target_column].astype(int).values
    model = joblib.load(mcfg.artifacts.model)

    # Predict PDs
    X = df[[c for c in df.columns if c not in {dcfg.target_column, dcfg.id_column}]]
    pd_hat = model.predict_proba(X)[:, 1]

    # Metrics
    auc = float(roc_auc_score(y, pd_hat))
    ks = float(ks_stat(y, pd_hat))
    policy = ProfitPolicy(**pcfg)
    policy_sel = select_threshold_by_profit(y, pd_hat, policy)

    metrics: dict[str, float] = {
        "auc": auc,
        "ks": ks,
        **{f"policy_{k}": v for k, v in policy_sel.items()},
    }

    # Optional fairness if a common sensitive column exists
    for cand in ["gender", "Gender", "sex", "Sex"]:
        if cand in df.columns:
            gm = group_metrics(
                y_true=y,
                scores=pd_hat,
                group=df[cand].astype(str).values,
                threshold=policy_sel["threshold"],
            )
            metrics["fairness_group"] = cand
            metrics["fairness_metrics"] = gm
            break

    save_json(metrics, mcfg.artifacts.metrics)
    print("Saved metrics ->", mcfg.artifacts.metrics)


if __name__ == "__main__":
    main()
