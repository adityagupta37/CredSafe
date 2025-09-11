from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from credsafe.data.preprocess import build_preprocess_pipeline
from credsafe.utils.io import ensure_dir, save_json, save_pickle


def train_pd(
    df: pd.DataFrame,
    target: str,
    id_column: str | None,
    cfg: Dict,
) -> Tuple[Pipeline, Dict[str, float]]:
    preproc, feature_cols = build_preprocess_pipeline(df, target, id_column)

    X = df[feature_cols]
    y = df[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg["random_state"]
    )

    base = LogisticRegression(max_iter=200, class_weight=cfg.get("class_weight", None))
    if cfg.get("calibration", "isotonic") in {"isotonic", "platt"}:
        method = "isotonic" if cfg["calibration"] == "isotonic" else "sigmoid"
        clf = CalibratedClassifierCV(base, method=method, cv=StratifiedKFold(n_splits=3))
    else:
        clf = base

    pipe = Pipeline([("preproc", preproc), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Evaluate
    p_test = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))
    metrics = {"auc": auc, "brier": brier}
    return pipe, metrics


def main() -> None:
    dcfg = OmegaConf.load("configs/dataset.yaml").dataset
    mcfg = OmegaConf.load("configs/model_pd.yaml")
    art = mcfg.artifacts

    ds_path = Path(dcfg.processed_dir) / "dataset.parquet"
    if not ds_path.exists():
        raise SystemExit("Processed dataset not found. Run `make data` first.")
    df = pd.read_parquet(ds_path)

    model, metrics = train_pd(df, dcfg.target_column, dcfg.id_column, mcfg.model_pd)

    ensure_dir(Path(art.dir))
    joblib.dump(model, art.model)
    save_json(metrics, art.metrics)
    save_pickle(model.named_steps["preproc"], art.preproc)
    print(f"Saved model to {art.model}; metrics: {metrics}")


if __name__ == "__main__":
    main()

