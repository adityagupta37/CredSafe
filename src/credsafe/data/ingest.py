from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from credsafe.utils.io import ensure_dir


def ingest(raw_glob: str, processed_dir: str) -> Path:
    files = sorted(glob.glob(raw_glob))
    if not files:
        raise SystemExit(f"No raw files found under {raw_glob}. Place CSVs in data/raw/.")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, axis=0, ignore_index=True)

    out_dir = ensure_dir(processed_dir)
    out_path = out_dir / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/dataset.yaml")
    path = ingest(cfg.dataset.raw_glob, cfg.dataset.processed_dir)
    print(f"Ingested -> {path}")
