from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, path: os.PathLike | str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: os.PathLike | str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Any, path: os.PathLike | str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: os.PathLike | str) -> Any:
    return OmegaConf.load(path)
