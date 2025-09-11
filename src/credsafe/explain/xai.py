from __future__ import annotations

import warnings

import numpy as np
import shap


def compute_shap_values(model, X_sample):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(model.named_steps["clf"], model.named_steps["preproc"])
        sv = explainer(X_sample)
    return sv

