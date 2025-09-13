# CredSafe

CredSafe is a production‑grade credit risk scoring and segmentation project with a Streamlit app, explainability, fairness checks, and lightweight MLOps.

This repository provides:
- A PD (Probability of Default) model pipeline (baseline Logistic Regression; hooks for LightGBM/CatBoost and calibration).
- A profit‑aware cutoff policy, segmentation, and fairness metrics.
- A Streamlit app (risk desk) with pages for Overview, Applicant Scoring, Segments, Fairness, Explainability, and Model Runs.
- MLOps touches: Hydra configs, MLflow hooks, Great Expectations stubs, CI, and pre‑commit.

Note: Start simple, then iterate. The initial scaffold is intentionally lightweight and easy to extend.

## Repo Layout

app/
  Home.py
  pages/
    1_Overview.py
    2_Applicant_Scoring.py
    3_Segments.py
    4_Fairness.py
    5_Explainability.py
    6_Model_Runs.py
  assets/

src/credsafe/
  data/
  features/
  models/
  explain/
  policy/
  fairness/
  utils/

configs/
notebooks/
expectations/
reports/
tests/

## Quickstart

Prereqs: Python 3.10+ recommended.

1) Create and activate a virtual environment, then install deps:

   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -U pip
   pip install -e .

2) Put your raw CSV(s) in `data/raw/`.

3) Ingest and validate data, then train:

   make data
   make train
   make eval

4) Run the app:

   make app

Artifacts (models, metrics) are written to `artifacts/` and referenced by the app.

## Configs

Hydra configs live in `configs/`. Edit `dataset.yaml` and `model_pd.yaml` to point at your dataset and model options. The `policy.yaml` covers profit assumptions (LGD, EAD, yield, cost).

## CI & Quality

- `pre-commit` hooks: ruff, black, isort.
- GitHub Actions builds: lint + tests + app import.

## Notes

- The initial pipeline uses scikit‑learn and a simple preprocessing setup. You can enable LightGBM/CatBoost later.
- Great Expectations is scaffolded but not enforced; add suites as you formalize the schema.
- Replace the demo CSV under `data/demo/` with your own samples.

