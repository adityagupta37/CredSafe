from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.title("Applicant Scoring")

model_path = Path("artifacts/pd_model.pkl")
if not model_path.exists():
    st.warning("Model not found. Train first (make train).")
    st.stop()

model = joblib.load(model_path)

st.subheader("Score a single applicant")
with st.form("single_form"):
    st.write("Enter a few feature:value pairs (JSON)")
    js = st.text_area("Features JSON", value='{"age": 35, "income": 50000, "tenor": 12}')
    submitted = st.form_submit_button("Score")
    if submitted:
        try:
            payload = json.loads(js)
            row = pd.DataFrame([payload])
            prob = float(model.predict_proba(row)[:, 1][0])
            st.success(f"PD: {prob:.3f}")
        except Exception as e:
            st.error(f"Failed to score: {e}")

st.subheader("Batch scoring (CSV)")
file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        df = pd.read_csv(file)
        pd_hat = model.predict_proba(df)[:, 1]
        out = df.copy()
        out["pd"] = pd_hat
        st.dataframe(out.head(20))
        buf = io.BytesIO()
        out.to_csv(buf, index=False)
        st.download_button("Download Scored CSV", buf.getvalue(), file_name="scored.csv")
    except Exception as e:
        st.error(f"Failed to score CSV: {e}")
