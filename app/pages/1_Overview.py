import json
from pathlib import Path

import streamlit as st

st.title("Overview")
st.write("KPIs, ROC/PR, KS, approval/profit vs baseline, recent runs.")

metrics_path = Path("artifacts/metrics.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        m = json.load(f)
    st.json(m)
else:
    st.info("Train and evaluate the model to see metrics.")
