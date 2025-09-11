import json
from pathlib import Path

import streamlit as st

st.title("Fairness")
st.write("Group metrics and parity deltas.")

mp = Path("artifacts/metrics.json")
if mp.exists():
    with open(mp) as f:
        m = json.load(f)
    if "fairness_metrics" in m:
        st.json(m["fairness_metrics"])
    else:
        st.info("No fairness group detected in data.")
else:
    st.info("Run evaluation to compute metrics.")
