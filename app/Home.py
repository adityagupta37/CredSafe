import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="CredSafe Risk Desk",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .credsafe-title {display:flex; align-items:center; gap: 12px}
    </style>
    """,
    unsafe_allow_html=True,
)

logo_path = Path(__file__).parent / "assets" / "credsafe_logo.svg"
st.markdown(
    f"<div class='credsafe-title'><img src='file://{logo_path}' height='36'/>"
    "<h2>CredSafe — Credit Risk Desk</h2></div>",
    unsafe_allow_html=True,
)

st.write("A compact credit risk app with scoring, policy, segments, fairness, and explainability.")

metrics_path = Path("artifacts/metrics.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        m = json.load(f)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", f"{m.get('auc', float('nan')):.3f}")
    c2.metric("KS", f"{m.get('ks', float('nan')):.3f}")
    c3.metric("Approval rate", f"{100*m.get('policy_approval_rate', 0):.1f}%")
    c4.metric("Profit/loan", f"{m.get('policy_profit_per_loan', 0):.3f}")
else:
    st.warning("No metrics found yet. Train and evaluate the model to populate KPIs.")

st.divider()
st.write("Use the sidebar pages to explore the app.")
