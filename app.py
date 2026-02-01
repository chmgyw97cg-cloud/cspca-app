# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="csPCa Risk Calculator", layout="centered")

@st.cache_resource
def load_prediction_system():
    return joblib.load("cspca_prediction_system.pkl")

try:
    packet = load_prediction_system()
    base_models = packet["base_models"]
    bootstrap_models = packet["bootstrap_models"]
    knots = packet["spline_knots"]
    THRESHOLD = packet.get("threshold", 0.2)
except Exception as e:
    st.error("❌ ERROR: Model file not found.")
    st.stop()

st.title("🛡️ csPCa Risk Assistant")
st.markdown("Predicting clinically significant prostate cancer (csPCa) with 95% Confidence Intervals.")

with st.sidebar:
    st.header("📋 Patient Characteristics")
    age = st.number_input("Age (years)", 40, 95, 65)
    psa = st.number_input("PSA (ng/mL)", 0.5, 200.0, 7.5)
    vol = st.number_input("Prostate Volume (mL)", 10, 250, 45)
    pirads = st.selectbox("PI-RADS Score", [3, 4, 5], index=1)
    st.divider()
    dre = st.radio("DRE", ["Normal", "Abnormal"])
    fam = st.radio("Family History of PCa", ["No", "Yes", "Unknown"])
    biopsy = st.radio("Biopsy History", ["Naïve", "Prior Negative", "Unknown"])

log_psa_val = np.log(psa)
input_data = {
    "age": [age], "log_PSA": [log_psa_val], "log_vol": [np.log(vol)], "pirads_max": [pirads],
    "tr_yes": [1 if dre == "Abnormal" else 0], "tr_unknown": [0],
    "fam_yes": [1 if fam == "Yes" else 0], "fam_unknown": [1 if fam == "Unknown" else 0],
    "atcd_yes": [1 if biopsy == "Prior Negative" else 0], "atcd_unknown": [1 if biopsy == "Unknown" else 0],
    "atcd": [1 if biopsy == "Prior Negative" else (2 if biopsy == "Unknown" else 0)],
    "fam": [1 if fam == "Yes" else (2 if fam == "Unknown" else 0)],
    "tr": [1 if dre == "Abnormal" else 0]
}

df_input = pd.DataFrame(input_data)
all_knots_vals = [log_psa_val] + knots
lb, ub = min(all_knots_vals) - 0.5, max(all_knots_vals) + 0.5
spline_df = dmatrix(
    "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)",
    {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": lb, "ub": ub},
    return_type="dataframe"
)
X_final = pd.concat([df_input, spline_df], axis=1)

if st.button("🚀 CALCULATE RISK"):
    base_probs = []
    for name, model in base_models.items():
        feat = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_final.columns
        base_probs.append(model.predict_proba(X_final[feat])[:, 1][0])
    
    meta_in = np.array([base_probs])
    boot_preds = [m.predict_proba(meta_in)[:, 1][0] for m in bootstrap_models]
    risk, low_ci, high_ci = np.mean(boot_preds), np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk:.1%}")
    c2.metric("Lower 95% CI", f"{low_ci:.1%}")
    c3.metric("Upper 95% CI", f"{high_ci:.1%}")

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.axvspan(0, THRESHOLD, color='#2ecc71', alpha=0.3)
    ax.axvspan(THRESHOLD, 1, color='#e74c3c', alpha=0.3)
    ax.errorbar(risk, 0.5, xerr=[[risk-low_ci], [high_ci-risk]], fmt='o', color='black', capsize=8)
    ax.set_xlim(0, 1); ax.set_yticks([]); st.pyplot(fig)

    if risk >= THRESHOLD: st.error(f"### RECOMMENDATION: BIOPSY")
    else: st.success(f"### RECOMMENDATION: MONITORING")
