# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Wide mode and compact CSS
st.set_page_config(page_title="csPCa Risk Assistant", layout="wide")

st.markdown("""
    <style>
    /* Reduce top padding */
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    /* Make titles smaller to save vertical space */
    h1 {font-size: 1.6rem !important; margin-bottom: 0.5rem;}
    h3 {font-size: 1.1rem !important; margin-top: 0.5rem; margin-bottom: 0.5rem;}
    /* Adjust sidebar padding */
    section[data-testid="stSidebar"] {padding-top: 0rem;}
    /* Hide the Streamlit header to save space */
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_prediction_system():
    return joblib.load("cspca_prediction_system.pkl")

try:
    packet = load_prediction_system()
    base_models, bootstrap_models, knots = packet["base_models"], packet["bootstrap_models"], packet["spline_knots"]
except Exception as e:
    st.error("❌ ERROR: Model file not found."); st.stop()

# ==========================================
# SIDEBAR (Compact Inputs)
# ==========================================
with st.sidebar:
    st.header("📋 Patient Data")
    age = st.number_input("Age (55-75)", 40, 95, 65)
    psa = st.number_input("PSA (0.4-50)", 0.1, 200.0, 7.5)
    vol = st.number_input("Volume (10-110)", 5, 300, 45)
    pirads = st.selectbox("PI-RADS Max Score (≥3)", [3, 4, 5], index=1)
    
    st.divider()
    dre = st.radio("DRE", ["Normal", "Abnormal"], horizontal=True)
    fam = st.radio("Family History", ["No", "Yes", "Unknown"], horizontal=True)
    biopsy = st.radio("Biopsy History", ["Naïve", "Prior Neg", "Unknown"], horizontal=True)
    
    run_btn = st.button("🚀 RUN ANALYSIS", use_container_width=True)

# ==========================================
# MAIN PANEL (Multi-column for no-scroll)
# ==========================================
st.title("🛡️ csPCa Risk & Uncertainty Analysis")

# Criteria in one single line
st.caption("Standard Range: Age 55-75 | PSA 0.4-50 | Vol 10-110 | PI-RADS ≥ 3")

if run_btn:
    # Prediction Logic
    log_psa_val = np.log(psa)
    input_data = {
        "age": [age], "log_PSA": [log_psa_val], "log_vol": [np.log(vol)], "pirads_max": [pirads],
        "tr_yes": [1 if dre == "Abnormal" else 0], "tr_unknown": [0],
        "fam_yes": [1 if fam == "Yes" else 0], "fam_unknown": [1 if fam == "Unknown" else 0],
        "atcd_yes": [1 if biopsy == "Prior Neg" else 0], "atcd_unknown": [1 if biopsy == "Unknown" else 0],
        "atcd": [1 if biopsy == "Prior Neg" else (2 if biopsy == "Unknown" else 0)],
        "fam": [1 if fam == "Yes" else (2 if fam == "Unknown" else 0)], "tr": [1 if dre == "Abnormal" else 0]
    }
    df_input = pd.DataFrame(input_data)
    
    all_knots_vals = [log_psa_val] + knots
    lb, ub = min(all_knots_vals) - 1.0, max(all_knots_vals) + 1.0
    spline_df = dmatrix("bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)",
                        {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": lb, "ub": ub}, return_type="dataframe")
    if 'Intercept' in spline_df.columns: spline_df = spline_df.drop(columns=['Intercept'])
    X_final = pd.concat([df_input.reset_index(drop=True), spline_df.reset_index(drop=True)], axis=1)

    base_probs = []
    for name, model in base_models.items():
        if hasattr(model, 'feature_names_in_'):
            expected = model.feature_names_in_
            for col in expected:
                if col not in X_final.columns: X_final[col] = 0
            X_predict = X_final[expected]
        else: X_predict = X_final
        base_probs.append(model.predict_proba(X_predict)[:, 1][0])
    
    boot_preds = [m.predict_proba(np.array([base_probs]))[:, 1][0] for m in bootstrap_models]
    risk, low_ci, high_ci = np.mean(boot_preds), np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)

    # --- RESULTS IN COLUMNS ---
    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        st.subheader("📊 Assessment")
        st.metric("Mean Risk", f"{risk:.1%}")
        st.metric("Lower 95% CI", f"{low_ci:.1%}")
        st.metric("Upper 95% CI", f"{high_ci:.1%}")
        st.info(f"CI Range: {(high_ci-low_ci):.1%}")

    with col2:
        st.subheader("🔍 Prediction Density")
        fig, ax = plt.subplots(figsize=(7, 3.2)) # Smaller chart
        sns.kdeplot(boot_preds, fill=True, color="skyblue", ax=ax)
        ax.axvline(risk, color="blue", linestyle="--", label="Mean")
        ax.axvspan(low_ci, high_ci, alpha=0.1, color='grey', label='95% CI')
        ax.set_xlabel("Probability", fontsize=8); ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7)
        st.pyplot(fig)

else:
    # Placeholder to keep screen full before button is clicked
    st.info("Fill in patient data on the left and click 'Run Analysis' to see results.")
