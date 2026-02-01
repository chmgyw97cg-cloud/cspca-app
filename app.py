# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compact Layout Configuration
st.set_page_config(page_title="csPCa Risk Assistant", layout="wide")

# Custom CSS to reduce padding and font sizes for a "Single Screen" feel
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {font-size: 1.8rem !important;}
    h3 {font-size: 1.2rem !important;}
    .stMetric {padding: 0.5rem !important;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_prediction_system():
    return joblib.load("cspca_prediction_system.pkl")

try:
    packet = load_prediction_system()
    base_models, bootstrap_models, knots = packet["base_models"], packet["bootstrap_models"], packet["spline_knots"]
except Exception as e:
    st.error("❌ Model file missing."); st.stop()

# ==========================================
# SIDEBAR (Input stays here to save main space)
# ==========================================
with st.sidebar:
    st.title("🛡️ Patient Data")
    age = st.number_input("Age (55-75)", 40, 95, 65)
    psa = st.number_input("PSA (0.4-50)", 0.1, 200.0, 7.5)
    vol = st.number_input("Volume (10-110)", 5, 300, 45)
    pirads = st.selectbox("PI-RADS Max (≥3)", [3, 4, 5], index=1)
    dre = st.radio("DRE", ["Normal", "Abnormal"], horizontal=True)
    fam = st.radio("Family History", ["No", "Yes", "Unknown"], horizontal=True)
    biopsy = st.radio("Biopsy History", ["Naïve", "Prior Neg", "Unknown"], horizontal=True)
    run_btn = st.button("🚀 RUN ANALYSIS", use_container_width=True)

# ==========================================
# MAIN PANEL (Optimized for no-scroll)
# ==========================================
st.title("csPCa Risk & Uncertainty Assistant")

# Inclusion criteria in a compact info box
st.caption("Target: Age 55-75 | PSA 0.4-50 | Vol 10-110 | PI-RADS ≥3")

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
    
    # Spline & Feature Alignment
    all_knots_vals = [log_psa_val] + knots
    lb, ub = min(all_knots_vals) - 1.0, max(all_knots_vals) + 1.0
    spline_df = dmatrix("bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)",
                        {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": lb, "ub": ub}, return_type="dataframe")
    if 'Intercept' in spline_df.columns: spline_df = spline_df.drop(columns=['Intercept'])
    X_final = pd.concat([df_input.reset_index(drop=True), spline_df.reset_index(drop=True)], axis=1)

    base_probs = []
    for name, model in base_models.items():
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in X_final.columns: X_final[col] = 0
            X_predict = X_final[model.feature_names_in_]
        else: X_predict = X_final
        base_probs.append(model.predict_proba(X_predict)[:, 1][0])
    
    boot_preds = [m.predict_proba(np.array([base_probs]))[:, 1][0] for m in bootstrap_models]
    risk, low_ci, high_ci = np.mean(boot_preds), np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)

    # --- SINGLE SCREEN LAYOUT (2 COLUMNS) ---
    col_results, col_chart = st.columns([1, 2])

    with col_results:
        st.subheader("📊 Metrics")
        st.metric("Mean Risk", f"{risk:.1%}")
        st.metric("95% CI Lower", f"{low_ci:.1%}")
        st.metric("95% CI Upper", f"{high_ci:.1%}")
        st.info(f"Uncertainty Range: {(high_ci-low_ci):.1%}")

    with col_chart:
        st.subheader("🔍 Uncertainty Profile")
        fig, ax = plt.subplots(figsize=(7, 3.5)) # Smaller figure size
        sns.kdeplot(boot_preds, fill=True, color="skyblue", ax=ax)
        ax.axvline(risk, color="blue", linestyle="--", label=f"Mean")
        ax.axvspan(low_ci, high_ci, alpha=0.1, color='grey', label='95% CI')
        ax.set_xlabel("Probability", fontsize=9); ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8); ax.legend(fontsize=7)
        st.pyplot(fig)

    st.divider()
    st.caption("Disclaimer: Research tool only. Clinical judgment required.")
