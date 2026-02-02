import streamlit as st
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="csPCa Risk Assistant", 
    page_icon="⚕️",
    layout="wide"
)

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_prediction_system():
    return joblib.load("cspca_prediction_system.pkl")

try:
    data_packet = load_prediction_system()
    
    # Unpack components
    base_models = data_packet["base_models"]
    knots = data_packet["spline_knots"]
    feature_mapping = data_packet.get("model_features", {})
    THRESHOLD = data_packet.get("threshold", 0.20)
    
    # LOAD TRỌNG SỐ & INTERCEPT
    meta_weights = data_packet.get("meta_weights")
    meta_intercept = data_packet.get("meta_intercept", 0.0) 
    
    bootstrap_weights = data_packet.get("bootstrap_weights")
    bootstrap_intercepts = data_packet.get("bootstrap_intercepts")

    if meta_weights is None:
        st.error("❌ Error: Missing 'meta_weights'.")
        st.stop()

except FileNotFoundError:
    st.error("❌ Critical Error: 'cspca_prediction_system.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🛡️ csPCa Risk & Uncertainty Analysis")
st.markdown(f"**Standardized Stacking Ensemble** | Decision Threshold: **{THRESHOLD:.0%}**")

with st.expander("📚 Clinical Standards & Inclusion Criteria", expanded=False):
    st.markdown("""
    * **Age:** 55 – 75 years.
    * **PSA Level:** 0.4 – 50.0 ng/mL.
    * **Prostate Volume:** 10 – 110 mL.
    * **MRI Requirement:** PI-RADS Max Score ≥ 3.
    """)

with st.sidebar:
    st.header("📋 Patient Data")
    age = st.number_input("Age (years)", 40, 95, 65)
    psa = st.number_input("Total PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.1, format="%.1f")
    vol = st.number_input("Prostate Volume (mL)", 5.0, 300.0, 45.0, step=0.1, format="%.1f")
    
    pirads = st.selectbox("PI-RADS Max Score (≥3)", [3, 4, 5], index=1)
    st.divider()
    dre_opt = st.radio("Digital Rectal Exam (DRE)", ["Normal", "Abnormal"], horizontal=True)
    fam_opt = st.radio("Family History", ["No", "Yes", "Unknown"], horizontal=True)
    biopsy_opt = st.radio("Biopsy History", ["Naïve", "Prior Negative"], horizontal=True)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("🚀 RUN ANALYSIS", type="primary"):
    
    # --- A. Pre-processing ---
    log_psa_val = np.log(psa)
    log_vol_val = np.log(vol)
    
    input_dict = {
        "age": [age], "PSA": [psa], "log_PSA": [log_psa_val], "log_vol": [log_vol_val], "pirads_max": [pirads],
        "tr_yes": [1 if dre_opt == "Abnormal" else 0], "fam_yes": [1 if fam_opt == "Yes" else 0], 
        "atcd_yes": [1 if biopsy_opt == "Prior Negative" else 0],
        "tr": [1 if dre_opt == "Abnormal" else 0], 
        "fam": [1 if fam_opt == "Yes" else (2 if fam_opt == "Unknown" else 0)],
        "atcd": [1 if biopsy_opt == "Prior Negative" else 0],
        "fam_unknown": [1 if fam_opt == "Unknown" else 0], "tr_unknown": [0], "atcd_unknown": [0]
    }
    df_input = pd.DataFrame(input_dict)
    
    # --- B. Spline Logic ---
    try:
        safe_lb = min(knots) - 5.0
        safe_ub = max(knots) + 5.0
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(spline_formula, 
                           {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub}, 
                           return_type="dataframe")
        
        rename_map = {}
        for col in spline_df.columns:
            if "Intercept" in col: continue
            match = re.search(r"\[(\d+)\]$", col)
            if match:
                idx = match.group(1)
                original_name = f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{idx}]"
                rename_map[col] = original_name
        spline_df = spline_df.rename(columns=rename_map)

        if "Intercept" not in spline_df.columns:
            spline_df["Intercept"] = 1.0
            
        df_full = pd.concat([df_input, spline_df], axis=1)

    except Exception as e:
        st.error(f"Spline Processing Error: {e}")
        st.stop()

    # --- C. Base Models Prediction ---
    base_preds = []
    model_names = list(base_models.keys())
    
    for name in model_names:
        model = base_models[name]
        if name in feature_mapping:
            required_cols = feature_mapping[name]
        else:
            required_cols = df_full.columns.tolist()
