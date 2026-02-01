# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="csPCa Risk Assistant", layout="centered")

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_prediction_system():
    return joblib.load("cspca_prediction_system.pkl")

try:
    packet = load_prediction_system()
    base_models = packet["base_models"]
    bootstrap_models = packet["bootstrap_models"]
    knots = packet["spline_knots"]
except Exception as e:
    st.error("❌ ERROR: Model file not found.")
    st.stop()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🛡️ csPCa Risk & Uncertainty Analysis")

# Tiêu chuẩn lâm sàng (Giữ lại để tham chiếu)
with st.expander("📚 Clinical Standards & Inclusion Criteria", expanded=True):
    st.markdown("""
    This model is optimized for patients meeting the combined criteria of **ERSPC** and **PCPT** trials:
    * **Age:** 55 – 75 years.
    * **PSA Level:** 0.4 – 50.0 ng/mL.
    * **Prostate Volume:** 10 – 110 mL.
    * **MRI Requirement:** PI-RADS Max Score ≥ 3.
    """)

# --- SIDEBAR: INPUTS (Đúng thứ tự yêu cầu) ---
with st.sidebar:
    st.header("📋 Patient Data")
    
    # 1. Age
    age = st.number_input("Age (years)", 40, 95, 65, help="Range: 55-75")
    
    # 2. PSA
    psa = st.number_input("PSA (ng/mL)", 0.1, 200.0, 7.5, help="Range: 0.4-50.0")
    
    # 3. Volume
    vol = st.number_input("Prostate Volume (mL)", 5, 300, 45, help="Range: 10-110")
    
    # 4. PI-RADS
    pirads = st.selectbox("PI-RADS Max Score (≥3)", [3, 4, 5], index=1)
    
    # 5. History (Family & Biopsy)
    # Gom lại gần nhau, dùng horizontal để gọn
    fam = st.radio("Family History", ["No", "Yes", "Unknown"], horizontal=True)
    biopsy = st.radio("Biopsy History", ["Naïve", "Prior Negative", "Unknown"], horizontal=True)
    
    # 6. DRE
    dre = st.radio("DRE Findings", ["Normal", "Abnormal"], horizontal=True)
    
    st.divider()

# ==========================================
# 4. PREDICTION LOGIC (Cấu trúc nút bấm CŨ)
# ==========================================
# Nút bấm nằm ở cột chính (Main Column), ngay dưới tiêu đề, giống mã cũ
if st.button("🚀 RUN ANALYSIS"):
    
    # --- A. Pre-processing ---
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
    
    # --- B. Spline Logic ---
    all_knots_vals = [log_psa_val] + knots
    lb, ub = min(all_knots_vals) - 1.0, max(all_knots_vals) + 1.0
    spline_df = dmatrix("bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)",
                        {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": lb, "ub": ub}, return_type="dataframe")
    if 'Intercept' in spline_df.columns: spline_df = spline_df.drop(columns=['Intercept'])
    X_final = pd.concat([df_input.reset_index(drop=True), spline_df.reset_index(drop=True)], axis=1)

    # --- C. Prediction Loop ---
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

    # ==========================================
    # 5. OUTPUT DISPLAY
    # ==========================================
    st.divider()
    st.subheader("📊 Quantitative Assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk:.1%}")
    c2.metric("Lower 95% CI", f"{low_ci:.1%}")
    c3.metric("Upper 95% CI", f"{high_ci:.1%}")

    st.write("### 🔍 Uncertainty Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(boot_preds, fill=True, color="skyblue", ax=ax, label="Prediction Density")
    ax.axvline(risk, color="blue", linestyle="--", label=f"Mean Risk ({risk:.1%})")
    ax.axvspan(low_ci, high_ci, alpha=0.1, color='grey', label='95% Confidence Interval')
    
    ax.set_title("Probability Distribution (1,000 Bootstrap Models)")
    ax.set_xlabel("Predicted Probability of csPCa")
    ax.set_ylabel("Density")
    ax.set_xlim(0, max(0.6, high_ci + 0.1))
    ax.legend()
    st.pyplot(fig)

    st.info(f"**Note:** 95% of model iterations fall between {low_ci:.1%} and {high_ci:.1%}. "
            f"The consistency of the prediction is high when this range is narrow.")
