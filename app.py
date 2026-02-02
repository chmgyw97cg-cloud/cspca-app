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
    base_models = data_packet["base_models"]
    knots = data_packet["spline_knots"]
    feature_mapping = data_packet.get("model_features", {})
    # Giữ Threshold cho mục đích tính toán nếu cần, nhưng sẽ không hiển thị lên biểu đồ
    THRESHOLD = data_packet.get("threshold", 0.20)
    meta_weights = data_packet.get("meta_weights")
    meta_intercept = data_packet.get("meta_intercept", 0.0) 
    bootstrap_weights = data_packet.get("bootstrap_weights")
    bootstrap_intercepts = data_packet.get("bootstrap_intercepts")
    
    if meta_weights is None: st.error("❌ Error: Missing weights."); st.stop()
except Exception as e:
    st.error(f"❌ Critical Error: {e}"); st.stop()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🛡️ csPCa Risk & Uncertainty Analysis")

st.markdown("**Standardized Stacking Ensemble** | Clinical Decision Support")
st.caption("**Definition:** csPCa (Clinically Significant Prostate Cancer) is defined as **ISUP Grade Group ≥ 2**.")
st.caption("**Scope:** Prediction applies to **MRI-Targeted Biopsy (ROI-only)**.")
with st.expander("📚 Clinical Standards & Inclusion Criteria", expanded=False):
    st.markdown("""
    * **Age:** 55 – 75 years.
    * **PSA Level:** 0.4 – 50.0 ng/mL.
    * **Prostate Volume:** 10 – 110 mL.
    * **MRI Requirement:** PI-RADS Max Score ≥ 3.
    """)
    
with st.sidebar:
    st.header("📋 Patient Data")
    
    # Định dạng dấu chấm thập phân được đảm bảo qua format="%.1f" hoặc "%.2f"
    age = st.number_input("Age (years)", 40, 95, 65)
    psa = st.number_input("Total PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.1, format="%.1f")
    vol = st.number_input("Prostate Volume (mL)", 5.0, 300.0, 45.0, step=0.1, format="%.1f")
    pirads = st.selectbox("PI-RADS Max Score (≥3)", [3, 4, 5], index=1)
    
    st.divider()
    dre_opt = st.radio("Digital Rectal Exam (DRE)", ["Normal", "Abnormal", "Unknown"], horizontal=True)
    fam_opt = st.radio("Family History", ["No", "Yes", "Unknown"], horizontal=True)
    biopsy_opt = st.radio("Biopsy History", ["Naïve", "Prior Negative", "Unknown"], horizontal=True)
    
    st.divider()
    with st.expander("⚙️ Calibration Details", expanded=True):
        st.markdown("**Standard: PRECISION Trial**")
        st.caption("Standard yield for MRI-Targeted Biopsy (ROI) in men with PI-RADS ≥ 3.")
        
        DEFAULT_TARGET = 38.0 # Based on PRECISION NEJM 2018
        
        local_prev_pct = st.number_input(
            "Target Yield within ROI (%):", 
            min_value=1.0, max_value=99.0, 
            value=DEFAULT_TARGET, 
            step=0.5, format="%.1f"
        )
        st.caption("*Ref: Kasivisvanathan et al., NEJM 2018.*")
        
        TRAIN_PREV = 0.452 # Development cohort prevalence
        
        target_prev = local_prev_pct / 100.0
        def logit(p): return np.log(p / (1 - p))
        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)
        
        st.info(f"✅ Adjusted: **{TRAIN_PREV:.1%}** ➔ **{local_prev_pct}%**")


# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("🚀 RUN ANALYSIS", type="primary"):
    # 0. CLINICAL VALIDATION
    warnings = []
    if not (55 <= age <= 75): 
        warnings.append(f"• **Age ({age})** is outside the study range (55-75).")
    if not (0.4 <= psa <= 50.0): 
        warnings.append(f"• **PSA ({psa})** is outside the validated range (0.4-50.0).")
    if not (10 <= vol <= 110): 
        warnings.append(f"• **Volume ({vol})** is outside the validated range (10-110).")
    if warnings:
        with st.warning("⚠️ **Clinical Caution: Out of Distribution**"):
            st.markdown("The patient's profile is outside the core inclusion criteria. Results should be interpreted with extra caution.")
            for w in warnings: st.markdown(w)
                
    # 1. PRE-PROCESSING
    log_psa_val = np.log(psa)
    log_vol_val = np.log(vol)
    psad = psa / vol
    
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
    
    # Spline Logic
    try:
        safe_lb, safe_ub = min(knots) - 5.0, max(knots) + 5.0
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(spline_formula, {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub}, return_type="dataframe")
        
        rename_map = {col: f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{re.search(r'\[(\d+)\]$', col).group(1)}]" 
                      for col in spline_df.columns if re.search(r"\[(\d+)\]$", col)}
        spline_df = spline_df.rename(columns=rename_map)
        if "Intercept" not in spline_df.columns: spline_df["Intercept"] = 1.0
        df_full = pd.concat([df_input, spline_df], axis=1)
    except Exception as e: st.error(f"Spline Error: {e}"); st.stop()

    # 2. INFERENCE
    base_preds = []
    for name in list(base_models.keys()):
        model = base_models[name]
        cols = feature_mapping.get(name, df_full.columns.tolist())
        p = model.predict_proba(df_full[cols])[:, 1][0] if hasattr(model, "predict_proba") else model.predict(df_full[cols])[0]
        base_preds.append(p)
    base_preds = np.array(base_preds)
    
    raw_log_odds = np.dot(base_preds, meta_weights) + meta_intercept
    risk_mean = 1 / (1 + np.exp(-(raw_log_odds + CALIBRATION_OFFSET)))
    
    if bootstrap_weights is not None:
        boot_log_odds = np.dot(bootstrap_weights, base_preds) + (bootstrap_intercepts if bootstrap_intercepts is not None else 0) + CALIBRATION_OFFSET
        boot_preds = 1 / (1 + np.exp(-boot_log_odds))
        low_ci, high_ci = np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)
        has_ci = True
    else:
        low_ci, high_ci, has_ci = risk_mean, risk_mean, False

    # 3. DISPLAY
    st.divider()
    st.subheader("📊 Quantitative Assessment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk_mean:.1%}")
    c2.metric("Lower 95% CI", f"{low_ci:.1%}" if has_ci else "N/A")
    c3.metric("Upper 95% CI", f"{high_ci:.1%}" if has_ci else "N/A")

    st.info(
        f"**Interpretation:** The model predicts a **{risk_mean:.1%}** probability of csPCa within the ROI.\n\n"
        f"**Uncertainty Note:** Based on 1,000 bootstrap simulations, the 95% CI is **{low_ci:.1%}** to **{high_ci:.1%}** "
        f"(uncertainty spread: **{high_ci - low_ci:.1%}**)."
        f" **A narrower distribution reflects higher model confidence**."
    )

    # --- UNCERTAINTY VISUALIZATION (CLEANED) ---
    st.write("### 🔍 Risk Probability Distribution")
    if has_ci:
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        
        # Plot KDE without background colors or threshold lines
        sns.kdeplot(boot_preds, fill=True, color="#2c3e50", alpha=0.3, ax=ax, linewidth=2, label="Risk Distribution")
        
        # Vertical line for mean risk
        ax.axvline(risk_mean, color="#d95f02", linestyle="-", linewidth=2.5, label=f"Point Estimate: {risk_mean:.1%}")
        
        # Formatting
        plt.title("Bootstrap Uncertainty Analysis", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Predicted Probability of csPCa", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, max(0.6, high_ci + 0.1))
        ax.legend(loc='best', fontsize=9)
        
        sns.despine()
        st.pyplot(fig, dpi=300)
        
    st.caption(f"**Calculated PSA Density (PSAD):** {psad:.2f} ng/mL²")
