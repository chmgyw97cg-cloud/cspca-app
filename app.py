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
    # Load the .pkl file located in the root directory
    return joblib.load("cspca_prediction_system.pkl")

try:
    data_packet = load_prediction_system()
    
    # Unpack components
    base_models = data_packet["base_models"]
    knots = data_packet["spline_knots"]
    feature_mapping = data_packet.get("model_features", {})
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

# --- HEADER ---
st.markdown(f"**Standardized Stacking Ensemble** | Decision Threshold: **{THRESHOLD:.0%}**")
st.caption("**Definition:** csPCa (Clinically Significant Prostate Cancer) is defined as **ISUP Grade Group ≥ 2** (Gleason Score ≥ 3+4).")
st.caption("**Scope:** Prediction applies to **MRI-Targeted Biopsy (ROI-only)**.")

with st.expander("📚 Clinical Standards & Inclusion Criteria", expanded=False):
    st.markdown("""
    * **Target Population:** Men undergoing **MRI-Targeted Biopsy** (ROI-only).
    * **Inclusion Criteria:** **PI-RADS Score ≥ 3**.
    * **Age:** 55 – 75 years | **PSA:** 0.4 – 50.0 ng/mL | **Vol:** 10 – 110 mL.
    """)

with st.sidebar:
    st.header("📋 Patient Data")
    
    # --- INPUTS ---
    age = st.number_input("Age (years)", 40, 95, 65)
    psa = st.number_input("Total PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.1, format="%.1f")
    vol = st.number_input("Prostate Volume (mL)", 5.0, 300.0, 45.0, step=0.1, format="%.1f")
    pirads = st.selectbox("PI-RADS Max Score (≥3)", [3, 4, 5], index=1)
    
    st.divider()
    dre_opt = st.radio("Digital Rectal Exam (DRE)", ["Normal", "Abnormal"], horizontal=True)
    fam_opt = st.radio("Family History", ["No", "Yes", "Unknown"], horizontal=True)
    biopsy_opt = st.radio("Biopsy History", ["Naïve", "Prior Negative"], horizontal=True)
    
    # --- AUTO CALIBRATION LOGIC (GỌN GÀNG THEO YÊU CẦU) ---
    st.divider()
    with st.expander("⚙️ Calibration Details", expanded=True):
        st.markdown("**Standard: PRECISION Trial**")
        st.caption("Standard yield for MRI-Targeted Biopsy (ROI) in men with PI-RADS ≥ 3.")
        
        # Mặc định 38.0% theo PRECISION Trial
        DEFAULT_TARGET = 38.0
        
        local_prev_pct = st.number_input(
            "Target Yield within ROI (%):", 
            min_value=1.0, max_value=99.0, 
            value=DEFAULT_TARGET, 
            step=0.5, format="%.1f"
        )
        st.caption("*Ref: Kasivisvanathan et al., NEJM 2018.*")
        
        # Training Prevalence (DỮ LIỆU CỦA BẠN)
        TRAIN_PREV = 0.452 
        
        # Tính toán Offset
        target_prev = local_prev_pct / 100.0
        def logit(p): return np.log(p / (1 - p))
        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)
        
        st.info(f"✅ Adjusted: **{TRAIN_PREV:.1%}** ➔ **{local_prev_pct}%**")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("🚀 RUN ANALYSIS", type="primary"):
    
    # 1. VALIDATION
    warnings = []
    if not (55 <= age <= 75): warnings.append(f"⚠️ **Age ({age})** outside 55-75.")
    if not (0.4 <= psa <= 50.0): warnings.append(f"⚠️ **PSA ({psa})** outside 0.4-50.0.")
    if not (10 <= vol <= 110): warnings.append(f"⚠️ **Vol ({vol})** outside 10-110.")
    if warnings:
        with st.warning("### ⚠️ Clinical Warning: Out of Distribution"):
            for w in warnings: st.markdown(f"* {w}")

    # 2. PRE-PROCESSING
    log_psa_val = np.log(psa)
    log_vol_val = np.log(vol)
    psa_density = psa / vol
    
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
        safe_lb = min(knots) - 5.0; safe_ub = max(knots) + 5.0
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(spline_formula, {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub}, return_type="dataframe")
        
        rename_map = {}
        for col in spline_df.columns:
            if "Intercept" in col: continue
            match = re.search(r"\[(\d+)\]$", col)
            if match: rename_map[col] = f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{match.group(1)}]"
        spline_df = spline_df.rename(columns=rename_map)
        if "Intercept" not in spline_df.columns: spline_df["Intercept"] = 1.0
        df_full = pd.concat([df_input, spline_df], axis=1)
    except Exception as e: st.error(f"Spline Error: {e}"); st.stop()

    # 3. INFERENCE
    base_preds = []
    for name in list(base_models.keys()):
        model = base_models[name]
        cols = feature_mapping.get(name, df_full.columns.tolist())
        try:
            p = model.predict_proba(df_full[cols])[:, 1][0] if hasattr(model, "predict_proba") else model.predict(df_full[cols])[0]
            base_preds.append(p)
        except: st.error(f"Error in {name}"); st.stop()
    base_preds = np.array(base_preds)
    
    # 4. META-PREDICTION (WITH OFFSET)
    raw_log_odds = np.dot(base_preds, meta_weights) + meta_intercept
    final_log_odds = raw_log_odds + CALIBRATION_OFFSET
    risk_mean = 1 / (1 + np.exp(-final_log_odds))
    
    if bootstrap_weights is not None:
        boot_log_odds = np.dot(bootstrap_weights, base_preds)
        if bootstrap_intercepts is not None: boot_log_odds += bootstrap_intercepts
        boot_log_odds += CALIBRATION_OFFSET
        boot_preds = 1 / (1 + np.exp(-boot_log_odds))
        low_ci, high_ci = np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)
        has_ci = True
    else:
        low_ci, high_ci = risk_mean, risk_mean
        has_ci = False

    # 5. DISPLAY
    st.divider()
    st.subheader("📊 Quantitative Assessment")

    GRAY_LOW = 0.10
    GRAY_HIGH = THRESHOLD

    if risk_mean < GRAY_LOW:
        risk_label = "Low Risk"; delta_color = "normal" 
    elif risk_mean < GRAY_HIGH:
        risk_label = "Intermediate Risk (Gray Zone)"; delta_color = "off"
    else:
        risk_label = "High Risk"; delta_color = "inverse"

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk_mean:.1%}", delta=risk_label, delta_color=delta_color)
    c2.metric("Lower 95% CI", f"{low_ci:.1%}" if has_ci else "N/A")
    c3.metric("Upper 95% CI", f"{high_ci:.1%}" if has_ci else "N/A")

    # --- CHÚ THÍCH XANH (GREEN NOTE) VỀ 95% CI ---
    ci_width = high_ci - low_ci
    st.info(
        f"**Interpretation:** The model predicts a **{risk_mean:.1%}** probability of csPCa within the ROI.\n\n"
        f"**Note on Uncertainty:** The true risk likely lies between **{low_ci:.1%}** and **{high_ci:.1%}** "
        f"(uncertainty spread: **{high_ci - low_ci:.1%}**). "
        f"**A narrower range implies higher certainty.**"
    )

    st.write("### 🔍 Uncertainty Visualization")
    if has_ci:
        sns.set_theme(style="ticks", context="paper", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(8, 3))
        
        color_low, color_mid, color_high = '#28a745', '#fd7e14', '#dc3545'
        ax.axvspan(0, GRAY_LOW, color=color_low, alpha=0.15, label='Low Risk', lw=0)
        ax.axvspan(GRAY_LOW, GRAY_HIGH, color=color_mid, alpha=0.15, label='Intermediate', lw=0)
        ax.axvspan(GRAY_HIGH, 1.0, color=color_high, alpha=0.1, label='High Risk', lw=0)

        sns.kdeplot(boot_preds, fill=True, color="#2c3e50", alpha=0.4, ax=ax, linewidth=1.5)
        ax.axvline(risk_mean, color="#d95f02", linestyle="-", linewidth=2, label=f"Mean: {risk_mean:.1%}")
        ax.axvline(GRAY_HIGH, color="black", linestyle="--", linewidth=1.2, label=f"Threshold: {GRAY_HIGH:.0%}")

        plt.suptitle("Estimated Risk Distribution & Confidence Intervals", y=1.02, fontsize=12, fontweight='bold', color='#333')
        plt.title(f"Target Yield (ROI-only): {local_prev_pct:.1f}% (Ref: PRECISION Trial)", fontsize=9, color='#666', style='italic', pad=10)
        
        ax.set_xlabel("Predicted Probability of csPCa"); ax.set_ylabel("Density")
        ax.set_xlim(0, max(0.6, high_ci + 0.15))
        ax.legend(loc='best', fontsize=8, framealpha=0.95)
        sns.despine(offset=5, trim=True)
        st.pyplot(fig, dpi=300, use_container_width=False)
        sns.reset_orig()

    st.caption(f"**Calculated PSA Density (PSAD):** {psa_density:.2f} ng/mL²")
