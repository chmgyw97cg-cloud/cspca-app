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
    # Default to 0.20 if not found
    THRESHOLD = data_packet.get("threshold", 0.20)
    meta_weights = data_packet.get("meta_weights")
    bootstrap_weights = data_packet.get("bootstrap_weights")

    if meta_weights is None:
        st.error("❌ Error: Missing 'meta_weights' in .pkl file.")
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
st.markdown(f"**Meta-stacking Ensemble Model** | Decision Threshold: **{THRESHOLD:.0%}**")

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
    psa = st.number_input("Total PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.1)
    vol = st.number_input("Prostate Volume (mL)", 5, 300, 45, step=1)
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
    
    # --- B. Spline Logic (Safety Fix + Renaming) ---
    try:
        safe_lb = min(knots) - 5.0
        safe_ub = max(knots) + 5.0
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(spline_formula, 
                           {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub}, 
                           return_type="dataframe")
        
        # Regex renaming to match training names
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

    # --- C. Prediction Loop ---
    base_preds = []
    model_names = list(base_models.keys())
    
    for name in model_names:
        model = base_models[name]
        if name in feature_mapping:
            required_cols = feature_mapping[name]
        else:
            required_cols = df_full.columns.tolist()
            
        missing = [c for c in required_cols if c not in df_full.columns]
        if missing:
            st.error(f"❌ Model '{name}' missing columns: {missing}")
            st.stop()
            
        try:
            X_subset = df_full[required_cols]
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_subset)[:, 1][0]
            else:
                p = model.predict(X_subset)[0]
            base_preds.append(p)
        except Exception as e:
            st.error(f"Error running model '{name}': {e}")
            st.stop()
    
    base_preds = np.array(base_preds)
    
    # --- D. Meta-Prediction ---
    risk_mean = np.dot(base_preds, meta_weights)
    
    if bootstrap_weights is not None:
        boot_preds = np.dot(bootstrap_weights, base_preds)
        low_ci, high_ci = np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)
        has_ci = True
    else:
        low_ci, high_ci = risk_mean, risk_mean
        has_ci = False

    # ==========================================
    # 5. OUTPUT DISPLAY (FINAL ENGLISH VERSION)
    # ==========================================
    st.divider()
    st.subheader("📊 Quantitative Assessment")

    # 1. Define Clinical Thresholds
    GRAY_LOW = 0.10        # < 10%: Safety Net (High NPV)
    GRAY_HIGH = THRESHOLD  # >= 20%: Optimal Biopsy Threshold (DCA)

    # 2. Determine Labels and Colors
    if risk_mean < GRAY_LOW:
        risk_label = "Low Risk"
        delta_color = "normal" 
    elif risk_mean < GRAY_HIGH:
        risk_label = "Intermediate Risk (Gray Zone)"
        delta_color = "off" # Neutral color
    else:
        risk_label = "High Risk"
        delta_color = "inverse" # Red highlighting

    # 3. Display Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk_mean:.1%}", delta=risk_label, delta_color=delta_color)

    if has_ci:
        c2.metric("Lower 95% CI", f"{low_ci:.1%}")
        c3.metric("Upper 95% CI", f"{high_ci:.1%}")
    else:
        c2.metric("Lower 95% CI", "N/A")
        c3.metric("Upper 95% CI", "N/A")

    # 4. Visual Clinical Risk Scale (HTML/CSS)
    st.write("### 🚦 Clinical Risk Scale")
    
    # Bar Color Logic
    if risk_mean < GRAY_LOW:
        bar_color = "#28a745" # Green
    elif risk_mean < GRAY_HIGH:
        bar_color = "#ffc107" # Amber/Yellow
    else:
        bar_color = "#dc3545" # Red

    bar_width = min(int(risk_mean * 100), 100)
    
    # Custom HTML Progress Bar
    # NOTE: unsafe_allow_html=True is REQUIRED for this to render as a bar instead of code
    st.markdown(f"""
    <div style="background-color: #e9ecef; border-radius: 10px; padding: 5px; position: relative; margin-bottom: 5px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);">
        <div style="position: absolute; left: {GRAY_LOW*100}%; top: 0; bottom: 0; border-left: 2px dashed #6c757d; opacity: 0.6;" title="Start of Gray Zone (10%)"></div>
        <div style="position: absolute; left: {GRAY_HIGH*100}%; top: 0; bottom: 0; border-left: 3px solid #343a40;" title="Biopsy Threshold ({GRAY_HIGH:.0%})"></div>
        
        <div style="width: {bar_width}%; background-color: {bar_color}; height: 30px; border-radius: 6px; 
                    text-align: right; padding-right: 10px; color: white; font-weight: bold; line-height: 30px; 
                    transition: width 0.6s ease; text-shadow: 0px 0px 2px rgba(0,0,0,0.5);">
            {risk_mean:.1%}
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6c757d; font-family: sans-serif;">
        <span>0%</span>
        <span style="text-align: center;">Gray Zone<br>({GRAY_LOW:.0%} - {GRAY_HIGH:.0%})</span>
        <span style="font-weight: bold; color: #343a40;">Biopsy Threshold<br>({GRAY_HIGH:.0%})</span>
        <span>100%</span>
    </div>
    """, unsafe_allow_html=True)

    # 5. Uncertainty Visualization (Matplotlib)
    
    st.write("### 🔍 Uncertainty Visualization")
    if has_ci:
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Background Zones
        ax.axvspan(0, GRAY_LOW, color='green', alpha=0.05, label='Low Risk Zone')
        ax.axvspan(GRAY_LOW, GRAY_HIGH, color='orange', alpha=0.1, label='Intermediate Zone')
        ax.axvspan(GRAY_HIGH, 1.0, color='red', alpha=0.05, label='High Risk Zone')

        # Density Plot
        sns.kdeplot(boot_preds, fill=True, color="#007bff", alpha=0.4, ax=ax, linewidth=1.5)
        
        # Mean Line
        ax.axvline(risk_mean, color="#d63384", linestyle="-", linewidth=2.5, label=f"Mean Risk: {risk_mean:.1%}")
        
        # Threshold Line
        ax.axvline(GRAY_HIGH, color="black", linestyle="--", linewidth=1.5, label=f"Threshold: {GRAY_HIGH:.0%}")

        # Formatting
        ax.set_title("Probability Distribution (1,000 Bootstrap Models)", fontsize=10)
        ax.set_xlabel("Predicted Probability of csPCa", fontsize=9)
        ax.set_yticks([]) # Hide y-axis for cleaner look
        ax.set_xlim(0, max(0.6, high_ci + 0.1))
        ax.legend(loc='upper right', fontsize='small', frameon=True)
        
        st.pyplot(fig)

    # 6. Clinical Recommendation (Three Levels)
    st.subheader("💡 Clinical Recommendation")
    
    if risk_mean >= GRAY_HIGH:
        st.error(f"""
        **🔴 HIGH RISK (≥ {GRAY_HIGH:.0%})**
        * **Probability:** {risk_mean:.1%} (CI: {low_ci:.1%} - {high_ci:.1%}).
        * **Interpretation:** The risk exceeds the optimal decision threshold.
        * **Action:** Strong indication for **mpMRI** and **Targeted Biopsy**.
        """)
    elif risk_mean >= GRAY_LOW:
        st.warning(f"""
        **🟡 INTERMEDIATE RISK ({GRAY_LOW:.0%} - {GRAY_HIGH:.0%})**
        * **Probability:** {risk_mean:.1%} (CI: {low_ci:.1%} - {high_ci:.1%}).
        * **Interpretation:** The patient falls into the diagnostic "Gray Zone".
        * **Action:** Consider **Shared Decision Making**. Evaluate secondary factors (e.g., PSA Density, Free/Total PSA ratio) before biopsying.
        """)
    else:
        st.success(f"""
        **🟢 LOW RISK (< {GRAY_LOW:.0%})**
        * **Probability:** {risk_mean:.1%} (CI: {low_ci:.1%} - {high_ci:.1%}).
        * **Interpretation:** High Negative Predictive Value (NPV).
        * **Action:** Immediate biopsy may be avoided. Continue **PSA Monitoring** (6-12 months).
        """)

    # Text Interpretation
    st.info(f"**Interpretation:** The model predicts a **{risk_mean:.1%}** probability of clinically significant Prostate Cancer (csPCa). "
            f"Considering model uncertainty, the true risk likely lies between **{low_ci:.1%}** and **{high_ci:.1%}**.")
