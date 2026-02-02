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
    # Load the .pkl file located in the same root directory
    return joblib.load("cspca_prediction_system.pkl")

try:
    data_packet = load_prediction_system()
    
    # Unpack data
    base_models = data_packet["base_models"]
    knots = data_packet["spline_knots"]
    feature_mapping = data_packet.get("model_features", {})
    THRESHOLD = data_packet.get("threshold", 0.2)
    
    # Retrieve Weights (Crucial for Meta-stacking)
    meta_weights = data_packet.get("meta_weights")
    bootstrap_weights = data_packet.get("bootstrap_weights")

    if meta_weights is None:
        st.error("❌ Error: Model file is outdated. Please re-export the .pkl file containing 'meta_weights'.")
        st.stop()

except FileNotFoundError:
    st.error("❌ Critical Error: 'cspca_prediction_system.pkl' not found. Please ensure it is in the root directory of your GitHub repo.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🛡️ csPCa Risk & Uncertainty Analysis")
st.markdown(f"**Meta-stacking Ensemble Model (Decision Threshold: {THRESHOLD:.0%})**")

with st.expander("📚 Clinical Standards & Inclusion Criteria", expanded=False):
    st.markdown("""
    This model is optimized for patients meeting the combined criteria of **ERSPC** and **PCPT** trials:
    * **Age:** 55 – 75 years.
    * **PSA Level:** 0.4 – 50.0 ng/mL.
    * **Prostate Volume:** 10 – 110 mL.
    * **MRI Requirement:** PI-RADS Max Score ≥ 3.
    """)

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("📋 Patient Data")
    
    # Numeric Inputs
    age = st.number_input("Age (years)", 40, 95, 65)
    psa = st.number_input("Total PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.1)
    vol = st.number_input("Prostate Volume (mL)", 5, 300, 45, step=1)
    pirads = st.selectbox("PI-RADS Max Score (≥3)", [3, 4, 5], index=1)
    
    st.divider()
    
    # Categorical Inputs
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
    
    # Create Input DataFrame (Mapping user input to model features)
    input_dict = {
        "age": [age],
        "PSA": [psa],
        "log_PSA": [log_psa_val],
        "log_vol": [log_vol_val],
        "pirads_max": [pirads],
        
        # Binary/One-Hot mappings
        "tr_yes": [1 if dre_opt == "Abnormal" else 0],
        "fam_yes": [1 if fam_opt == "Yes" else 0],
        "atcd_yes": [1 if biopsy_opt == "Prior Negative" else 0],
        
        # Label encoded mappings (for Trees)
        "tr": [1 if dre_opt == "Abnormal" else 0],
        "fam": [1 if fam_opt == "Yes" else (2 if fam_opt == "Unknown" else 0)],
        "atcd": [1 if biopsy_opt == "Prior Negative" else 0],
        
        # 'Unknown' columns
        "fam_unknown": [1 if fam_opt == "Unknown" else 0],
        "tr_unknown": [0],
        "atcd_unknown": [0]
    }
    df_input = pd.DataFrame(input_dict)
    
   # --- B. Spline Logic (FIXED NAMES & BOUNDS) ---
    try:
        # 1. Tính toán biên an toàn
        safe_lb = min(knots) - 5.0
        safe_ub = max(knots) + 5.0
        
        # 2. Tạo Spline với biên an toàn
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        
        spline_df = dmatrix(
            spline_formula, 
            {
                "log_PSA": df_input["log_PSA"], 
                "knots": knots,
                "lb": safe_lb,
                "ub": safe_ub
            }, 
            return_type="dataframe"
        )
        
        # 3. SỬA LỖI TÊN CỘT (QUAN TRỌNG)
        # Model Lasso mong đợi tên cột cũ (không có lower_bound=...), ta phải đổi tên lại cho khớp.
        new_column_names = {}
        for col in spline_df.columns:
            if "Intercept" in col:
                continue # Intercept xử lý sau
            
            # Lấy chỉ số đuôi [0], [1], [2]...
            if "[" in col and "]" in col:
                idx = col.split("[")[-1].split("]")[0] 
                # Tạo lại tên chuẩn mà Model đã học lúc train
                original_name = f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{idx}]"
                new_column_names[col] = original_name
        
        # Áp dụng đổi tên
        spline_df = spline_df.rename(columns=new_column_names)

        # 4. Xử lý Intercept (Dựa theo lỗi báo thiếu Intercept)
        # Nếu model cần Intercept, ta thêm cột Intercept = 1.0 thủ công
        if "Intercept" not in spline_df.columns:
            spline_df["Intercept"] = 1.0
            
        # 5. Ghép vào DataFrame chính
        df_full = pd.concat([df_input, spline_df], axis=1)

    except Exception as e:
        st.error(f"Spline Error: {e}")
        st.stop()

    # --- C. Prediction Loop ---
    base_preds = []
    model_names = list(base_models.keys())
    
    for name in model_names:
        model = base_models[name]
        
        # Select correct columns for this specific model
        if name in feature_mapping:
            required_cols = feature_mapping[name]
        else:
            required_cols = df_full.columns.tolist() # Fallback
            
        # Check for missing columns
        missing = [c for c in required_cols if c not in df_full.columns]
        if missing:
            st.error(f"Model '{name}' missing columns: {missing}")
            st.stop()
            
        # Predict
        X_subset = df_full[required_cols]
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_subset)[:, 1][0]
            else:
                p = model.predict(X_subset)[0]
            base_preds.append(p)
        except Exception as e:
            st.error(f"Error running model '{name}': {e}")
            st.stop()
    
    base_preds = np.array(base_preds)

    # --- D. Meta-Prediction (Weighted Average) ---
    # Dot product of predictions and meta-weights
    risk_mean = np.dot(base_preds, meta_weights)
    
    # Calculate Confidence Intervals using Bootstrap Weights
    if bootstrap_weights is not None:
        boot_preds = np.dot(bootstrap_weights, base_preds)
        low_ci = np.percentile(boot_preds, 2.5)
        high_ci = np.percentile(boot_preds, 97.5)
        has_ci = True
    else:
        low_ci = risk_mean
        high_ci = risk_mean
        has_ci = False

    # ==========================================
    # 5. OUTPUT DISPLAY
    # ==========================================
    st.divider()
    st.subheader("📊 Quantitative Assessment")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk_mean:.1%}", delta="High Risk" if risk_mean > THRESHOLD else "Low Risk", delta_color="inverse")
    
    if has_ci:
        c2.metric("Lower 95% CI", f"{low_ci:.1%}")
        c3.metric("Upper 95% CI", f"{high_ci:.1%}")
    else:
        c2.metric("Lower 95% CI", "N/A")
        c3.metric("Upper 95% CI", "N/A")

    # Visual Chart
    st.write("### 🔍 Uncertainty Visualization")
    if has_ci:
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Density plot
        sns.kdeplot(boot_preds, fill=True, color="skyblue", alpha=0.3, ax=ax, label="Bootstrap Distribution")
        
        # Indicators
        ax.axvline(risk_mean, color="red", linestyle="-", linewidth=2, label=f"Mean Risk ({risk_mean:.1%})")
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Biopsy Threshold ({THRESHOLD:.0%})")
        
        # Confidence Interval Area
        ax.axvspan(low_ci, high_ci, color='gray', alpha=0.1, label='95% Confidence Interval')
        
        ax.set_title("Risk Probability Distribution (Uncertainty Analysis)")
        ax.set_xlabel("Predicted Probability of csPCa")
        ax.set_xlim(0, 1)
        ax.legend()
        st.pyplot(fig)

    # Text Interpretation
    st.info(f"**Interpretation:** The model predicts a **{risk_mean:.1%}** probability of clinically significant Prostate Cancer (csPCa). "
            f"Considering model uncertainty, the true risk likely lies between **{low_ci:.1%}** and **{high_ci:.1%}**.")
