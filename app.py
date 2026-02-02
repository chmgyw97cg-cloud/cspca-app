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
    
    # Clinical Thresholds (Default to 20% if not specified)
    THRESHOLD = data_packet.get("threshold", 0.20)
    
    # --- CẬP NHẬT QUAN TRỌNG: LOAD TRỌNG SỐ & INTERCEPT ---
    meta_weights = data_packet.get("meta_weights")
    # Lấy Intercept (nếu không có thì mặc định là 0)
    meta_intercept = data_packet.get("meta_intercept", 0.0) 
    
    bootstrap_weights = data_packet.get("bootstrap_weights")
    bootstrap_intercepts = data_packet.get("bootstrap_intercepts")

    if meta_weights is None:
        st.error("❌ Error: Missing 'meta_weights' in model file.")
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
    psa = st.number_input("Total PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.1, format="%.1f")
    # Giữ nguyên volume số thực như bạn yêu cầu
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
    
    # Input Dictionary Mapping
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
    
    # --- B. Spline Logic (Robust Fix) ---
    try:
        # Define safety bounds to prevent Patsy errors
        safe_lb = min(knots) - 5.0
        safe_ub = max(knots) + 5.0
        
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(spline_formula, 
                           {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub}, 
                           return_type="dataframe")
        
        # Rename columns to match training data
        rename_map = {}
        for col in spline_df.columns:
            if "Intercept" in col: continue
            match = re.search(r"\[(\d+)\]$", col)
            if match:
                idx = match.group(1)
                original_name = f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{idx}]"
                rename_map[col] = original_name
        spline_df = spline_df.rename(columns=rename_map)

        # Add Intercept if required by Lasso
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
        
        # Feature selection
        if name in feature_mapping:
            required_cols = feature_mapping[name]
        else:
            required_cols = df_full.columns.tolist()
            
        # Check missing columns
        missing = [c for c in required_cols if c not in df_full.columns]
        if missing:
            st.error(f"❌ Model '{name}' missing columns: {missing}")
            st.stop()
            
        # Inference
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
    
    # --- D. Meta-Prediction (SỬA LỖI 169%: THÊM SIGMOID) ---
    
    # 1. Tính Log-odds (Linear combination: w*x + b)
    log_odds = np.dot(base_preds, meta_weights) + meta_intercept
    
    # 2. Chuyển sang Xác suất (Sigmoid Function: 1 / (1 + e^-z))
    # Bước này ép giá trị về khoảng [0, 1] -> Hết lỗi > 100%
    risk_mean = 1 / (1 + np.exp(-log_odds))
    
    # 3. Bootstrap Uncertainty (Cũng dùng Sigmoid)
    if bootstrap_weights is not None:
        # Tính Log-odds cho 1000 mẫu
        boot_log_odds = np.dot(bootstrap_weights, base_preds)
        
        # Cộng intercept nếu có
        if bootstrap_intercepts is not None:
            boot_log_odds += bootstrap_intercepts
            
        # Chuyển tất cả sang Xác suất
        boot_preds = 1 / (1 + np.exp(-boot_log_odds))
        
        low_ci, high_ci = np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)
        has_ci = True
    else:
        low_ci, high_ci = risk_mean, risk_mean
        has_ci = False

    # ==========================================
    # 5. OUTPUT DISPLAY (SCIENTIFIC VERSION)
    # ==========================================
    st.divider()
    st.subheader("📊 Quantitative Assessment")

    # 1. Define Clinical Thresholds
    GRAY_LOW = 0.10        # < 10%: Safety Net
    GRAY_HIGH = THRESHOLD  # >= 20%: Biopsy Threshold

    # 2. Determine Labels
    if risk_mean < GRAY_LOW:
        risk_label = "Low Risk"
        delta_color = "normal" 
    elif risk_mean < GRAY_HIGH:
        risk_label = "Intermediate Risk (Gray Zone)"
        delta_color = "off"
    else:
        risk_label = "High Risk"
        delta_color = "inverse"

    # 3. Display Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{risk_mean:.1%}", delta=risk_label, delta_color=delta_color)

    if has_ci:
        c2.metric("Lower 95% CI", f"{low_ci:.1%}")
        c3.metric("Upper 95% CI", f"{high_ci:.1%}")
    else:
        c2.metric("Lower 95% CI", "N/A")
        c3.metric("Upper 95% CI", "N/A")

  # 4. Uncertainty Visualization (High Contrast Scientific Style)
    st.write("### 🔍 Uncertainty Visualization")
    if has_ci:
        # --- NATURE STYLE SETUP ---
        sns.set_theme(style="ticks", context="paper", font_scale=1.1)
        
        # figsize=(width, height). Giữ kích thước gọn (8, 3)
        fig, ax = plt.subplots(figsize=(8, 3))

        # --- BẢNG MÀU MỚI (HIGH CONTRAST) ---
        # Dùng tông Xanh Dương - Cam - Đỏ (Blue - Orange - Red)
        # Đây là bảng màu chuẩn "Paired" giúp phân biệt cực rõ vùng An toàn và Vùng xám
        color_low = '#a6cee3'  # Light Blue (Thay cho xanh lá - Rất dễ nhìn)
        color_mid = '#fdbf6f'  # Light Orange (Thay cho vàng nhạt - Tương phản tốt với xanh)
        color_high = '#fb9a99' # Light Red (Giữ nguyên màu đỏ cảnh báo)
        
        # Background Zones
        ax.axvspan(0, GRAY_LOW, color=color_low, alpha=0.3, label='Low Risk Zone', lw=0)
        ax.axvspan(GRAY_LOW, GRAY_HIGH, color=color_mid, alpha=0.3, label='Intermediate Zone', lw=0)
        ax.axvspan(GRAY_HIGH, 1.0, color=color_high, alpha=0.3, label='High Risk Zone', lw=0)

        # Density Plot 
        # Đổi màu đường viền density sang màu Tím than (Dark Slate) để nổi bật trên mọi nền
        sns.kdeplot(boot_preds, fill=True, color="#2c3e50", alpha=0.4, ax=ax, linewidth=1.5)
        
        # Indicator Lines
        ax.axvline(risk_mean, color="#e31a1c", linestyle="-", linewidth=2, label=f"Mean Prediction: {risk_mean:.1%}")
        ax.axvline(GRAY_HIGH, color="black", linestyle="--", linewidth=1.2, label=f"Biopsy Threshold: {GRAY_HIGH:.0%}")

        # Titles and Subtitles
        n_boot = len(bootstrap_weights) if bootstrap_weights is not None else 0
        plt.suptitle("Estimated Risk Distribution & Confidence Intervals", 
                     y=1.02, fontsize=12, fontweight='bold', color='#333')
        plt.title(f"Method: Kernel Density Estimation (n = {n_boot} bootstrap iterations)", 
                  fontsize=9, color='#666', style='italic', pad=10)

        # Axis Formatting
        ax.set_xlabel("Predicted Probability of csPCa", labelpad=5)
        ax.set_ylabel("Density (Bootstrap)", labelpad=5)
        
        # Set X-axis limit
        x_max = max(0.6, high_ci + 0.15)
        ax.set_xlim(0, x_max)
        
        # Legend
        ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='#e0e0e0', framealpha=0.95, shadow=False)
        
        # Despine
        sns.despine(offset=5, trim=True)
        
        # Hiển thị
        st.pyplot(fig, dpi=300, use_container_width=False)
        
        sns.reset_orig()

    # 5. Clinical Recommendation (3 Levels)
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
        * **Action:** Consider **Shared Decision Making**. Evaluate secondary factors (e.g., PSA Density, Free/Total PSA) before deciding on biopsy.
        """)
    else:
        st.success(f"""
        **🟢 LOW RISK (< {GRAY_LOW:.0%})**
        * **Probability:** {risk_mean:.1%} (CI: {low_ci:.1%} - {high_ci:.1%}).
        * **Interpretation:** High Negative Predictive Value (NPV).
        * **Action:** Immediate biopsy may be avoided. Continue **PSA Monitoring**.
        """)

    # Footer Interpretation
    st.info(f"**Interpretation:** The model predicts a **{risk_mean:.1%}** probability of clinically significant Prostate Cancer (csPCa). "
            f"Considering model uncertainty, the true risk likely lies between **{low_ci:.1%}** and **{high_ci:.1%}**.")
