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
    THRESHOLD = data_packet.get("threshold", 0.20)
    meta_weights = data_packet.get("meta_weights")
    meta_intercept = data_packet.get("meta_intercept", 0.0) 
    bootstrap_weights = data_packet.get("bootstrap_weights")
    bootstrap_intercepts = data_packet.get("bootstrap_intercepts")

    if meta_weights is None:
        st.error("❌ Error: Missing 'meta_weights'.")
        st.stop()

except FileNotFoundError:
    st.error("❌ Error: 'cspca_prediction_system.pkl' not found.")
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
    
    # -----------------------------------------------------------
    # ⚙️ TÍNH NĂNG HIỆU CHỈNH THÔNG MINH (SMART CALIBRATION)
    # -----------------------------------------------------------
    st.divider()
    with st.expander("⚙️ Calibration (Hiệu chỉnh Mô hình)", expanded=True):
        st.caption("Để kết quả chính xác nhất, vui lòng nhập thông tin dịch tễ tại cơ sở của bạn.")
        
        # 1. Nhập tỷ lệ thực tế (User Input)
        local_prev_pct = st.number_input(
            "Tỷ lệ ung thư (csPCa) trung bình tại cơ sở của bạn (%):", 
            min_value=1.0, max_value=80.0, value=15.0, step=0.5, format="%.1f",
            help="Trong 100 bệnh nhân có chỉ định sinh thiết tại nơi bạn làm việc, trung bình có bao nhiêu người dương tính?"
        )
        
        # 2. Định nghĩa tỷ lệ gốc của mô hình (Hardcoded từ tập Training)
        # Giả sử tập train của bạn là bệnh viện tuyến cuối, tỷ lệ bệnh là 40% (0.40)
        TRAIN_PREV = 0.40 
        
        # 3. Tính toán Offset tự động
        target_prev = local_prev_pct / 100.0
        
        def logit(p): return np.log(p / (1 - p))
        
        # Công thức Bayes: Offset = Logit(Target) - Logit(Train)
        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)
        
        # Hiển thị thông tin cho bác sĩ yên tâm
        st.info(f"✅ Hệ thống đã tự động hiệu chỉnh.\n(Mức nền được điều chỉnh từ {TRAIN_PREV:.0%} xuống {target_prev:.1%})")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("🚀 RUN ANALYSIS", type="primary"):
    
    # 1. VALIDATION CHECK
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
        safe_lb = min(knots) - 5.0
        safe_ub = max(knots) + 5.0
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
    except Exception as e:
        st.error(f"Spline Error: {e}")
        st.stop()

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
    
    # 4. META-PREDICTION (WITH AUTO CALIBRATION)
    raw_log_odds = np.dot(base_preds, meta_weights) + meta_intercept
    
    # --- ÁP DỤNG OFFSET TỰ ĐỘNG ---
    final_log_odds = raw_log_odds + CALIBRATION_OFFSET
    risk_mean = 1 / (1 + np.exp(-final_log_odds))
    
    if bootstrap_weights is not None:
        boot_log_odds = np.dot(bootstrap_weights, base_preds)
        if bootstrap_intercepts is not None: boot_log_odds += bootstrap_intercepts
        
        # Calibration cho bootstrap
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
        # Chú thích rõ ràng cho bác sĩ hiểu biểu đồ đang vẽ theo tỷ lệ nào
        plt.title(f"Calibrated to {local_prev_pct:.1f}% Prevalence (n={len(bootstrap_weights)})", fontsize=9, color='#666', style='italic', pad=10)
        
        ax.set_xlabel("Predicted Probability of csPCa"); ax.set_ylabel("Density")
        ax.set_xlim(0, max(0.6, high_ci + 0.15))
        ax.legend(loc='best', fontsize=8, framealpha=0.95)
        sns.despine(offset=5, trim=True)
        st.pyplot(fig, dpi=300, use_container_width=False)
        sns.reset_orig()

    st.subheader("💡 Clinical Recommendation")
    if risk_mean >= GRAY_HIGH:
        st.error(f"**🔴 HIGH RISK**: Strong indication for **Targeted Biopsy**.")
    elif risk_mean >= GRAY_LOW:
        st.warning(f"**🟡 INTERMEDIATE RISK**: Consider **Shared Decision Making** & secondary markers.")
    else:
        st.success(f"**🟢 LOW RISK**: Biopsy may be avoided. Continue **PSA Monitoring**.")
    
    # Footer giải thích
    st.info(f"**Note:** Model results have been calibrated to match your facility's local prevalence of **{local_prev_pct}%**.")
