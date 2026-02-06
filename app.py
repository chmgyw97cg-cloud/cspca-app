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
# 2. LANGUAGE DICTIONARY (Y KHOA CHUẨN)
# ==========================================
TRANS = {
    "🇬🇧 English": {
        "title": "🛡️ csPCa Risk & Uncertainty Analysis",
        "subtitle": "**Standardized Meta-Stacking Ensemble** | Clinical Decision Support",
        "def": "**Definition:** csPCa (Clinically Significant Prostate Cancer) is defined as **ISUP Grade Group ≥ 2**.",
        "scope": "**Scope:** Prediction applies to **MRI-Targeted Biopsy (ROI-only)**.",
        "expander_title": "📚 Clinical Standards & Inclusion Criteria",
        "expander_content": """
        * **Age:** 55 – 75 years.
        * **PSA Level:** 0.4 – 50.0 ng/mL.
        * **Prostate Volume:** 10 – 110 mL.
        * **MRI Requirement:** PI-RADS Max Score ≥ 3.
        """,
        "sidebar_header": "📋 Patient Data",
        "lbl_age": "Age (years)",
        "lbl_psa": "Total PSA (ng/mL)",
        "lbl_vol": "Prostate Volume (mL)",
        "lbl_pirads": "PI-RADS Max Score (≥3)",
        "lbl_dre": "Digital Rectal Exam (DRE)",
        "opt_dre": ["Normal", "Abnormal", "Unknown"],
        "lbl_fam": "Family History",
        "opt_fam": ["No", "Yes", "Unknown"],
        "lbl_biopsy": "Biopsy History",
        "opt_biopsy": ["Naïve", "Prior Negative", "Unknown"],
        "calib_title": "⚙️ Calibration Details",
        "calib_desc": "**Standard: PRECISION Trial**\n\nStandard yield for MRI-Targeted Biopsy (ROI) in men with PI-RADS ≥ 3.",
        "calib_input": "Target Yield within ROI (%):",
        "calib_info": "✅ Adjusted:",
        "btn_run": "🚀 RUN ANALYSIS",
        "warn_age": "⚠️ **Age ({})** is outside the model's primary range (55-75).",
        "warn_psa": "⚠️ **PSA ({:.1f})** is outside the model's primary range (0.4-50.0).",
        "warn_vol": "⚠️ **Prostate Volume ({:.1f})** is outside the model's primary range (10-110).",
        "warn_title": "### ⚠️ Clinical Warning: Out of Distribution",
        "warn_footer": "The prediction may be less reliable for patients outside these criteria.",
        "res_title": "📊 Quantitative Assessment",
        "res_risk": "Predicted Risk",
        "res_low": "Lower 95% CI",
        "res_high": "Upper 95% CI",
        "res_interp": "**Interpretation:** The model predicts a **{:.1%}** probability of csPCa within the ROI.",
        "res_uncert": "**Uncertainty Note:** Based on 1,000 bootstrap simulations, the 95% CI is **{:.1%}** to **{:.1%}** (uncertainty spread: **{:.1%}**). **A narrower distribution reflects higher model confidence**.",
        "plot_title": "🔍 Risk Probability Distribution",
        "plot_xlabel": "Predicted Probability of csPCa",
        "plot_ylabel": "Probability density",
        "plot_legend_dist": "Risk Distribution",
        "plot_legend_point": "Point Estimate"
    },
    "🇫🇷 Français": {
        "title": "🛡️ Analyse de Risque csPCa & Incertitude",
        "subtitle": "**Ensemble Meta-Stacking Standardisé** | Aide à la Décision Médicale",
        "def": "**Définition :** csPCa (Cancer de la Prostate Cliniquement Significatif) défini par **ISUP Grade Group ≥ 2**.",
        "scope": "**Portée :** Prédiction applicable à la **Biopsie Ciblée par IRM (ROI uniquement)**.",
        "expander_title": "📚 Standards Cliniques & Critères d'Inclusion",
        "expander_content": """
        * **Âge :** 55 – 75 ans.
        * **Taux de PSA :** 0.4 – 50.0 ng/mL.
        * **Volume Prostatique :** 10 – 110 mL.
        * **Critère IRM :** Score PI-RADS Max ≥ 3.
        """,
        "sidebar_header": "📋 Données Patient",
        "lbl_age": "Âge (ans)",
        "lbl_psa": "PSA Total (ng/mL)",
        "lbl_vol": "Volume Prostatique (mL)",
        "lbl_pirads": "Score PI-RADS Max (≥3)",
        "lbl_dre": "Toucher Rectal (TR)",
        "opt_dre": ["Normal", "Anormal", "Inconnu"],
        "lbl_fam": "Antécédents Familiaux",
        "opt_fam": ["Non", "Oui", "Inconnu"],
        "lbl_biopsy": "Historique de Biopsie",
        "opt_biopsy": ["Première biopsie (Naïf)", "Négative Précédente", "Inconnu"],
        "calib_title": "⚙️ Détails de Calibrage",
        "calib_desc": "**Standard : Essai PRECISION**\n\nRendement standard pour la biopsie ciblée IRM (ROI) chez les hommes avec PI-RADS ≥ 3.",
        "calib_input": "Taux de positivité des biopsies:",
        "calib_info": "✅ Ajusté :",
        "btn_run": "🚀 LANCER L'ANALYSE",
        "warn_age": "⚠️ **Âge ({})** est hors de la plage primaire du modèle (55-75).",
        "warn_psa": "⚠️ **PSA ({:.1f})** est hors de la plage primaire du modèle (0.4-50.0).",
        "warn_vol": "⚠️ **Volume ({:.1f})** est hors de la plage primaire du modèle (10-110).",
        "warn_title": "### ⚠️ Avertissement Clinique : Hors Distribution",
        "warn_footer": "La prédiction peut être moins fiable pour les patients hors de ces critères.",
        "res_title": "📊 Évaluation Quantitative",
        "res_risk": "Risque Prédit",
        "res_low": "IC 95% Inf",
        "res_high": "IC 95% Sup",
        "res_interp": "**Interprétation :** Le modèle prédit une probabilité de **{:.1%}** de csPCa dans la ROI.",
        "res_uncert": "**Note d'Incertitude :** Basé sur 1 000 simulations bootstrap, l'IC 95% est de **{:.1%}** à **{:.1%}** (écart : **{:.1%}**). **Une distribution plus étroite reflète une confiance plus élevée**.",
        "plot_title": "🔍 Distribution de Probabilité du Risque",
        "plot_xlabel": "Probabilité prédite de csPCa",
        "plot_ylabel": "Densité de la distribution bootstrap",
        "plot_legend_dist": "Distribution du Risque",
        "plot_legend_point": "Estimation Ponctuelle"
    },
    "🇻🇳 Tiếng Việt": {
        "title": "🛡️ Phân tích Nguy cơ Ung thư tuyến tiền liệt có ý nghĩa lâm sàng & Độ bất định",
        "subtitle": "**Mô hình Meta-Stacking Ensemble** | Hỗ trợ Ra quyết định Lâm sàng",
        "def": "**Định nghĩa:** csPCa (Ung thư tiền liệt tuyến có ý nghĩa lâm sàng) được định nghĩa là **ISUP Grade Group ≥ 2**.",
        "scope": "**Phạm vi:** Dự báo áp dụng cho **Sinh thiết trúng đích MRI (chỉ vùng ROI)**.",
        "expander_title": "📚 Tiêu chuẩn Lâm sàng & Tiêu chí Lựa chọn",
        "expander_content": """
        * **Tuổi:** 55 – 75 tuổi.
        * **Nồng độ PSA:** 0.4 – 50.0 ng/mL.
        * **Thể tích tuyến:** 10 – 110 mL.
        * **Yêu cầu MRI:** Điểm PI-RADS Max ≥ 3.
        """,
        "sidebar_header": "📋 Dữ liệu Bệnh nhân",
        "lbl_age": "Tuổi (năm)",
        "lbl_psa": "PSA Toàn phần (ng/mL)",
        "lbl_vol": "Thể tích Tuyến tiền liệt (mL)",
        "lbl_pirads": "Điểm PI-RADS Max (≥3)",
        "lbl_dre": "Thăm trực tràng",
        "opt_dre": ["Bình thường", "Bất thường", "Không rõ"],
        "lbl_fam": "Tiền sử Gia đình",
        "opt_fam": ["Không", "Có", "Không rõ"],
        "lbl_biopsy": "Tiền sử Sinh thiết",
        "opt_biopsy": ["Chưa từng (Naïve)", "Đã từng (Âm tính)", "Không rõ"],
        "calib_title": "⚙️ Hiệu chỉnh (Calibration)",
        "calib_desc": "**Tiêu chuẩn: Thử nghiệm PRECISION**\n\nTỷ lệ phát hiện chuẩn cho sinh thiết trúng đích MRI (ROI) ở bệnh nhân PI-RADS ≥ 3.",
        "calib_input": "Tỷ lệ hiện mắc tại cơ sở (%):",
        "calib_info": "✅ Đã hiệu chỉnh:",
        "btn_run": "🚀 CHẠY PHÂN TÍCH",
        "warn_age": "⚠️ **Tuổi ({})** nằm ngoài phạm vi chính của mô hình (55-75).",
        "warn_psa": "⚠️ **PSA ({:.1f})** nằm ngoài phạm vi chính của mô hình (0.4-50.0).",
        "warn_vol": "⚠️ **Thể tích ({:.1f})** nằm ngoài phạm vi chính của mô hình (10-110).",
        "warn_title": "### ⚠️ Cảnh báo Lâm sàng: Ngoài vùng dữ liệu",
        "warn_footer": "Kết quả dự báo có thể kém tin cậy đối với bệnh nhân nằm ngoài các tiêu chuẩn này.",
        "res_title": "📊 Đánh giá Định lượng",
        "res_risk": "Nguy cơ Dự báo",
        "res_low": "KTC 95% (Dưới)",
        "res_high": "KTC 95% (Trên)",
        "res_interp": "**Diễn giải:** Mô hình dự báo xác suất **{:.1%}** mắc csPCa trong vùng ROI.",
        "res_uncert": "**Ghi chú về Độ bất định:** Dựa trên 1,000 mô phỏng bootstrap, khoảng tin cậy (CI) 95% là từ **{:.1%}** đến **{:.1%}** (độ rộng phân tán: **{:.1%}**). **Phân phối càng hẹp thể hiện độ tin cậy của mô hình càng cao**.",
        "plot_title": "🔍 Phân phối Xác suất Nguy cơ",
        "plot_xlabel": "Xác suất Dự báo csPCa",
        "plot_ylabel": "Mật độ phân phối",
        "plot_legend_dist": "Phân phối Nguy cơ",
        "plot_legend_point": "Điểm Ước lượng"
    }
}

# ==========================================
# 3. MODEL LOADING
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
    
    if meta_weights is None: st.error("❌ Error: Missing weights."); st.stop()
except Exception as e:
    st.error(f"❌ Critical Error: {e}"); st.stop()

# ==========================================
# 4. USER INTERFACE
# ==========================================

# --- LANGUAGE SELECTOR (TOP RIGHT) ---
# Sử dụng cột để đẩy nút chọn sang bên phải
col_header, col_lang = st.columns([6, 2])

with col_lang:
    # Chọn ngôn ngữ, mặc định là English (index 0)
    selected_lang = st.selectbox(
        "Language / Langue / Ngôn ngữ", 
        ["🇬🇧 English", "🇫🇷 Français", "🇻🇳 Tiếng Việt"], 
        index=0,
        label_visibility="collapsed" # Ẩn nhãn cho gọn
    )

# Lấy từ điển ngôn ngữ hiện tại
T = TRANS[selected_lang]

# --- MAIN HEADER ---
st.title(T["title"])
st.markdown(T["subtitle"])
st.caption(T["def"])
st.caption(T["scope"])

with st.expander(T["expander_title"], expanded=False):
    st.markdown(T["expander_content"])
    
with st.sidebar:
    st.header(T["sidebar_header"])
    
    # Định dạng input
    age = st.number_input(T["lbl_age"], 40, 95, 65)
    psa = st.number_input(T["lbl_psa"], 0.1, 200.0, 7.5, step=0.1, format="%.1f")
    vol = st.number_input(T["lbl_vol"], 5.0, 300.0, 45.0, step=0.1, format="%.1f")
    pirads = st.selectbox(T["lbl_pirads"], [3, 4, 5], index=1)
    
    st.divider()
    
    # Logic xử lý lựa chọn ngôn ngữ cho Radio button
    # Vì giá trị trả về là Text đã dịch, ta cần map ngược lại về giá trị Logic tiếng Anh để tính toán
    
    # DRE
    dre_display = st.radio(T["lbl_dre"], T["opt_dre"], horizontal=True)
    # Map selection to English logic keys
    dre_map = dict(zip(T["opt_dre"], ["Normal", "Abnormal", "Unknown"]))
    dre_opt = dre_map[dre_display]

    # Family History
    fam_display = st.radio(T["lbl_fam"], T["opt_fam"], horizontal=True)
    fam_map = dict(zip(T["opt_fam"], ["No", "Yes", "Unknown"]))
    fam_opt = fam_map[fam_display]

    # Biopsy History
    biopsy_display = st.radio(T["lbl_biopsy"], T["opt_biopsy"], horizontal=True)
    biopsy_map = dict(zip(T["opt_biopsy"], ["Naïve", "Prior Negative", "Unknown"]))
    biopsy_opt = biopsy_map[biopsy_display]
    
    st.divider()
    with st.expander(T["calib_title"], expanded=True):
        st.markdown(T["calib_desc"])
        
        DEFAULT_TARGET = 38.0 # Based on PRECISION NEJM 2018
        
        local_prev_pct = st.number_input(
            T["calib_input"], 
            min_value=1.0, max_value=99.0, 
            value=DEFAULT_TARGET, 
            step=0.5, format="%.1f"
        )
        st.caption("*Ref: Kasivisvanathan et al., NEJM 2018.*")
        
        TRAIN_PREV = 0.452 # Development cohort prevalence
        
        target_prev = local_prev_pct / 100.0
        def logit(p): return np.log(p / (1 - p))
        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)
        
        st.info(f"{T['calib_info']} **{TRAIN_PREV:.1%}** ➔ **{local_prev_pct}%**")


# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if st.button(T["btn_run"], type="primary"):
    # 0. CLINICAL VALIDATION
    warnings = []
    if not (55 <= age <= 75):
        warnings.append(T["warn_age"].format(age))
    if not (0.4 <= psa <= 50.0):
        warnings.append(T["warn_psa"].format(psa))
    if not (10 <= vol <= 110):
        warnings.append(T["warn_vol"].format(vol))
    
    if warnings:
        with st.container():
            st.warning(T["warn_title"])
            for w in warnings:
                st.markdown(w)
            st.caption(T["warn_footer"])
                
    # 1. PRE-PROCESSING
    log_psa_val = np.log(psa)
    log_vol_val = np.log(vol)
    psad = psa / vol
    
    # Input dictionary uses the English Logic Variables (dre_opt, etc.)
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
    st.subheader(T["res_title"])

    c1, c2, c3 = st.columns(3)
    c1.metric(T["res_risk"], f"{risk_mean:.1%}")
    c2.metric(T["res_low"], f"{low_ci:.1%}" if has_ci else "N/A")
    c3.metric(T["res_high"], f"{high_ci:.1%}" if has_ci else "N/A")

    st.info(
        T["res_interp"].format(risk_mean) + "\n\n" +
        T["res_uncert"].format(low_ci, high_ci, high_ci - low_ci)
    )

    # --- UNCERTAINTY VISUALIZATION (TRANSLATED) ---
    st.write(f"### {T['plot_title']}")
    if has_ci:
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        
        # Plot KDE
        sns.kdeplot(boot_preds, fill=True, color="#2c3e50", alpha=0.3, ax=ax, linewidth=2, label=T["plot_legend_dist"])
        
        # Vertical line for mean risk
        ax.axvline(risk_mean, color="#d95f02", linestyle="-", linewidth=2.5, label=f"{T['plot_legend_point']}: {risk_mean:.1%}")
        
        # Formatting
        plt.title("Bootstrap Uncertainty Analysis", fontsize=12, fontweight='bold', pad=15) # Giữ tiêu đề kỹ thuật tiếng Anh hoặc đổi nếu muốn
        ax.set_xlabel(T["plot_xlabel"], fontsize=10)
        ax.set_ylabel(T["plot_ylabel"], fontsize=10)
        ax.set_xlim(0, max(0.6, high_ci + 0.1))
        ax.legend(loc='best', fontsize=9)
        
        sns.despine()
        st.pyplot(fig, dpi=300)
        
    st.caption(f"**Calculated PSA Density (PSAD):** {psad:.2f} ng/mL²")
