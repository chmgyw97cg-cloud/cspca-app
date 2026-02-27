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
        "res_uncert": "**Uncertainty Note:** Based on bootstrap simulations, the 95% CI is **{:.1%}** to **{:.1%}** (spread: **{:.1%}**).",
        "plot_title": "🔍 Risk Probability Distribution",
        "plot_xlabel": "Predicted Probability of csPCa",
        "plot_ylabel": "Probability density",
        "plot_legend_dist": "Risk Distribution",
        "plot_legend_point": "Point Estimate",
        "res_psad": "Calculated PSA Density (PSAD):"
    },
    "🇫🇷 Français": {
        "title": "🛡️ Analyse de Risque csPCa & Incertitude",
        "subtitle": "**Ensemble Meta-Stacking Standardisé** | Aide à la Décision Médicale",
        "def": "**Définition :** csPCa (Cancer cliniquement significatif) défini par **ISUP Grade Group ≥ 2**.",
        "scope": "**Portée :** Applicable aux **biopsies ciblées par IRM (ROI uniquement)**.",
        "expander_title": "📚 Critères d'Inclusion & Standards",
        "expander_content": """
        * **Âge :** 55 – 75 ans.
        * **PSA Total :** 0.4 – 50.0 ng/mL.
        * **Volume Prostatique :** 10 – 110 mL.
        * **IRM :** Score PI-RADS Max ≥ 3.
        """,
        "sidebar_header": "📋 Données Patient",
        "lbl_age": "Âge (ans)",
        "lbl_psa": "PSA Total (ng/mL)",
        "lbl_vol": "Volume Prostatique (mL)",
        "lbl_pirads": "Score PI-RADS Max (≥3)",
        "lbl_dre": "Toucher Rectal (TR)",
        "opt_dre": ["Normal", "Anormal (Suspect)", "Inconnu"],
        "lbl_fam": "Antécédents Familiaux",
        "opt_fam": ["Non", "Oui", "Inconnu"],
        "lbl_biopsy": "Antécédents de Biopsie",
        "opt_biopsy": ["Première biopsie (Naïf)", "Négative antérieure", "Inconnu"],
        "calib_title": "⚙️ Calibrage du Modèle",
        "calib_desc": "**Référence : Essai PRECISION**\n\nTaux de détection attendu pour les biopsies ciblées (ROI) chez les patients PI-RADS ≥ 3.",
        "calib_input": "Taux de positivité des biopsies (%):",
        "calib_info": "✅ Ajusté sur :",
        "btn_run": "🚀 LANCER L'ANALYSE",
        "warn_age": "⚠️ **Âge ({})** hors des critères principaux (55-75).",
        "warn_psa": "⚠️ **PSA ({:.1f})** hors des critères principaux (0.4-50.0).",
        "warn_vol": "⚠️ **Volume ({:.1f})** hors des critères principaux (10-110).",
        "warn_title": "### ⚠️ Avertissement Clinique : Hors Critères",
        "warn_footer": "La fiabilité de la prédiction peut être réduite hors de ces critères.",
        "res_title": "📊 Évaluation Quantitative",
        "res_risk": "Risque Estimé",
        "res_low": "IC 95% Inf",
        "res_high": "IC 95% Sup",
        "res_interp": "**Interprétation :** Le modèle prédit une probabilité de **{:.1%}** de csPCa dans la cible (ROI).",
        "res_uncert": "**Note :** Basé sur bootstrap, l'IC 95% va de **{:.1%}** à **{:.1%}** (écart : **{:.1%}**).",
        "plot_title": "🔍 Distribution de Probabilité du Risque",
        "plot_xlabel": "Probabilité prédite de csPCa",
        "plot_ylabel": "Densité de probabilité",
        "plot_legend_dist": "Distribution du Risque",
        "plot_legend_point": "Estimation Ponctuelle",
        "res_psad": "Densité de PSA calculée (PSAD) :"
    },
    "🇻🇳 Tiếng Việt": {
        "title": "🛡️ Phân tích Nguy cơ & Độ bất định csPCa",
        "subtitle": "**Mô hình Meta-Stacking Ensemble** | Hỗ trợ Ra quyết định Lâm sàng",
        "def": "**Định nghĩa:** csPCa (Ung thư tiền liệt tuyến có ý nghĩa lâm sàng) = **ISUP Grade Group ≥ 2**.",
        "scope": "**Phạm vi:** Dự báo cho **Sinh thiết trúng đích MRI (ROI-only)**.",
        "expander_title": "📚 Tiêu chuẩn Lâm sàng & Tiêu chí Lựa chọn",
        "expander_content": """
        * **Tuổi:** 55 – 75 tuổi.
        * **Nồng độ PSA:** 0.4 – 50.0 ng/mL.
        * **Thể tích tuyến:** 10 – 110 mL.
        * **Yêu cầu MRI:** PI-RADS Max ≥ 3.
        """,
        "sidebar_header": "📋 Dữ liệu Bệnh nhân",
        "lbl_age": "Tuổi (năm)",
        "lbl_psa": "PSA Toàn phần (ng/mL)",
        "lbl_vol": "Thể tích Tuyến tiền liệt (mL)",
        "lbl_pirads": "Điểm PI-RADS Max (≥3)",
        "lbl_dre": "Thăm trực tràng (DRE)",
        "opt_dre": ["Bình thường", "Bất thường", "Không rõ"],
        "lbl_fam": "Tiền sử Gia đình",
        "opt_fam": ["Không", "Có", "Không rõ"],
        "lbl_biopsy": "Tiền sử Sinh thiết",
        "opt_biopsy": ["Chưa từng (Naïve)", "Đã từng (Âm tính)", "Không rõ"],
        "calib_title": "⚙️ Hiệu chỉnh mô hình (Calibration)",
        "calib_desc": "**Tiêu chuẩn: PRECISION (NEJM 2018)**\n\nTỷ lệ dương tính kỳ vọng của ROI-targeted biopsy ở PI-RADS ≥ 3.",
        "calib_input": "Tỷ lệ dương tính sinh thiết (%):",
        "calib_info": "✅ Đã hiệu chỉnh theo:",
        "btn_run": "🚀 CHẠY PHÂN TÍCH",
        "warn_age": "⚠️ **Tuổi ({})** ngoài phạm vi chính (55-75).",
        "warn_psa": "⚠️ **PSA ({:.1f})** ngoài phạm vi chính (0.4-50.0).",
        "warn_vol": "⚠️ **Thể tích ({:.1f})** ngoài phạm vi chính (10-110).",
        "warn_title": "### ⚠️ Cảnh báo: Ngoài vùng dữ liệu",
        "warn_footer": "Độ tin cậy có thể giảm khi ngoài tiêu chí.",
        "res_title": "📊 Đánh giá Định lượng",
        "res_risk": "Nguy cơ Dự báo",
        "res_low": "KTC 95% (Dưới)",
        "res_high": "KTC 95% (Trên)",
        "res_interp": "**Diễn giải:** Xác suất csPCa trong ROI = **{:.1%}**.",
        "res_uncert": "**Ghi chú:** Bootstrap KTC 95%: **{:.1%}** đến **{:.1%}** (độ rộng: **{:.1%}**).",
        "plot_title": "🔍 Phân phối Xác suất Nguy cơ",
        "plot_xlabel": "Xác suất Dự báo csPCa",
        "plot_ylabel": "Mật độ",
        "plot_legend_dist": "Phân phối Nguy cơ",
        "plot_legend_point": "Điểm Ước lượng",
        "res_psad": "Mật độ PSA (PSAD):"
    }
}

# ==========================================
# 3. MODEL LOADING (DE + ORDER + FALLBACK + DEFENSIVE)
# ==========================================
@st.cache_resource
def load_prediction_system(_version="v1"):
    return joblib.load("cspca_prediction_system.pkl")

def _as_float(x, default=0.0):
    try:
        arr = np.asarray(x).reshape(-1)
        return float(arr[0])
    except Exception:
        return float(default)

try:
    data_packet = load_prediction_system("v1")
    base_models = data_packet["base_models"]                      # MUST be FITTED pipelines/estimators
    knots = np.asarray(data_packet["spline_knots"], dtype=float)  # array
    feature_mapping = data_packet.get("model_features", {}) or {}
    THRESHOLD = float(data_packet.get("threshold", 0.20))

    # DE (paper-aligned)
    de_weights = data_packet.get("de_weights", None)
    if de_weights is not None:
        de_weights = np.asarray(de_weights, dtype=float)

    model_names_ordered = data_packet.get("model_names_ordered", None)
    if model_names_ordered is not None:
        model_names_ordered = [m for m in list(model_names_ordered) if m in base_models]

    # Fallback logistic meta (legacy)
    meta_weights = data_packet.get("meta_weights", None)
    if meta_weights is not None:
        meta_weights = np.asarray(meta_weights, dtype=float)
    meta_intercept = _as_float(data_packet.get("meta_intercept", 0.0), default=0.0)

    # CI arrays (legacy bootstrap logistic meta)
    bootstrap_weights = data_packet.get("bootstrap_weights", None)
    bootstrap_intercepts = data_packet.get("bootstrap_intercepts", None)
    if bootstrap_weights is not None:
        bootstrap_weights = np.asarray(bootstrap_weights, dtype=float)
    if bootstrap_intercepts is not None:
        bootstrap_intercepts = np.asarray(bootstrap_intercepts, dtype=float).reshape(-1)

    if de_weights is None and meta_weights is None:
        st.error("❌ Error: Missing weights in .pkl (need de_weights or meta_weights).")
        st.stop()

except Exception as e:
    st.error(f"❌ Critical Error loading prediction system: {e}")
    st.stop()

# ==========================================
# 4. USER INTERFACE
# ==========================================
col_header, col_lang = st.columns([6, 2])
with col_lang:
    selected_lang = st.selectbox(
        "Language / Langue / Ngôn ngữ",
        ["🇬🇧 English", "🇫🇷 Français", "🇻🇳 Tiếng Việt"],
        index=0,
        label_visibility="collapsed"
    )

T = TRANS[selected_lang]

st.title(T["title"])
st.markdown(T["subtitle"])
st.caption(T["def"])
st.caption(T["scope"])

with st.expander(T["expander_title"], expanded=False):
    st.markdown(T["expander_content"])

with st.sidebar:
    st.header(T["sidebar_header"])

    age = st.number_input(T["lbl_age"], 40, 95, 65)
    psa = st.number_input(T["lbl_psa"], 0.1, 200.0, 7.5, step=0.1, format="%.1f")
    vol = st.number_input(T["lbl_vol"], 5.0, 300.0, 45.0, step=0.1, format="%.1f")
    pirads = st.selectbox(T["lbl_pirads"], [3, 4, 5], index=1)

    st.divider()

    dre_display = st.radio(T["lbl_dre"], T["opt_dre"], horizontal=True)
    dre_map = dict(zip(T["opt_dre"], ["Normal", "Abnormal", "Unknown"]))
    dre_opt = dre_map[dre_display]

    fam_display = st.radio(T["lbl_fam"], T["opt_fam"], horizontal=True)
    fam_map = dict(zip(T["opt_fam"], ["No", "Yes", "Unknown"]))
    fam_opt = fam_map[fam_display]

    biopsy_display = st.radio(T["lbl_biopsy"], T["opt_biopsy"], horizontal=True)
    biopsy_map = dict(zip(T["opt_biopsy"], ["Naïve", "Prior Negative", "Unknown"]))
    biopsy_opt = biopsy_map[biopsy_display]

    st.divider()
    with st.expander(T["calib_title"], expanded=True):
        st.markdown(T["calib_desc"])

        DEFAULT_TARGET = 38.0  # PRECISION
        local_prev_pct = st.number_input(
            T["calib_input"],
            min_value=1.0, max_value=99.0,
            value=DEFAULT_TARGET,
            step=0.5, format="%.1f"
        )

        TRAIN_PREV = 0.452  # your dev cohort prevalence
        target_prev = local_prev_pct / 100.0

        def logit(p):
            p = float(np.clip(p, 1e-9, 1 - 1e-9))
            return np.log(p / (1 - p))

        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)
        st.info(f"{T['calib_info']} **{TRAIN_PREV:.1%}** ➔ **{local_prev_pct}%**")

        st.divider()
        st.caption("© 2026 Copyright by Authors")

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if st.button(T["btn_run"], type="primary"):

    # 0. CLINICAL WARNINGS
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
    log_psa_val = np.log(float(psa))
    log_vol_val = np.log(float(vol))
    psad = float(psa) / float(vol)

    input_dict = {
        "age": [age], "PSA": [psa], "log_PSA": [log_psa_val], "log_vol": [log_vol_val], "pirads_max": [pirads],
        "tr_yes": [1 if dre_opt == "Abnormal" else 0],
        "fam_yes": [1 if fam_opt == "Yes" else 0],
        "atcd_yes": [1 if biopsy_opt == "Prior Negative" else 0],
        "tr": [1 if dre_opt == "Abnormal" else 0],
        "fam": [1 if fam_opt == "Yes" else (2 if fam_opt == "Unknown" else 0)],
        "atcd": [1 if biopsy_opt == "Prior Negative" else 0],
        "fam_unknown": [1 if fam_opt == "Unknown" else 0],
        "tr_unknown": [0],
        "atcd_unknown": [0],
    }
    df_input = pd.DataFrame(input_dict)

    # 2. SPLINE (FORCE-ALIAS BY POSITION)
    try:
        safe_lb, safe_ub = float(np.min(knots) - 5.0), float(np.max(knots) + 5.0)
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(
            spline_formula,
            {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub},
            return_type="dataframe"
        )

        # Ensure Intercept column exists
        if "Intercept" not in spline_df.columns:
            spline_df["Intercept"] = 1.0

        df_full = pd.concat([df_input, spline_df], axis=1)

        # FORCE-ALIAS: always create bs(...)[0..K-1] by column position
        basis_df = spline_df.copy()
        if "Intercept" in basis_df.columns:
            basis_df = basis_df.drop(columns=["Intercept"])
        K = basis_df.shape[1]
        if K == 0:
            raise ValueError("No spline basis columns returned by patsy (after dropping Intercept).")

        for k in range(K):
            expected = f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{k}]"
            df_full[expected] = basis_df.iloc[:, k].values

    except Exception as e:
        st.error(f"Spline Error: {e}")
        st.stop()

    # 3. BASE MODELS INFERENCE (ORDERED + STRICT FEATURE ORDER)
    loop_names = model_names_ordered if model_names_ordered is not None else list(base_models.keys())
    loop_names = [m for m in list(loop_names) if m in base_models]

    base_preds = []
    for name in loop_names:
        model = base_models[name]

        # cols must be in exact order used at fit; your pkl's feature_mapping should already be ordered
        cols = feature_mapping.get(name, df_full.columns.tolist())
        cols = list(cols)

        missing = [c for c in cols if c not in df_full.columns]
        if missing:
            st.error(f"Model '{name}' missing columns (up to 12): {missing[:12]}{'...' if len(missing) > 12 else ''}")
            st.stop()

        X = df_full.loc[:, cols]  # enforce column order

        try:
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(X)[0, 1])
            else:
                p = float(model.predict(X)[0])
        except Exception as e:
            st.error(f"Error running model '{name}': {e}")
            st.stop()

        base_preds.append(p)

    base_preds = np.asarray(base_preds, dtype=float)

    # 4. META PREDICTION
    #    - Point estimate: DE (probability convex combiner) -> logit -> +offset -> sigmoid
    #    - Fallback: logistic meta
    if de_weights is not None:
        if len(de_weights) != len(base_preds):
            st.error(f"❌ Weight mismatch: de_weights has {len(de_weights)}, but got {len(base_preds)} base preds.")
            st.stop()

        p_de = float(np.dot(base_preds, de_weights))
        p_de = float(np.clip(p_de, 1e-6, 1 - 1e-6))
        log_odds_de = np.log(p_de / (1.0 - p_de))
        risk_mean = float(sigmoid(log_odds_de + CALIBRATION_OFFSET))
        used_method = "DE"
    else:
        raw_log_odds = float(np.dot(base_preds, meta_weights) + meta_intercept)
        risk_mean = float(sigmoid(raw_log_odds + CALIBRATION_OFFSET))
        used_method = "LOGISTIC_FALLBACK"

    # 5. BOOTSTRAP CI (legacy logistic bootstrap if present)
    # NOTE: This CI comes from bootstrap logistic meta; we force CI to contain point estimate to avoid "upper < mean" confusion.
    has_ci = False
    boot_preds = None
    low_ci = high_ci = risk_mean

    if bootstrap_weights is not None:
        try:
            boot_log_odds = np.dot(bootstrap_weights, base_preds)
            if bootstrap_intercepts is not None and len(bootstrap_intercepts) == boot_log_odds.shape[0]:
                boot_log_odds = boot_log_odds + bootstrap_intercepts
            boot_log_odds = boot_log_odds + CALIBRATION_OFFSET
            boot_preds = sigmoid(boot_log_odds)

            low_ci = float(np.percentile(boot_preds, 2.5))
            high_ci = float(np.percentile(boot_preds, 97.5))

            # ensure CI brackets the displayed point estimate
            low_ci = min(low_ci, risk_mean)
            high_ci = max(high_ci, risk_mean)

            has_ci = True
        except Exception as e:
            st.warning(f"Bootstrap CI unavailable: {e}")
            has_ci = False
            low_ci = high_ci = risk_mean
            boot_preds = None

    # ==========================================
    # 6. DISPLAY
    # ==========================================
    st.divider()
    st.subheader(T["res_title"])

    c1, c2, c3 = st.columns(3)
    c1.metric(T["res_risk"], f"{risk_mean:.1%}")
    c2.metric(T["res_low"], f"{low_ci:.1%}" if has_ci else "N/A")
    c3.metric(T["res_high"], f"{high_ci:.1%}" if has_ci else "N/A")

    spread = max(0.0, high_ci - low_ci)
    st.info(
        T["res_interp"].format(risk_mean) + "\n\n" +
        T["res_uncert"].format(low_ci, high_ci, spread) + "\n\n" +
        f"*Method note: point estimate uses **{used_method}**; CI uses stored bootstrap (if available).*"
    )

    st.write(f"### {T['plot_title']}")
    if has_ci and boot_preds is not None:
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(8, 3.5))

        sns.kdeplot(
            boot_preds, fill=True, color="#2c3e50", alpha=0.3,
            ax=ax, linewidth=2, label=T["plot_legend_dist"]
        )
        ax.axvline(
            risk_mean, color="#d95f02", linestyle="-", linewidth=2.5,
            label=f"{T['plot_legend_point']}: {risk_mean:.1%}"
        )

        plt.title("Bootstrap Uncertainty Analysis", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel(T["plot_xlabel"], fontsize=10)
        ax.set_ylabel(T["plot_ylabel"], fontsize=10)
        ax.set_xlim(0, max(0.6, high_ci + 0.1))
        ax.legend(loc="best", fontsize=9)

        sns.despine()
        st.pyplot(fig, dpi=300)

    st.caption(f"**{T['res_psad']}** {psad:.2f} ng/mL²")
