import streamlit as st
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==========================================
# 1) PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="csPCa Risk Assistant",
    page_icon="⚕️",
    layout="wide"
)

# ==========================================
# 2) LANGUAGE DICTIONARY  (MUST be defined BEFORE using it)
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
        "res_uncert": "**Uncertainty Note:** 95% CI **{:.1%}** to **{:.1%}** (spread **{:.1%}**).",
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
        "def": "**Définition :** csPCa défini par **ISUP Grade Group ≥ 2**.",
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
        "calib_desc": "**Référence : Essai PRECISION**\n\nTaux attendu pour ROI-targeted biopsy en PI-RADS ≥ 3.",
        "calib_input": "Taux de positivité (%):",
        "calib_info": "✅ Ajusté sur :",
        "btn_run": "🚀 LANCER L'ANALYSE",
        "warn_age": "⚠️ **Âge ({})** hors 55–75.",
        "warn_psa": "⚠️ **PSA ({:.1f})** hors 0.4–50.0.",
        "warn_vol": "⚠️ **Volume ({:.1f})** hors 10–110.",
        "warn_title": "### ⚠️ Avertissement : Hors Critères",
        "warn_footer": "Fiabilité potentiellement réduite.",
        "res_title": "📊 Évaluation Quantitative",
        "res_risk": "Risque Estimé",
        "res_low": "IC 95% Inf",
        "res_high": "IC 95% Sup",
        "res_interp": "**Interprétation :** Probabilité prédite = **{:.1%}**.",
        "res_uncert": "**Note :** IC 95% **{:.1%}** à **{:.1%}** (écart **{:.1%}**).",
        "plot_title": "🔍 Distribution de Probabilité",
        "plot_xlabel": "Probabilité prédite",
        "plot_ylabel": "Densité",
        "plot_legend_dist": "Distribution",
        "plot_legend_point": "Estimation",
        "res_psad": "Densité de PSA (PSAD) :"
    },
    "🇻🇳 Tiếng Việt": {
        "title": "🛡️ Phân tích Nguy cơ & Độ bất định csPCa",
        "subtitle": "**Mô hình Meta-Stacking Ensemble** | Hỗ trợ Ra quyết định Lâm sàng",
        "def": "**Định nghĩa:** csPCa = **ISUP Grade Group ≥ 2**.",
        "scope": "**Phạm vi:** Dự báo cho **Sinh thiết trúng đích MRI (ROI-only)**.",
        "expander_title": "📚 Tiêu chuẩn Lâm sàng & Tiêu chí Lựa chọn",
        "expander_content": """
        * **Tuổi:** 55 – 75.
        * **PSA:** 0.4 – 50.0 ng/mL.
        * **Thể tích:** 10 – 110 mL.
        * **MRI:** PI-RADS Max ≥ 3.
        """,
        "sidebar_header": "📋 Dữ liệu Bệnh nhân",
        "lbl_age": "Tuổi (năm)",
        "lbl_psa": "PSA Toàn phần (ng/mL)",
        "lbl_vol": "Thể tích (mL)",
        "lbl_pirads": "PI-RADS Max (≥3)",
        "lbl_dre": "Thăm trực tràng (DRE)",
        "opt_dre": ["Bình thường", "Bất thường", "Không rõ"],
        "lbl_fam": "Tiền sử Gia đình",
        "opt_fam": ["Không", "Có", "Không rõ"],
        "lbl_biopsy": "Tiền sử Sinh thiết",
        "opt_biopsy": ["Chưa từng (Naïve)", "Đã từng (Âm tính)", "Không rõ"],
        "calib_title": "⚙️ Hiệu chỉnh (Calibration)",
        "calib_desc": "**Chuẩn: PRECISION (NEJM 2018)**\n\nTỷ lệ dương tính kỳ vọng cho ROI-targeted biopsy (PI-RADS ≥ 3).",
        "calib_input": "Tỷ lệ dương tính (%):",
        "calib_info": "✅ Đã hiệu chỉnh:",
        "btn_run": "🚀 CHẠY PHÂN TÍCH",
        "warn_age": "⚠️ **Tuổi ({})** ngoài 55–75.",
        "warn_psa": "⚠️ **PSA ({:.1f})** ngoài 0.4–50.0.",
        "warn_vol": "⚠️ **Thể tích ({:.1f})** ngoài 10–110.",
        "warn_title": "### ⚠️ Cảnh báo: Ngoài vùng dữ liệu",
        "warn_footer": "Kết quả có thể kém tin cậy.",
        "res_title": "📊 Đánh giá Định lượng",
        "res_risk": "Nguy cơ Dự báo",
        "res_low": "KTC 95% (Dưới)",
        "res_high": "KTC 95% (Trên)",
        "res_interp": "**Diễn giải:** Xác suất dự báo = **{:.1%}**.",
        "res_uncert": "**Ghi chú:** KTC 95% **{:.1%}** đến **{:.1%}** (độ rộng **{:.1%}**).",
        "plot_title": "🔍 Phân phối Xác suất",
        "plot_xlabel": "Xác suất dự báo",
        "plot_ylabel": "Mật độ",
        "plot_legend_dist": "Phân phối",
        "plot_legend_point": "Điểm ước lượng",
        "res_psad": "Mật độ PSA (PSAD):"
    }
}

# ==========================================
# 3) MODEL LOADING (DE + ORDER + FALLBACK)
# ==========================================
@st.cache_resource
def load_prediction_system(_version="v1"):
    # bump _version to force cache refresh when you update the pkl
    return joblib.load("cspca_prediction_system.pkl")

def _as_float(x, default=0.0) -> float:
    try:
        arr = np.asarray(x).reshape(-1)
        return float(arr[0])
    except Exception:
        return float(default)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

try:
    data_packet = load_prediction_system("v2")  # change tag when you upload new PKL

    base_models = data_packet["base_models"]                      # fitted models/pipelines
    knots = np.asarray(data_packet["spline_knots"], dtype=float)  # knots array
    feature_mapping = data_packet.get("model_features", {}) or {}
    THRESHOLD = float(data_packet.get("threshold", 0.20))

    # DE (main model)
    de_weights = data_packet.get("de_weights", None)
    if de_weights is not None:
        de_weights = np.asarray(de_weights, dtype=float)

    de_weights_matrix = data_packet.get("de_weights_matrix", None)  # optional: bootstrap DE weights
    if de_weights_matrix is not None:
        de_weights_matrix = np.asarray(de_weights_matrix, dtype=float)

    model_names_ordered = data_packet.get("model_names_ordered", None)
    if model_names_ordered is not None:
        model_names_ordered = [m for m in list(model_names_ordered) if m in base_models]

    # fallback logistic meta (legacy)
    meta_weights = data_packet.get("meta_weights", None)
    if meta_weights is not None:
        meta_weights = np.asarray(meta_weights, dtype=float)
    meta_intercept = _as_float(data_packet.get("meta_intercept", 0.0), default=0.0)

    # legacy bootstrap (logistic meta) – optional
    bootstrap_weights = data_packet.get("bootstrap_weights", None)
    bootstrap_intercepts = data_packet.get("bootstrap_intercepts", None)
    if bootstrap_weights is not None:
        bootstrap_weights = np.asarray(bootstrap_weights, dtype=float)
    if bootstrap_intercepts is not None:
        bootstrap_intercepts = np.asarray(bootstrap_intercepts, dtype=float).reshape(-1)

    if de_weights is None and meta_weights is None:
        st.error("❌ Missing weights in .pkl (need de_weights or meta_weights).")
        st.stop()

except Exception as e:
    st.error(f"❌ Critical Error loading prediction system: {e}")
    st.stop()

# ==========================================
# 4) UI: language selector + header
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

# ==========================================
# 5) Sidebar inputs + calibration offset
# ==========================================
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

        TRAIN_PREV = 0.452  # dev cohort prevalence
        target_prev = local_prev_pct / 100.0

        def logit(p):
            p = float(np.clip(p, 1e-9, 1 - 1e-9))
            return np.log(p / (1 - p))

        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)
        st.info(f"{T['calib_info']} **{TRAIN_PREV:.1%}** ➔ **{local_prev_pct}%**")

        st.divider()
        st.caption("© 2026 Copyright by Authors")

# ==========================================
# 6) Prediction logic
# ==========================================
if st.button(T["btn_run"], type="primary"):
    # Warnings
    warnings = []
    if not (55 <= age <= 75):
        warnings.append(T["warn_age"].format(age))
    if not (0.4 <= psa <= 50.0):
        warnings.append(T["warn_psa"].format(psa))
    if not (10 <= vol <= 110):
        warnings.append(T["warn_vol"].format(vol))

    if warnings:
        st.warning(T["warn_title"])
        for w in warnings:
            st.markdown(w)
        st.caption(T["warn_footer"])

    # Basic features
    psa = float(psa); vol = float(vol)
    log_psa_val = np.log(psa)
    log_vol_val = np.log(vol)
    psad = psa / vol

    input_dict = {
        "age": [int(age)],
        "PSA": [psa],
        "log_PSA": [log_psa_val],
        "log_vol": [log_vol_val],
        "pirads_max": [int(pirads)],

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

    # Spline + FORCE-ALIAS
    try:
        safe_lb, safe_ub = float(np.min(knots) - 5.0), float(np.max(knots) + 5.0)
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(
            spline_formula,
            {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub},
            return_type="dataframe"
        )

        # ensure Intercept
        if "Intercept" not in spline_df.columns:
            spline_df["Intercept"] = 1.0

        df_full = pd.concat([df_input, spline_df], axis=1)

        # FORCE-ALIAS by position (basis columns only)
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

    # Base models inference (ordered)
    loop_names = model_names_ordered if model_names_ordered is not None else list(base_models.keys())
    loop_names = [m for m in list(loop_names) if m in base_models]

    base_preds = []
    for name in loop_names:
        model = base_models[name]
        cols = feature_mapping.get(name, df_full.columns.tolist())
        cols = list(cols)

        missing = [c for c in cols if c not in df_full.columns]
        if missing:
            st.error(f"Model '{name}' missing columns (up to 12): {missing[:12]}{'...' if len(missing) > 12 else ''}")
            st.stop()

        X = df_full.loc[:, cols]  # enforce feature order
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

    # Meta prediction
    used_method = "DE" if de_weights is not None else "LOGISTIC_FALLBACK"
    if de_weights is not None:
        if len(de_weights) != len(base_preds):
            st.error(f"❌ Weight mismatch: de_weights has {len(de_weights)} but got {len(base_preds)} base preds.")
            st.stop()

        p_de = float(np.dot(base_preds, de_weights))
        p_de = float(np.clip(p_de, 1e-6, 1 - 1e-6))
        log_odds_de = np.log(p_de / (1.0 - p_de))
        risk_mean = float(sigmoid(log_odds_de + CALIBRATION_OFFSET))
    else:
        raw_log_odds = float(np.dot(base_preds, meta_weights) + meta_intercept)
        risk_mean = float(sigmoid(raw_log_odds + CALIBRATION_OFFSET))

    # CI (prefer DE bootstrap if available)
    has_ci = False
    low_ci = high_ci = risk_mean
    boot_preds = None

    try:
        if de_weights_matrix is not None:
            # bootstrap via DE weights distribution
            W = np.asarray(de_weights_matrix, dtype=float)
            if W.ndim != 2 or W.shape[1] != len(base_preds):
                raise ValueError(f"de_weights_matrix shape {W.shape} incompatible with {len(base_preds)} base preds.")
            # convex probs
            p_boot = W @ base_preds
            p_boot = np.clip(p_boot, 1e-6, 1 - 1e-6)
            log_odds_boot = np.log(p_boot / (1.0 - p_boot)) + CALIBRATION_OFFSET
            boot_preds = sigmoid(log_odds_boot)
            low_ci = float(np.percentile(boot_preds, 2.5))
            high_ci = float(np.percentile(boot_preds, 97.5))
            # ensure brackets point estimate
            low_ci = min(low_ci, risk_mean)
            high_ci = max(high_ci, risk_mean)
            has_ci = True

        elif bootstrap_weights is not None:
            # legacy logistic bootstrap CI
            boot_log_odds = np.dot(bootstrap_weights, base_preds)
            if bootstrap_intercepts is not None and len(bootstrap_intercepts) == boot_log_odds.shape[0]:
                boot_log_odds = boot_log_odds + bootstrap_intercepts
            boot_preds = sigmoid(boot_log_odds + CALIBRATION_OFFSET)
            low_ci = float(np.percentile(boot_preds, 2.5))
            high_ci = float(np.percentile(boot_preds, 97.5))
            low_ci = min(low_ci, risk_mean)
            high_ci = max(high_ci, risk_mean)
            has_ci = True

    except Exception as e:
        st.warning(f"Bootstrap CI unavailable: {e}")
        has_ci = False
        low_ci = high_ci = risk_mean
        boot_preds = None

    # ==========================================
    # 7) DISPLAY
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
        f"*Method note: point estimate uses **{used_method}**; CI uses {'DE bootstrap' if de_weights_matrix is not None else 'stored bootstrap' if bootstrap_weights is not None else 'N/A'}.*"
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
