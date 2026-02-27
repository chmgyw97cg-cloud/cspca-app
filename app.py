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
st.set_page_config(page_title="csPCa Risk Assistant", page_icon="⚕️", layout="wide")

# ==========================================
# 2. LANGUAGE DICTIONARY (GIỮ NGUYÊN TRANS CỦA BẠN)
# ==========================================
# --- dán nguyên TRANS của bạn ở đây ---
# TRANS = {...}

# ==========================================
# 3. MODEL LOADING
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

def logit(p):
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

try:
    data_packet = load_prediction_system("v1")
    base_models = data_packet["base_models"]
    knots = np.asarray(data_packet["spline_knots"], dtype=float)

    # IMPORTANT: feature list per model MUST be ordered
    feature_mapping = data_packet.get("model_features", {}) or {}
    model_names_ordered = data_packet.get("model_names_ordered", None)
    if model_names_ordered is not None:
        model_names_ordered = list(model_names_ordered)

    # DE
    de_weights = data_packet.get("de_weights", None)
    de_weights_matrix = data_packet.get("de_weights_matrix", None)
    if de_weights is not None:
        de_weights = np.asarray(de_weights, dtype=float)
    if de_weights_matrix is not None:
        de_weights_matrix = np.asarray(de_weights_matrix, dtype=float)

    THRESHOLD = float(data_packet.get("threshold", 0.20))

    if de_weights is None:
        st.error("❌ Missing de_weights in .pkl (you said DE is the main IP).")
        st.stop()

except Exception as e:
    st.error(f"❌ Critical Error loading prediction system: {e}")
    st.stop()

# ==========================================
# 4. USER INTERFACE (GIỮ NGUYÊN UI/TRANS của bạn)
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

        DEFAULT_TARGET = 38.0
        local_prev_pct = st.number_input(
            T["calib_input"],
            min_value=1.0, max_value=99.0,
            value=DEFAULT_TARGET,
            step=0.5, format="%.1f"
        )

        TRAIN_PREV = 0.452
        target_prev = local_prev_pct / 100.0
        CALIBRATION_OFFSET = logit(target_prev) - logit(TRAIN_PREV)

        st.info(f"{T['calib_info']} **{TRAIN_PREV:.1%}** ➔ **{local_prev_pct}%**")
        st.divider()
        st.caption("© 2026 Copyright by Authors")

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if st.button(T["btn_run"], type="primary"):

    # 0. WARNINGS
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

    # 1. BASIC FEATURES
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

    # 2. SPLINE + FORCE-ALIAS (chỉ 1 cách, giống exporter)
    try:
        safe_lb, safe_ub = float(np.min(knots) - 5.0), float(np.max(knots) + 5.0)
        spline_formula = "bs(log_PSA, knots=knots, degree=3, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        spline_df = dmatrix(
            spline_formula,
            {"log_PSA": df_input["log_PSA"], "knots": knots, "lb": safe_lb, "ub": safe_ub},
            return_type="dataframe"
        )
        if "Intercept" not in spline_df.columns:
            spline_df["Intercept"] = 1.0

        df_full = pd.concat([df_input, spline_df], axis=1)

        basis_df = spline_df.copy()
        if "Intercept" in basis_df.columns:
            basis_df = basis_df.drop(columns=["Intercept"])
        K = basis_df.shape[1]
        if K <= 0:
            raise ValueError("No spline basis columns returned by patsy.")
        for k in range(K):
            expected = f"bs(log_PSA, knots=knots, degree=3, include_intercept=False)[{k}]"
            df_full[expected] = basis_df.iloc[:, k].values

        # IMPORTANT: string columns
        df_full.columns = [str(c) for c in df_full.columns]

    except Exception as e:
        st.error(f"Spline Error: {e}")
        st.stop()

    # 3. BASE PREDICTIONS (đúng order + đúng feature order theo mapping đã lưu)
    loop_names = model_names_ordered if model_names_ordered is not None else list(base_models.keys())
    loop_names = [m for m in list(loop_names) if m in base_models]

    base_preds = []
    for name in loop_names:
        model = base_models[name]

        cols = feature_mapping.get(name, None)
        if cols is None:
            st.error(f"Missing feature list for model '{name}' in pkl. Re-export pkl with model_features_out.")
            st.stop()

        cols = [str(c) for c in list(cols)]  # keep order
        missing = [c for c in cols if c not in df_full.columns]
        if missing:
            st.error(f"Model '{name}' missing columns (up to 12): {missing[:12]}{'...' if len(missing) > 12 else ''}")
            st.stop()

        X = df_full.loc[:, cols]  # EXACT ORDER
        try:
            p = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(X)[0])
        except Exception as e:
            st.error(f"Error running model '{name}': {e}")
            st.stop()

        base_preds.append(p)

    base_preds = np.asarray(base_preds, dtype=float)

    # 4. META (DE point estimate) + prevalence offset in logit space
    if len(de_weights) != len(base_preds):
        st.error(f"❌ Weight mismatch: de_weights has {len(de_weights)}, base_preds has {len(base_preds)}")
        st.stop()

    p_de = float(np.dot(base_preds, de_weights))
    p_de = float(np.clip(p_de, 1e-6, 1 - 1e-6))
    risk_mean = float(sigmoid(np.log(p_de / (1 - p_de)) + CALIBRATION_OFFSET))

    # 5. CI theo DE (dùng de_weights_matrix)
    has_ci = False
    low_ci = high_ci = risk_mean
    boot_preds = None

    if de_weights_matrix is not None and de_weights_matrix.ndim == 2 and de_weights_matrix.shape[1] == len(base_preds):
        try:
            # mỗi row là 1 bộ trọng số (outer fold) -> tạo phân phối xác suất
            p_mat = np.dot(de_weights_matrix, base_preds)  # shape (n_outer,)
            p_mat = np.clip(p_mat, 1e-6, 1 - 1e-6)
            boot_preds = sigmoid(np.log(p_mat / (1 - p_mat)) + CALIBRATION_OFFSET)

            low_ci = float(np.percentile(boot_preds, 2.5))
            high_ci = float(np.percentile(boot_preds, 97.5))

            # luôn đảm bảo CI bao risk_mean (tránh upper < mean gây khó chịu)
            low_ci = min(low_ci, risk_mean)
            high_ci = max(high_ci, risk_mean)
            has_ci = True
        except Exception as e:
            st.warning(f"DE-based CI unavailable: {e}")

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
        "*Method note: point estimate + CI are both derived from the DE-optimised convex ensemble (CI uses outer-fold DE weight variability).*"
    )

    st.write(f"### {T['plot_title']}")
    if has_ci and boot_preds is not None:
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(8, 3.5))

        sns.kdeplot(boot_preds, fill=True, color="#2c3e50", alpha=0.3, ax=ax, linewidth=2, label=T["plot_legend_dist"])
        ax.axvline(risk_mean, color="#d95f02", linestyle="-", linewidth=2.5, label=f"{T['plot_legend_point']}: {risk_mean:.1%}")

        plt.title("DE-based Uncertainty (Outer-fold Weight Variability)", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel(T["plot_xlabel"], fontsize=10)
        ax.set_ylabel(T["plot_ylabel"], fontsize=10)
        ax.set_xlim(0, max(0.6, high_ci + 0.1))
        ax.legend(loc="best", fontsize=9)

        sns.despine()
        st.pyplot(fig, dpi=300)

    st.caption(f"**{T['res_psad']}** {psad:.2f} ng/mL²")
