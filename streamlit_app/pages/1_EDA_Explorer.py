"""
1_EDA_Explorer.py — RF-Sentinel Streamlit page: EDA Explorer.

Displays all 48 EDA visualisations across 3 datasets (NASA C-MAPSS,
UCI SECOM, Kaggle AI4I 2020) and the unified RF schema. Each plot is
accompanied by a one-line domain insight.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── Paths ─────────────────────────────────────────────────────────────────────
EDA_DIR = Path(__file__).resolve().parents[2] / "outputs" / "eda"

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════

st.title("EDA Explorer")
st.markdown(
    "Explore all 48 EDA visualisations across 3 datasets "
    "and the unified RF schema."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# DATASET SELECTOR
# ══════════════════════════════════════════════════════════════

sel_col, info_col = st.columns([1.3, 1], gap="large")

with sel_col:
    dataset = st.selectbox(
        "Select Dataset",
        options=[
            "NASA CMAPSS — Turbofan Engine Degradation",
            "UCI SECOM — Semiconductor Manufacturing",
            "Kaggle AI4I 2020 — Predictive Maintenance",
            "Unified RF Schema — Cross-dataset Analysis",
        ],
    )

_dataset_info = {
    "NASA CMAPSS — Turbofan Engine Degradation":
        "160,359 rows · 100-260 engines · 14 sensors · 4 sub-datasets",
    "UCI SECOM — Semiconductor Manufacturing":
        "1,567 samples · 562 features · 6.6% failure rate",
    "Kaggle AI4I 2020 — Predictive Maintenance":
        "10,000 samples · 6 features · 5 failure types",
    "Unified RF Schema — Cross-dataset Analysis":
        "32,198 rows · 8 RF params · 3 sources combined",
}

with info_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(_dataset_info[dataset])

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPER — display image + insight
# ══════════════════════════════════════════════════════════════

def _show_plot(img_path: Path, insight: str) -> None:
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
        st.caption(insight)
    else:
        st.warning(f"Plot not found: {img_path.name}")
        st.info("Run: python -m layer2_eda.eda_runner")


# ══════════════════════════════════════════════════════════════
# CMAPSS SECTION
# ══════════════════════════════════════════════════════════════

if dataset == "NASA CMAPSS — Turbofan Engine Degradation":

    # Sub-dataset selector
    sub = st.radio(
        "Select Sub-dataset",
        options=["FD001", "FD002", "FD003", "FD004"],
        horizontal=True,
    )
    ds = sub.lower()   # "fd001" etc.

    _sub_info = {
        "FD001": "1 fault mode · 1 operating condition · 100 engines · simplest",
        "FD002": "1 fault mode · 6 operating conditions · 260 engines",
        "FD003": "2 fault modes · 1 operating condition · 100 engines",
        "FD004": "2 fault modes · 6 operating conditions · 249 engines · hardest",
    }
    st.caption(_sub_info[sub])
    st.markdown("")

    # Plot selector
    _cmapss_plots = {
        "Sensor Degradation Waves":   f"{ds}_sensor_degradation_waves.png",
        "RUL Distribution":           f"{ds}_rul_distribution.png",
        "Sensor Correlation Heatmap": f"{ds}_sensor_correlation.png",
        "Sensor vs RUL":              f"{ds}_sensor_vs_rul.png",
        "Operating Conditions":       f"{ds}_operating_conditions.png",
        "Degradation Index":          f"{ds}_degradation_index.png",
        "Failure Rate by Cycle":      f"{ds}_failure_rate_by_cycle.png",
        "Sensor Variance Ranking":    f"{ds}_sensor_variance_ranking.png",
    }

    _cmapss_insights = {
        "Sensor Degradation Waves":
            "Sensors s3, s4, s9 show clear degradation trend as engine approaches "
            "failure. Red shaded zone = last 30 cycles (failure threshold).",
        "RUL Distribution":
            "Most engines live 150-250 cycles. "
            "Failure label threshold = 30 cycles remaining.",
        "Sensor Correlation Heatmap":
            "s3/s4 highly correlated (both temperature sensors). "
            "s9/s14 highly correlated (speed measurements).",
        "Sensor vs RUL":
            "Scatter shows linear degradation signal in s11, s9 against RUL.",
        "Operating Conditions":
            "FD001/FD003 have 1 operating condition; FD002/FD004 have 6 "
            "distinct regimes visible as clusters.",
        "Degradation Index":
            "Composite health score rises from 0 (healthy) to 1 (failed). "
            "Built from normalized values of all 14 useful sensors.",
        "Failure Rate by Cycle":
            "Failure rate near zero until cycle ~180, then spikes sharply — "
            "matches the 30-cycle threshold labelling strategy.",
        "Sensor Variance Ranking":
            "s2, s3, s4 carry the most signal variance; "
            "s1, s5, s10, s16 are near-constant (dropped in feature engineering).",
    }

    plot_name = st.selectbox("Select Plot", options=list(_cmapss_plots.keys()))
    img_path  = EDA_DIR / "cmapss" / _cmapss_plots[plot_name]
    _show_plot(img_path, _cmapss_insights[plot_name])

    # ── Show all ──────────────────────────────────────────────
    st.divider()
    if st.checkbox("Show all plots for this dataset"):
        cols = st.columns(2)
        for i, (name, fname) in enumerate(_cmapss_plots.items()):
            p = EDA_DIR / "cmapss" / fname
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.warning(f"Not found: {fname}")


# ══════════════════════════════════════════════════════════════
# SECOM SECTION
# ══════════════════════════════════════════════════════════════

elif dataset == "UCI SECOM — Semiconductor Manufacturing":

    _secom_plots = {
        "Class Imbalance Analysis":     "secom_class_imbalance.png",
        "PCA Dimensionality Reduction": "secom_pca_analysis.png",
        "Feature-Failure Correlation":  "secom_feature_correlation.png",
        "Top Feature Distributions":    "secom_feature_distributions.png",
        "Missing Values vs Failure":    "secom_missing_vs_failure.png",
        "Feature Variance Overview":    "secom_feature_variance.png",
    }

    _secom_insights = {
        "Class Imbalance Analysis":
            "14:1 imbalance — SMOTE adds 1,359 synthetic samples "
            "to balance the training set.",
        "PCA Dimensionality Reduction":
            "562 features need 100+ components to capture 95% of variance — "
            "high intrinsic dimensionality.",
        "Feature-Failure Correlation":
            "feature_59 and feature_103 are strongest predictors of yield failure.",
        "Top Feature Distributions":
            "sep=0.59 for feature_59 — best class separation of any single feature.",
        "Missing Values vs Failure":
            "Missing values NOT correlated with failure — safe to impute with median.",
        "Feature Variance Overview":
            "265 near-zero variance features dropped by VarianceThreshold "
            "before modelling.",
    }

    plot_name = st.selectbox("Select Plot", options=list(_secom_plots.keys()))
    img_path  = EDA_DIR / "secom" / _secom_plots[plot_name]
    _show_plot(img_path, _secom_insights[plot_name])

    # ── Show all ──────────────────────────────────────────────
    st.divider()
    if st.checkbox("Show all plots for this dataset"):
        cols = st.columns(2)
        for i, (name, fname) in enumerate(_secom_plots.items()):
            p = EDA_DIR / "secom" / fname
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.warning(f"Not found: {fname}")


# ══════════════════════════════════════════════════════════════
# AI4I SECTION
# ══════════════════════════════════════════════════════════════

elif dataset == "Kaggle AI4I 2020 — Predictive Maintenance":

    _ai4i_plots = {
        "Failure Type Distribution":       "ai4i_failure_type_donut.png",
        "Temperature vs Failure":          "ai4i_temperature_vs_failure.png",
        "Torque vs Speed Operating Space": "ai4i_torque_speed_contour.png",
        "Tool Wear Survival Analysis":     "ai4i_tool_wear_survival.png",
        "Failure Correlation Matrix":      "ai4i_failure_correlation.png",
    }

    _ai4i_insights = {
        "Failure Type Distribution":
            "HDF most common (33%). RNF has only 18 samples — nearly unlearnable.",
        "Temperature vs Failure":
            "HDF clusters at slightly higher process temperatures — heat driven.",
        "Torque vs Speed Operating Space":
            "Each failure type occupies a distinct region of the operating space.",
        "Tool Wear Survival Analysis":
            "Failure rate jumps sharply after 208 min tool wear.",
        "Failure Correlation Matrix":
            "Torque is the best predictor of OSF. Air temp is best for HDF.",
    }

    plot_name = st.selectbox("Select Plot", options=list(_ai4i_plots.keys()))
    img_path  = EDA_DIR / "ai4i" / _ai4i_plots[plot_name]
    _show_plot(img_path, _ai4i_insights[plot_name])

    # ── Show all ──────────────────────────────────────────────
    st.divider()
    if st.checkbox("Show all plots for this dataset"):
        cols = st.columns(2)
        for i, (name, fname) in enumerate(_ai4i_plots.items()):
            p = EDA_DIR / "ai4i" / fname
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.warning(f"Not found: {fname}")


# ══════════════════════════════════════════════════════════════
# UNIFIED SECTION
# ══════════════════════════════════════════════════════════════

elif dataset == "Unified RF Schema — Cross-dataset Analysis":

    _unified_plots = {
        "Dataset Source Breakdown":   "unified_dataset_source_breakdown.png",
        "Failure Rate Comparison":    "unified_failure_rate_comparison.png",
        "RF Parameter Distributions": "unified_rf_param_distributions.png",
        "RF Parameters vs Failure":   "unified_rf_params_vs_failure.png",
        "Correlation Heatmap":        "unified_correlation_heatmap.png",
    }

    _unified_insights = {
        "Dataset Source Breakdown":
            "CMAPSS 64% · AI4I 31% · SECOM 5% of the unified dataset.",
        "Failure Rate Comparison":
            "CMAPSS 15% · SECOM 6.6% · AI4I 3.4% failure rates across sources.",
        "RF Parameter Distributions":
            "Each dataset maps to the same 8 RF parameter columns via the unified schema.",
        "RF Parameters vs Failure":
            "rf_param_1 (S21 proxy) is the strongest failure discriminator.",
        "Correlation Heatmap":
            "11% overall failure rate across the unified dataset.",
    }

    plot_name = st.selectbox("Select Plot", options=list(_unified_plots.keys()))
    img_path  = EDA_DIR / "unified" / _unified_plots[plot_name]
    _show_plot(img_path, _unified_insights[plot_name])

    # ── Show all ──────────────────────────────────────────────
    st.divider()
    if st.checkbox("Show all plots for this dataset"):
        cols = st.columns(2)
        for i, (name, fname) in enumerate(_unified_plots.items()):
            p = EDA_DIR / "unified" / fname
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.warning(f"Not found: {fname}")
