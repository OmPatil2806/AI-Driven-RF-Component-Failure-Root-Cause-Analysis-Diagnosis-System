"""
2_RCA_Analyzer.py — RF-Sentinel Streamlit page: Root Cause Analyzer.

Upload sensor readings (CSV) or adjust sliders manually → instant failure
prediction (XGBoost) → SHAP explanation → knowledge graph traversal →
ranked repair action table.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from layer1_data_ingestion.config import (
    CMAPSS_SENSOR_LABELS,
    CMAPSS_USEFUL_SENSORS,
    ROOT_DIR,
)
from layer3_models.xgb_classifier import RFSentinelXGB
from layer5_explainability.shap_explainer import RFSentinelSHAP
from layer6_knowledge_graph.build_graph import build_rf_knowledge_graph
from layer6_knowledge_graph.recommender import RFSentinelRecommender
from layer6_knowledge_graph.shap_to_graph import SHAPToGraphLinker

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════

st.title("RCA Analyzer")
st.markdown(
    "Upload sensor readings → "
    "get instant failure prediction + "
    "root cause + ranked repair actions."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# LOAD MODELS (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model() -> RFSentinelXGB:
    m = RFSentinelXGB()
    m.build()
    m.load()
    return m


@st.cache_resource
def load_recommender() -> RFSentinelRecommender:
    return RFSentinelRecommender()


with st.spinner("Loading AI models..."):
    model       = load_model()
    recommender = load_recommender()
st.success("Models loaded")

# ══════════════════════════════════════════════════════════════
# INPUT SECTION — 2 tabs
# ══════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["Upload CSV", "Manual Input"])

X_input: np.ndarray | None = None   # set by whichever tab is active

# ── Tab 1: Upload CSV ─────────────────────────────────────────────────────────

with tab1:
    st.markdown("Upload a CSV with sensor readings.")
    st.markdown(
        "**Required columns:** " + ", ".join(CMAPSS_USEFUL_SENSORS)
    )

    with st.expander("Show example CSV format"):
        example_df = pd.DataFrame(
            [np.random.uniform(0.5, 1.5, len(CMAPSS_USEFUL_SENSORS))],
            columns=CMAPSS_USEFUL_SENSORS,
        ).round(3)
        st.dataframe(example_df)
        st.caption("Each row = one measurement cycle")

    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV with sensor columns matching CMAPSS_USEFUL_SENSORS",
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)

        missing = [c for c in CMAPSS_USEFUL_SENSORS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_input = df[CMAPSS_USEFUL_SENSORS].values

# ── Tab 2: Manual Input ───────────────────────────────────────────────────────

with tab2:
    st.markdown("Adjust sensor values manually.")
    st.info("Values normalized 0–1  (0 = min observed, 1 = max observed)")

    col1, col2 = st.columns(2)

    with col1:
        s2  = st.slider("s2 — LPC outlet temp",     0.0, 1.0, 0.5, 0.01)
        s3  = st.slider("s3 — HPC outlet temp",     0.0, 1.0, 0.5, 0.01)
        s4  = st.slider("s4 — LPT outlet temp",     0.0, 1.0, 0.5, 0.01)
        s7  = st.slider("s7 — HPC pressure",        0.0, 1.0, 0.5, 0.01)
        s8  = st.slider("s8 — Fan speed",           0.0, 1.0, 0.5, 0.01)
        s9  = st.slider("s9 — Core speed",          0.0, 1.0, 0.5, 0.01)
        s11 = st.slider("s11 — HPC static pres.",   0.0, 1.0, 0.5, 0.01)

    with col2:
        s12 = st.slider("s12 — Fuel flow ratio",    0.0, 1.0, 0.5, 0.01)
        s13 = st.slider("s13 — Corrected fan spd",  0.0, 1.0, 0.5, 0.01)
        s14 = st.slider("s14 — Corrected core spd", 0.0, 1.0, 0.5, 0.01)
        s15 = st.slider("s15 — Bypass ratio",       0.0, 1.0, 0.5, 0.01)
        s17 = st.slider("s17 — Bleed enthalpy",     0.0, 1.0, 0.5, 0.01)
        s20 = st.slider("s20 — HPT coolant bleed",  0.0, 1.0, 0.5, 0.01)
        s21 = st.slider("s21 — LPT coolant bleed",  0.0, 1.0, 0.5, 0.01)

    X_manual = np.array([[s2, s3, s4, s7, s8, s9, s11,
                           s12, s13, s14, s15, s17, s20, s21]])

    analyze_btn = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
    )
    if analyze_btn:
        X_input = X_manual

# ══════════════════════════════════════════════════════════════
# ANALYSIS SECTION
# ══════════════════════════════════════════════════════════════

if X_input is not None:

    with st.spinner("Running analysis..."):

        # ── Step 1: Predict ───────────────────────────────────────────────────
        preds      = model.predict(X_input)
        probas     = model.predict_proba(X_input)
        raw_pred = preds[0]
        CLASS_MAP = {
            0: "pass",
            1: "sensor_degradation",
            "0": "pass",
            "1": "sensor_degradation",
            0.0: "pass",
            1.0: "sensor_degradation",
        }
        pred_class = CLASS_MAP.get(raw_pred, str(raw_pred))
        if pred_class not in CLASS_MAP.values():
            pred_class = str(raw_pred).replace("_", " ").title()
        confidence = float(probas[0].max())
        is_failure = pred_class not in ["pass", "0", 0]

        # ── Step 2: Show prediction ───────────────────────────────────────────
        st.divider()
        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Prediction",
            pred_class.replace("_", " ").title(),
        )
        c2.metric(
            "Confidence",
            f"{confidence:.1%}",
        )
        c3.metric(
            "Status",
            "FAIL" if is_failure else "PASS",
        )

        if is_failure:
            st.error(
                f"Failure detected: **{pred_class}** "
                f"({confidence:.1%} confidence)"
            )
        else:
            st.success("Component is healthy (PASS)")

        # ── Step 3: SHAP explanation ──────────────────────────────────────────
        st.subheader("SHAP Explanation — Why this prediction?")

        explainer = RFSentinelSHAP(model)
        explainer.fit(X_input, max_samples=len(X_input))

        top = explainer.get_top_features(sample_idx=0, n_top=5)
        top_feats = [f["feature"]    for f in top["top_features"]]
        shap_vals = [f["shap_value"] for f in top["top_features"]]

        # Horizontal bar chart
        shap_df = pd.DataFrame({
            "Feature":    top_feats,
            "SHAP Value": shap_vals,
        }).set_index("Feature")

        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#141414")
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]
        ax.barh(top_feats[::-1], shap_vals[::-1], color=colors[::-1])
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.set_xlabel("SHAP value", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_title("Top 5 Feature Contributions", color="#e0e0e0", pad=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Red bars push toward FAILURE · Blue bars push toward PASS"
        )

        # Waterfall plot if saved
        waterfall_path = (
            ROOT_DIR / "outputs" / "explainability" / "shap" /
            "shap_waterfall_0.png"
        )
        if waterfall_path.exists():
            with st.expander("Show SHAP Waterfall Plot"):
                st.image(str(waterfall_path), use_container_width=True)

        # ── Step 4: Knowledge graph traversal ────────────────────────────────
        if is_failure:
            st.subheader("Root Cause Analysis")

            with st.spinner("Traversing knowledge graph..."):
                report = recommender.recommend(
                    top_features    = top_feats,
                    shap_values     = shap_vals,
                    predicted_class = pred_class,
                    confidence      = confidence,
                    device_id       = "streamlit_input",
                    save_report     = False,
                )

            primary_cause = report.get("primary_cause", "Unknown")
            st.success(f"Primary cause identified: **{primary_cause}**")

            # ── Step 5: Repair actions table ──────────────────────────────────
            st.subheader("Recommended Repair Actions")

            repairs = report.get("repair_actions", [])

            if repairs:
                repair_df = pd.DataFrame(repairs)

                # Select display columns that actually exist
                display_cols = [
                    c for c in
                    ["rank", "priority", "action", "cause",
                     "confidence_pct", "estimated_time"]
                    if c in repair_df.columns
                ]

                def _row_color(row: pd.Series) -> list[str]:
                    pri = row.get("priority", "P3")
                    color_map = {
                        "P1": "background-color: #3d1010; color: #f0b0b0",
                        "P2": "background-color: #3d2410; color: #f0c890",
                        "P3": "background-color: #101a2a; color: #90b8f0",
                    }
                    style = color_map.get(pri, "")
                    return [style] * len(row)

                styled = (
                    repair_df[display_cols]
                    .style.apply(_row_color, axis=1)
                )
                st.dataframe(styled, use_container_width=True)

                # Top P1 action callout
                p1 = repairs[0]
                action     = p1.get("action",          "inspect component")
                est_time   = p1.get("estimated_time",  "unknown")
                conf_pct   = p1.get("confidence_pct",  0)
                st.error(
                    f"**P1 ACTION:** {action} "
                    f"— {est_time} "
                    f"({conf_pct}% confidence)"
                )
            else:
                st.info("No repair actions returned by the knowledge graph.")

        # ── Step 6: Diagnosis card ────────────────────────────────────────────
        card_path = (
            ROOT_DIR / "outputs" / "knowledge_graph" /
            "diagnosis_card_DUT_FD001_sample0.png"
        )
        if card_path.exists():
            with st.expander("View Full Diagnosis Card"):
                st.image(str(card_path), use_container_width=True)
