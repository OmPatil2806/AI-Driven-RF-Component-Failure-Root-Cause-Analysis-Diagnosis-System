"""
5_Live_Simulator.py — RF-Sentinel Streamlit page: Live Simulator.

Adjust sensor values with sliders → XGBoost predicts instantly →
approximate SHAP contributions update live → failure probability gauge.
Preset scenarios (Healthy / Early Degradation / Near Failure / Critical)
load realistic baseline values from the CMAPSS FD001 training distribution.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from layer1_data_ingestion.config import (
    CMAPSS_SENSOR_LABELS,
    CMAPSS_USEFUL_SENSORS,
)
from layer1_data_ingestion.loaders import load_cmapss
from layer3_models.xgb_classifier import RFSentinelXGB

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════

st.title("Live Simulator")
st.markdown(
    "Adjust sensor values with sliders → "
    "model predicts instantly → "
    "SHAP values update live."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# LOAD MODEL + SENSOR RANGES (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model() -> RFSentinelXGB:
    m = RFSentinelXGB()
    m.build()
    m.load()
    return m


@st.cache_data
def get_sensor_ranges() -> dict:
    data      = load_cmapss("FD001")
    train_raw = data["train_raw"]
    ranges: dict = {}
    for s in CMAPSS_USEFUL_SENSORS:
        col = train_raw[s]
        ranges[s] = {
            "min":          float(col.min()),
            "max":          float(col.max()),
            "mean":         float(col.mean()),
            "healthy_mean": float(train_raw[train_raw["fail_soon"] == 0][s].mean()),
            "fail_mean":    float(train_raw[train_raw["fail_soon"] == 1][s].mean()),
        }
    return ranges


model  = load_model()
ranges = get_sensor_ranges()

# ══════════════════════════════════════════════════════════════
# PRESET SCENARIOS
# ══════════════════════════════════════════════════════════════

st.subheader("Quick Scenarios")
st.markdown("Load a preset scenario or adjust sliders manually.")

sc1, sc2, sc3, sc4 = st.columns(4)
scenario: str | None = None
if sc1.button("Healthy Engine",    use_container_width=True):
    scenario = "healthy"
if sc2.button("Early Degradation", use_container_width=True):
    scenario = "early"
if sc3.button("Near Failure",      use_container_width=True):
    scenario = "near_failure"
if sc4.button("Critical Failure",  use_container_width=True):
    scenario = "critical"

PRESETS: dict[str, dict] = {
    "healthy": {
        s: ranges[s]["healthy_mean"]
        for s in CMAPSS_USEFUL_SENSORS
    },
    "early": {
        s: (
            ranges[s]["healthy_mean"] * 0.97 if s in ["s9", "s11", "s14"]
            else ranges[s]["healthy_mean"] * 1.01 if s in ["s3", "s4"]
            else ranges[s]["healthy_mean"]
        )
        for s in CMAPSS_USEFUL_SENSORS
    },
    "near_failure": {
        s: (
            ranges[s]["fail_mean"] * 0.98 if s in ["s9", "s11", "s14", "s7"]
            else ranges[s]["fail_mean"] * 1.02 if s in ["s3", "s4", "s17"]
            else ranges[s]["fail_mean"]
        )
        for s in CMAPSS_USEFUL_SENSORS
    },
    "critical": {
        s: ranges[s]["fail_mean"]
        for s in CMAPSS_USEFUL_SENSORS
    },
}

# Initialise slider keys to healthy baseline on first load
for _s in CMAPSS_USEFUL_SENSORS:
    if f"slider_{_s}" not in st.session_state:
        st.session_state[f"slider_{_s}"] = PRESETS["healthy"][_s]

# Preset buttons write directly to slider session-state keys
if scenario is not None:
    for _s in CMAPSS_USEFUL_SENSORS:
        st.session_state[f"slider_{_s}"] = PRESETS[scenario][_s]
    st.rerun()

# ══════════════════════════════════════════════════════════════
# SENSOR SLIDERS
# ══════════════════════════════════════════════════════════════

st.divider()
st.subheader("Sensor Values")
st.markdown("Drag sliders to simulate sensor readings.")

slider_left, slider_right = st.columns(2)

for i, sensor in enumerate(CMAPSS_USEFUL_SENSORS):
    col   = slider_left if i < 7 else slider_right
    label = CMAPSS_SENSOR_LABELS.get(sensor, sensor)
    r     = ranges[sensor]
    span  = r["max"] - r["min"]

    col.slider(
        f"{sensor} — {label}",
        min_value=r["min"] * 0.95,
        max_value=r["max"] * 1.05,
        value=float(st.session_state.get(f"slider_{sensor}", r["mean"])),
        step=span / 100 if span > 0 else 0.01,
        key=f"slider_{sensor}",
    )

# Read current slider values from session state after all widgets are rendered
sensor_values = {
    s: st.session_state.get(f"slider_{s}", ranges[s]["mean"])
    for s in CMAPSS_USEFUL_SENSORS
}

# ══════════════════════════════════════════════════════════════
# LIVE PREDICTION + SHAP + GAUGE  (all inside empty containers
# so they refresh cleanly on every slider interaction)
# ══════════════════════════════════════════════════════════════

st.divider()

prediction_container = st.empty()
shap_container       = st.empty()
gauge_container      = st.empty()

# ── Compute prediction ────────────────────────────────────────
X_input = np.array([[sensor_values[s] for s in CMAPSS_USEFUL_SENSORS]])

preds  = model.predict(X_input)
probas = model.predict_proba(X_input)

CLASS_MAP = {
    0:   "pass", 1:   "sensor_degradation",
    "0": "pass", "1": "sensor_degradation",
}
pred_class = CLASS_MAP.get(preds[0], str(preds[0]))
confidence = float(probas[0].max())
fail_prob  = (
    float(probas[0][1]) if probas.shape[1] > 1
    else (confidence if pred_class != "pass" else 1.0 - confidence)
)

# ── Prediction container ──────────────────────────────────────
with prediction_container.container():
    st.subheader("Live Prediction")

    p1, p2, p3 = st.columns(3)
    p1.metric("Prediction",   pred_class.replace("_", " ").title())
    p2.metric("Confidence",   f"{confidence:.1%}")
    p3.metric("Failure Prob", f"{fail_prob:.1%}")

    if pred_class != "pass":
        st.error(
            f"FAILURE DETECTED: {pred_class} "
            f"({confidence:.1%} confidence) — "
            "Check repair recommendations in RCA Analyzer"
        )
    else:
        st.success(f"Engine is HEALTHY ({confidence:.1%} confidence)")

# ── SHAP container ────────────────────────────────────────────
with shap_container.container():
    st.subheader("Live SHAP — Feature Contributions")
    st.markdown(
        "Red bars push toward **FAILURE**. "
        "Blue bars push toward **PASS**. "
        "Updates every time you move a slider."
    )

    current_proba = probas[0]
    sensitivity: list[float] = []
    for i, sensor in enumerate(CMAPSS_USEFUL_SENSORS):
        perturbed       = X_input.copy()
        perturbed[0][i] = ranges[sensor]["healthy_mean"]
        perturbed_proba = model.predict_proba(perturbed)[0]
        diff = (
            float(current_proba[1] - perturbed_proba[1])
            if len(current_proba) > 1 else 0.0
        )
        sensitivity.append(diff)

    sensitivity_arr   = np.array(sensitivity)
    labels_all        = [CMAPSS_SENSOR_LABELS.get(s, s) for s in CMAPSS_USEFUL_SENSORS]
    order             = np.argsort(np.abs(sensitivity_arr))[::-1][:10]
    sensitivity_sorted = sensitivity_arr[order]
    labels_sorted      = [labels_all[j] for j in order]
    colors_sorted      = ["crimson" if v > 0 else "steelblue" for v in sensitivity_sorted]

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=sensitivity_sorted,
        y=labels_sorted,
        orientation="h",
        marker_color=colors_sorted,
        text=[f"{v:+.3f}" for v in sensitivity_sorted],
        textposition="outside",
    ))
    fig_shap.update_layout(
        title="Feature Contributions to Failure Probability",
        xaxis_title="Contribution to failure probability",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="#ffffff",
        height=400,
        xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Deviation table inside SHAP container
    st.subheader("Sensor Status vs Healthy Baseline")
    deviation_data = []
    for sensor in CMAPSS_USEFUL_SENSORS:
        current   = sensor_values[sensor]
        healthy   = ranges[sensor]["healthy_mean"]
        deviation = (current - healthy) / healthy * 100 if healthy != 0 else 0.0
        status    = "Normal"
        if abs(deviation) > 5:
            status = "Warning"
        if abs(deviation) > 10:
            status = "Critical"
        deviation_data.append({
            "Sensor":        sensor,
            "Description":   CMAPSS_SENSOR_LABELS.get(sensor, sensor),
            "Current Value": round(current, 3),
            "Healthy Mean":  round(healthy, 3),
            "Deviation %":   round(deviation, 2),
            "Status":        status,
        })
    st.dataframe(pd.DataFrame(deviation_data), use_container_width=True)

# ── Gauge container ───────────────────────────────────────────
with gauge_container.container():
    st.subheader("Failure Probability Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fail_prob * 100,
        delta={"reference": 15, "valueformat": ".1f"},
        title={"text": "Failure Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "crimson"},
            "steps": [
                {"range": [0,  30], "color": "#1D9E75"},
                {"range": [30, 60], "color": "#E09040"},
                {"range": [60, 100], "color": "#D85A30"},
            ],
            "threshold": {
                "line":      {"color": "white", "width": 4},
                "thickness": 0.75,
                "value":     50,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#111111",
        font_color="#ffffff",
        height=300,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption(
        "Green = healthy (0–30%) · "
        "Orange = warning (30–60%) · "
        "Red = critical (60–100%)"
    )
