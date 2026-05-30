"""
3_Model_Compare.py — RF-Sentinel Streamlit page: Model Compare.

Side-by-side comparison of all trained models: XGBoost (binary + multiclass),
1D-CNN, and Ensemble. Shows validation scores, HPO best parameters, training
plots saved to disk, and a directory listing of serialised model files.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parents[2]
MODELS_DIR = _ROOT / "outputs" / "models"
MODELS_BIN = _ROOT / "models"

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════

st.title("Model Compare")
st.markdown("Compare all trained models — scores, parameters, plots.")
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — Performance summary table
# ══════════════════════════════════════════════════════════════

st.subheader("Model Performance Summary")

_data = {
    "Model":       ["XGBoost CMAPSS", "XGBoost AI4I",
                    "1D-CNN CMAPSS",  "Ensemble",     "XGBoost HPO"],
    "Dataset":     ["NASA CMAPSS FD001", "Kaggle AI4I 2020",
                    "NASA CMAPSS FD001", "CMAPSS FD001",
                    "NASA CMAPSS FD001"],
    "Val Score":   [0.9521, 0.9477, 0.8666, 0.9466, 0.9589],
    "Score Type":  ["F1 weighted", "F1 weighted",
                    "Accuracy",    "F1 weighted", "F1 weighted"],
    "Train Score": [0.9813, 0.9917, 0.9392, 0.9466, 0.9834],
    "Type":        ["XGBoost", "XGBoost", "PyTorch CNN",
                    "Ensemble", "XGBoost+HPO"],
}
df = pd.DataFrame(_data)

st.dataframe(
    df.style.highlight_max(subset=["Val Score"], color="#1D9E75"),
    use_container_width=True,
)

# ══════════════════════════════════════════════════════════════
# SECTION 2 — Bar chart comparison
# ══════════════════════════════════════════════════════════════

st.subheader("Validation Score Comparison")

fig = px.bar(
    df,
    x="Model",
    y="Val Score",
    color="Type",
    text="Val Score",
    title="Validation Score by Model",
    color_discrete_map={
        "XGBoost":     "#378ADD",
        "PyTorch CNN": "#E09040",
        "Ensemble":    "#1D9E75",
        "XGBoost+HPO": "#D85A30",
    },
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(
    yaxis_range=[0.8, 1.0],
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font_color="#ffffff",
    showlegend=True,
)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SECTION 3 — HPO results
# ══════════════════════════════════════════════════════════════

st.subheader("Optuna HPO — Best Hyperparameters Found")

hpo_left, hpo_right = st.columns(2)

hpo_left.markdown("**Best XGBoost params (20 trials):**")
best_params = {
    "n_estimators":     250,
    "max_depth":        8,
    "learning_rate":    0.1206,
    "subsample":        0.8395,
    "colsample_bytree": 0.6624,
    "reg_alpha":        0.3120,
    "reg_lambda":       0.7614,
}
params_df = pd.DataFrame(
    best_params.items(),
    columns=["Parameter", "Best Value"],
)
hpo_left.dataframe(params_df, use_container_width=True)

hpo_right.markdown("**HPO improvement:**")
hpo_right.metric("Before HPO",  "0.9521", "baseline")
hpo_right.metric("After HPO",   "0.9589", "+0.0068")
hpo_right.metric("Trials run",  "20",     "Optuna TPE sampler")
hpo_right.metric("Best trial",  "Trial #2", "n_estimators=250")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — Saved model plots
# ══════════════════════════════════════════════════════════════

st.subheader("Model Training Plots")

model_choice = st.selectbox(
    "Select model",
    ["XGBoost CMAPSS", "XGBoost AI4I", "1D-CNN", "Ensemble"],
)

if model_choice == "XGBoost CMAPSS":
    _folder = MODELS_DIR / "xgboost"
    _plots = {
        "Training Curves":    "xgb_training_curves.png",
        "Feature Importance": "xgb_feature_importance.png",
        "Confusion Matrix":   "xgb_classifier_confusion_matrix.png",
        "SHAP Summary":       "xgb_shap_summary.png",
        "SHAP Waterfall":     "xgb_classifier_shap_waterfall_0.png",
    }
elif model_choice == "XGBoost AI4I":
    _folder = MODELS_DIR / "xgboost"
    _plots = {
        "Confusion Matrix":   "xgb_classifier_ai4i_confusion_matrix.png",
        "SHAP Waterfall":     "xgb_classifier_ai4i_shap_waterfall_0.png",
    }
elif model_choice == "1D-CNN":
    _folder = MODELS_DIR / "cnn1d"
    _plots = {
        "Architecture":    "cnn1d_model_summary.png",
        "Training Curves": "cnn1d_training_curves.png",
        "Confusion Matrix":"cnn1d_confusion_matrix.png",
    }
else:  # Ensemble
    _folder = MODELS_DIR / "ensemble"
    _plots = {
        "Model Comparison": "ensemble_model_comparison.png",
    }

plot_choice = st.selectbox("Select plot", list(_plots.keys()))
img_path    = _folder / _plots[plot_choice]

if img_path.exists():
    st.image(str(img_path), use_container_width=True)
else:
    st.warning(f"Plot not found: {img_path.name}")
    st.info("Run the relevant layer3_models training script to generate plots.")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — Saved model files on disk
# ══════════════════════════════════════════════════════════════

st.subheader("Saved Model Files")

if MODELS_BIN.exists():
    files = [f for f in sorted(MODELS_BIN.iterdir()) if f.is_file()]
    if files:
        files_data = [
            {
                "File":       f.name,
                "Size (KB)":  round(f.stat().st_size / 1024, 1),
                "Type":       f.suffix.upper(),
            }
            for f in files
        ]
        st.dataframe(pd.DataFrame(files_data), use_container_width=True)
    else:
        st.info("No model files found in models/")
else:
    st.warning(f"models/ directory not found at: {MODELS_BIN}")
    st.info("Run layer3_models training scripts to create model files.")
