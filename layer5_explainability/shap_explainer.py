# Unified SHAP explainability for XGBoost and ensemble predictions
"""
shap_explainer.py — Dedicated SHAP explainability module for RF-Sentinel.

Explains WHY a model predicted a specific failure type for any given RF
component measurement. Richer than the basic SHAP built into xgb_classifier:
supports waterfall, beeswarm, force, decision, and bar plots; generates full
JSON diagnosis reports; and extracts top-N features for Layer 6.

Usage
-----
    python -m layer5_explainability.shap_explainer
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from pathlib import Path
from datetime import datetime
from loguru import logger

from layer1_data_ingestion.config import (
    ROOT_DIR, CMAPSS_USEFUL_SENSORS,
    CMAPSS_SENSOR_LABELS, AI4I_FEATURE_COLS,
)
from layer1_data_ingestion.loaders import load_cmapss, load_ai4i
from layer1_data_ingestion.preprocessor import preprocess_cmapss, preprocess_ai4i
from layer3_models.xgb_classifier import RFSentinelXGB

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Layer 5: SHAP Explainability Module
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# Dedicated SHAP explainability module for RF-Sentinel.
# Explains WHY the model predicted a specific failure type
# for any given RF component measurement.
#
# DIFFERENCE FROM LAYER 3 SHAP:
# ──────────────────────────────
# Layer 3 xgb_classifier.py → basic SHAP inside the model
# Layer 5 shap_explainer.py → dedicated explainability module
#   - Works for any model (XGBoost, ensemble)
#   - Richer plots (waterfall, beeswarm, force, decision)
#   - Explains multiple samples at once
#   - Produces full diagnosis report JSON
#   - Extracts top-3 features for Layer 6 knowledge graph
#
# WHAT SHAP VALUES MEAN:
# ──────────────────────
# Positive SHAP → feature pushes TOWARD failure prediction
# Negative SHAP → feature pushes TOWARD pass prediction
# Magnitude     → how strongly that feature influenced result
#
# Example output for one prediction:
#   Predicted: sensor_degradation (confidence 89%)
#   s4  (LPT outlet temp)   : +0.82  ← strongest failure signal
#   s9  (core speed)        : +0.61  ← second strongest
#   s3  (HPC outlet temp)   : +0.38  ← third strongest
#   s14 (corrected speed)   : -0.08  ← pushes toward pass
#
# OUTPUT FILES:
# ─────────────
# outputs/explainability/shap/
#     shap_waterfall_{sample_idx}.png
#     shap_beeswarm.png
#     shap_force_{sample_idx}.png
#     shap_decision.png
#     shap_bar_importance.png
#     diagnosis_report_{sample_idx}.json
# ══════════════════════════════════════════════════════════════

SHAP_DIR = ROOT_DIR / "outputs" / "explainability" / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)


# ── Class: RFSentinelSHAP ─────────────────────────────────────────────────────

class RFSentinelSHAP:
    """
    SHAP explainability wrapper for RF-Sentinel XGBoost models.

    Handles both binary (CMAPSS) and multiclass (AI4I) XGBoost outputs.
    The key challenge is that shap_values shape differs between binary (2D),
    multiclass-list (list of 2D arrays), and multiclass-array (3D) cases —
    all extraction helpers normalise this variation internally.
    """

    def __init__(self, model: RFSentinelXGB) -> None:
        self.model         = model
        self.explainer     = None
        self.shap_values   = None
        self.feature_names = model.feature_names
        self.classes_      = model.classes_
        self.X_explain     = None
        self.is_fitted     = False

    # ── Method 1: fit ─────────────────────────────────────────────────────────

    def fit(self, X, max_samples: int = 500) -> "RFSentinelSHAP":
        """
        Calculate SHAP values for a dataset.

        TreeExplainer is fast and exact for XGBoost/tree models — no
        approximation needed unlike KernelExplainer used for black-box models.
        We cap at max_samples to keep computation under a minute for large sets.

        Parameters
        ----------
        X           : feature array or DataFrame (val set is typical input)
        max_samples : maximum rows to compute SHAP values for (default 500)

        Returns
        -------
        self  (fluent interface)
        """
        X_arr = np.array(X)

        if len(X_arr) > max_samples:
            logger.info(
                f"[SHAP] Sampling {max_samples} from {len(X_arr)} for SHAP"
            )
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_arr), size=max_samples, replace=False)
            X_arr = X_arr[idx]

        self.X_explain   = X_arr
        self.explainer   = shap.TreeExplainer(self.model.model)
        self.shap_values = self.explainer.shap_values(self.X_explain)
        self.is_fitted   = True

        n, n_features = self.X_explain.shape
        logger.success("[SHAP] SHAP values calculated")
        logger.info(f"  Samples   : {n}")
        logger.info(f"  Features  : {n_features}")
        logger.info(f"  Classes   : {self.classes_}")
        return self

    # ── Method 2: get_top_features ────────────────────────────────────────────

    def get_top_features(self, sample_idx: int = 0, n_top: int = 3) -> dict:
        """
        Extract top-N features by absolute SHAP value for one sample.

        This is the primary output fed to Layer 6 knowledge graph. The
        direction ("toward_failure" / "toward_pass") encodes the sign of
        the SHAP value in a human-readable form for the report.

        Parameters
        ----------
        sample_idx : index of the sample to explain
        n_top      : number of top features to return

        Returns
        -------
        dict with keys: sample_idx, predicted_class, confidence, top_features
        """
        self.model.check_is_trained()

        # Predicted class index for this sample
        raw_pred = self.model.model.predict(
            self.X_explain[sample_idx: sample_idx + 1]
        )
        pred_enc   = int(np.squeeze(raw_pred).flat[0])
        pred_class = str(self.classes_[pred_enc])

        # Confidence = max probability across classes
        proba     = self.model.model.predict_proba(
            self.X_explain[sample_idx: sample_idx + 1]
        )
        confidence = float(np.max(proba))

        # Extract SHAP values for this sample and predicted class
        shap_vals = self._get_sample_shap(sample_idx)

        # Sort features by absolute SHAP value descending
        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"f{i}" for i in range(len(shap_vals))]
        )
        pairs = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:n_top]

        top_features = [
            {
                "feature":    name,
                "shap_value": round(float(val), 4),
                "direction":  "toward_failure" if val > 0 else "toward_pass",
            }
            for name, val in pairs
        ]

        return {
            "sample_idx":      sample_idx,
            "predicted_class": pred_class,
            "confidence":      confidence,
            "top_features":    top_features,
        }

    # ── Method 3: plot_waterfall ──────────────────────────────────────────────

    def plot_waterfall(self, sample_idx: int = 0, save: bool = True):
        """
        Horizontal waterfall bar chart for ONE sample.

        Shows exactly how much each feature pushed the prediction toward or
        away from failure. Red = toward failure, blue = toward pass.
        Readable sensor labels replace raw column names (e.g. s4 → LPT outlet).

        Parameters
        ----------
        sample_idx : which sample from X_explain to explain
        save       : write PNG to SHAP_DIR (default True)

        Returns
        -------
        matplotlib Figure
        """
        top_info   = self.get_top_features(sample_idx)
        pred_class = top_info["predicted_class"]
        confidence = top_info["confidence"]

        shap_vals     = self._get_sample_shap(sample_idx)
        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"f{i}" for i in range(len(shap_vals))]
        )

        # Build sorted DataFrame — show top 15 features
        df = pd.DataFrame({
            "feature":    feature_names,
            "shap_value": shap_vals,
        }).assign(abs_shap=lambda d: d["shap_value"].abs())
        df = df.sort_values("abs_shap", ascending=False).head(15)

        # Readable labels for CMAPSS sensors
        df["label"] = df["feature"].map(
            lambda f: f"{f} — {CMAPSS_SENSOR_LABELS[f]}"
            if f in CMAPSS_SENSOR_LABELS else f
        )

        colors = ["crimson" if v > 0 else "steelblue" for v in df["shap_value"]]

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(df["label"], df["shap_value"], color=colors, edgecolor="white")

        # Annotate bars with SHAP value
        for bar, val in zip(bars, df["shap_value"]):
            offset = 0.005 if val >= 0 else -0.005
            ax.text(
                val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title(
            f"SHAP Waterfall — Sample {sample_idx}\n"
            f"Predicted: {pred_class} | Confidence: {confidence:.1%}",
            fontweight="bold",
        )

        # Legend
        red_patch  = mpatches.Patch(color="crimson",   label="→ failure")
        blue_patch = mpatches.Patch(color="steelblue", label="→ pass")
        ax.legend(handles=[red_patch, blue_patch], loc="lower right")

        # Top-3 summary text box
        top3 = top_info["top_features"][:3]
        summary = "\n".join(
            f"{i+1}. {t['feature']}: {t['shap_value']:+.3f}"
            for i, t in enumerate(top3)
        )
        ax.text(
            0.02, 0.02, f"Top drivers:\n{summary}",
            transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()

        if save:
            path = SHAP_DIR / f"shap_waterfall_{sample_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[SHAP] Waterfall saved → {path}")

        return fig

    # ── Method 4: plot_beeswarm ───────────────────────────────────────────────

    def plot_beeswarm(self, save: bool = True):
        """
        Beeswarm summary plot for global feature importance.

        Each dot = one sample. x-position = SHAP value. y-position = feature
        (sorted by mean |SHAP|). Dot colour = original feature value (red=high,
        blue=low). Best plot for understanding which features matter globally
        and in which direction they drive predictions.

        Parameters
        ----------
        save : write PNG to SHAP_DIR (default True)

        Returns
        -------
        matplotlib Figure
        """
        # Use SHAP values for the dominant class (index 1 or last class)
        shap_vals = self._get_class_shap_matrix()

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"f{i}" for i in range(shap_vals.shape[1])]
        )

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_vals,
            self.X_explain,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
        plt.title("SHAP Beeswarm — Global Feature Importance", fontweight="bold")
        plt.tight_layout()

        if save:
            path = SHAP_DIR / "shap_beeswarm.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[SHAP] Beeswarm saved → {path}")

        return fig

    # ── Method 5: plot_bar_importance ─────────────────────────────────────────

    def plot_bar_importance(self, save: bool = True):
        """
        Bar chart of mean absolute SHAP values per feature.

        Cleaner than beeswarm for presentations and reports. Shows a single
        ranked importance score per feature without the distributional detail.
        Top 5 bars are highlighted in crimson for emphasis.

        Parameters
        ----------
        save : write PNG to SHAP_DIR (default True)

        Returns
        -------
        matplotlib Figure
        """
        shap_matrix = self._get_class_shap_matrix()
        mean_abs    = np.abs(shap_matrix).mean(axis=0)

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"f{i}" for i in range(len(mean_abs))]
        )

        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        df["label"] = df["feature"].map(
            lambda f: f"{f} — {CMAPSS_SENSOR_LABELS[f]}"
            if f in CMAPSS_SENSOR_LABELS else f
        )
        df = df.sort_values("mean_abs_shap", ascending=True)

        colors = [
            "crimson" if i >= len(df) - 5 else "steelblue"
            for i in range(len(df))
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(df["label"], df["mean_abs_shap"], color=colors, edgecolor="white")

        for bar, val in zip(bars, df["mean_abs_shap"]):
            ax.text(
                val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8,
            )

        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Mean Absolute SHAP — Feature Importance Ranking", fontweight="bold")

        red_patch = mpatches.Patch(color="crimson", label="Top 5 features")
        ax.legend(handles=[red_patch], loc="lower right")

        plt.tight_layout()

        if save:
            path = SHAP_DIR / "shap_bar_importance.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[SHAP] Bar importance saved → {path}")

        return fig

    # ── Method 6: plot_decision ───────────────────────────────────────────────

    def plot_decision(self, n_samples: int = 50, save: bool = True):
        """
        SHAP decision plot showing cumulative feature contributions.

        Each line traces one sample from the base value (average model output)
        to the final prediction, with each feature's SHAP value added step by
        step. Ideal for understanding how features combine, not just their
        individual importances.

        Parameters
        ----------
        n_samples : number of samples to plot (default 50 — more gets cluttered)
        save      : write PNG to SHAP_DIR (default True)

        Returns
        -------
        matplotlib Figure
        """
        shap_matrix = self._get_class_shap_matrix()
        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"f{i}" for i in range(shap_matrix.shape[1])]
        )

        # Expected value for the class being explained
        expected_val = (
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )

        fig = plt.figure(figsize=(12, 8))
        shap.decision_plot(
            expected_val,
            shap_matrix[:n_samples],
            feature_names=feature_names,
            show=False,
        )
        plt.title(
            f"SHAP Decision Plot — {n_samples} samples",
            fontweight="bold",
        )
        plt.tight_layout()

        if save:
            path = SHAP_DIR / "shap_decision.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[SHAP] Decision plot saved → {path}")

        return fig

    # ── Method 7: generate_diagnosis_report ──────────────────────────────────

    def generate_diagnosis_report(self, sample_idx: int = 0) -> dict:
        """
        Generate a complete JSON diagnosis report for one sample.

        This is the primary output that feeds into Layer 6 knowledge graph.
        The report is self-contained: it includes the prediction, confidence,
        top-5 SHAP feature drivers, human-readable RF parameter mappings,
        a recommendation string, and the full SHAP vector for the sample.

        Parameters
        ----------
        sample_idx : which sample from X_explain to generate report for

        Returns
        -------
        dict  diagnosis report (also saved to JSON in SHAP_DIR)
        """
        top = self.get_top_features(sample_idx, n_top=5)

        report = {
            "timestamp":       datetime.now().isoformat(),
            "sample_idx":      sample_idx,
            "predicted_class": top["predicted_class"],
            "confidence_pct":  round(top["confidence"] * 100, 1),
            "top_features":    top["top_features"],
            "diagnosis": {
                "primary_cause":   top["top_features"][0]["feature"],
                "secondary_cause": (
                    top["top_features"][1]["feature"]
                    if len(top["top_features"]) > 1 else None
                ),
                "recommendation": (
                    f"Investigate {top['top_features'][0]['feature']} — "
                    f"strongest failure signal "
                    f"(SHAP={top['top_features'][0]['shap_value']:.3f})"
                ),
            },
            "rf_mapping": {
                feat["feature"]: CMAPSS_SENSOR_LABELS.get(
                    feat["feature"], feat["feature"]
                )
                for feat in top["top_features"]
            },
            "all_shap_values": {
                name: float(val)
                for name, val in zip(
                    self.feature_names
                    if self.feature_names is not None
                    else [f"f{i}" for i in range(len(self._get_sample_shap(sample_idx)))],
                    self._get_sample_shap(sample_idx),
                )
            },
        }

        out_path = SHAP_DIR / f"diagnosis_report_{sample_idx}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.success(
            f"[SHAP] Report saved — sample={sample_idx} | "
            f"predicted={top['predicted_class']} | "
            f"confidence={top['confidence']:.1%}"
        )
        return report

    # ── Private helper: _get_sample_shap ─────────────────────────────────────

    def _get_sample_shap(self, sample_idx: int) -> np.ndarray:
        """
        Safely extract SHAP vector for one sample.

        XGBoost with multi:softprob returns shap_values as a list of 2D arrays
        (one per class), while binary XGBoost returns a single 2D array.
        This helper normalises both cases so callers don't need to branch.

        Parameters
        ----------
        sample_idx : row index into X_explain

        Returns
        -------
        np.ndarray  shape (n_features,) — SHAP values for the predicted class
        """
        raw_pred = self.model.model.predict(
            self.X_explain[sample_idx: sample_idx + 1]
        )
        pred_enc = int(np.squeeze(raw_pred).flat[0])

        if isinstance(self.shap_values, list):
            return self.shap_values[pred_enc][sample_idx]
        if self.shap_values.ndim == 3:
            return self.shap_values[sample_idx, :, pred_enc]
        return self.shap_values[sample_idx]

    # ── Private helper: _get_class_shap_matrix ───────────────────────────────

    def _get_class_shap_matrix(self) -> np.ndarray:
        """
        Return a 2D (n_samples × n_features) SHAP matrix for the dominant class.

        Used by beeswarm, bar, and decision plots which need a matrix rather
        than per-sample extraction. For binary models this is class 1 (failure).
        For multiclass this is the last class index.

        Returns
        -------
        np.ndarray  shape (n_samples, n_features)
        """
        if isinstance(self.shap_values, list):
            # Multiclass list: use last class (or index 1 for binary)
            class_idx = min(1, len(self.shap_values) - 1)
            return self.shap_values[class_idx]
        if self.shap_values.ndim == 3:
            return self.shap_values[:, :, -1]
        return self.shap_values


# ── Standalone function: run_shap_analysis ────────────────────────────────────

def run_shap_analysis(
    model_name: str = "xgb_classifier",
    dataset:    str = "FD001",
    n_samples:  int = 5,
) -> tuple["RFSentinelSHAP", list]:
    """
    Load a pre-trained model, run full SHAP analysis, save all outputs.

    Convenience entry-point that wires together model loading, explainer
    fitting, all five plot types, and per-sample diagnosis reports. Designed
    to be called from the notebook or as __main__.

    Parameters
    ----------
    model_name : model_name attribute of the saved model (used to find .pkl)
    dataset    : "FD001"–"FD004" for CMAPSS binary, "AI4I" for multiclass
    n_samples  : number of individual diagnosis reports to generate

    Returns
    -------
    (explainer, reports)  — fitted RFSentinelSHAP and list of report dicts
    """
    # Step 1: Load model and validation data
    model = RFSentinelXGB()
    model.model_name = model_name
    model.build()
    model.load()

    if dataset != "AI4I":
        data      = load_cmapss(dataset)
        processed = preprocess_cmapss(data)
    else:
        data      = load_ai4i()
        processed = preprocess_ai4i(data, target="multiclass")

    X_val = processed["X_val"]

    # Step 2: Fit SHAP explainer
    explainer = RFSentinelSHAP(model)
    explainer.fit(X_val, max_samples=200)

    # Step 3: Generate all plot types
    explainer.plot_waterfall(sample_idx=0, save=True)
    explainer.plot_waterfall(sample_idx=1, save=True)
    explainer.plot_beeswarm(save=True)
    explainer.plot_bar_importance(save=True)
    try:
        explainer.plot_decision(save=True)
    except Exception as exc:
        logger.warning(f"[SHAP] Decision plot skipped: {exc}")

    # Step 4: Generate per-sample diagnosis reports
    reports = []
    for i in range(n_samples):
        try:
            report = explainer.generate_diagnosis_report(i)
            reports.append(report)
            logger.info(
                f"  Sample {i}: {report['predicted_class']} "
                f"({report['confidence_pct']}%)"
            )
            logger.info(
                f"  Top feature: {report['top_features'][0]['feature']} "
                f"= {report['top_features'][0]['shap_value']:.3f}"
            )
        except Exception as exc:
            logger.warning(f"[SHAP] Report for sample {i} skipped: {exc}")

    # Step 5: Print summary — count most common failure-driving features
    from collections import Counter
    feature_counts: Counter = Counter()
    for r in reports:
        for feat in r.get("top_features", []):
            if feat["direction"] == "toward_failure":
                feature_counts[feat["feature"]] += 1

    print()
    print("SHAP Analysis Complete")
    print(f"  Plots saved : {SHAP_DIR}")
    print(f"  Reports     : {len(reports)} JSON files")
    print("  Top failure-driving features across samples:")
    for feat, count in feature_counts.most_common(3):
        label = CMAPSS_SENSOR_LABELS.get(feat, feat)
        print(f"    {feat} ({label}) — appeared in {count}/{len(reports)} samples")

    return explainer, reports


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    explainer, reports = run_shap_analysis(
        model_name="xgb_classifier",
        dataset="FD001",
        n_samples=5,
    )
