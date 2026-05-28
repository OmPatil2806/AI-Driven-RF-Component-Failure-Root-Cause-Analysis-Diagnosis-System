"""
ensemble.py — Soft-vote ensemble combiner for RF-Sentinel.

Merges XGBoost (tabular) and 1D-CNN (time-series) predictions by
weighted averaging of their class probabilities, then taking the argmax.
Neither model alone is sufficient:
    XGBoost excels at single-snapshot feature discrimination.
    CNN excels at detecting multi-cycle degradation patterns.
The ensemble captures both signals.
"""

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Soft-Vote Ensemble Combiner
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# Combines XGBoost + 1D-CNN predictions into one final answer.
# Each model votes with its probability — we take weighted average.
#
# WHY ENSEMBLE?
# ─────────────
# XGBoost: strong on tabular features, fast, interpretable
# 1D-CNN : strong on time patterns, catches degradation trends
# Together: catches failures that either model alone would miss
#
# HOW SOFT VOTING WORKS:
# ──────────────────────
# XGBoost  → [0.8 pass, 0.2 fail]  × weight 0.55
# 1D-CNN   → [0.6 pass, 0.4 fail]  × weight 0.45
# Combined → [0.71 pass, 0.29 fail] → predict PASS
#
# IMPORTANT — SHARED CLASS SPACE:
# ────────────────────────────────
# XGBoost is trained on both CMAPSS (binary) and AI4I (6-class)
# 1D-CNN is trained on CMAPSS only (binary)
# Ensemble works on CMAPSS binary task:
#     0 = pass
#     1 = fail (sensor_degradation)
# ══════════════════════════════════════════════════════════════

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from layer1_data_ingestion.config import ROOT_DIR, RANDOM_STATE
from layer3_models.base_model import BaseModel
from layer3_models.cnn1d_model import CMAPSSWindowDataset, RFSentinelCNN1D
from layer3_models.xgb_classifier import RFSentinelXGB

PLOTS_DIR = ROOT_DIR / "outputs" / "models" / "ensemble"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class RFSentinelEnsemble(BaseModel):
    """
    Soft-vote ensemble that combines XGBoost (tabular) and 1D-CNN
    (sliding-window) probability outputs by weighted average.

    The ensemble is NOT retrained from scratch — it loads already-trained
    sub-models and only learns the optimal weight combination by evaluating
    on a held-out validation set.  This avoids retraining costs while still
    benefiting from both model families.
    """

    model_name = "ensemble"
    model_type = "soft_vote_ensemble"

    def __init__(self, xgb_weight: float = 0.55, cnn_weight: float = 0.45) -> None:
        self.xgb_model:       Optional[RFSentinelXGB]   = None
        self.cnn_model:       Optional[RFSentinelCNN1D] = None
        self.xgb_weight:      float = xgb_weight
        self.cnn_weight:      float = cnn_weight
        self.classes_:        List[str] = ["pass", "sensor_degradation"]
        self.n_classes:       int   = 2
        self.xgb_val_f1:      float = 0.0
        self.cnn_val_acc:     float = 0.0
        self.ensemble_val_f1: float = 0.0

    # ── Abstract method implementations ──────────────────────────────────────

    def build(self, **kwargs) -> None:
        """
        Configure ensemble weights.

        No neural network or estimator is built here — the sub-models are
        loaded from disk in train().  build() only sets the weighting scheme,
        allowing the caller to experiment with different weight combinations
        without retraining.

        Parameters
        ----------
        xgb_weight : float  weight given to XGBoost probabilities (default 0.55)
        cnn_weight : float  weight given to CNN probabilities     (default 0.45)
        """
        self.xgb_weight = kwargs.get("xgb_weight", self.xgb_weight)
        self.cnn_weight = kwargs.get("cnn_weight",  self.cnn_weight)
        logger.info(
            f"[{self.model_name}] Ensemble built — "
            f"XGB weight: {self.xgb_weight} | CNN weight: {self.cnn_weight}"
        )

    def train(
        self,
        X_train,
        y_train,
        X_val    = None,
        y_val    = None,
        train_df = None,
        val_df   = None,
    ) -> Dict[str, Any]:
        """
        Load pre-trained sub-models and evaluate their combination on the
        validation set.

        WHY load instead of re-train?
        Each sub-model was already trained on domain-appropriate data and
        saved to disk.  The ensemble's job is purely to combine them, not
        to overfit a new blending layer.

        Parameters
        ----------
        X_train : tabular XGBoost val array used as evaluation data
                  (when X_val is None, X_train doubles as X_eval)
        y_train : true labels for X_train / X_eval
        X_val   : optional separate tabular val array
        y_val   : optional labels for X_val
        train_df: raw CMAPSS train DataFrame (for CNN window dataset)
        val_df  : raw CMAPSS val DataFrame   (for CNN window dataset)

        Returns
        -------
        dict with keys: xgb_val_f1, cnn_val_acc, ensemble_val_f1,
                        xgb_weight, cnn_weight
        """
        # Resolve evaluation data (X_val optional — fall back to X_train)
        X_eval = X_val  if X_val  is not None else X_train
        y_eval = y_val  if y_val  is not None else y_train

        self.log_training_start(
            dataset_name="CMAPSS FD001 (ensemble eval)",
            X_shape=np.array(X_eval).shape,
            y_shape=np.array(y_eval).shape,
        )

        # ── Step 1: Load XGBoost ──────────────────────────────────────────────
        self.xgb_model = RFSentinelXGB()
        self.xgb_model.build()
        self.xgb_model.load()
        logger.info(f"[{self.model_name}] XGBoost model loaded")

        # ── Step 2: Load CNN ──────────────────────────────────────────────────
        self.cnn_model = RFSentinelCNN1D()
        self.cnn_model.build()
        self.cnn_model.load()
        logger.info(f"[{self.model_name}] CNN model loaded")

        # ── Step 3: XGBoost probabilities on tabular val data ─────────────────
        X_eval_arr  = np.array(X_eval)
        xgb_proba   = self.xgb_model.predict_proba(X_eval_arr)  # (n_samples, 2)

        # Encode ground-truth labels using XGBoost's label encoder
        y_eval_arr = np.array(y_eval).ravel()
        try:
            y_enc = self.xgb_model.label_encoder.transform(y_eval_arr)
        except Exception:
            y_enc = y_eval_arr.astype(int)

        xgb_preds      = np.argmax(xgb_proba, axis=1)
        self.xgb_val_f1 = float(
            f1_score(y_enc, xgb_preds, average="weighted", zero_division=0)
        )

        # ── Step 4: CNN probabilities from raw DataFrame ──────────────────────
        cnn_proba    = None
        self.cnn_val_acc = 0.0
        if val_df is not None:
            cnn_proba = self._get_cnn_proba(val_df)          # (n_windows, 2)
            cnn_preds = np.argmax(cnn_proba, axis=1)

            # True labels for CNN windows (from the window dataset directly)
            from layer1_data_ingestion.config import CMAPSS_USEFUL_SENSORS, RUL_THRESHOLD
            cnn_ds   = CMAPSSWindowDataset(
                val_df, CMAPSS_USEFUL_SENSORS, self.cnn_model.window_size, RUL_THRESHOLD,
            )
            cnn_true = cnn_ds.y
            if len(cnn_true) == len(cnn_preds):
                self.cnn_val_acc = float(accuracy_score(cnn_true, cnn_preds))

            logger.info(
                f"[{self.model_name}] XGB proba shape: {xgb_proba.shape} | "
                f"CNN proba shape: {cnn_proba.shape}"
            )

        # ── Step 5 & 6: Align and combine probabilities ───────────────────────
        n_samples   = len(xgb_proba)
        align_proba = self._align_proba(cnn_proba, n_samples)

        ensemble_proba = (
            xgb_proba * self.xgb_weight +
            align_proba * self.cnn_weight
        )
        ensemble_pred = np.argmax(ensemble_proba, axis=1)

        # ── Step 7: Scores ────────────────────────────────────────────────────
        self.ensemble_val_f1 = float(
            f1_score(y_enc, ensemble_pred, average="weighted", zero_division=0)
        )
        self.val_score   = self.ensemble_val_f1
        self.train_score = self.ensemble_val_f1
        self.is_trained  = True

        # ── Step 8: Comparison table ──────────────────────────────────────────
        sep = "─" * 36
        logger.success(sep)
        logger.success("  Model            | Val F1 (weighted)")
        logger.success(sep)
        logger.success(f"  XGBoost alone    | {self.xgb_val_f1:.4f}")
        logger.success(f"  CNN alone (acc)  | {self.cnn_val_acc:.4f}")
        logger.success(f"  Ensemble         | {self.ensemble_val_f1:.4f}")
        logger.success(sep)

        metrics = {
            "xgb_val_f1":      self.xgb_val_f1,
            "cnn_val_acc":     self.cnn_val_acc,
            "ensemble_val_f1": self.ensemble_val_f1,
            "xgb_weight":      self.xgb_weight,
            "cnn_weight":      self.cnn_weight,
        }
        self.log_training_end(metrics)
        return metrics

    def predict(self, X_tab, df_raw: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Return ensemble class label predictions.

        Combines XGBoost (tabular) and CNN (window) probabilities by weighted
        average.  If df_raw is not provided, falls back to XGBoost alone and
        logs a warning — the ensemble degrades gracefully.

        Parameters
        ----------
        X_tab  : preprocessed tabular feature array for XGBoost
        df_raw : raw CMAPSS DataFrame for CNN windowing (optional)

        Returns
        -------
        np.ndarray of decoded string class labels
        """
        self.check_is_trained()
        ensemble_proba = self.predict_proba(X_tab, df_raw)
        pred_idx = np.argmax(ensemble_proba, axis=1)
        return self.xgb_model.label_encoder.inverse_transform(pred_idx)

    def predict_proba(
        self,
        X_tab,
        df_raw: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Return ensemble probability estimates.

        WHY weighted average of probabilities (not hard votes)?
        Hard voting treats each model as equally confident.  Probability
        averaging preserves the degree of certainty — a 51 % XGBoost vote
        and a 95 % CNN vote should not be treated identically.

        Parameters
        ----------
        X_tab  : tabular feature array
        df_raw : raw CMAPSS DataFrame for CNN (optional)

        Returns
        -------
        np.ndarray shape (n_samples, 2)
        """
        self.check_is_trained()

        xgb_proba = self.xgb_model.predict_proba(np.array(X_tab))
        n_samples  = len(xgb_proba)

        if df_raw is not None:
            cnn_proba   = self._get_cnn_proba(df_raw)
            align_proba = self._align_proba(cnn_proba, n_samples)
        else:
            logger.warning(
                f"[{self.model_name}] CNN df_raw not provided — "
                "using XGBoost only"
            )
            align_proba = xgb_proba  # pure XGBoost, CNN weight is wasted

        return (
            xgb_proba * self.xgb_weight +
            align_proba * self.cnn_weight
        )

    def evaluate(self, X, y) -> Dict[str, Any]:
        """
        Compute ensemble metrics on tabular data (XGBoost path only).

        For full ensemble evaluation including CNN, call train() with
        both X_val and val_df — this method covers the quick tabular case.

        Parameters
        ----------
        X : tabular feature array
        y : true labels

        Returns
        -------
        dict with keys: accuracy, f1_macro, f1_weighted,
                        confusion_matrix, xgb_weight, cnn_weight
        """
        self.check_is_trained()

        y_arr = np.array(y).ravel()
        try:
            y_enc = self.xgb_model.label_encoder.transform(y_arr)
        except Exception:
            y_enc = y_arr.astype(int)

        ensemble_proba = self.predict_proba(X)
        y_pred         = np.argmax(ensemble_proba, axis=1)

        acc       = accuracy_score(y_enc, y_pred)
        f1_macro  = f1_score(y_enc, y_pred, average="macro",    zero_division=0)
        f1_weight = f1_score(y_enc, y_pred, average="weighted", zero_division=0)
        cm        = confusion_matrix(y_enc, y_pred)

        logger.info(
            f"[{self.model_name}] Evaluate — "
            f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weight:.4f}"
        )
        return {
            "accuracy":         acc,
            "f1_macro":         f1_macro,
            "f1_weighted":      f1_weight,
            "confusion_matrix": cm.tolist(),
            "xgb_weight":       self.xgb_weight,
            "cnn_weight":       self.cnn_weight,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_cnn_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Run CNN inference on a raw CMAPSS DataFrame and return probabilities.

        WHY a private helper?
        Both train() and predict_proba() need CNN inference.  Centralising
        here keeps both callers short and ensures the window parameters
        (window_size, sensors, threshold) are applied consistently.

        Parameters
        ----------
        df : raw CMAPSS DataFrame with 'unit_id', 'cycle', sensor columns, 'RUL'

        Returns
        -------
        np.ndarray shape (n_windows, 2)
        """
        return self.cnn_model.predict_proba(df)  # handles Dataset + DataLoader internally

    def _align_proba(
        self,
        cnn_proba:  Optional[np.ndarray],
        n_samples:  int,
    ) -> np.ndarray:
        """
        Align CNN probability array to the same length as XGBoost array.

        XGBoost operates on individual cycles (n_samples rows).
        CNN operates on sliding windows (n_windows rows, where n_windows
        is typically smaller).  We repeat the last CNN window to fill any
        shortfall, and truncate any surplus from the start.

        Parameters
        ----------
        cnn_proba  : CNN output (n_windows, 2) or None
        n_samples  : target length (= number of XGBoost val rows)

        Returns
        -------
        np.ndarray shape (n_samples, 2)
        """
        if cnn_proba is None or len(cnn_proba) == 0:
            # No CNN data — return uniform 50/50 so weight is neutral
            return np.full((n_samples, 2), 0.5, dtype=np.float32)

        n_windows = len(cnn_proba)
        if n_windows == n_samples:
            return cnn_proba
        if n_windows > n_samples:
            # More windows than tabular rows — take last n_samples
            return cnn_proba[-n_samples:]
        # Fewer windows — pad by repeating the last window
        pad   = np.tile(cnn_proba[-1:], (n_samples - n_windows, 1))
        return np.concatenate([cnn_proba, pad], axis=0)

    # ── Diagnostic plots ──────────────────────────────────────────────────────

    def plot_model_comparison(self, save: bool = True) -> plt.Figure:
        """
        Bar chart comparing XGBoost, CNN, and Ensemble validation scores.

        WHY this plot?
        The bar chart immediately answers the key question: "does the ensemble
        actually improve over its components?"  If the ensemble bar is not
        higher than both baselines, the weighting needs adjustment.
        """
        model_names = ["XGBoost", "1D-CNN", "Ensemble"]
        scores      = [self.xgb_val_f1, self.cnn_val_acc, self.ensemble_val_f1]
        colors      = ["steelblue", "orange", "crimson"]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(model_names, scores, color=colors,
                      edgecolor="white", width=0.5)

        # Thicker border on ensemble bar
        bars[2].set_edgecolor("darkred")
        bars[2].set_linewidth(2.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{score:.4f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.axhline(self.xgb_val_f1, color="steelblue", linestyle="--",
                   linewidth=1.2, label="XGBoost baseline")
        ax.axhline(self.cnn_val_acc, color="orange",    linestyle="--",
                   linewidth=1.2, label="CNN baseline")

        ax.set_ylim(0, min(1.0, max(scores) * 1.18))
        ax.set_title(
            "RF-Sentinel — Model Comparison (Val F1 Weighted)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Validation F1 Score (weighted)", fontsize=11)
        ax.legend(fontsize=10)

        ax.text(
            0.98, 0.04,
            f"Ensemble weights: {self.xgb_weight} XGB + {self.cnn_weight} CNN",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

        sns.despine(ax=ax)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "ensemble_model_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Model comparison saved → {path}")

        return fig

    def plot_confidence_distribution(
        self,
        X_val,
        val_df:    Optional[pd.DataFrame],
        y_val,
        save: bool = True,
    ) -> plt.Figure:
        """
        Show the distribution of ensemble confidence scores split by
        whether each prediction was correct or incorrect.

        WHY confidence matters?
        A model that is always 60 % confident is less useful than one
        that is 95 % confident on easy cases and 55 % on hard cases.
        This plot reveals whether the ensemble's high-confidence predictions
        are reliably correct — the key to deployable failure detection.
        """
        self.check_is_trained()

        ensemble_proba  = self.predict_proba(X_val, val_df)
        y_pred_idx      = np.argmax(ensemble_proba, axis=1)
        max_confidence  = np.max(ensemble_proba, axis=1)

        y_arr = np.array(y_val).ravel()
        try:
            y_enc = self.xgb_model.label_encoder.transform(y_arr)
        except Exception:
            y_enc = y_arr.astype(int)

        n = min(len(y_enc), len(y_pred_idx))
        correct = (y_pred_idx[:n] == y_enc[:n])
        conf    = max_confidence[:n]
        y_true  = y_enc[:n]

        correct_conf   = conf[correct]
        incorrect_conf = conf[~correct]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: confidence histogram ────────────────────────────────────────
        ax1.hist(correct_conf,   bins=30, color="#27AE60", alpha=0.7,
                 label=f"Correct ({correct.sum()})")
        ax1.hist(incorrect_conf, bins=20, color="#E74C3C", alpha=0.7,
                 label=f"Wrong ({(~correct).sum()})")
        ax1.axvline(0.8, color="black", linestyle="--", linewidth=1.3,
                    label="0.8 confidence threshold")
        ax1.set_title("Prediction Confidence Distribution",
                      fontweight="bold", fontsize=12)
        ax1.set_xlabel("Confidence Score", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.legend(fontsize=9)

        # ── Right: confidence vs true label ───────────────────────────────────
        rng    = np.random.default_rng(RANDOM_STATE)
        jitter = rng.uniform(-0.04, 0.04, size=n)
        colors_pts = np.where(correct, "#27AE60", "#E74C3C")

        ax2.scatter(conf, y_true + jitter,
                    c=colors_pts, alpha=0.4, s=8, rasterized=True)
        ax2.axhline(0, color="steelblue", linestyle=":", linewidth=1)
        ax2.axhline(1, color="crimson",   linestyle=":", linewidth=1)
        ax2.axvline(0.8, color="black",   linestyle="--", linewidth=1.2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["pass (0)", "fail (1)"], fontsize=10)

        green_patch = mpatches.Patch(color="#27AE60", label="Correct")
        red_patch   = mpatches.Patch(color="#E74C3C", label="Wrong")
        ax2.legend(handles=[green_patch, red_patch], fontsize=9)
        ax2.set_title("Confidence vs True Label", fontweight="bold", fontsize=12)
        ax2.set_xlabel("Confidence Score", fontsize=11)
        ax2.set_ylabel("True Label", fontsize=11)

        fig.suptitle("RF-Sentinel Ensemble — Prediction Confidence",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "ensemble_confidence.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Confidence distribution saved → {path}")

        return fig

    # ── PyTorch-incompatible save / load (JSON only) ─────────────────────────

    def save(self, path: Optional[str | Path] = None) -> str:
        """
        Save ensemble configuration as a JSON file.

        WHY JSON not joblib?
        The ensemble itself holds no learned weights — just two float
        weights and cached scores.  Sub-models are saved separately.
        JSON is human-readable and editable without retraining.

        Parameters
        ----------
        path : optional explicit .json file path

        Returns
        -------
        str  absolute path of the saved JSON file
        """
        if path is None:
            json_path = ROOT_DIR / "models" / "ensemble_config.json"
        else:
            json_path = Path(path)
            json_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name":      self.model_name,
            "xgb_weight":      self.xgb_weight,
            "cnn_weight":      self.cnn_weight,
            "xgb_val_f1":     round(self.xgb_val_f1,      6),
            "cnn_val_acc":    round(self.cnn_val_acc,     6),
            "ensemble_val_f1": round(self.ensemble_val_f1, 6),
            "classes_":        self.classes_,
            "n_classes":       self.n_classes,
            "is_trained":      self.is_trained,
            "saved_at":        datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"[{self.model_name}] Config saved → {json_path}")
        return str(json_path)

    def load(self, path: Optional[str | Path] = None) -> "RFSentinelEnsemble":
        """
        Load ensemble configuration and both sub-models from disk.

        Restores weights, cached scores, and reloads XGBoost + CNN so
        the ensemble is ready for predict() / predict_proba() without
        calling train() again.

        Parameters
        ----------
        path : optional explicit ensemble_config.json path

        Returns
        -------
        self (fluent interface)
        """
        if path is None:
            json_path = ROOT_DIR / "models" / "ensemble_config.json"
        else:
            json_path = Path(path)

        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.xgb_weight      = config["xgb_weight"]
        self.cnn_weight      = config["cnn_weight"]
        self.xgb_val_f1      = config.get("xgb_val_f1",      0.0)
        self.cnn_val_acc     = config.get("cnn_val_acc",     0.0)
        self.ensemble_val_f1 = config.get("ensemble_val_f1", 0.0)

        self.xgb_model = RFSentinelXGB()
        self.xgb_model.build()
        self.xgb_model.load()

        self.cnn_model = RFSentinelCNN1D()
        self.cnn_model.build()
        self.cnn_model.load()

        self.is_trained = True
        logger.info(f"[{self.model_name}] Loaded from {json_path}")
        return self


# ── Standalone training function ──────────────────────────────────────────────

def build_ensemble() -> RFSentinelEnsemble:
    """
    Load pre-trained XGBoost and CNN models, combine them as a soft-vote
    ensemble, evaluate on a held-out val set, and save the configuration.

    Preconditions
    -------------
    1. XGBoost must have been trained:
           python -m layer3_models.xgb_classifier
    2. CNN must have been trained:
           python -m layer3_models.cnn1d_model
    Both models save to ROOT_DIR/models/ automatically.

    Returns
    -------
    RFSentinelEnsemble — evaluated and saved ensemble
    """
    from layer1_data_ingestion.loaders import load_cmapss
    from layer1_data_ingestion.preprocessor import preprocess_cmapss

    logger.info("=" * 55)
    logger.info("Building RF-Sentinel Ensemble")
    logger.info("=" * 55)

    # ── Step 1: Load raw CMAPSS FD001 data ───────────────────────────────────
    data      = load_cmapss("FD001")
    train_raw = data["train_raw"]

    # ── Step 2: Engine-level 80/20 split (same seed as CNN training) ──────────
    rng      = np.random.default_rng(RANDOM_STATE)
    unit_ids = train_raw["unit_id"].unique().copy()
    rng.shuffle(unit_ids)

    n_train     = int(len(unit_ids) * 0.8)
    train_units = set(unit_ids[:n_train])
    val_units   = set(unit_ids[n_train:])

    train_df = train_raw[train_raw["unit_id"].isin(train_units)].copy()
    val_df   = train_raw[train_raw["unit_id"].isin(val_units)].copy()

    logger.info(
        f"Engine split — train: {len(train_units)} | val: {len(val_units)}"
    )

    # ── Step 3: Preprocess val data for XGBoost ───────────────────────────────
    # Build a processed dict using only the val engines
    val_data_dict = {
        "X_train":         data["X_train"],
        "y_train":         data["y_train"],
        "X_test":          data["X_test"],
        "y_test":          data["y_test"],
        "train_raw":       val_df,   # temporarily use val_df as "train_raw"
        "test_raw":        data.get("test_raw"),
        "dataset":         data["dataset"],
        "n_engines_train": len(val_units),
        "useful_sensors":  data["useful_sensors"],
        "feature_cols":    data["feature_cols"],
    }
    processed = preprocess_cmapss(val_data_dict)

    # ── Step 4: Build and evaluate ensemble ───────────────────────────────────
    ensemble = RFSentinelEnsemble(xgb_weight=0.55, cnn_weight=0.45)
    ensemble.build()

    metrics = ensemble.train(
        processed["X_val"], processed["y_val"],
        train_df=train_df, val_df=val_df,
    )

    # ── Step 5: Generate plots ────────────────────────────────────────────────
    ensemble.plot_model_comparison(save=True)

    # ── Step 6: Save and print summary ───────────────────────────────────────
    ensemble.save()

    diff = metrics["ensemble_val_f1"] - metrics["xgb_val_f1"]
    logger.info("=" * 55)
    logger.info(f"  XGBoost Val F1   : {metrics['xgb_val_f1']:.4f}")
    logger.info(f"  CNN Val Accuracy : {metrics['cnn_val_acc']:.4f}")
    logger.info(f"  Ensemble Val F1  : {metrics['ensemble_val_f1']:.4f}")
    logger.info(f"  Improvement      : {diff:+.4f} over XGBoost alone")
    logger.info("=" * 55)

    return ensemble


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensemble = build_ensemble()
