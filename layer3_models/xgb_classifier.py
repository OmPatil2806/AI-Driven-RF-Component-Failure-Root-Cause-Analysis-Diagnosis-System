"""
xgb_classifier.py — XGBoost multi-class failure classifier for RF-Sentinel.

Inherits from BaseModel and is trained on:
    - CMAPSS FD001  (binary: pass vs sensor_degradation)
    - AI4I 2020     (5-class: 5 distinct RF failure modes)

Explains predictions via TreeSHAP and saves five diagnostic plots:
    - SHAP beeswarm summary
    - SHAP waterfall for a single prediction
    - Built-in XGBoost feature importance
    - Confusion matrix heatmap
    - Training loss curves (mlogloss per boosting round)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from layer1_data_ingestion.config import ROOT_DIR
from layer3_models.base_model import BaseModel

PLOTS_DIR = ROOT_DIR / "outputs" / "models" / "xgboost"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class RFSentinelXGB(BaseModel):
    """
    XGBoost multi-class failure classifier for RF-Sentinel.

    Wraps XGBClassifier with a LabelEncoder so it can handle both integer
    and string class labels. Built-in SHAP explanation via TreeExplainer.

    Usage
    -----
    model = RFSentinelXGB()
    model.build(n_estimators=200)
    model.train(X_train, y_train, X_val, y_val,
                feature_names=..., class_names=...)
    model.evaluate(X_val, y_val)
    model.plot_feature_importance()
    model.save()
    """

    model_name = "xgb_classifier"
    model_type = "xgboost_multiclass"

    def __init__(self) -> None:
        self.model:            Optional[XGBClassifier] = None
        self.label_encoder:    Optional[LabelEncoder]  = None
        self.feature_names:    Optional[List[str]]     = None
        self.classes_:         Optional[np.ndarray]    = None
        self.shap_values:      Optional[Any]           = None
        self.explainer:        Optional[Any]           = None
        self.n_classes:        Optional[int]           = None
        self.training_history: Dict[str, Any]          = {}

    # ── Abstract method implementations ──────────────────────────────────────

    def build(self, **kwargs) -> None:
        """
        Construct the XGBClassifier with sensible RF-Sentinel defaults.

        All hyperparameters can be overridden via kwargs.  Defaults are
        tuned for tabular sensor data with moderate class imbalance.

        Parameters
        ----------
        **kwargs : any XGBClassifier hyperparameter to override
        """
        # sample_weight="balanced" in fit() handles imbalance
        # better than aggressive regularization alone.
        # These balanced hyperparameters let the model learn
        # enough complexity while sample weights guide it
        # to focus on rare failure classes.
        params = {
            "n_estimators":     300,
            "max_depth":        5,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma":            0.1,
            "reg_alpha":        0.3,
            "reg_lambda":       2.0,
            "objective":        "multi:softprob",
            "eval_metric":      "mlogloss",
            "random_state":     42,
            "n_jobs":           -1,
            "tree_method":      "hist",
            "verbosity":        0,
        }
        params.update(kwargs)
        self.model = XGBClassifier(**params)
        logger.info(f"[{self.model_name}] Built XGBClassifier — params: {params}")

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names: Optional[List[str]] = None,
        class_names:   Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fit the XGBoost model and record all training metrics.

        Steps
        -----
        1. Log training header.
        2. Encode labels with LabelEncoder (works for int or string labels).
        3. Resolve feature names.
        4. Fit with eval_set for tracking mlogloss per round.
        5. Compute macro-F1 train and val scores.
        6. Store evals_result_ in training_history.
        7. Log footer with final metrics.

        Parameters
        ----------
        X_train, y_train : training split (numpy or DataFrame)
        X_val,   y_val   : validation split
        feature_names    : column names for SHAP plots (optional)
        class_names      : human-readable class labels (optional)

        Returns
        -------
        dict with keys:
            train_score, val_score, training_time,
            n_classes, classes, n_features
        """
        if self.model is None:
            self.build()

        X_train = np.array(X_train)
        X_val   = np.array(X_val)
        y_train = np.array(y_train).ravel()
        y_val   = np.array(y_val).ravel()

        self.log_training_start(
            dataset_name=self.model_name,
            X_shape=X_train.shape,
            y_shape=y_train.shape,
        )

        # ── Step 2: Encode string labels to integers ──────────────────────────
        # XGBoost needs integer class indices not strings.
        # We always encode here — preprocessor now returns
        # raw string labels for multiclass targets.
        y_train_arr = np.array(y_train).ravel()
        y_val_arr   = np.array(y_val).ravel()

        self.label_encoder = LabelEncoder()
        # Fit on all unique labels from both train and val
        # so no unseen labels appear at prediction time
        all_labels = np.concatenate([y_train_arr, y_val_arr])
        self.label_encoder.fit(all_labels)

        y_train_encoded = self.label_encoder.transform(y_train_arr)
        y_val_encoded   = self.label_encoder.transform(y_val_arr)

        self.classes_  = self.label_encoder.classes_
        self.n_classes = len(self.classes_)

        # Override display names if caller provided them (e.g. CMAPSS binary
        # labels arrive as 0/1 integers but we want "pass"/"sensor_degradation")
        if class_names is not None:
            self.classes_ = np.array(class_names)

        # Update XGBoost to expect correct number of classes
        self.model.set_params(num_class=self.n_classes)

        logger.info(
            f"[{self.model_name}] Classes: {self.classes_} "
            f"| n_classes: {self.n_classes}"
        )

        # ── Feature names ─────────────────────────────────────────────────────
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        )

        # ── Fit ───────────────────────────────────────────────────────────────
        # Compute per-sample weights so rare failure classes cost more to
        # misclassify — more effective than aggressive regularisation alone.
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train_encoded,
        )

        t0 = time.time()
        self.model.fit(
            X_train, y_train_encoded,
            sample_weight=sample_weights,
            eval_set=[
                (X_train, y_train_encoded),
                (X_val,   y_val_encoded),
            ],
            verbose=False,
        )
        self.training_time = time.time() - t0
        self.is_trained    = True

        # ── Scores ────────────────────────────────────────────────────────────
        # With multi:softprob, model.predict() can return a 2-D probability
        # matrix instead of 1-D class indices.  Always derive integer class
        # indices via argmax on predict_proba to guarantee 1-D output.
        # Use weighted F1: accounts for class size so rare classes (RNF=18)
        # don't unfairly dominate the overall score as with macro averaging.
        train_preds = np.argmax(self.model.predict_proba(X_train), axis=1)
        val_preds   = np.argmax(self.model.predict_proba(X_val),   axis=1)

        self.train_score = f1_score(
            y_train_encoded,
            train_preds,
            average="weighted",
            zero_division=0,
        )
        self.val_score = f1_score(
            y_val_encoded,
            val_preds,
            average="weighted",
            zero_division=0,
        )

        # ── Training history from evals_result_ ──────────────────────────────
        self.training_history = self.model.evals_result()

        metrics = {
            "train_score":   self.train_score,
            "val_score":     self.val_score,
            "training_time": self.training_time,
            "n_classes":     self.n_classes,
            "classes":       list(self.classes_),
            "n_features":    X_train.shape[1],
        }
        self.log_training_end(metrics)
        return metrics

    def predict(self, X) -> np.ndarray:
        """
        Return decoded class labels for each sample in X.

        Parameters
        ----------
        X : feature array, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of decoded labels (same dtype as original y)
        """
        self.check_is_trained()
        int_preds = np.argmax(self.model.predict_proba(np.array(X)), axis=1)
        return self.label_encoder.inverse_transform(int_preds)

    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probability estimates.

        Parameters
        ----------
        X : feature array, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        self.check_is_trained()
        return self.model.predict_proba(np.array(X))

    def evaluate(self, X, y) -> Dict[str, Any]:
        """
        Compute a full classification metrics report on the given split.

        Parameters
        ----------
        X : feature array
        y : true labels (same encoding as training y)

        Returns
        -------
        dict with keys:
            accuracy, f1_macro, f1_weighted, roc_auc,
            confusion_matrix (list), classification_report (str)
        """
        self.check_is_trained()
        X     = np.array(X)
        y_arr = np.array(y).ravel()
        try:
            y_enc = self.label_encoder.transform(y_arr)
        except Exception:
            y_enc = y_arr.astype(int)
        y_pred = np.argmax(self.model.predict_proba(X), axis=1)  # safe 1-D ints

        acc       = accuracy_score(y_enc, y_pred)
        f1_macro  = f1_score(y_enc, y_pred, average="macro",    zero_division=0)
        f1_weight = f1_score(y_enc, y_pred, average="weighted", zero_division=0)

        try:
            proba   = self.predict_proba(X)
            roc_auc = roc_auc_score(
                y_enc, proba,
                multi_class="ovr", average="weighted",
            )
        except Exception:
            roc_auc = 0.0

        cm     = confusion_matrix(y_enc, y_pred)
        report = classification_report(
            y_enc, y_pred,
            target_names=self.classes_.astype(str),
            zero_division=0,
        )

        logger.info(
            f"[{self.model_name}] Evaluation — "
            f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | "
            f"f1_weighted={f1_weight:.4f} | roc_auc={roc_auc:.4f}"
        )

        return {
            "accuracy":               acc,
            "f1_macro":               f1_macro,
            "f1_weighted":            f1_weight,
            "roc_auc":                roc_auc,
            "confusion_matrix":       cm.tolist(),
            "classification_report":  report,
        }

    # ── Explainability ────────────────────────────────────────────────────────

    def explain(self, X, max_samples: int = 500) -> Any:
        """
        Compute SHAP values using TreeExplainer.

        For large datasets, X is randomly sub-sampled to max_samples rows
        to keep computation time reasonable.

        Parameters
        ----------
        X           : feature array to explain
        max_samples : maximum rows to use for SHAP computation

        Returns
        -------
        shap_values : list of arrays (one per class) for multiclass,
                      or 2-D array for binary
        """
        self.check_is_trained()
        X = np.array(X)
        if X.shape[0] > max_samples:
            rng     = np.random.default_rng(42)
            idx     = rng.choice(X.shape[0], size=max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        self.explainer  = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        logger.info(
            f"[{self.model_name}] SHAP values calculated "
            f"for {X_sample.shape[0]} samples"
        )
        return self.shap_values

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_shap_summary(self, X, save: bool = True) -> plt.Figure:
        """
        SHAP beeswarm summary plot showing global feature importance.

        Each dot is one sample; x-axis = SHAP value impact; colour = feature value.
        For multiclass, summary_plot shows importance across all classes.

        Parameters
        ----------
        X    : feature array (used to compute SHAP values if not yet done)
        save : write PNG to PLOTS_DIR if True

        Returns
        -------
        matplotlib Figure
        """
        if self.shap_values is None:
            self.explain(X)

        X_arr = np.array(X)
        if X_arr.shape[0] > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_arr.shape[0], 500, replace=False)
            X_arr = X_arr[idx]

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X_arr,
            feature_names=self.feature_names,
            class_names=list(self.classes_.astype(str)),
            show=False,
        )
        plt.title("XGBoost — SHAP Feature Importance Summary",
                  fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "xgb_shap_summary.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Saved SHAP summary → {path}")

        return fig

    def plot_shap_waterfall(self, X, sample_idx=0, save=True):
        """
        SHAP waterfall for a single prediction, showing which features
        drove the model toward the predicted failure class.

        Positive SHAP values (red) push the prediction toward the predicted
        class; negative values (blue) push away from it.

        Parameters
        ----------
        X          : feature array
        sample_idx : which row to explain
        save       : write PNG if True

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.patches as mpatches

        self.check_is_trained()

        if self.shap_values is None:
            self.explain(X)

        # Get single sample as 2D array (1, n_features)
        X_arr    = np.array(X)
        X_sample = X_arr[sample_idx:sample_idx + 1]

        # predict() with multi:softprob may return a probability array
        # not class indices — squeeze to scalar safely via numpy
        raw_pred = self.model.predict(X_sample)
        squeezed = np.squeeze(raw_pred)
        pred_enc = int(squeezed.item() if squeezed.ndim == 0
                       else squeezed.flat[0])
        n_classes  = len(self.classes_) if self.classes_ is not None else 0
        pred_label = (
            str(self.classes_[pred_enc])
            if (n_classes > 0 and pred_enc < n_classes)
            else str(pred_enc)
        )

        # Resolve SHAP values for this sample and predicted class.
        # shap_values can be:
        #   list of arrays  — multiclass, one array per class
        #   3-D array        — (n_samples, n_features, n_classes)
        #   2-D array        — binary (n_samples, n_features)
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            sample_shap = shap_vals[pred_enc][sample_idx]
        elif shap_vals.ndim == 3:
            sample_shap = shap_vals[sample_idx, :, pred_enc]
        else:
            sample_shap = shap_vals[sample_idx]

        feat_names = (
            self.feature_names
            if self.feature_names
            else [f"feature_{i}" for i in range(len(sample_shap))]
        )

        # Sort by absolute SHAP value and keep top-15
        shap_df = (
            pd.DataFrame({"feature": feat_names, "shap_value": sample_shap})
            .sort_values("shap_value", key=abs, ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        colors  = ["crimson" if v > 0 else "steelblue"
                   for v in shap_df["shap_value"]]

        bars = ax.barh(
            shap_df["feature"][::-1],
            shap_df["shap_value"][::-1],
            color=colors[::-1],
            edgecolor="white",
            height=0.6,
        )

        for bar, val in zip(bars, shap_df["shap_value"][::-1]):
            x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
            ha    = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=9)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"SHAP Waterfall — Sample {sample_idx}\n"
            f"Predicted: {pred_label}",
            fontweight="bold", fontsize=13,
        )
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)

        red_patch  = mpatches.Patch(color="crimson",   label="Pushes toward failure")
        blue_patch = mpatches.Patch(color="steelblue", label="Pushes toward pass")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")

        plt.tight_layout()

        if save:
            path = PLOTS_DIR / f"{self.model_name}_shap_waterfall_{sample_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] SHAP waterfall saved → {path}")

        return fig

    def plot_feature_importance(
        self,
        top_n: int = 20,
        save: bool = True,
    ) -> plt.Figure:
        """
        Horizontal bar chart of XGBoost's built-in feature importance (F-score).

        Bars are coloured along a RdYlGn gradient — highest importance is
        green, lowest is red — so the most diagnostic features stand out.

        Parameters
        ----------
        top_n : how many features to display (default 20)
        save  : write PNG if True

        Returns
        -------
        matplotlib Figure
        """
        self.check_is_trained()

        importance = self.model.feature_importances_
        feat_df = (
            pd.DataFrame({
                "feature":    self.feature_names,
                "importance": importance,
            })
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        cmap   = plt.cm.RdYlGn
        norm   = plt.Normalize(feat_df["importance"].min(),
                               feat_df["importance"].max())
        colors = [cmap(norm(v)) for v in feat_df["importance"]]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(
            feat_df["feature"][::-1],
            feat_df["importance"][::-1],
            color=colors[::-1], edgecolor="white",
        )
        for bar, val in zip(bars, feat_df["importance"][::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax.set_title(f"XGBoost — Top {top_n} Feature Importances",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance (F-score)", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "xgb_feature_importance.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Saved feature importance → {path}")

        return fig

    def plot_confusion_matrix(self, X, y, save: bool = True) -> plt.Figure:
        """
        Seaborn heatmap confusion matrix with accuracy annotated in the title.

        Rows = true label, columns = predicted label.  Off-diagonal cells reveal
        which failure modes the model confuses most — critical for root-cause
        analysis reliability.

        Parameters
        ----------
        X    : feature array
        y    : true labels
        save : write PNG if True

        Returns
        -------
        matplotlib Figure
        """
        self.check_is_trained()

        # Encode y to integers — handle both string and integer inputs safely
        y_arr = np.array(y).ravel()
        try:
            y_enc = self.label_encoder.transform(y_arr)
        except Exception:
            y_enc = y_arr.astype(int)

        # Integer predictions via argmax (safe for multi:softprob objective)
        X_arr  = np.array(X)
        y_pred = np.argmax(self.model.predict_proba(X_arr), axis=1)

        acc = accuracy_score(y_enc, y_pred)
        cm  = confusion_matrix(y_enc, y_pred)

        class_labels = (
            self.classes_.astype(str)
            if self.classes_ is not None
            else [str(i) for i in range(self.n_classes)]
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(
            f"XGBoost Confusion Matrix\n(Accuracy: {acc:.3f})",
            fontweight="bold", fontsize=13, pad=12,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / f"{self.model_name}_confusion_matrix.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Confusion matrix saved → {path}")

        return fig

    def plot_training_curves(self, save: bool = True) -> plt.Figure:
        """
        Plot mlogloss per boosting round for train and validation splits.

        The gap between curves reveals overfitting; the red dot marks
        the round with the lowest validation loss.

        Parameters
        ----------
        save : write PNG if True

        Returns
        -------
        matplotlib Figure
        """
        self.check_is_trained()
        if not self.training_history:
            raise RuntimeError("No training history found. Call train() first.")

        hist_keys  = list(self.training_history.keys())
        train_loss = self.training_history[hist_keys[0]]["mlogloss"]
        val_loss   = self.training_history[hist_keys[1]]["mlogloss"]
        rounds     = np.arange(1, len(train_loss) + 1)
        best_round = int(np.argmin(val_loss)) + 1
        best_val   = min(val_loss)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, train_loss, color="steelblue", linewidth=2, label="Train")
        ax.plot(rounds, val_loss,   color="orange",    linewidth=2, label="Validation")
        ax.fill_between(rounds, train_loss, val_loss, alpha=0.1, color="gray")
        ax.scatter(best_round, best_val, color="red", s=80, zorder=5,
                   label=f"Best val @ round {best_round} ({best_val:.4f})")
        ax.annotate(
            f"round {best_round}\nloss={best_val:.4f}",
            xy=(best_round, best_val),
            xytext=(best_round + len(rounds) * 0.05, best_val + 0.02),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red", fontsize=9,
        )
        ax.set_title("XGBoost Training Curves — Log Loss",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Boosting Round", fontsize=11)
        ax.set_ylabel("Log Loss (mlogloss)", fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "xgb_training_curves.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Saved training curves → {path}")

        return fig


# ── Standalone training function ──────────────────────────────────────────────

def train_xgb_on_all_datasets() -> Dict[str, RFSentinelXGB]:
    """
    Train XGBoost on CMAPSS FD001 (binary) and AI4I (multiclass) and
    save each model with its full diagnostic plot suite.

    Returns
    -------
    dict with keys "cmapss" and "ai4i", each holding a trained RFSentinelXGB
    """
    from layer1_data_ingestion.config import CMAPSS_USEFUL_SENSORS
    from layer1_data_ingestion.loaders import load_ai4i, load_cmapss
    from layer1_data_ingestion.preprocessor import preprocess_ai4i, preprocess_cmapss

    results: Dict[str, RFSentinelXGB] = {}
    summary_rows: List[Dict] = []

    # ── CMAPSS FD001 — binary failure detection ───────────────────────────────
    logger.info("=" * 55)
    logger.info("Training XGBoost on CMAPSS FD001 (binary)")
    logger.info("=" * 55)

    cmapss_data = load_cmapss("FD001")
    pp_cmapss   = preprocess_cmapss(cmapss_data)

    model_cmapss = RFSentinelXGB()
    model_cmapss.build(n_estimators=300)
    metrics_c = model_cmapss.train(
        pp_cmapss["X_train"], pp_cmapss["y_train"],
        pp_cmapss["X_val"],   pp_cmapss["y_val"],
        feature_names=CMAPSS_USEFUL_SENSORS,
        class_names=["pass", "sensor_degradation"],
    )
    model_cmapss.evaluate(pp_cmapss["X_val"], pp_cmapss["y_val"])

    model_cmapss.plot_confusion_matrix(pp_cmapss["X_val"], pp_cmapss["y_val"])
    model_cmapss.plot_feature_importance()
    model_cmapss.plot_training_curves()
    model_cmapss.plot_shap_summary(pp_cmapss["X_val"])
    model_cmapss.save()

    summary_rows.append({
        "dataset":       "CMAPSS FD001",
        "train_score":   round(metrics_c["train_score"], 4),
        "val_score":     round(metrics_c["val_score"], 4),
        "n_classes":     metrics_c["n_classes"],
        "training_time": f"{metrics_c['training_time']:.1f}s",
    })
    results["cmapss"] = model_cmapss

    # ── AI4I 2020 — 5-class failure mode classification ───────────────────────
    logger.info("=" * 55)
    logger.info("Training XGBoost on AI4I 2020 (multiclass)")
    logger.info("=" * 55)

    ai4i_data  = load_ai4i()
    pp_ai4i    = preprocess_ai4i(ai4i_data, target="multiclass")

    # Recover string class names from the preprocessor's LabelEncoder
    class_names_ai4i = (
        list(pp_ai4i["label_encoder"].classes_.astype(str))
        if pp_ai4i["label_encoder"] is not None
        else None
    )

    model_ai4i = RFSentinelXGB()
    model_ai4i.model_name = "xgb_classifier_ai4i"
    model_ai4i.build(n_estimators=300)
    metrics_a = model_ai4i.train(
        pp_ai4i["X_train"], pp_ai4i["y_train"],
        pp_ai4i["X_val"],   pp_ai4i["y_val"],
        feature_names=ai4i_data["feature_names"],
        class_names=class_names_ai4i,
    )
    model_ai4i.evaluate(pp_ai4i["X_val"], pp_ai4i["y_val"])

    model_ai4i.plot_confusion_matrix(pp_ai4i["X_val"], pp_ai4i["y_val"])
    model_ai4i.plot_feature_importance()
    model_ai4i.plot_training_curves()
    model_ai4i.plot_shap_summary(pp_ai4i["X_val"])
    model_ai4i.save()

    summary_rows.append({
        "dataset":       "AI4I 2020",
        "train_score":   round(metrics_a["train_score"], 4),
        "val_score":     round(metrics_a["val_score"], 4),
        "n_classes":     metrics_a["n_classes"],
        "training_time": f"{metrics_a['training_time']:.1f}s",
    })
    results["ai4i"] = model_ai4i

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("XGBoost Training Summary")
    logger.info("=" * 55)
    header = f"{'Dataset':<18} {'Train F1':>10} {'Val F1':>10} {'Classes':>8} {'Time':>8}"
    logger.info(header)
    logger.info("-" * 55)
    for row in summary_rows:
        logger.info(
            f"{row['dataset']:<18} {row['train_score']:>10.4f} "
            f"{row['val_score']:>10.4f} {row['n_classes']:>8} {row['training_time']:>8}"
        )
    logger.info("=" * 55)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = train_xgb_on_all_datasets()
