"""
cnn1d_model.py — 1D-CNN time-series failure classifier for RF-Sentinel.

Processes sequences of CMAPSS sensor cycles through three stacked
Conv1d blocks, learning degradation patterns at multiple time scales.
Unlike XGBoost (which sees a single-cycle snapshot), the CNN sees how
sensors evolve over time — giving it access to early-warning signals
that only emerge across multiple cycles.

Pipeline:
    load_cmapss() → engine-level train/val split → CMAPSSWindowDataset
    (sliding windows 30 cycles × 14 sensors) → CNN1DArchitecture
    → binary classification (pass / fail)
"""

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — 1D-CNN Sensor Sequence Failure Classifier
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# This file builds the second AI model in RF-Sentinel.
# It uses a 1D Convolutional Neural Network (PyTorch) to
# classify engine failure from TIME-SERIES sensor sequences.
#
# DIFFERENCE FROM XGBOOST:
# ─────────────────────────
# XGBoost sees ONE snapshot:  [s2=445, s3=550, s9=8900, ...]
# 1D-CNN sees a SEQUENCE:     30 consecutive cycles of all 14 sensors
#                             It learns HOW sensors CHANGE over time
#                             not just their current value
#
# ARCHITECTURE:
# ─────────────
# Input shape : (batch, 14 sensors, 30 time steps)
# Conv1d_1    : 14 → 32 filters, kernel=7, captures slow trends
# Conv1d_2    : 32 → 64 filters, kernel=5, captures medium patterns
# Conv1d_3    : 64 → 128 filters, kernel=3, captures sharp changes
# BatchNorm   : stabilizes training after each conv layer
# ReLU        : non-linear activation
# Dropout     : prevents overfitting (p=0.3)
# GlobalAvgPool: reduces sequence to single vector
# FC layers   : 128 → 64 → n_classes
#
# SLIDING WINDOW:
# ───────────────
# We slide a window of 30 cycles across each engine's lifetime
# Window 1: cycles 1-30   → label = pass (RUL=200)
# Window 2: cycles 2-31   → label = pass (RUL=199)
# ...
# Last window: cycles N-29 to N → label = fail (RUL=0)
# This creates many training samples from each engine
#
# DATA:
# ─────
# Dataset : CMAPSS FD001
# Engines : 100 training engines
# Window  : 30 cycles × 14 sensors per sample
# Label   : 0=pass, 1=fail (RUL <= 30)
#
# OUTPUTS:
# ────────
# models/cnn1d_model.pt          ← saved model weights
# models/cnn1d_model.json        ← metadata
# outputs/models/cnn1d/
#     cnn1d_training_curves.png  ← loss + accuracy curves
#     cnn1d_confusion_matrix.png ← per-class performance
#     cnn1d_gradcam.png          ← which time steps matter
# ══════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from layer1_data_ingestion.config import (
    CMAPSS_USEFUL_SENSORS,
    RANDOM_STATE,
    ROOT_DIR,
    RUL_THRESHOLD,
)
from layer3_models.base_model import BaseModel

PLOTS_DIR = ROOT_DIR / "outputs" / "models" / "cnn1d"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Class 1: Sliding-window PyTorch Dataset ───────────────────────────────────

class CMAPSSWindowDataset(Dataset):
    """
    Convert a raw CMAPSS DataFrame into fixed-length sliding windows.

    Each window is a contiguous block of `window_size` consecutive cycles
    for a single engine.  The label is derived from the RUL of the last
    cycle in the window: 1 (fail) if RUL <= rul_threshold, else 0 (pass).

    WHY engine-level windows?
    An engine's sensor readings are temporally correlated.  Mixing cycles
    from different engines in a single window would make no physical sense —
    the CNN needs to see how ONE engine degrades continuously.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensors: List[str],
        window_size: int = 30,
        rul_threshold: int = RUL_THRESHOLD,
    ) -> None:
        self.window_size   = window_size
        self.sensors       = sensors
        self.rul_threshold = rul_threshold

        windows_X: List[np.ndarray] = []
        windows_y: List[int]        = []

        for unit_id, engine_df in df.groupby("unit_id"):
            engine_df = engine_df.sort_values("cycle").reset_index(drop=True)
            n_cycles  = len(engine_df)

            if n_cycles < window_size:
                continue  # engine too short for even one window

            sensor_vals = engine_df[sensors].values   # (n_cycles, n_sensors)
            rul_vals    = engine_df["RUL"].values      # (n_cycles,)

            for start in range(n_cycles - window_size + 1):
                end     = start + window_size
                window  = sensor_vals[start:end]       # (window_size, n_sensors)
                last_rul = rul_vals[end - 1]
                label   = 1 if last_rul <= rul_threshold else 0

                # Transpose to (n_sensors, window_size) for Conv1d
                windows_X.append(window.T.astype(np.float32))
                windows_y.append(label)

        self.X = np.stack(windows_X)   # (n_windows, n_sensors, window_size)
        self.y = np.array(windows_y, dtype=np.int64)

        fail_rate = self.y.mean() * 100
        logger.info(
            f"[CMAPSSWindowDataset] Created {len(self.X):,} windows "
            f"(window={window_size}, fail_rate={fail_rate:.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.LongTensor([self.y[idx]]).squeeze(),
        )


# ── Class 2: PyTorch CNN architecture ────────────────────────────────────────

class CNN1DArchitecture(nn.Module):
    """
    Three-block 1D-CNN for sensor sequence classification.

    Each conv block uses progressively smaller kernels to capture
    degradation signals at three time scales:
        Block 1 (kernel=7): slow drifts over ~7 cycles
        Block 2 (kernel=5): medium-speed transitions
        Block 3 (kernel=3): sharp sudden changes near failure

    Global average pooling collapses the time dimension, making the
    model input-length agnostic (works for any window size).
    """

    def __init__(
        self,
        n_sensors: int = 14,
        n_classes: int = 2,
        dropout:   float = 0.3,
    ) -> None:
        super().__init__()

        # Block 1 — slow trend detection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_sensors, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Block 2 — medium-speed pattern detection
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Block 3 — sharp change / spike detection
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Reduces (128, window_size) → (128, 1) regardless of window size
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three conv blocks and classifier.

        Parameters
        ----------
        x : (batch, n_sensors, window_size)

        Returns
        -------
        logits : (batch, n_classes)
        """
        x = self.conv_block1(x)       # → (batch, 32,  window_size)
        x = self.conv_block2(x)       # → (batch, 64,  window_size)
        x = self.conv_block3(x)       # → (batch, 128, window_size)
        x = self.global_avg_pool(x)   # → (batch, 128, 1)
        x = x.squeeze(-1)             # → (batch, 128)
        x = self.classifier(x)        # → (batch, n_classes)
        return x


# ── Class 3: RF-Sentinel model wrapper ───────────────────────────────────────

class RFSentinelCNN1D(BaseModel):
    """
    1D-CNN failure classifier wrapping CNN1DArchitecture with the BaseModel
    interface (build / train / predict / predict_proba / evaluate / save / load).

    Accepts raw CMAPSS DataFrames (not preprocessed arrays) so that the
    sliding-window construction happens inside the model, keeping the
    temporal structure intact.
    """

    model_name = "cnn1d_model"
    model_type = "pytorch_1d_cnn"

    def __init__(self) -> None:
        self.network:       Optional[CNN1DArchitecture] = None
        self.optimizer:     Optional[optim.Optimizer]   = None
        self.criterion:     Optional[nn.Module]         = None
        self.window_size:   int   = 30
        self.n_sensors:     int   = 14
        self.n_classes:     int   = 2
        self.classes_:      List[str] = ["pass", "fail"]
        self.train_losses:  List[float] = []
        self.val_losses:    List[float] = []
        self.train_accs:    List[float] = []
        self.val_accs:      List[float] = []
        self.device:        torch.device = DEVICE
        self._n_epochs_trained: int = 0

    # ── Abstract method implementations ──────────────────────────────────────

    def build(self, **kwargs) -> None:
        """
        Instantiate the CNN architecture, Adam optimiser, and cross-entropy loss.

        All hyperparameters can be overridden via kwargs.  CrossEntropyLoss
        handles the multi-class case and works with the raw logits output by
        the network (no softmax needed inside forward()).

        Parameters
        ----------
        window_size  : cycles per window (default 30)
        n_sensors    : number of input sensor channels (default 14)
        n_classes    : number of output classes (default 2)
        dropout      : dropout probability throughout the network
        lr           : Adam learning rate
        weight_decay : L2 regularisation coefficient
        """
        window_size  = kwargs.get("window_size",  30)
        n_sensors    = kwargs.get("n_sensors",    14)
        n_classes    = kwargs.get("n_classes",     2)
        dropout      = kwargs.get("dropout",      0.3)
        lr           = kwargs.get("lr",           1e-3)
        weight_decay = kwargs.get("weight_decay", 1e-4)

        self.window_size = window_size
        self.n_sensors   = n_sensors
        self.n_classes   = n_classes

        self.network = CNN1DArchitecture(
            n_sensors=n_sensors,
            n_classes=n_classes,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        n_params = sum(p.numel() for p in self.network.parameters())
        logger.info(
            f"[{self.model_name}] Built CNN1D — "
            f"input=({n_sensors}, {window_size}) | "
            f"n_classes={n_classes} | "
            f"params={n_params:,} | "
            f"device={self.device}"
        )

        # Auto-generate architecture summary plot when model is built
        try:
            self.plot_model_summary(save=True)
            logger.info("[cnn1d_model] Architecture summary plot saved")
        except Exception:
            pass  # Don't fail build if plot fails

    def plot_model_summary(self, save: bool = True) -> plt.Figure:
        """
        Plot a visual summary table of the CNN architecture.
        Shows each layer, input/output shape, parameters count,
        and purpose of each layer in plain English.
        """
        layers_info = [
            {
                "layer":        "Input",
                "type":         "Tensor",
                "input_shape":  f"(batch, {self.n_sensors}, {self.window_size})",
                "output_shape": f"(batch, {self.n_sensors}, {self.window_size})",
                "params":       0,
                "purpose":      f"{self.n_sensors} sensors × {self.window_size} time steps",
            },
            {
                "layer":        "Conv1d Block 1",
                "type":         "Conv1d + BN + ReLU + Dropout",
                "input_shape":  f"(batch, {self.n_sensors}, {self.window_size})",
                "output_shape": f"(batch, 32, {self.window_size})",
                "params":       self.n_sensors * 32 * 7 + 32,
                "purpose":      "Captures slow degradation trends (kernel=7)",
            },
            {
                "layer":        "Conv1d Block 2",
                "type":         "Conv1d + BN + ReLU + Dropout",
                "input_shape":  f"(batch, 32, {self.window_size})",
                "output_shape": f"(batch, 64, {self.window_size})",
                "params":       32 * 64 * 5 + 64,
                "purpose":      "Captures medium patterns (kernel=5)",
            },
            {
                "layer":        "Conv1d Block 3",
                "type":         "Conv1d + BN + ReLU + Dropout",
                "input_shape":  f"(batch, 64, {self.window_size})",
                "output_shape": f"(batch, 128, {self.window_size})",
                "params":       64 * 128 * 3 + 128,
                "purpose":      "Captures sharp sudden changes (kernel=3)",
            },
            {
                "layer":        "GlobalAvgPool1d",
                "type":         "AdaptiveAvgPool1d",
                "input_shape":  f"(batch, 128, {self.window_size})",
                "output_shape": "(batch, 128)",
                "params":       0,
                "purpose":      "Reduces sequence to single vector",
            },
            {
                "layer":        "FC Layer 1",
                "type":         "Linear + ReLU + Dropout",
                "input_shape":  "(batch, 128)",
                "output_shape": "(batch, 64)",
                "params":       128 * 64 + 64,
                "purpose":      "Learns failure patterns from features",
            },
            {
                "layer":        "FC Layer 2 (Output)",
                "type":         "Linear",
                "input_shape":  "(batch, 64)",
                "output_shape": f"(batch, {self.n_classes})",
                "params":       64 * self.n_classes + self.n_classes,
                "purpose":      f"Final prediction: {self.n_classes} classes",
            },
        ]

        total_params = sum(row["params"] for row in layers_info)

        col_labels = [
            "Layer", "Type", "Input Shape",
            "Output Shape", "Parameters", "Purpose",
        ]
        col_widths = [0.12, 0.18, 0.14, 0.14, 0.09, 0.28]

        table_data = [
            [
                row["layer"],
                row["type"],
                row["input_shape"],
                row["output_shape"],
                f"{row['params']:,}",
                row["purpose"],
            ]
            for row in layers_info
        ]

        palette = [
            "#E8F4FD", "#E1F5EE", "#E8F4FD", "#E1F5EE",
            "#FFF3E0", "#E8F4FD", "#FFEBEE",
        ]
        row_colors = [
            [palette[i % len(palette)]] * len(col_labels)
            for i in range(len(table_data))
        ]

        fig, ax = plt.subplots(figsize=(18, 6))
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            cellColours=row_colors,
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)

        # Style header row
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2C3E50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Style output layer row
        for j in range(len(col_labels)):
            table[len(table_data), j].set_facecolor("#FFEBEE")
            table[len(table_data), j].set_text_props(
                color="#C0392B", fontweight="bold"
            )

        ax.text(
            0.5, 0.02,
            f"Total trainable parameters: {total_params:,}  |  "
            f"Device: {self.device}  |  "
            f"Window: {self.window_size} cycles × {self.n_sensors} sensors",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10, color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", alpha=0.8),
        )

        ax.set_title(
            "RF-Sentinel — 1D-CNN Architecture Summary",
            fontsize=14, fontweight="bold", pad=20,
        )
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "cnn1d_model_summary.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[cnn1d_model] Model summary saved → {path}")

        return fig

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_epochs:     int = 30,
        batch_size:   int = 64,
        feature_names = None,
        class_names   = None,
    ) -> Dict[str, Any]:
        """
        Fit the CNN on CMAPSS DataFrames using sliding-window batches.

        WHY pass DataFrames (not arrays)?
        The CNN needs to respect engine boundaries when building windows.
        Passing a flat array would mix cycles from different engines into
        single windows, destroying the temporal meaning of the sequences.

        Parameters
        ----------
        X_train  : raw CMAPSS train DataFrame with 'unit_id', 'cycle', 'RUL'
        y_train  : ignored (labels derived from RUL inside the Dataset)
        X_val    : raw CMAPSS val DataFrame
        y_val    : ignored
        n_epochs : training epochs
        batch_size : DataLoader batch size
        """
        if self.network is None:
            self.build()

        if class_names is not None:
            self.classes_ = list(class_names)

        self.log_training_start(
            dataset_name="CMAPSS (sliding windows)",
            X_shape=X_train.shape,
            y_shape=(len(X_train),),
        )

        t0 = time.time()

        # ── Datasets ──────────────────────────────────────────────────────────
        train_dataset = CMAPSSWindowDataset(
            X_train, CMAPSS_USEFUL_SENSORS, self.window_size, RUL_THRESHOLD,
        )
        val_dataset = CMAPSSWindowDataset(
            X_val, CMAPSS_USEFUL_SENSORS, self.window_size, RUL_THRESHOLD,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
        )

        # ── Epoch loop ────────────────────────────────────────────────────────
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_accs.clear()
        self.val_accs.clear()

        for epoch in range(1, n_epochs + 1):
            # ── Train phase ───────────────────────────────────────────────────
            self.network.train()
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.network(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss    += loss.item() * len(y_batch)
                preds          = logits.argmax(dim=1)
                epoch_correct += (preds == y_batch).sum().item()
                epoch_total   += len(y_batch)

            train_loss = epoch_loss    / epoch_total
            train_acc  = epoch_correct / epoch_total

            # ── Validation phase ──────────────────────────────────────────────
            self.network.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits  = self.network(X_batch)
                    loss    = self.criterion(logits, y_batch)
                    val_loss_sum  += loss.item() * len(y_batch)
                    preds          = logits.argmax(dim=1)
                    val_correct   += (preds == y_batch).sum().item()
                    val_total     += len(y_batch)

            val_loss = val_loss_sum / val_total
            val_acc  = val_correct  / val_total

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:>3}/{n_epochs} — "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

        self.training_time     = time.time() - t0
        self.train_score       = self.train_accs[-1]
        self.val_score         = self.val_accs[-1]
        self.is_trained        = True
        self._n_epochs_trained = n_epochs

        metrics = {
            "train_score":   self.train_score,
            "val_score":     self.val_score,
            "training_time": self.training_time,
            "n_epochs":      n_epochs,
            "train_losses":  self.train_losses,
            "val_losses":    self.val_losses,
        }
        self.log_training_end(metrics)
        return metrics

    def predict(self, X) -> np.ndarray:
        """
        Return binary class predictions (0=pass, 1=fail).

        Accepts either a raw CMAPSS DataFrame (creates a Dataset internally
        preserving engine boundaries) or a pre-windowed numpy array.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray of int class indices, shape (n_windows,)
        """
        self.check_is_trained()
        self.network.eval()

        if isinstance(X, pd.DataFrame):
            dataset = CMAPSSWindowDataset(
                X, CMAPSS_USEFUL_SENSORS, self.window_size, RUL_THRESHOLD,
            )
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            all_preds = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    logits = self.network(X_batch.to(self.device))
                    all_preds.append(logits.argmax(dim=1).cpu().numpy())
            return np.concatenate(all_preds)

        # numpy array already windowed: (n_windows, n_sensors, window_size)
        tensor = torch.FloatTensor(np.array(X)).to(self.device)
        with torch.no_grad():
            logits = self.network(tensor)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probability estimates via softmax.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray of shape (n_windows, n_classes)
        """
        self.check_is_trained()
        self.network.eval()

        if isinstance(X, pd.DataFrame):
            dataset = CMAPSSWindowDataset(
                X, CMAPSS_USEFUL_SENSORS, self.window_size, RUL_THRESHOLD,
            )
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            all_proba = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    logits = self.network(X_batch.to(self.device))
                    proba  = torch.softmax(logits, dim=1)
                    all_proba.append(proba.cpu().numpy())
            return np.concatenate(all_proba)

        tensor = torch.FloatTensor(np.array(X)).to(self.device)
        with torch.no_grad():
            logits = self.network(tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def evaluate(self, X, y) -> Dict[str, Any]:
        """
        Compute classification metrics on a raw CMAPSS DataFrame.

        y is ignored — true labels are derived from the RUL column inside X.
        This matches the pattern used by predict() and avoids any label-alignment
        mismatches between manually split arrays and windowed sequences.

        Parameters
        ----------
        X : pd.DataFrame (raw CMAPSS with RUL column)
        y : ignored

        Returns
        -------
        dict with keys:
            accuracy, f1_macro, f1_weighted, confusion_matrix, classification_report
        """
        self.check_is_trained()

        dataset = CMAPSSWindowDataset(
            X, CMAPSS_USEFUL_SENSORS, self.window_size, RUL_THRESHOLD,
        )
        loader  = DataLoader(dataset, batch_size=256, shuffle=False)
        self.network.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.network(X_batch.to(self.device))
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_targets.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        acc       = accuracy_score(y_true, y_pred)
        f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm        = confusion_matrix(y_true, y_pred)
        report    = classification_report(
            y_true, y_pred, target_names=self.classes_, zero_division=0,
        )

        logger.info(
            f"[{self.model_name}] Evaluation — "
            f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weight:.4f}"
        )
        return {
            "accuracy":              acc,
            "f1_macro":              f1_macro,
            "f1_weighted":           f1_weight,
            "confusion_matrix":      cm.tolist(),
            "classification_report": report,
        }

    # ── PyTorch-specific save / load (override BaseModel) ─────────────────────

    def save(self, path=None) -> str:
        """
        Save network weights with torch.save() and write a JSON metadata file.

        Overrides BaseModel.save() because PyTorch state dicts cannot be
        serialised with joblib without losing gradient information.

        Parameters
        ----------
        path : optional explicit .pt file path

        Returns
        -------
        str  absolute path of the saved .pt file
        """
        if path is None:
            pt_path   = MODELS_DIR / "cnn1d_model.pt"
            json_path = MODELS_DIR / "cnn1d_model.json"
        else:
            pt_path   = Path(path)
            json_path = pt_path.with_suffix(".json")
            pt_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.network.state_dict(), pt_path)

        metadata = {
            "model_name":    self.model_name,
            "model_type":    self.model_type,
            "is_trained":    self.is_trained,
            "train_score":   round(self.train_score,   6),
            "val_score":     round(self.val_score,     6),
            "training_time": round(self.training_time, 2),
            "window_size":   self.window_size,
            "n_sensors":     self.n_sensors,
            "n_classes":     self.n_classes,
            "n_epochs":      self._n_epochs_trained,
            "saved_at":      datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[{self.model_name}] Saved weights → {pt_path}")
        return str(pt_path)

    def load(self, path=None) -> "RFSentinelCNN1D":
        """
        Load network weights from a .pt file.

        Overrides BaseModel.load() for PyTorch state dict format.
        Calls build() automatically if the network has not been initialised.

        Parameters
        ----------
        path : optional .pt file path

        Returns
        -------
        self (fluent interface)
        """
        if path is None:
            pt_path = MODELS_DIR / "cnn1d_model.pt"
        else:
            pt_path = Path(path)

        if self.network is None:
            self.build()

        state = torch.load(pt_path, map_location=self.device)
        self.network.load_state_dict(state)
        self.is_trained = True

        logger.info(f"[{self.model_name}] Loaded weights ← {pt_path}")
        return self

    # ── Diagnostic plots ──────────────────────────────────────────────────────

    def plot_training_curves(self, save: bool = True) -> plt.Figure:
        """
        Side-by-side loss and accuracy curves across training epochs.

        WHY both loss AND accuracy?
        Loss drives parameter updates; accuracy is the human-interpretable
        metric.  The gap between train and val curves reveals overfitting.
        """
        self.check_is_trained()

        epochs = np.arange(1, len(self.train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ── Left: loss ────────────────────────────────────────────────────────
        ax1.plot(epochs, self.train_losses, color="steelblue",
                 linewidth=2, label="Train")
        ax1.plot(epochs, self.val_losses,   color="orange",
                 linewidth=2, label="Validation")
        ax1.fill_between(epochs, self.train_losses, self.val_losses,
                         alpha=0.1, color="gray")

        best_val_loss_idx = int(np.argmin(self.val_losses))
        ax1.scatter(epochs[best_val_loss_idx], self.val_losses[best_val_loss_idx],
                    color="red", s=80, zorder=5,
                    label=f"Best val @ epoch {epochs[best_val_loss_idx]}")

        ax1.set_title("Training & Validation Loss", fontweight="bold", fontsize=12)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax1.legend(fontsize=10)

        # ── Right: accuracy ───────────────────────────────────────────────────
        ax2.plot(epochs, self.train_accs, color="steelblue",
                 linewidth=2, label="Train")
        ax2.plot(epochs, self.val_accs,   color="orange",
                 linewidth=2, label="Validation")

        best_acc_idx = int(np.argmax(self.val_accs))
        ax2.scatter(epochs[best_acc_idx], self.val_accs[best_acc_idx],
                    color="green", s=80, zorder=5,
                    label=f"Best val @ epoch {epochs[best_acc_idx]}")
        ax2.axhline(self.val_accs[-1], color="gray", linestyle="--",
                    linewidth=1, label=f"Final val = {self.val_accs[-1]:.4f}")

        ax2.set_title("Training & Validation Accuracy", fontweight="bold", fontsize=12)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=10)

        fig.suptitle("1D-CNN Training Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "cnn1d_training_curves.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Training curves saved → {path}")

        return fig

    def plot_confusion_matrix(self, X, y=None, save: bool = True) -> plt.Figure:
        """
        Heatmap confusion matrix from a raw CMAPSS DataFrame.

        y is ignored — labels come from the RUL column in X via
        CMAPSSWindowDataset, matching the same source used in train/evaluate.

        Parameters
        ----------
        X    : raw CMAPSS DataFrame
        y    : ignored
        save : write PNG if True
        """
        self.check_is_trained()

        dataset = CMAPSSWindowDataset(
            X, CMAPSS_USEFUL_SENSORS, self.window_size, RUL_THRESHOLD,
        )
        loader  = DataLoader(dataset, batch_size=256, shuffle=False)
        self.network.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.network(X_batch.to(self.device))
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_targets.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        acc    = accuracy_score(y_true, y_pred)
        cm     = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(
            f"1D-CNN Confusion Matrix (Acc: {acc:.3f})",
            fontweight="bold", fontsize=13, pad=12,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "cnn1d_confusion_matrix.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[{self.model_name}] Confusion matrix saved → {path}")

        return fig


# ── Standalone training function ──────────────────────────────────────────────

def train_cnn_on_cmapss() -> RFSentinelCNN1D:
    """
    Train a 1D-CNN on CMAPSS FD001 with an engine-level train/val split.

    WHY split by engine (not by row)?
    A sliding window creates overlapping samples from the same engine.
    If we split by row, a window from engine 42 cycle 100-130 would
    appear in train while cycle 101-131 (almost identical) is in val —
    causing data leakage and inflated val accuracy.  Splitting by engine
    guarantees the model never sees any cycle from a val engine during training.

    Returns
    -------
    RFSentinelCNN1D — trained model instance
    """
    from layer1_data_ingestion.loaders import load_cmapss

    logger.info("=" * 55)
    logger.info("Training 1D-CNN on CMAPSS FD001")
    logger.info("=" * 55)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    data      = load_cmapss("FD001")
    train_raw = data["train_raw"]

    # ── Step 2: Engine-level train / val split ────────────────────────────────
    rng         = np.random.default_rng(RANDOM_STATE)
    unit_ids    = train_raw["unit_id"].unique()
    rng.shuffle(unit_ids)

    n_train     = int(len(unit_ids) * 0.8)
    train_units = set(unit_ids[:n_train])
    val_units   = set(unit_ids[n_train:])

    train_df = train_raw[train_raw["unit_id"].isin(train_units)].copy()
    val_df   = train_raw[train_raw["unit_id"].isin(val_units)].copy()

    logger.info(
        f"Engine split — train: {len(train_units)} engines "
        f"({len(train_df):,} rows) | "
        f"val: {len(val_units)} engines ({len(val_df):,} rows)"
    )

    # ── Step 3: Build and train ───────────────────────────────────────────────
    model = RFSentinelCNN1D()
    model.build(
        window_size=30,
        n_sensors=len(CMAPSS_USEFUL_SENSORS),
        n_classes=2,
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
    )

    metrics = model.train(
        train_df, None,
        val_df,   None,
        n_epochs=30,
        batch_size=64,
        class_names=["pass", "fail"],
    )

    # ── Step 4: Diagnostic plots ──────────────────────────────────────────────
    model.plot_training_curves(save=True)
    model.plot_confusion_matrix(val_df, save=True)

    # ── Step 5: Save and report ───────────────────────────────────────────────
    model.save()

    ev = model.evaluate(val_df, None)
    logger.info("=" * 55)
    logger.info("1D-CNN Training Complete")
    logger.info(f"  Val Accuracy : {ev['accuracy']:.4f}")
    logger.info(f"  Val F1 Macro : {ev['f1_macro']:.4f}")
    logger.info(f"  Training time: {metrics['training_time']:.1f}s")
    logger.info("=" * 55)

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = train_cnn_on_cmapss()
