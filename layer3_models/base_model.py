"""
base_model.py — Abstract base class for all RF-Sentinel models.

Every model in layer3_models must inherit from BaseModel and implement
the 5 abstract methods: build, train, predict, predict_proba, evaluate.

Provides a shared interface for:
    - Saving / loading model weights and metadata (joblib + JSON)
    - Consistent logging of training start, progress, and end
    - Guarded method calls via check_is_trained()
    - Unified model info reporting via get_model_info()
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from layer1_data_ingestion.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"


class BaseModel(ABC):
    """
    Abstract base class that every RF-Sentinel model must subclass.

    Subclasses must implement:
        build()        — configure model architecture / hyperparameters
        train()        — fit the model, record scores and timing
        predict()      — return class label predictions
        predict_proba()— return class probability estimates
        evaluate()     — compute a standard metrics dict

    Concrete helpers provided:
        save() / load()       — joblib serialisation + JSON metadata
        get_model_info()      — dict summary of current model state
        check_is_trained()    — guard that raises if model not fitted
        log_training_start()  — formatted log header
        log_training_end()    — formatted log footer with metrics
    """

    # ── Class-level defaults (overridden per subclass) ────────────────────────
    model_name:    str   = "base_model"
    model_type:    str   = "classifier"
    is_trained:    bool  = False
    training_time: float = 0.0
    train_score:   float = 0.0
    val_score:     float = 0.0

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    def build(self, **kwargs) -> None:
        """
        Build and configure the model architecture.

        Called once before training to set up internal estimator,
        hyperparameters, and any preprocessing components. All
        subclass-specific configuration belongs here, not in __init__.

        Parameters
        ----------
        **kwargs
            Model-specific hyperparameters (e.g. n_estimators, lr, layers).
        """

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val:   np.ndarray | pd.DataFrame,
        y_val:   np.ndarray | pd.Series,
    ) -> Dict[str, Any]:
        """
        Fit the model on training data and evaluate on validation data.

        Implementations must:
            - Set self.is_trained = True
            - Set self.training_time (wall-clock seconds)
            - Set self.train_score and self.val_score

        Parameters
        ----------
        X_train, y_train : training features and labels
        X_val,   y_val   : validation features and labels

        Returns
        -------
        dict containing at minimum:
            train_score, val_score, training_time
        """

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Return predicted class labels for X.

        Parameters
        ----------
        X : feature array, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with integer class labels
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Return class probability estimates for X.

        Parameters
        ----------
        X : feature array, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes) with probabilities
        """

    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate the fitted model on the given dataset.

        Parameters
        ----------
        X : feature array
        y : true labels

        Returns
        -------
        dict with keys:
            accuracy, f1_macro, f1_weighted, roc_auc, confusion_matrix
        """

    # ── Concrete methods ──────────────────────────────────────────────────────

    def save(self, path: Optional[str | Path] = None) -> str:
        """
        Serialise the model to disk with joblib and write a JSON metadata file.

        If path is None, saves to:
            ROOT_DIR/models/{model_name}.pkl
            ROOT_DIR/models/{model_name}.json

        Parameters
        ----------
        path : optional explicit file path (without extension)

        Returns
        -------
        str  absolute path of the saved .pkl file
        """
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            pkl_path  = MODELS_DIR / f"{self.model_name}.pkl"
            json_path = MODELS_DIR / f"{self.model_name}.json"
        else:
            pkl_path  = Path(path)
            json_path = pkl_path.with_suffix(".json")
            pkl_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, pkl_path)

        metadata = {
            "model_name":    self.model_name,
            "model_type":    self.model_type,
            "is_trained":    self.is_trained,
            "train_score":   round(self.train_score, 6),
            "val_score":     round(self.val_score, 6),
            "training_time": round(self.training_time, 2),
            "saved_at":      datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[{self.model_name}] Saved model → {pkl_path}")
        return str(pkl_path)

    def load(self, path: Optional[str | Path] = None) -> "BaseModel":
        """
        Load a serialised model from disk using joblib.

        If path is None, loads from:
            ROOT_DIR/models/{model_name}.pkl

        Sets self.is_trained = True after a successful load.

        Parameters
        ----------
        path : optional explicit .pkl file path

        Returns
        -------
        self  (fluent interface)
        """
        if path is None:
            pkl_path = MODELS_DIR / f"{self.model_name}.pkl"
        else:
            pkl_path = Path(path)

        loaded = joblib.load(pkl_path)

        # Copy all attributes from the loaded object into self
        self.__dict__.update(loaded.__dict__)
        self.is_trained = True

        logger.info(f"[{self.model_name}] Loaded model ← {pkl_path}")
        return self

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return a summary dict of the current model state and log it.

        Returns
        -------
        dict with keys:
            model_name, model_type, is_trained,
            train_score, val_score, training_time
        """
        info = {
            "model_name":    self.model_name,
            "model_type":    self.model_type,
            "is_trained":    self.is_trained,
            "train_score":   self.train_score,
            "val_score":     self.val_score,
            "training_time": self.training_time,
        }
        logger.info(
            f"[{self.model_name}] Model info — "
            f"type={self.model_type} | "
            f"trained={self.is_trained} | "
            f"train={self.train_score:.4f} | "
            f"val={self.val_score:.4f} | "
            f"time={self.training_time:.1f}s"
        )
        return info

    def check_is_trained(self) -> None:
        """
        Guard method — raises RuntimeError if the model has not been fitted.

        Call at the start of predict(), predict_proba(), and evaluate()
        in subclasses to give a clear error before a silent crash.

        Raises
        ------
        RuntimeError if self.is_trained is False
        """
        if not self.is_trained:
            raise RuntimeError(
                f"{self.model_name} has not been trained yet. "
                "Call train() first."
            )

    def log_training_start(
        self,
        dataset_name: str,
        X_shape: tuple,
        y_shape: tuple,
    ) -> None:
        """
        Log a formatted header at the start of a training run.

        Parameters
        ----------
        dataset_name : human-readable name of the dataset being trained on
        X_shape      : shape tuple of the training feature matrix
        y_shape      : shape tuple of the training label vector
        """
        sep = "─" * 52
        logger.info(sep)
        logger.info(f"  Training {self.model_name} on {dataset_name}")
        logger.info(f"  X shape : {X_shape} | y shape: {y_shape}")
        logger.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(sep)

    def log_training_end(self, metrics: Dict[str, Any]) -> None:
        """
        Log a formatted summary at the end of a training run.

        Prints train_score, val_score, and training_time from self,
        then logs any additional keys present in the metrics dict.

        Parameters
        ----------
        metrics : dict returned by train(); must contain at minimum
                  train_score, val_score, training_time
        """
        sep = "─" * 52
        logger.success(sep)
        logger.success(f"  Training complete — {self.model_name}")
        logger.success(f"  Train score : {self.train_score:.4f}")
        logger.success(f"  Val score   : {self.val_score:.4f}")
        logger.success(f"  Time        : {self.training_time:.1f}s")

        extra = {
            k: v for k, v in metrics.items()
            if k not in {"train_score", "val_score", "training_time"}
        }
        for k, v in extra.items():
            logger.success(f"  {k:<14}: {v}")

        logger.success(sep)
