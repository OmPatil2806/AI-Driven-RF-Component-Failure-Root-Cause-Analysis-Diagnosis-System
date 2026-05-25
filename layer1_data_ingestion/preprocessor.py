"""
preprocessor.py — Dataset-specific preprocessing pipelines for RF-Sentinel.

Responsibilities
----------------
- Build reusable sklearn / imbalanced-learn pipelines (imputation, scaling,
  feature selection, SMOTE oversampling).
- Produce train / validation splits that are ready to feed into model layers.
- Never touch raw files — call loaders.py first, pass its output dicts here.

Public API
----------
    build_tabular_pipeline(use_smote, k_features) -> Pipeline
    preprocess_cmapss(cmapss_data)               -> dict
    preprocess_secom(secom_data)                 -> dict
    preprocess_ai4i(ai4i_data, target)           -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

from layer1_data_ingestion.config import (
    MAX_FEATURES_SECOM,
    RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    TEST_SIZE,
)


# ── Function 1: build_tabular_pipeline ───────────────────────────────────────

def build_tabular_pipeline(
    use_smote: bool = True,
    k_features: Optional[int] = None,
) -> ImbPipeline:
    """
    Construct an imbalanced-learn Pipeline with optional feature selection and SMOTE.

    Pipeline step order
    -------------------
    1. SimpleImputer       — median imputation for NaN values
    2. VarianceThreshold   — drop zero-variance (constant) features
    3. StandardScaler      — zero-mean, unit-variance normalisation
    4. SelectKBest         — univariate feature selection (only if k_features given)
    5. SMOTE               — minority-class oversampling (only if use_smote=True)

    Parameters
    ----------
    use_smote : bool
        Whether to append a SMOTE resampling step (default True).
    k_features : int or None
        Number of top features to keep via SelectKBest. Skipped if None.

    Returns
    -------
    ImbPipeline
        Unfitted imbalanced-learn pipeline ready for fit_resample / fit_transform.
    """
    steps: List[tuple] = [
        ("imputer",  SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler",   StandardScaler()),
    ]

    if k_features is not None:
        steps.append(
            ("select_k", SelectKBest(score_func=f_classif, k=k_features))
        )

    if use_smote:
        steps.append(
            ("smote", SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE))
        )

    step_names = [name for name, _ in steps]
    logger.debug(f"[Pipeline] Built pipeline steps: {step_names}")
    return ImbPipeline(steps=steps)


# ── Function 2: preprocess_cmapss ────────────────────────────────────────────

def preprocess_cmapss(cmapss_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess one C-MAPSS dataset dict (output of loaders.load_cmapss).

    Steps
    -----
    1. Extract X_train, y_train from the input dict.
    2. Stratified train / val split (TEST_SIZE, RANDOM_STATE).
    3. Fit a simple sklearn Pipeline (SimpleImputer → StandardScaler) on train.
    4. Transform train and val; transform test set if present.

    Parameters
    ----------
    cmapss_data : dict
        Output dict from loaders.load_cmapss(), must contain keys:
        X_train, y_train, feature_cols. Optionally X_test, y_test.

    Returns
    -------
    dict with keys:
        X_train, y_train   — processed training arrays
        X_val,   y_val     — processed validation arrays
        X_test,  y_test    — processed test arrays (None if unavailable)
        pipeline           — fitted sklearn Pipeline
        feature_names      — list of feature column names
        dataset            — subset name string (e.g. "FD001")
    """
    dataset_name: str = cmapss_data.get("dataset", "unknown")
    feature_names: List[str] = cmapss_data["feature_cols"]

    X: pd.DataFrame = cmapss_data["X_train"]
    y: pd.Series    = cmapss_data["y_train"]

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Lightweight sklearn pipeline — no SMOTE needed for C-MAPSS
    # (class balance is controlled by RUL_THRESHOLD tuning)
    pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    X_tr  = pipeline.fit_transform(X_tr_raw)
    X_val = pipeline.transform(X_val_raw)

    # Process held-out test set if loaders provided it
    X_test_out = y_test_out = None
    if cmapss_data.get("X_test") is not None:
        X_test_out = pipeline.transform(cmapss_data["X_test"])
        y_test_out = (
            cmapss_data["y_test"].values
            if cmapss_data.get("y_test") is not None
            else None
        )

    logger.info(
        f"[C-MAPSS | {dataset_name}] "
        f"train={X_tr.shape} | "
        f"val={X_val.shape} | "
        f"test={'yes' if X_test_out is not None else 'no'}"
    )

    return {
        "X_train":      X_tr,
        "y_train":      y_tr.values,
        "X_val":        X_val,
        "y_val":        y_val.values,
        "X_test":       X_test_out,
        "y_test":       y_test_out,
        "pipeline":     pipeline,
        "feature_names": feature_names,
        "dataset":      dataset_name,
    }


# ── Function 3: preprocess_secom ─────────────────────────────────────────────

def preprocess_secom(secom_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the SECOM dataset dict (output of loaders.load_secom).

    Steps
    -----
    1. Extract X and y from the input dict.
    2. Stratified train / val split (TEST_SIZE, RANDOM_STATE).
    3. Build pipeline: median imputation → variance filter → scaling
       → SelectKBest(MAX_FEATURES_SECOM) → SMOTE.
    4. fit_resample on training data (SMOTE applied here).
    5. Transform validation data through all steps except SMOTE
       via pipeline[:-1].transform(X_val).

    Parameters
    ----------
    secom_data : dict
        Output dict from loaders.load_secom(), must contain keys: X, y.

    Returns
    -------
    dict with keys:
        X_train, y_train   — resampled training arrays (after SMOTE)
        X_val,   y_val     — validation arrays (no SMOTE)
        pipeline           — fitted ImbPipeline
    """
    X: pd.DataFrame = secom_data["X"]
    y: pd.Series    = secom_data["y"]

    X_tr_raw, X_val_raw, y_tr_raw, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Class counts before SMOTE
    counts_before = pd.Series(y_tr_raw).value_counts().sort_index()

    pipeline = build_tabular_pipeline(use_smote=True, k_features=MAX_FEATURES_SECOM)

    # fit_resample runs the full pipeline including SMOTE on training data
    X_tr, y_tr = pipeline.fit_resample(X_tr_raw, y_tr_raw)

    # Validation: run through all transformer steps, skip SMOTE (last step)
    X_val = pipeline[:-1].transform(X_val_raw)

    counts_after = pd.Series(y_tr).value_counts().sort_index()

    logger.info(
        f"[SECOM] "
        f"original_train={len(y_tr_raw)} | "
        f"after_SMOTE={len(y_tr)} | "
        f"val={X_val.shape} | "
        f"class_before={counts_before.to_dict()} | "
        f"class_after={counts_after.to_dict()}"
    )

    return {
        "X_train":  X_tr,
        "y_train":  y_tr,
        "X_val":    X_val,
        "y_val":    y_val.values,
        "pipeline": pipeline,
    }


# ── Function 4: preprocess_ai4i ──────────────────────────────────────────────

def preprocess_ai4i(
    ai4i_data: Dict[str, Any],
    target: str = "binary",
) -> Dict[str, Any]:
    """
    Preprocess the AI4I dataset dict (output of loaders.load_ai4i).

    Steps
    -----
    1. Select y_binary or y_multiclass based on `target` parameter.
    2. If multiclass: encode string labels with LabelEncoder.
    3. Stratified train / val split (TEST_SIZE, RANDOM_STATE).
    4. Build pipeline: median imputation → variance filter → scaling → SMOTE.
    5. fit_resample on training data; pipeline[:-1].transform on validation.

    Parameters
    ----------
    ai4i_data : dict
        Output dict from loaders.load_ai4i(), must contain keys:
        X, y_binary, y_multiclass, feature_names.
    target : {"binary", "multiclass"}
        Which label to use. Default is "binary".

    Returns
    -------
    dict with keys:
        X_train       — resampled training feature array
        y_train       — resampled training labels
        X_val         — validation feature array
        y_val         — validation labels
        pipeline      — fitted ImbPipeline
        label_encoder — fitted LabelEncoder (None when target=="binary")
        target        — echoed target string
    """
    if target not in ("binary", "multiclass"):
        raise ValueError(f"target must be 'binary' or 'multiclass', got '{target}'.")

    X: pd.DataFrame = ai4i_data["X"]
    label_encoder: Optional[LabelEncoder] = None

    if target == "binary":
        y = ai4i_data["y_binary"]
    else:
        # Keep string labels — xgb_classifier will encode them.
        # This prevents double-encoding (strings → int → '0','1','2').
        label_encoder = None
        y = ai4i_data["y_multiclass"]  # keep original strings

    X_tr_raw, X_val_raw, y_tr_raw, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_tabular_pipeline(use_smote=True, k_features=None)

    X_tr, y_tr = pipeline.fit_resample(X_tr_raw, y_tr_raw)
    X_val_out  = pipeline[:-1].transform(X_val_raw)

    logger.info(
        f"[AI4I | target={target}] "
        f"train_after_SMOTE={X_tr.shape} | "
        f"val={X_val_out.shape}"
    )

    return {
        "X_train":      X_tr,
        "y_train":      y_tr,
        "X_val":        X_val_out,
        "y_val":        y_val.values,
        "pipeline":     pipeline,
        "label_encoder": label_encoder,
        "target":       target,
    }
