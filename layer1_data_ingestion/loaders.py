"""
loaders.py — Raw-file readers for all three RF-Sentinel datasets.

Provides five public functions:
    load_cmapss(dataset)   — single C-MAPSS subset → ML-ready split dict
    load_all_cmapss()      — all four C-MAPSS subsets
    load_secom()           — SECOM semiconductor dataset with NaN audit
    load_ai4i()            — AI4I 2020 with failure-type labelling
    load_all_datasets()    — convenience wrapper for all three sources

All heavy feature engineering and scaling live in preprocessor.py;
these functions only parse files, engineer RUL/labels, and return
ML-ready arrays.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from layer1_data_ingestion.config import (
    AI4I_FAILURE_COLS,
    AI4I_FILE,
    AI4I_TARGET_COL,
    AI4I_TO_RF_MAP,
    CMAPSS_COLS,
    CMAPSS_DATASETS,
    CMAPSS_RUL_FILES,
    CMAPSS_TEST_FILES,
    CMAPSS_TRAIN_FILES,
    CMAPSS_USEFUL_SENSORS,
    RUL_THRESHOLD,
    SECOM_DATA_FILE,
    SECOM_LABELS_FILE,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _read_cmapss_file(path) -> pd.DataFrame:
    """Parse a space-delimited C-MAPSS text file and strip trailing empty cols."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLS,
        index_col=False,
    )
    df.dropna(axis=1, how="all", inplace=True)
    return df


# ── Function 1: load_cmapss ───────────────────────────────────────────────────

def load_cmapss(dataset: str = "FD001") -> Dict[str, Any]:
    """
    Load and prepare one C-MAPSS subset for binary classification (fail_soon).

    Steps
    -----
    1. Read the training txt file; assign CMAPSS_COLS; drop trailing empty cols.
    2. Engineer RUL  = max_cycle_per_engine - current_cycle.
    3. Engineer fail_soon = 1 if RUL <= RUL_THRESHOLD else 0.
    4. If a test file exists: take the last observed cycle per engine and pair
       it with the ground-truth RUL file converted to a binary fail_soon label.

    Returns
    -------
    dict with keys:
        X_train         pd.DataFrame  training feature matrix
        y_train         pd.Series     binary fail_soon labels
        X_test          pd.DataFrame | None
        y_test          pd.Series    | None
        train_raw       pd.DataFrame  full training frame (with RUL, fail_soon)
        dataset         str           e.g. "FD001"
        n_engines_train int
        useful_sensors  list[str]
        feature_cols    list[str]
    """
    if dataset not in CMAPSS_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {CMAPSS_DATASETS}.")

    train_path = CMAPSS_TRAIN_FILES[dataset]
    if not train_path.exists():
        raise FileNotFoundError(f"C-MAPSS train file not found: {train_path}")

    # ── Load training data ────────────────────────────────────────────────────
    train_df = _read_cmapss_file(train_path)

    # RUL = remaining cycles until the engine reached end-of-life in training
    max_cycle = train_df.groupby("unit_id")["cycle"].max()
    train_df["RUL"] = train_df.apply(
        lambda r: max_cycle[r["unit_id"]] - r["cycle"], axis=1
    )
    train_df["fail_soon"] = (train_df["RUL"] <= RUL_THRESHOLD).astype(int)

    feature_cols: List[str] = CMAPSS_USEFUL_SENSORS
    X_train = train_df[feature_cols].copy()
    y_train = train_df["fail_soon"].copy()

    n_engines = train_df["unit_id"].nunique()
    fail_rate = y_train.mean() * 100

    # ── Load test data (optional) ─────────────────────────────────────────────
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None
    test_raw: pd.DataFrame | None = None  # full test time series for FE

    if dataset in CMAPSS_TEST_FILES and CMAPSS_TEST_FILES[dataset].exists():
        test_df = _read_cmapss_file(CMAPSS_TEST_FILES[dataset])
        test_raw = test_df.copy()  # store full history so pipeline can engineer features on it

        # C-MAPSS test files record sensor history up to a cut-off point;
        # the last cycle per engine is the observation we evaluate on.
        last_cycles = (
            test_df.sort_values("cycle")
            .groupby("unit_id")
            .tail(1)
            .reset_index(drop=True)
        )
        X_test = last_cycles[feature_cols].copy()

        if dataset in CMAPSS_RUL_FILES and CMAPSS_RUL_FILES[dataset].exists():
            rul_series = pd.read_csv(
                CMAPSS_RUL_FILES[dataset], header=None, names=["RUL"]
            ).squeeze("columns")
            # Align index with X_test rows (one RUL value per engine)
            y_test = (rul_series.values <= RUL_THRESHOLD).astype(int)
            y_test = pd.Series(y_test, name="fail_soon")

    logger.success(
        f"[C-MAPSS | {dataset}] "
        f"engines={n_engines} | "
        f"train_rows={len(train_df):,} | "
        f"fail_rate={fail_rate:.1f}% | "
        f"features={len(feature_cols)} | "
        f"test={'yes' if X_test is not None else 'no'}"
    )

    return {
        "X_train":         X_train,
        "y_train":         y_train,
        "X_test":          X_test,
        "y_test":          y_test,
        "train_raw":       train_df,
        "test_raw":        test_raw,   # full test time series; used by pipeline for feature engineering
        "dataset":         dataset,
        "n_engines_train": n_engines,
        "useful_sensors":  CMAPSS_USEFUL_SENSORS,
        "feature_cols":    feature_cols,
    }


# ── Function 2: load_all_cmapss ───────────────────────────────────────────────

def load_all_cmapss() -> Dict[str, Dict[str, Any]]:
    """
    Load all four C-MAPSS subsets by calling load_cmapss for each.

    Returns
    -------
    dict keyed by dataset name, e.g.:
        {
            "FD001": {X_train, y_train, X_test, y_test, ...},
            "FD002": {X_train, y_train, X_test=None, y_test=None, ...},
            ...
        }
    """
    result: Dict[str, Dict[str, Any]] = {}
    for ds in CMAPSS_DATASETS:
        result[ds] = load_cmapss(ds)
    logger.success(
        f"[C-MAPSS] All subsets loaded — "
        f"total train rows: {sum(len(v['train_raw']) for v in result.values()):,}"
    )
    return result


# ── Function 3: load_secom ────────────────────────────────────────────────────

def load_secom() -> Dict[str, Any]:
    """
    Load the SECOM semiconductor manufacturing dataset.

    Steps
    -----
    1. Read secom.data (space-delimited, 1567×591) with NaN variants flagged.
    2. Name columns feature_0 … feature_N.
    3. Read secom_labels.data; remap -1 → 0 (pass), 1 → 1 (fail).
    4. Compute missing-value percentage per column.
    5. Drop columns where missing > 50 %.

    Returns
    -------
    dict with keys:
        X                  pd.DataFrame  cleaned feature matrix
        y                  pd.Series     binary labels (0=pass, 1=fail)
        feature_names      list[str]     surviving column names
        missing_pct_per_col pd.Series    missing % for every original column
        cols_dropped       list[str]     columns removed (>50 % missing)
        class_counts       pd.Series     value counts of y
    """
    for path in (SECOM_DATA_FILE, SECOM_LABELS_FILE):
        if not path.exists():
            raise FileNotFoundError(f"SECOM file not found: {path}")

    # Feature matrix
    X_raw = pd.read_csv(
        SECOM_DATA_FILE,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan", "NA", ""],
    )
    X_raw.columns = [f"feature_{i}" for i in range(X_raw.shape[1])]

    # Labels: -1 (pass) → 0, 1 (fail) → 1
    label_df = pd.read_csv(
        SECOM_LABELS_FILE,
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
    )
    y = label_df["label"].map({-1: 0, 1: 1}).astype(int).rename("label")

    # Missing-value audit
    missing_pct = X_raw.isna().mean() * 100
    cols_dropped = missing_pct[missing_pct > 50].index.tolist()
    X_clean = X_raw.drop(columns=cols_dropped)

    fail_count = y.sum()
    logger.success(
        f"[SECOM] "
        f"samples={len(X_clean):,} | "
        f"features_kept={X_clean.shape[1]} | "
        f"features_dropped={len(cols_dropped)} (>50% missing) | "
        f"failures={fail_count} / {len(y)} ({fail_count / len(y) * 100:.1f}%)"
    )

    return {
        "X":                   X_clean,
        "y":                   y,
        "feature_names":       X_clean.columns.tolist(),
        "missing_pct_per_col": missing_pct,
        "cols_dropped":        cols_dropped,
        "class_counts":        y.value_counts().sort_index(),
    }


# ── Function 4: load_ai4i ─────────────────────────────────────────────────────

def load_ai4i() -> Dict[str, Any]:
    """
    Load the AI4I 2020 predictive maintenance dataset and engineer labels.

    Steps
    -----
    1. Read ai4i2020.csv.
    2. Build failure_type by scanning TWF→HDF→PWF→OSF→RNF in order;
       use AI4I_TO_RF_MAP to convert to RF-Sentinel canonical names.
       If no flag is set, label is "pass".
    3. Encode the Type column: L=0, M=1, H=2 → Type_encoded.
    4. Assemble X from [Type_encoded, numeric process features].
    5. y_binary    = Machine failure (0/1).
       y_multiclass = failure_type string.

    Returns
    -------
    dict with keys:
        X                   pd.DataFrame  feature matrix
        y_binary            pd.Series     binary failure label
        y_multiclass        pd.Series     canonical failure-type string
        failure_type_counts pd.Series     counts per failure type
        feature_names       list[str]
        df_raw              pd.DataFrame  full original DataFrame
    """
    if not AI4I_FILE.exists():
        raise FileNotFoundError(f"AI4I file not found: {AI4I_FILE}")

    df = pd.read_csv(AI4I_FILE)

    # ── Failure-type labelling ────────────────────────────────────────────────
    # Scan failure columns in priority order; use first flag found per row
    def _assign_failure_type(row: pd.Series) -> str:
        for col in AI4I_FAILURE_COLS:          # TWF, HDF, PWF, OSF, RNF
            if row[col] == 1:
                return AI4I_TO_RF_MAP[col]
        return AI4I_TO_RF_MAP["no_failure"]    # "pass"

    df["failure_type"] = df.apply(_assign_failure_type, axis=1)

    # ── Type encoding ─────────────────────────────────────────────────────────
    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type_encoded"] = df["Type"].map(type_map)

    # ── Feature matrix ────────────────────────────────────────────────────────
    feature_cols = [
        "Type_encoded",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    X = df[feature_cols].copy()
    y_binary     = df[AI4I_TARGET_COL].astype(int).rename("failure")
    y_multiclass = df["failure_type"].rename("failure_type")

    fail_count   = y_binary.sum()
    unique_types = y_multiclass.nunique()

    logger.success(
        f"[AI4I] "
        f"samples={len(df):,} | "
        f"features={len(feature_cols)} | "
        f"failures={fail_count} ({fail_count / len(df) * 100:.2f}%) | "
        f"failure_types={unique_types}"
    )

    return {
        "X":                   X,
        "y_binary":            y_binary,
        "y_multiclass":        y_multiclass,
        "failure_type_counts": y_multiclass.value_counts(),
        "feature_names":       feature_cols,
        "df_raw":              df,
    }


# ── Function 5: load_all_datasets ─────────────────────────────────────────────

def load_all_datasets() -> Dict[str, Any]:
    """
    Load all three RF-Sentinel datasets in one call.

    Returns
    -------
    dict with keys:
        "cmapss"  — output of load_all_cmapss()
        "secom"   — output of load_secom()
        "ai4i"    — output of load_ai4i()
    """
    logger.info("=" * 60)
    logger.info("RF-Sentinel — loading all raw datasets")
    logger.info("=" * 60)

    data = {
        "cmapss": load_all_cmapss(),
        "secom":  load_secom(),
        "ai4i":   load_ai4i(),
    }

    logger.success("All RF-Sentinel datasets loaded and ready.")
    return data
