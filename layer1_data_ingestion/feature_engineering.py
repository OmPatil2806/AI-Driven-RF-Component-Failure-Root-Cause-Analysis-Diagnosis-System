"""
feature_engineering.py — All feature engineering for RF-Sentinel.

Design philosophy
-----------------
Every feature here is physically motivated, not a random combination:

  C-MAPSS (turbofan time-series)
  --------------------------------
  - Rolling statistics    capture the local trend window; a sudden spike in
    std or a rising mean signals the onset of degradation.
  - Delta / acceleration  reveal the rate of change; healthy engines show
    near-zero first derivatives, while failing ones show increasing gradients.
  - Exponential MA        gives the model a smoothed signal that weights recent
    cycles more heavily — exactly what matters as failure approaches.
  - Cycle features        provide temporal context: where is this engine in its
    expected life? A sensor reading means different things at cycle 10 vs 300.
  - Degradation index     collapses 14 sensors into a single 0–1 health score,
    useful as a standalone diagnostic and as an input feature.
  - Cross-sensor ratios   encode known thermodynamic relationships (temp/pressure
    ratios, coolant differentials) that are more stable across operating
    conditions than raw readings.

  SECOM (semiconductor, 591 sensors)
  ------------------------------------
  - PCA dimensionality reduction retains 95 % of variance while removing
    collinear sensor noise — critical before any tree or distance-based model.

Public API
----------
    add_rolling_features(df, sensors, window)
    add_delta_features(df, sensors)
    add_exponential_moving_average(df, sensors, span)
    add_cycle_features(df)
    add_degradation_index(df, sensors)
    add_cross_sensor_features(df)
    apply_pca_secom(X, n_components)
    engineer_cmapss_features(df, sensors)   ← master wrapper
    get_feature_summary(df_original, df_engineered)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from layer1_data_ingestion.config import (
    CMAPSS_USEFUL_SENSORS,
    PCA_N_COMPONENTS,
    RANDOM_STATE,
    ROLLING_WINDOW,
)


# ── Function 1: add_rolling_features ─────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    sensors: List[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Add per-engine rolling statistics for each sensor over a sliding cycle window.

    WHY: A single cycle reading is noisy. The rolling mean smooths sensor
    drift and reveals true trends. Rolling std detects volatility onset — a
    healthy engine shows low, stable std; a degrading one shows rising variance.
    Rolling min/max bracket the operating envelope per window, exposing
    anomalous excursions that point means cannot capture.

    Parameters
    ----------
    df : pd.DataFrame
        Training frame with columns 'unit_id', 'cycle', and sensor columns.
    sensors : list[str]
        Sensor column names to compute rolling features for.
    window : int
        Cycle window size (default ROLLING_WINDOW from config).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with rolling feature columns appended in-place.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    new_cols: List[str] = []

    for sensor in sensors:
        grouped = df.groupby("unit_id")[sensor]

        df[f"{sensor}_roll_mean"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_roll_std"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"{sensor}_roll_min"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f"{sensor}_roll_max"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        new_cols += [
            f"{sensor}_roll_mean", f"{sensor}_roll_std",
            f"{sensor}_roll_min",  f"{sensor}_roll_max",
        ]

    logger.info(
        f"[FE] Rolling features added: {len(new_cols)} columns "
        f"({len(sensors)} sensors × 4 stats, window={window})"
    )
    return df


# ── Function 2: add_delta_features ───────────────────────────────────────────

def add_delta_features(
    df: pd.DataFrame,
    sensors: List[str],
) -> pd.DataFrame:
    """
    Add first and second differences (velocity and acceleration) for each sensor.

    WHY: The raw sensor value tells you *where* the engine is; the first delta
    tells you *how fast* it is changing; the second delta (acceleration) tells
    you whether that change is itself accelerating — the clearest early warning
    of impending failure. Near-zero deltas indicate steady-state; diverging
    values indicate active degradation.

    Parameters
    ----------
    df : pd.DataFrame
        Training frame sorted by (unit_id, cycle).
    sensors : list[str]
        Sensor column names to differentiate.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with delta and delta² columns appended.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    new_cols: List[str] = []

    for sensor in sensors:
        grouped = df.groupby("unit_id")[sensor]

        df[f"{sensor}_delta"] = (
            grouped.transform(lambda x: x.diff(1)).fillna(0)
        )
        df[f"{sensor}_delta2"] = (
            grouped.transform(lambda x: x.diff(1).diff(1)).fillna(0)
        )
        new_cols += [f"{sensor}_delta", f"{sensor}_delta2"]

    logger.info(
        f"[FE] Delta features added: {len(new_cols)} columns "
        f"({len(sensors)} sensors × 2 orders)"
    )
    return df


# ── Function 3: add_exponential_moving_average ───────────────────────────────

def add_exponential_moving_average(
    df: pd.DataFrame,
    sensors: List[str],
    span: int = 10,
) -> pd.DataFrame:
    """
    Add exponential weighted moving average (EMA) for each sensor per engine.

    WHY: Unlike simple rolling mean, EMA assigns exponentially decreasing
    weights to older observations — so the signal reacts faster to recent
    changes. This is especially valuable for degradation modelling because
    the last 10–20 cycles before failure carry the most diagnostic information.
    Span=10 balances smoothing with responsiveness.

    Parameters
    ----------
    df : pd.DataFrame
        Training frame sorted by (unit_id, cycle).
    sensors : list[str]
        Sensor column names to compute EMA for.
    span : int
        Decay span in cycles (default 10).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with EMA columns appended.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    new_cols: List[str] = []

    for sensor in sensors:
        df[f"{sensor}_ema"] = (
            df.groupby("unit_id")[sensor]
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            .fillna(0)
        )
        new_cols.append(f"{sensor}_ema")

    logger.info(
        f"[FE] EMA features added: {len(new_cols)} columns "
        f"({len(sensors)} sensors, span={span})"
    )
    return df


# ── Function 4: add_cycle_features ───────────────────────────────────────────

def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal position features that contextualise each sensor reading
    within an engine's observed lifetime.

    WHY: The same sensor value at cycle 5 vs cycle 250 has completely different
    diagnostic meaning. cycle_pct normalises position to [0, 1] regardless
    of engine life length, making the model invariant to absolute cycle count.
    cycles_remaining_est provides the inverse view. cycle_log compresses the
    long-tail of extended engines so early cycles aren't dominated by outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Training frame with 'unit_id' and 'cycle' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with cycle_pct, cycles_remaining_est, cycle_log added.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()

    max_cycle_per_unit = df.groupby("unit_id")["cycle"].transform("max")

    df["cycle_pct"]             = df["cycle"] / max_cycle_per_unit
    df["cycles_remaining_est"]  = max_cycle_per_unit - df["cycle"]
    df["cycle_log"]             = np.log1p(df["cycle"])

    logger.info(
        "[FE] Cycle features added: cycle_pct, cycles_remaining_est, cycle_log"
    )
    return df


# ── Function 5: add_degradation_index ────────────────────────────────────────

def add_degradation_index(
    df: pd.DataFrame,
    sensors: List[str],
) -> pd.DataFrame:
    """
    Compute a composite 0–1 health degradation index per cycle.

    WHY: Individual sensors give partial pictures. The degradation index
    collapses all useful sensors into a single scalar that represents overall
    component health. Per-engine normalisation removes manufacturing variance —
    each engine starts at ~0 and drifts toward 1 as it degrades. The smoothed
    version reduces noise while preserving the degradation trend, making it
    both a useful input feature and a human-readable health gauge.

    Construction
    ------------
    1. Normalise each sensor per engine: (x − min) / (max − min + ε).
    2. Average normalised values across all useful sensors → degradation_index.
    3. Apply rolling mean (ROLLING_WINDOW) → degradation_index_smooth.

    Parameters
    ----------
    df : pd.DataFrame
        Training frame with sensor columns and 'unit_id'.
    sensors : list[str]
        Sensors to include in the composite score.

    Returns
    -------
    pd.DataFrame
        DataFrame with degradation_index and degradation_index_smooth columns.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    norm_cols: List[str] = []

    for sensor in sensors:
        col_min = df.groupby("unit_id")[sensor].transform("min")
        col_max = df.groupby("unit_id")[sensor].transform("max")
        norm_col = f"_norm_{sensor}"
        df[norm_col] = (df[sensor] - col_min) / (col_max - col_min + 1e-8)
        norm_cols.append(norm_col)

    df["degradation_index"] = df[norm_cols].mean(axis=1)

    df["degradation_index_smooth"] = (
        df.groupby("unit_id")["degradation_index"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    # Drop intermediate normalised columns — internal calculation only
    df.drop(columns=norm_cols, inplace=True)

    mean_di = df["degradation_index"].mean()
    logger.info(
        f"[FE] Degradation index added "
        f"(mean={mean_di:.4f}, sensors={len(sensors)})"
    )
    return df


# ── Function 6: add_cross_sensor_features ────────────────────────────────────

def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add thermodynamically motivated interaction features between sensor pairs.

    WHY: Raw sensor readings are condition-dependent — the same s3 value means
    different things at different throttle settings. Ratios and differences
    between related sensors are more stable across operating conditions and
    encode known physical relationships from turbofan thermodynamics:

    - temp_ratio          (s3/s2): LPC temperature amplification factor.
                          A rising ratio indicates compressor efficiency loss.
    - pressure_temp_ratio (s7/s3): Higher pressure with lower temperature
                          deviation signals HPC degradation early.
    - speed_torque_proxy  (s9×s14): Correlated under normal operation; their
                          product diverging signals core speed anomaly.
    - coolant_diff        (s21−s20): Coolant imbalance between LPT and HPT;
                          negative shifts indicate thermal management issues.
    - thermal_load        ((s3+s4)/2): Mean turbine outlet temperature — the
                          primary driver of thermal fatigue and component wear.

    Parameters
    ----------
    df : pd.DataFrame
        Training frame containing columns s2, s3, s4, s7, s9, s14, s20, s21.

    Returns
    -------
    pd.DataFrame
        DataFrame with five cross-sensor feature columns added.
    """
    df = df.copy()

    df["temp_ratio"]           = df["s3"] / (df["s2"] + 1e-8)
    df["pressure_temp_ratio"]  = df["s7"] / (df["s3"] + 1e-8)
    df["speed_torque_proxy"]   = df["s9"] * df["s14"]
    df["coolant_diff"]         = df["s21"] - df["s20"]
    df["thermal_load"]         = (df["s3"] + df["s4"]) / 2

    logger.info(
        "[FE] Cross-sensor features added: "
        "temp_ratio, pressure_temp_ratio, speed_torque_proxy, "
        "coolant_diff, thermal_load"
    )
    return df


# ── Function 7: apply_pca_secom ──────────────────────────────────────────────

def apply_pca_secom(
    X: pd.DataFrame,
    n_components: float = PCA_N_COMPONENTS,
) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    Reduce SECOM's high-dimensional sensor space via PCA.

    WHY: SECOM has ~560 features after NaN-column removal. Most are correlated
    — semiconductor processes have tight physical constraints linking sensors.
    PCA removes redundancy, stabilises distance-based models, and reduces
    overfitting risk. Retaining 95 % variance (n_components=0.95) preserves
    nearly all discriminative signal while typically reducing dimensions by 5–10×.

    Steps
    -----
    1. Fill residual NaN with column median (imputation at FE stage).
    2. StandardScaler — PCA is sensitive to scale; all sensors must contribute
       on equal footing before decomposition.
    3. PCA(n_components) — extract principal components.
    4. Return named DataFrame (pca_0, pca_1, …) for downstream compatibility.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from loaders.load_secom() after initial NaN-column drop.
    n_components : float
        Fraction of variance to retain (default 0.95 from config).

    Returns
    -------
    X_pca   : pd.DataFrame  — transformed data, columns pca_0…pca_N
    pca     : PCA           — fitted PCA object (for inference transforms)
    scaler  : StandardScaler — fitted scaler (for inference transforms)
    """
    n_original = X.shape[1]

    # Step 1: impute residual NaN with column median
    X_filled = X.copy()
    for col in X_filled.columns:
        median_val = X_filled[col].median()
        X_filled[col] = X_filled[col].fillna(median_val)

    # Step 2: scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Step 3: PCA
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X_scaled)

    n_components_kept = X_reduced.shape[1]
    variance_explained = pca.explained_variance_ratio_.sum() * 100

    # Step 4: named DataFrame
    col_names = [f"pca_{i}" for i in range(n_components_kept)]
    X_pca = pd.DataFrame(X_reduced, columns=col_names, index=X.index)

    logger.info(
        f"[FE | SECOM PCA] "
        f"original_features={n_original} → "
        f"pca_components={n_components_kept} | "
        f"variance_explained={variance_explained:.2f}%"
    )
    return X_pca, pca, scaler


# ── Function 8: engineer_cmapss_features (master wrapper) ────────────────────

def engineer_cmapss_features(
    df: pd.DataFrame,
    sensors: List[str] = CMAPSS_USEFUL_SENSORS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Master pipeline that applies all C-MAPSS feature engineering in order.

    Applies in sequence:
        1. Rolling statistics  (×4 per sensor)
        2. Delta features      (×2 per sensor)
        3. Exponential MA      (×1 per sensor)
        4. Cycle features      (3 columns)
        5. Degradation index   (2 columns)
        6. Cross-sensor ratios (5 columns)

    Total new features = len(sensors) × 7 + 10
    For 14 useful sensors: 14 × 7 + 10 = 108 new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw C-MAPSS training frame (output of loaders.load_cmapss or train_raw).
    sensors : list[str]
        Sensors to engineer (default CMAPSS_USEFUL_SENSORS from config).

    Returns
    -------
    df_engineered : pd.DataFrame
        Fully enriched DataFrame.
    feature_cols  : list[str]
        All column names usable as model features (excludes unit_id, cycle,
        RUL, fail_soon, and internal columns).
    """
    original_cols = set(df.columns)
    n_original = len(original_cols)

    logger.info(
        f"[FE | C-MAPSS] Starting feature engineering — "
        f"input: {df.shape[0]:,} rows × {n_original} cols"
    )

    df = add_rolling_features(df, sensors)
    df = add_delta_features(df, sensors)
    df = add_exponential_moving_average(df, sensors)
    df = add_cycle_features(df)
    df = add_degradation_index(df, sensors)
    df = add_cross_sensor_features(df)

    # Columns added by each group for reporting
    new_cols = set(df.columns) - original_cols
    n_final  = len(df.columns)

    feature_groups = {
        "rolling":     [c for c in new_cols if "_roll_"  in c],
        "delta":       [c for c in new_cols if "_delta"  in c],
        "ema":         [c for c in new_cols if "_ema"    in c],
        "cycle":       [c for c in new_cols if "cycle"   in c],
        "degradation": [c for c in new_cols if "degrad"  in c],
        "cross_sensor":[c for c in new_cols if c in {
                            "temp_ratio","pressure_temp_ratio",
                            "speed_torque_proxy","coolant_diff","thermal_load"
                        }],
    }

    logger.success(
        f"[FE | C-MAPSS] Done — "
        f"original={n_original} → final={n_final} | "
        f"new_features={len(new_cols)} | "
        f"groups: "
        + ", ".join(f"{k}={len(v)}" for k, v in feature_groups.items())
    )

    # Exclude metadata and label columns from the returned feature list
    exclude = {"unit_id", "cycle", "RUL", "fail_soon"}
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols


# ── Function 9: get_feature_summary ──────────────────────────────────────────

def get_feature_summary(
    df_original: pd.DataFrame,
    df_engineered: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute and log a structured summary of features added during engineering.

    WHY: Tracking feature counts per group makes it easy to audit pipeline
    output, catch regressions when sensors are added/removed, and communicate
    the feature space to stakeholders without reading the full column list.

    Parameters
    ----------
    df_original : pd.DataFrame
        DataFrame before feature engineering.
    df_engineered : pd.DataFrame
        DataFrame after feature engineering.

    Returns
    -------
    dict with keys:
        original_features    int         columns before engineering
        engineered_features  int         new columns added
        total_features       int         total columns after engineering
        feature_groups       dict[str,int] count per feature group
    """
    original_cols   = set(df_original.columns)
    engineered_cols = set(df_engineered.columns)
    new_cols        = engineered_cols - original_cols

    group_keywords = {
        "rolling":      "_roll_",
        "delta":        "_delta",
        "ema":          "_ema",
        "cycle":        "cycle",
        "degradation":  "degrad",
        "cross_sensor": ("temp_ratio", "pressure_temp_ratio",
                         "speed_torque_proxy", "coolant_diff", "thermal_load"),
        "original":     None,
    }

    feature_groups: Dict[str, int] = {}
    for group, keyword in group_keywords.items():
        if group == "original":
            feature_groups[group] = len(original_cols)
        elif isinstance(keyword, tuple):
            feature_groups[group] = sum(1 for c in new_cols if c in keyword)
        else:
            feature_groups[group] = sum(1 for c in new_cols if keyword in c)

    summary = {
        "original_features":   len(original_cols),
        "engineered_features": len(new_cols),
        "total_features":      len(engineered_cols),
        "feature_groups":      feature_groups,
    }

    # Formatted log output
    separator = "─" * 48
    logger.info(separator)
    logger.info("  Feature Engineering Summary")
    logger.info(separator)
    logger.info(f"  Original features   : {summary['original_features']}")
    logger.info(f"  New features added  : {summary['engineered_features']}")
    logger.info(f"  Total features      : {summary['total_features']}")
    logger.info(separator)
    for group, count in feature_groups.items():
        logger.info(f"  {group:<18}: {count}")
    logger.info(separator)

    return summary
