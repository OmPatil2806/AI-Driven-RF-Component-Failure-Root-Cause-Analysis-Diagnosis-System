"""
schema_mapper.py — Unified RF parameter schema for RF-Sentinel.

Design philosophy
-----------------
RF component failure analysis requires measuring how parameters like insertion
loss (S21), noise figure, output power (P1dB), and linearity (IP3) degrade
over time. None of our three datasets directly measures these RF parameters —
they come from turbofan engines, semiconductor manufacturing lines, and
industrial machines. The schema mapper bridges this gap by finding the closest
physical analogue in each dataset to each RF measurement:

  S21 insertion loss (rf_param_1)
      Gain drops as internal temperature rises. CMAPSS s3 (HPC outlet
      temperature) tracks this thermal gain-compression mechanism directly.

  Noise figure (rf_param_2)
      Downstream temperature elevation degrades SNR. CMAPSS s4 (LPT outlet
      temperature) represents the downstream thermal environment.

  P1dB / output power (rf_param_3)
      Power handling capacity scales with rotational speed in turbomachinery
      and RF amplifier bias current. CMAPSS s9 (physical core speed) and
      AI4I rotational speed are the closest proxies.

  IP3 / linearity (rf_param_4)
      Linearity degrades under mechanical and thermal stress. CMAPSS s11
      (HPC static pressure) and AI4I torque represent equivalent stress loads.

  Temperature (rf_param_5)
      Direct ambient/junction temperature — the most universal RF degradation
      driver across all three datasets.

  Frequency proxy (rf_param_6)
      Operating frequency in RF systems corresponds to rotational speed in
      mechanical systems; both set the fundamental operating regime.

  Pressure / bias proxy (rf_param_7)
      Supply voltage / bias current in RF maps to hydraulic pressure in
      turbomachinery and tool wear accumulation in manufacturing.

  Secondary parameter (rf_param_8)
      Coolant flow (CMAPSS), anonymous high-variance sensor (SECOM), or
      product quality grade (AI4I) — dataset-specific secondary context.

By mapping to this common 8-parameter schema, a single model can learn
failure signatures across heterogeneous physical systems and transfer
diagnostic knowledge between them.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from layer1_data_ingestion.config import (
    AI4I_FAILURE_COLS,
    AI4I_TO_RF_MAP,
    CMAPSS_USEFUL_SENSORS,
)

# ── Unified schema column order ───────────────────────────────────────────────
UNIFIED_SCHEMA_COLS: List[str] = [
    "device_id",
    "cycle_or_sample",
    "rf_param_1",     # S21 insertion loss proxy
    "rf_param_2",     # noise figure proxy
    "rf_param_3",     # output power / P1dB proxy
    "rf_param_4",     # linearity / IP3 proxy
    "rf_param_5",     # temperature
    "rf_param_6",     # frequency / speed proxy
    "rf_param_7",     # pressure / bias proxy
    "rf_param_8",     # secondary parameter
    "failure_label",  # 0 = pass, 1 = fail
    "failure_type",   # string description of failure mode
    "dataset_source", # origin dataset identifier
    "rul",            # remaining useful life (NaN if not applicable)
]


# ── Function 1: map_cmapss_to_rf_schema ──────────────────────────────────────

def map_cmapss_to_rf_schema(
    train_raw: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Map a C-MAPSS training DataFrame to the unified RF parameter schema.

    Mapping rationale
    -----------------
    Each CMAPSS sensor is chosen because it captures the same physical
    degradation mechanism as its RF counterpart:

        rf_param_1 = s3  — HPC outlet temperature
                          Thermal gain compression: as HPC degrades, outlet
                          temperature rises, analogous to S21 gain dropping
                          in an amplifier running hot.

        rf_param_2 = s4  — LPT outlet temperature
                          Downstream thermal elevation degrades noise
                          performance, exactly as elevated junction temp
                          raises noise figure in RF transistors.

        rf_param_3 = s9  — Physical core speed (rpm)
                          Rotational energy → power handling capacity.
                          Speed drop = P1dB compression, just as bias
                          current reduction limits RF output power.

        rf_param_4 = s11 — HPC outlet static pressure
                          Pressure load drives mechanical stress and
                          nonlinearity, mirroring how drive level raises
                          intermodulation and reduces IP3.

        rf_param_5 = s2  — LPC outlet temperature (fan inlet / ambient)
                          Primary operating temperature — the universal
                          RF degradation driver.

        rf_param_6 = s14 — Corrected core speed
                          Speed normalised to standard conditions; analogous
                          to RF operating frequency setting the regime.

        rf_param_7 = s7  — HPC outlet pressure
                          Supply/bias pressure equivalent — sets the
                          operating point and its drift signals wear.

        rf_param_8 = s21 — LPT coolant bleed
                          Secondary thermal management parameter; maps to
                          heat-sink effectiveness in RF assemblies.

    Parameters
    ----------
    train_raw : pd.DataFrame
        Raw training frame from loaders.load_cmapss()["train_raw"].
        Must contain columns: unit_id, cycle, s2–s21, fail_soon, RUL.
    dataset_name : str
        Subset identifier, e.g. "FD001".

    Returns
    -------
    pd.DataFrame
        Rows in the unified schema column order.
    """
    df = train_raw.copy()

    mapped = pd.DataFrame({
        "device_id":       df["unit_id"].astype(str),
        "cycle_or_sample": df["cycle"],
        "rf_param_1":      df["s3"],   # HPC outlet temp → S21 gain loss
        "rf_param_2":      df["s4"],   # LPT outlet temp → noise figure
        "rf_param_3":      df["s9"],   # core speed → P1dB proxy
        "rf_param_4":      df["s11"],  # HPC static pressure → IP3 proxy
        "rf_param_5":      df["s2"],   # LPC outlet temp → operating temperature
        "rf_param_6":      df["s14"],  # corrected core speed → frequency proxy
        "rf_param_7":      df["s7"],   # HPC outlet pressure → bias/supply proxy
        "rf_param_8":      df["s21"],  # LPT coolant bleed → secondary thermal
        "failure_label":   df["fail_soon"].astype(int),
        "failure_type":    df["fail_soon"].map({1: "sensor_degradation", 0: "pass"}),
        "dataset_source":  f"cmapss_{dataset_name}",
        "rul":             df["RUL"],
    })

    fail_rate = mapped["failure_label"].mean() * 100
    logger.info(
        f"[SchemaMapper | C-MAPSS/{dataset_name}] "
        f"rows={len(mapped):,} | "
        f"failure_rate={fail_rate:.2f}%"
    )
    return mapped[UNIFIED_SCHEMA_COLS]


# ── Function 2: map_secom_to_rf_schema ───────────────────────────────────────

def map_secom_to_rf_schema(secom_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Map the SECOM semiconductor dataset to the unified RF parameter schema.

    SECOM's 562 surviving features have no named physical interpretation.
    The highest-variance features are selected as RF parameter proxies because
    variance is a proxy for information content — low-variance sensors are
    near-constant and carry little diagnostic signal. The top 8 by variance
    span the most dynamic measurement channels in the semiconductor process,
    which physically correspond to etch rate, deposition uniformity, chamber
    pressure, temperature uniformity, and RF power delivery in semiconductor
    manufacturing — the closest analogues to RF measurement parameters.

    Parameters
    ----------
    secom_data : dict
        Output of loaders.load_secom(). Must contain keys: X (DataFrame), y (Series).

    Returns
    -------
    pd.DataFrame
        Rows in the unified schema column order.
    """
    X: pd.DataFrame = secom_data["X"]
    y: pd.Series    = secom_data["y"]

    # Step 1-2: select top 8 features by variance (ignoring NaN)
    variances = X.var(skipna=True).sort_values(ascending=False)
    top8_cols = variances.index[:8].tolist()

    logger.info(
        f"[SchemaMapper | SECOM] Top-8 features by variance: {top8_cols}"
    )

    # Step 3: map to rf_param_1…rf_param_8 in variance-rank order
    mapped = pd.DataFrame({
        "device_id":       [f"secom_{i}" for i in X.index],
        "cycle_or_sample": X.index,
        "rf_param_1":      X[top8_cols[0]].values,
        "rf_param_2":      X[top8_cols[1]].values,
        "rf_param_3":      X[top8_cols[2]].values,
        "rf_param_4":      X[top8_cols[3]].values,
        "rf_param_5":      X[top8_cols[4]].values,
        "rf_param_6":      X[top8_cols[5]].values,
        "rf_param_7":      X[top8_cols[6]].values,
        "rf_param_8":      X[top8_cols[7]].values,
        "failure_label":   y.values.astype(int),
        "failure_type":    pd.Series(y.values).map({1: "manufacturing_defect", 0: "pass"}).values,
        "dataset_source":  "secom",
        "rul":             np.nan,
    })

    fail_rate = mapped["failure_label"].mean() * 100
    logger.info(
        f"[SchemaMapper | SECOM] "
        f"rows={len(mapped):,} | "
        f"failure_rate={fail_rate:.2f}%"
    )
    return mapped[UNIFIED_SCHEMA_COLS]


# ── Function 3: map_ai4i_to_rf_schema ────────────────────────────────────────

def map_ai4i_to_rf_schema(ai4i_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Map the AI4I 2020 predictive maintenance dataset to the unified RF schema.

    Mapping rationale
    -----------------
        rf_param_1 = Air temperature [K]
                     Operating temperature is the dominant S21 degradation
                     driver — thermal expansion detunes matching networks and
                     raises substrate loss.

        rf_param_2 = Process temperature [K]
                     Elevated process temperature degrades transistor noise
                     performance directly, mirroring semiconductor junction
                     temperature effects on noise figure.

        rf_param_3 = Rotational speed [rpm]
                     Speed determines power delivery capacity, analogous to
                     bias current setting the P1dB compression point.

        rf_param_4 = Torque [Nm]
                     Torque represents mechanical stress loading, which in RF
                     corresponds to drive level pushing the device into
                     compression and reducing IP3.

        rf_param_5 = Air temperature [K]
                     Repeated as the primary ambient temperature input — RF
                     thermal models weight ambient temperature most heavily.

        rf_param_6 = Rotational speed [rpm]
                     Speed as frequency proxy — both define the operating
                     regime and scale of dynamic stresses.

        rf_param_7 = Tool wear [min]
                     Accumulated wear is the clearest single-number
                     degradation indicator; maps to bias drift / aging
                     in RF components.

        rf_param_8 = Type_encoded (L=0, M=1, H=2)
                     Product quality grade; maps to RF component tolerance
                     grade (commercial / industrial / military spec).

    Parameters
    ----------
    ai4i_data : dict
        Output of loaders.load_ai4i(). Must contain keys: X, y_binary, df_raw.

    Returns
    -------
    pd.DataFrame
        Rows in the unified schema column order.
    """
    df_raw: pd.DataFrame = ai4i_data["df_raw"]
    X: pd.DataFrame      = ai4i_data["X"]
    y_binary: pd.Series  = ai4i_data["y_binary"]

    mapped = pd.DataFrame({
        "device_id":       [f"ai4i_{i}" for i in df_raw.index],
        "cycle_or_sample": df_raw.index,
        "rf_param_1":      df_raw["Air temperature [K]"].values,      # operating temp → S21 proxy
        "rf_param_2":      df_raw["Process temperature [K]"].values,  # process temp → noise figure
        "rf_param_3":      df_raw["Rotational speed [rpm]"].values,   # speed → P1dB proxy
        "rf_param_4":      df_raw["Torque [Nm]"].values,              # torque → IP3/linearity proxy
        "rf_param_5":      df_raw["Air temperature [K]"].values,      # ambient temperature
        "rf_param_6":      df_raw["Rotational speed [rpm]"].values,   # speed → frequency proxy
        "rf_param_7":      df_raw["Tool wear [min]"].values,          # wear → degradation/bias proxy
        "rf_param_8":      df_raw["Type_encoded"].values,             # quality grade → spec proxy
        "failure_label":   y_binary.values.astype(int),
        "failure_type":    df_raw["failure_type"].values,
        "dataset_source":  "ai4i",
        "rul":             np.nan,
    })

    fail_rate  = mapped["failure_label"].mean() * 100
    type_dist  = mapped["failure_type"].value_counts().to_dict()

    logger.info(
        f"[SchemaMapper | AI4I] "
        f"rows={len(mapped):,} | "
        f"failure_rate={fail_rate:.2f}% | "
        f"types={type_dist}"
    )
    return mapped[UNIFIED_SCHEMA_COLS]


# ── Function 4: build_unified_dataset ────────────────────────────────────────

def build_unified_dataset(
    cmapss_all: Dict[str, Dict[str, Any]],
    secom_data: Dict[str, Any],
    ai4i_data: Dict[str, Any],
) -> pd.DataFrame:
    """
    Combine all three datasets into a single unified RF-schema DataFrame.

    Only C-MAPSS FD001 is used because it is the cleanest baseline:
    single operating condition, single fault mode — no condition-normalisation
    required before merging with the other two sources. Including FD002–FD004
    without normalisation would bias the combined dataset toward multi-regime
    operating signatures.

    Steps
    -----
    1. Map CMAPSS FD001 train_raw via map_cmapss_to_rf_schema.
    2. Map SECOM via map_secom_to_rf_schema.
    3. Map AI4I via map_ai4i_to_rf_schema.
    4. Concatenate all three; reset index.
    5. Add sequential row_id column.
    6. Verify all schema columns are present.
    7. Report NaN counts per column.

    Parameters
    ----------
    cmapss_all : dict
        Output of loaders.load_all_cmapss().
    secom_data : dict
        Output of loaders.load_secom().
    ai4i_data : dict
        Output of loaders.load_ai4i().

    Returns
    -------
    pd.DataFrame
        Unified DataFrame with UNIFIED_SCHEMA_COLS + row_id.
    """
    logger.info("[SchemaMapper] Building unified RF dataset...")

    # Step 1: CMAPSS FD001
    cmapss_fd001_raw = cmapss_all["FD001"]["train_raw"]
    df_cmapss = map_cmapss_to_rf_schema(cmapss_fd001_raw, "FD001")

    # Step 2: SECOM
    df_secom = map_secom_to_rf_schema(secom_data)

    # Step 3: AI4I
    df_ai4i = map_ai4i_to_rf_schema(ai4i_data)

    n_cmapss = len(df_cmapss)
    n_secom  = len(df_secom)
    n_ai4i   = len(df_ai4i)

    # Step 4: Concatenate
    unified = pd.concat([df_cmapss, df_secom, df_ai4i], ignore_index=True)

    # Step 5: Sequential row_id
    unified.insert(0, "row_id", range(len(unified)))

    # Step 6: Verify schema completeness
    missing_cols = [c for c in UNIFIED_SCHEMA_COLS if c not in unified.columns]
    if missing_cols:
        logger.warning(f"[SchemaMapper] Missing schema columns: {missing_cols}")
    else:
        logger.info("[SchemaMapper] Schema verification: all columns present.")

    # Step 7: NaN report
    nan_report = unified[UNIFIED_SCHEMA_COLS].isna().sum()
    nan_cols   = nan_report[nan_report > 0]
    if not nan_cols.empty:
        nan_summary = " | ".join(f"{c}={v}" for c, v in nan_cols.items())
        logger.info(f"[SchemaMapper] NaN counts: {nan_summary}")

    # Summary stats
    overall_fail_rate = unified["failure_label"].mean() * 100
    type_dist = unified["failure_type"].value_counts().to_dict()

    logger.success(
        f"[SchemaMapper] Unified dataset ready — "
        f"cmapss={n_cmapss:,} | secom={n_secom:,} | ai4i={n_ai4i:,} | "
        f"total={len(unified):,} | "
        f"overall_failure_rate={overall_fail_rate:.2f}%"
    )
    logger.info(f"[SchemaMapper] Failure type distribution: {type_dist}")

    return unified


# ── Function 5: validate_schema ──────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that a DataFrame conforms to the unified RF-Sentinel schema.

    Checks performed
    ----------------
    1. All UNIFIED_SCHEMA_COLS are present.
    2. failure_label contains only values in {0, 1}.
    3. No NaN in failure_label or dataset_source (critical label columns).
    4. rf_param_1 through rf_param_8 are all numeric dtype.
    5. NaN count reported per rf_param column (NaN is acceptable, just audited).

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame that should conform to the unified schema.

    Returns
    -------
    dict with keys:
        passed  bool          True only if all hard checks pass
        issues  list[str]     descriptions of any failed checks
    """
    issues: List[str] = []
    rf_params = [f"rf_param_{i}" for i in range(1, 9)]

    # Check 1: required columns present
    missing_cols = [c for c in UNIFIED_SCHEMA_COLS if c not in df.columns]
    if missing_cols:
        msg = f"Missing columns: {missing_cols}"
        issues.append(msg)
        logger.error(f"[Validate] FAIL — {msg}")
    else:
        logger.info("[Validate] PASS — all required columns present")

    # Check 2: failure_label values are only 0 or 1
    if "failure_label" in df.columns:
        invalid_labels = df["failure_label"].dropna()
        invalid_vals   = set(invalid_labels.unique()) - {0, 1}
        if invalid_vals:
            msg = f"failure_label contains unexpected values: {invalid_vals}"
            issues.append(msg)
            logger.error(f"[Validate] FAIL — {msg}")
        else:
            logger.info("[Validate] PASS — failure_label contains only 0 and 1")

    # Check 3: no NaN in failure_label or dataset_source
    for critical_col in ["failure_label", "dataset_source"]:
        if critical_col in df.columns:
            n_nan = df[critical_col].isna().sum()
            if n_nan > 0:
                msg = f"{critical_col} has {n_nan} NaN values"
                issues.append(msg)
                logger.error(f"[Validate] FAIL — {msg}")
            else:
                logger.info(f"[Validate] PASS — no NaN in {critical_col}")

    # Check 4: rf_param columns are numeric
    non_numeric = []
    for col in rf_params:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        msg = f"Non-numeric rf_param columns: {non_numeric}"
        issues.append(msg)
        logger.error(f"[Validate] FAIL — {msg}")
    else:
        present_params = [c for c in rf_params if c in df.columns]
        logger.info(
            f"[Validate] PASS — all {len(present_params)} rf_param columns are numeric"
        )

    # Check 5: NaN audit per rf_param (informational, not a hard failure)
    if all(c in df.columns for c in rf_params):
        nan_counts = df[rf_params].isna().sum()
        nan_present = nan_counts[nan_counts > 0]
        if nan_present.empty:
            logger.info("[Validate] INFO — no NaN in any rf_param column")
        else:
            for col, count in nan_present.items():
                pct = count / len(df) * 100
                logger.info(
                    f"[Validate] INFO — {col}: {count} NaN ({pct:.1f}%)"
                )

    passed = len(issues) == 0
    status = "PASSED" if passed else f"FAILED ({len(issues)} issues)"
    logger.info(f"[Validate] Schema validation {status}")

    return {"passed": passed, "issues": issues}
