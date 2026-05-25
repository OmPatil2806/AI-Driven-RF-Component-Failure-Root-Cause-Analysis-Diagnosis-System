"""
config.py — Single source of truth for all paths, constants, and pipeline
settings in RF-Sentinel. Every other module imports from here; nothing is
hard-coded elsewhere.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
# config.py lives at rf_sentinel/layer1_data_ingestion/config.py
# so 2 .parent calls bring us to rf_sentinel/
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Ensure output directories exist on import
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

# ── C-MAPSS file paths ────────────────────────────────────────────────────────
CMAPSS_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

CMAPSS_TRAIN_FILES = {ds: RAW_DIR / f"train_{ds}.txt" for ds in CMAPSS_DATASETS}
CMAPSS_TEST_FILES  = {ds: RAW_DIR / f"test_{ds}.txt"  for ds in ["FD001", "FD004"]}
CMAPSS_RUL_FILES   = {ds: RAW_DIR / f"RUL_{ds}.txt"   for ds in ["FD001", "FD004"]}

# ── SECOM file paths ──────────────────────────────────────────────────────────
SECOM_DATA_FILE   = RAW_DIR / "secom.data"
SECOM_LABELS_FILE = RAW_DIR / "secom_labels.data"
SECOM_NAMES_FILE  = RAW_DIR / "secom.names"

# ── AI4I file path ────────────────────────────────────────────────────────────
AI4I_FILE = RAW_DIR / "ai4i2020.csv"

# ── Processed output paths ────────────────────────────────────────────────────
CMAPSS_PROCESSED = PROCESSED_DIR / "cmapss_unified.parquet"
SECOM_PROCESSED  = PROCESSED_DIR / "secom_clean.parquet"
AI4I_PROCESSED   = PROCESSED_DIR / "ai4i_clean.parquet"
UNIFIED_DATASET  = PROCESSED_DIR / "rf_sentinel_unified.parquet"
LAYER1_SUMMARY   = PROCESSED_DIR / "layer1_summary.json"

# ── C-MAPSS column definitions ────────────────────────────────────────────────
# 26 columns: unit_id, cycle, 3 operational settings, 21 sensor readings
CMAPSS_COLS = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensors with near-zero variance across all operating conditions — dropped
CMAPSS_DROP_SENSORS = ["s1", "s5", "s10", "s16", "s18", "s19"]

# Sensors that carry meaningful degradation signal
CMAPSS_USEFUL_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

# Human-readable physical meaning for each useful sensor
CMAPSS_SENSOR_LABELS = {
    "s2":  "Fan inlet temperature (°R)",
    "s3":  "LPC outlet temperature (°R)",
    "s4":  "HPC outlet temperature (°R)",
    "s7":  "HPC outlet pressure (psia)",
    "s8":  "Physical fan speed (rpm)",
    "s9":  "Physical core speed (rpm)",
    "s11": "HPC outlet static pressure (psia)",
    "s12": "Ratio of fuel flow to Ps30 (pps/psia)",
    "s13": "Corrected fan speed (rpm)",
    "s14": "Corrected core speed (rpm)",
    "s15": "Bypass ratio",
    "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed (lbm/s)",
    "s21": "LPT coolant bleed (lbm/s)",
}

# Cycles remaining below this threshold are labelled as degraded / near-failure
RUL_THRESHOLD = 30

# Fault mode and operating condition metadata per subset
CMAPSS_FAULT_MODES = {
    "FD001": {"fault_modes": 1, "operating_conditions": 1, "faults": "HPC degradation"},
    "FD002": {"fault_modes": 1, "operating_conditions": 6, "faults": "HPC degradation"},
    "FD003": {"fault_modes": 2, "operating_conditions": 1, "faults": "HPC + Fan degradation"},
    "FD004": {"fault_modes": 2, "operating_conditions": 6, "faults": "HPC + Fan degradation"},
}

# ── AI4I column definitions ───────────────────────────────────────────────────
AI4I_FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

AI4I_TARGET_COL   = "Machine failure"
AI4I_FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Maps AI4I failure codes to RF-Sentinel canonical failure mode names
AI4I_TO_RF_MAP = {
    "TWF":        "thermal_wear_failure",
    "HDF":        "heat_dissipation_failure",
    "PWF":        "power_failure",
    "OSF":        "overstrain_failure",
    "RNF":        "random_failure",
    "no_failure": "pass",
}

# ── Pipeline settings ─────────────────────────────────────────────────────────
RANDOM_STATE       = 42    # global seed for reproducibility
TEST_SIZE          = 0.20  # 80/20 train-test split
SMOTE_K_NEIGHBORS  = 5     # SMOTE neighbours for minority class oversampling
PCA_N_COMPONENTS   = 0.95  # retain 95 % of variance in SECOM PCA reduction
ROLLING_WINDOW     = 5     # cycle window for rolling-mean sensor smoothing
MAX_FEATURES_SECOM = 50    # max features selected after PCA / importance ranking
