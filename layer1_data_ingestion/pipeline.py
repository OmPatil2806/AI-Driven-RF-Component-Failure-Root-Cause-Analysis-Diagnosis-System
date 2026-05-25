"""
pipeline.py — Master orchestration script for RF-Sentinel Layer 1.

Responsibilities
----------------
Calls every Layer 1 module in the correct order:
    loaders         → load raw datasets from disk
    feature_engineering → enrich C-MAPSS with time-series features
    preprocessor    → scale, impute, SMOTE, PCA
    schema_mapper   → map all three sources to a unified RF schema
    save outputs    → write parquet files + JSON summary to data/processed/

Usage
-----
Run directly (recommended):
    python -m src.data.layer1_data_ingestion.pipeline

Dry-run (no files written):
    python -m src.data.layer1_data_ingestion.pipeline --no-save

Print last run report:
    python -m src.data.layer1_data_ingestion.pipeline --report

Import in other modules:
    from layer1_data_ingestion.pipeline import run_layer1_pipeline
    results = run_layer1_pipeline(save_outputs=True)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from layer1_data_ingestion.config import (
    AI4I_PROCESSED,
    CMAPSS_DATASETS,
    CMAPSS_PROCESSED,
    LAYER1_SUMMARY,
    PROCESSED_DIR,
    SECOM_PROCESSED,
    UNIFIED_DATASET,
)
from layer1_data_ingestion.feature_engineering import (
    apply_pca_secom,
    engineer_cmapss_features,
    get_feature_summary,
)
from layer1_data_ingestion.loaders import load_all_datasets
from layer1_data_ingestion.preprocessor import (
    preprocess_ai4i,
    preprocess_cmapss,
    preprocess_secom,
)
from layer1_data_ingestion.schema_mapper import (
    build_unified_dataset,
    validate_schema,
)


# ── Function 1: run_layer1_pipeline ──────────────────────────────────────────

def run_layer1_pipeline(save_outputs: bool = True) -> Dict[str, Any]:
    """
    Execute all six Layer 1 steps in order and return a result dict.

    Steps
    -----
    1. Load all raw datasets via loaders.load_all_datasets().
    2. Apply feature engineering to every C-MAPSS subset.
    3. Preprocess all datasets (scale, impute, SMOTE splits).
    4. Apply PCA to SECOM.
    5. Build and validate the unified RF-schema dataset.
    6. Save processed parquet files and a JSON summary report.

    Parameters
    ----------
    save_outputs : bool
        Write parquet files and JSON summary to data/processed/ (default True).
        Set False for a dry run that validates the pipeline without writing.

    Returns
    -------
    dict with keys:
        raw_data, engineered_cmapss, preprocessed, pca_result,
        unified_df, summary
    """
    pipeline_start = time.time()

    # ── STEP 1 — Load raw datasets ────────────────────────────────────────────
    _banner("STEP 1/6 — Loading raw datasets")
    step_start = time.time()

    raw_data = load_all_datasets()
    cmapss_all: Dict[str, Any] = raw_data["cmapss"]
    secom_data: Dict[str, Any] = raw_data["secom"]
    ai4i_data:  Dict[str, Any] = raw_data["ai4i"]

    logger.info(f"STEP 1 done in {time.time() - step_start:.1f}s")

    # ── STEP 2 — Feature engineering on C-MAPSS ───────────────────────────────
    _banner("STEP 2/6 — Feature engineering on C-MAPSS")
    step_start = time.time()

    engineered_cmapss: Dict[str, Any] = {}
    total_new_features = 0

    _exclude = {"unit_id", "cycle", "RUL", "fail_soon", "op1", "op2", "op3"}

    for ds in CMAPSS_DATASETS:
        train_raw = cmapss_all[ds]["train_raw"]
        df_eng, feat_cols = engineer_cmapss_features(train_raw)
        summary = get_feature_summary(train_raw, df_eng)

        new_features = summary["engineered_features"]
        total_new_features += new_features

        # Columns that go into the model (no metadata or labels)
        eng_feat_cols = [c for c in feat_cols if c not in _exclude]

        # Update the dict so preprocessor and schema_mapper see enriched data
        cmapss_all[ds]["train_raw_engineered"] = df_eng
        cmapss_all[ds]["feature_cols"]         = feat_cols
        cmapss_all[ds]["X_train"]              = df_eng[eng_feat_cols].copy()

        # Engineer features on the full test time series, then re-extract the
        # last cycle per engine so X_test has the same columns as X_train.
        if cmapss_all[ds].get("test_raw") is not None:
            test_eng, _ = engineer_cmapss_features(cmapss_all[ds]["test_raw"])
            last_test = (
                test_eng.sort_values("cycle")
                .groupby("unit_id")
                .tail(1)
                .reset_index(drop=True)
            )
            cmapss_all[ds]["X_test"] = last_test[eng_feat_cols].copy()

        engineered_cmapss[ds] = {
            "df_engineered": df_eng,
            "feature_cols":  feat_cols,
            "new_features":  new_features,
        }

    logger.success(
        f"STEP 2 done in {time.time() - step_start:.1f}s | "
        f"total new features across all subsets: {total_new_features}"
    )

    # ── STEP 3 — Preprocess all datasets ─────────────────────────────────────
    _banner("STEP 3/6 — Preprocessing all datasets")
    step_start = time.time()

    pp_cmapss = preprocess_cmapss(cmapss_all["FD001"])
    logger.info(
        f"  C-MAPSS/FD001 — "
        f"train={pp_cmapss['X_train'].shape} | "
        f"val={pp_cmapss['X_val'].shape}"
    )

    pp_secom = preprocess_secom(secom_data)
    logger.info(
        f"  SECOM — "
        f"train={pp_secom['X_train'].shape} | "
        f"val={pp_secom['X_val'].shape}"
    )

    pp_ai4i_bin = preprocess_ai4i(ai4i_data, target="binary")
    logger.info(
        f"  AI4I/binary — "
        f"train={pp_ai4i_bin['X_train'].shape} | "
        f"val={pp_ai4i_bin['X_val'].shape}"
    )

    pp_ai4i_multi = preprocess_ai4i(ai4i_data, target="multiclass")
    logger.info(
        f"  AI4I/multiclass — "
        f"train={pp_ai4i_multi['X_train'].shape} | "
        f"val={pp_ai4i_multi['X_val'].shape}"
    )

    logger.success(f"STEP 3 done in {time.time() - step_start:.1f}s")

    # ── STEP 4 — PCA on SECOM ────────────────────────────────────────────────
    _banner("STEP 4/6 — Applying PCA to SECOM")
    step_start = time.time()

    X_pca, pca_obj, scaler_obj = apply_pca_secom(secom_data["X"])

    n_components_kept   = X_pca.shape[1]
    variance_explained  = float(pca_obj.explained_variance_ratio_.sum() * 100)

    logger.success(
        f"STEP 4 done in {time.time() - step_start:.1f}s | "
        f"components={n_components_kept} | "
        f"variance_explained={variance_explained:.2f}%"
    )

    pca_result = {
        "X_pca":              X_pca,
        "pca":                pca_obj,
        "scaler":             scaler_obj,
        "n_components":       n_components_kept,
        "variance_explained": variance_explained,
    }

    # ── STEP 5 — Build and validate unified schema ────────────────────────────
    _banner("STEP 5/6 — Building and validating unified schema")
    step_start = time.time()

    unified_df = build_unified_dataset(cmapss_all, secom_data, ai4i_data)
    validation = validate_schema(unified_df)

    if not validation["passed"]:
        for issue in validation["issues"]:
            logger.error(f"  Schema issue: {issue}")
        raise ValueError(
            f"Unified schema validation failed with "
            f"{len(validation['issues'])} issue(s). See logs above."
        )

    logger.success(
        f"STEP 5 done in {time.time() - step_start:.1f}s | "
        f"unified_rows={len(unified_df):,} | schema=VALID"
    )

    # ── STEP 6 — Save outputs ─────────────────────────────────────────────────
    _banner("STEP 6/6 — Saving outputs to data/processed/")
    step_start = time.time()

    total_runtime = time.time() - pipeline_start
    saved_files: List[str] = []

    # Collect stats for the summary report
    # CMAPSS stats
    cmapss_total_rows    = sum(len(cmapss_all[ds]["train_raw"]) for ds in CMAPSS_DATASETS)
    cmapss_total_engines = sum(cmapss_all[ds]["n_engines_train"] for ds in CMAPSS_DATASETS)
    cmapss_fail_rate     = float(
        pd.concat([cmapss_all[ds]["y_train"] for ds in CMAPSS_DATASETS])
        .mean() * 100
    )
    cmapss_feat_count    = len(cmapss_all["FD001"]["feature_cols"])

    # SECOM stats
    secom_fail_rate = float(secom_data["y"].mean() * 100)
    secom_orig_feat = secom_data["X"].shape[1]

    # AI4I stats
    ai4i_fail_rate   = float(ai4i_data["y_binary"].mean() * 100)
    ai4i_type_counts = ai4i_data["failure_type_counts"].to_dict()

    # Unified stats
    unified_fail_rate  = float(unified_df["failure_label"].mean() * 100)
    unified_type_dist  = unified_df["failure_type"].value_counts().to_dict()

    summary: Dict[str, Any] = {
        "run_timestamp":         datetime.now().isoformat(),
        "total_runtime_seconds": round(total_runtime, 2),
        "datasets": {
            "cmapss": {
                "total_rows":                cmapss_total_rows,
                "total_engines":             cmapss_total_engines,
                "datasets_loaded":           len(CMAPSS_DATASETS),
                "failure_rate_pct":          round(cmapss_fail_rate, 2),
                "features_after_engineering": cmapss_feat_count,
            },
            "secom": {
                "samples":               secom_data["X"].shape[0],
                "original_features":     secom_orig_feat,
                "features_after_pca":    n_components_kept,
                "variance_explained_pct": round(variance_explained, 2),
                "failure_rate_pct":      round(secom_fail_rate, 2),
            },
            "ai4i": {
                "samples":             len(ai4i_data["y_binary"]),
                "features":            ai4i_data["X"].shape[1],
                "failure_rate_pct":    round(ai4i_fail_rate, 2),
                "failure_type_counts": {k: int(v) for k, v in ai4i_type_counts.items()},
            },
        },
        "unified": {
            "total_rows":               len(unified_df),
            "sources":                  unified_df["dataset_source"].unique().tolist(),
            "failure_rate_pct":         round(unified_fail_rate, 2),
            "failure_type_distribution": {k: int(v) for k, v in unified_type_dist.items()},
            "schema_valid":             validation["passed"],
        },
        "output_files": [],
    }

    if save_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # 1. CMAPSS unified — all 4 subsets combined with engineered features
        cmapss_frames = []
        for ds in CMAPSS_DATASETS:
            frame = cmapss_all[ds]["train_raw_engineered"].copy()
            frame["dataset"] = ds
            cmapss_frames.append(frame)
        cmapss_unified_df = pd.concat(cmapss_frames, ignore_index=True)
        cmapss_unified_df.to_parquet(CMAPSS_PROCESSED, index=False)
        saved_files.append(str(CMAPSS_PROCESSED))
        _log_saved(CMAPSS_PROCESSED)

        # 2. SECOM clean — PCA-reduced features + label
        secom_pca_df = X_pca.copy()
        secom_pca_df["label"] = secom_data["y"].values
        secom_pca_df.to_parquet(SECOM_PROCESSED, index=False)
        saved_files.append(str(SECOM_PROCESSED))
        _log_saved(SECOM_PROCESSED)

        # 3. AI4I clean — full df_raw with failure_type and Type_encoded
        ai4i_data["df_raw"].to_parquet(AI4I_PROCESSED, index=False)
        saved_files.append(str(AI4I_PROCESSED))
        _log_saved(AI4I_PROCESSED)

        # 4. Unified RF schema dataset
        unified_df.to_parquet(UNIFIED_DATASET, index=False)
        saved_files.append(str(UNIFIED_DATASET))
        _log_saved(UNIFIED_DATASET)

        # 5. Layer 1 summary JSON
        summary["output_files"] = saved_files
        with open(LAYER1_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        saved_files.append(str(LAYER1_SUMMARY))
        _log_saved(LAYER1_SUMMARY)

        logger.success(
            f"STEP 6 done in {time.time() - step_start:.1f}s | "
            f"{len(saved_files)} files written"
        )
    else:
        logger.info("STEP 6 skipped (save_outputs=False — dry run)")

    _banner(f"Layer 1 complete in {total_runtime:.1f}s")

    return {
        "raw_data":          raw_data,
        "engineered_cmapss": engineered_cmapss,
        "preprocessed": {
            "cmapss":       pp_cmapss,
            "secom":        pp_secom,
            "ai4i_binary":  pp_ai4i_bin,
            "ai4i_multi":   pp_ai4i_multi,
        },
        "pca_result":  pca_result,
        "unified_df":  unified_df,
        "summary":     summary,
    }


# ── Function 2: load_processed_data ──────────────────────────────────────────

def load_processed_data() -> Dict[str, pd.DataFrame]:
    """
    Load all processed parquet files written by run_layer1_pipeline.

    Checks each file exists before loading and logs a warning for any that
    are missing (run the pipeline first to generate them).

    Returns
    -------
    dict with keys:
        cmapss_unified  pd.DataFrame  all four C-MAPSS subsets, engineered
        secom_clean     pd.DataFrame  PCA-reduced SECOM features + label
        ai4i_clean      pd.DataFrame  full AI4I frame with failure_type
        unified         pd.DataFrame  RF-schema unified dataset
    """
    file_map = {
        "cmapss_unified": CMAPSS_PROCESSED,
        "secom_clean":    SECOM_PROCESSED,
        "ai4i_clean":     AI4I_PROCESSED,
        "unified":        UNIFIED_DATASET,
    }

    result: Dict[str, pd.DataFrame] = {}

    for key, path in file_map.items():
        if not path.exists():
            logger.warning(
                f"[load_processed] '{path.name}' not found — "
                f"run run_layer1_pipeline() first."
            )
            result[key] = None
            continue

        df = pd.read_parquet(path)
        result[key] = df
        logger.info(
            f"[load_processed] Loaded {path.name}: "
            f"{df.shape[0]:,} rows × {df.shape[1]} cols"
        )

    return result


# ── Function 3: print_layer1_report ──────────────────────────────────────────

def print_layer1_report() -> None:
    """
    Read layer1_summary.json and print a formatted report using rich.

    Displays dataset-level stats (rows, features, failure rate), the unified
    dataset summary, and pipeline runtime. Prompts to run the pipeline first
    if the JSON file does not exist.
    """
    if not LAYER1_SUMMARY.exists():
        print(
            "\n  No Layer 1 report found.\n"
            "  Run the pipeline first:\n"
            "    python -m src.data.layer1_data_ingestion.pipeline\n"
        )
        return

    with open(LAYER1_SUMMARY, "r", encoding="utf-8") as f:
        report = json.load(f)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()

        console.print(
            f"\n[bold cyan]RF-Sentinel — Layer 1 Report[/bold cyan]  "
            f"[dim]{report['run_timestamp']}[/dim]"
        )
        console.print(
            f"[dim]Total runtime: {report['total_runtime_seconds']:.1f}s[/dim]\n"
        )

        # ── Dataset stats table ───────────────────────────────────────────────
        tbl = Table(
            title="Dataset Statistics",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        tbl.add_column("Dataset",         style="cyan",  min_width=14)
        tbl.add_column("Rows",            justify="right")
        tbl.add_column("Features",        justify="right")
        tbl.add_column("Failure Rate %",  justify="right", style="yellow")
        tbl.add_column("Notes",           style="dim")

        ds = report["datasets"]

        tbl.add_row(
            "C-MAPSS (all)",
            f"{ds['cmapss']['total_rows']:,}",
            str(ds['cmapss']['features_after_engineering']),
            f"{ds['cmapss']['failure_rate_pct']:.2f}",
            f"{ds['cmapss']['total_engines']} engines, "
            f"{ds['cmapss']['datasets_loaded']} subsets",
        )
        tbl.add_row(
            "SECOM",
            f"{ds['secom']['samples']:,}",
            f"{ds['secom']['original_features']} → "
            f"{ds['secom']['features_after_pca']} PCA",
            f"{ds['secom']['failure_rate_pct']:.2f}",
            f"{ds['secom']['variance_explained_pct']:.1f}% variance kept",
        )
        tbl.add_row(
            "AI4I 2020",
            f"{ds['ai4i']['samples']:,}",
            str(ds['ai4i']['features']),
            f"{ds['ai4i']['failure_rate_pct']:.2f}",
            f"{len(ds['ai4i']['failure_type_counts'])} failure types",
        )
        console.print(tbl)

        # ── Unified dataset table ─────────────────────────────────────────────
        utbl = Table(
            title="Unified RF Schema Dataset",
            box=box.SIMPLE_HEAVY,
            header_style="bold green",
        )
        utbl.add_column("Metric",  style="cyan")
        utbl.add_column("Value",   justify="right")

        uni = report["unified"]
        utbl.add_row("Total rows",      f"{uni['total_rows']:,}")
        utbl.add_row("Sources",         ", ".join(uni["sources"]))
        utbl.add_row("Failure rate",    f"{uni['failure_rate_pct']:.2f}%")
        utbl.add_row("Schema valid",    "✓ YES" if uni["schema_valid"] else "✗ NO")
        console.print(utbl)

        # ── Failure type breakdown ────────────────────────────────────────────
        ftbl = Table(
            title="Failure Type Distribution (Unified)",
            box=box.SIMPLE,
            header_style="bold yellow",
        )
        ftbl.add_column("Failure Type", style="cyan")
        ftbl.add_column("Count",        justify="right")

        for ftype, count in sorted(
            uni["failure_type_distribution"].items(),
            key=lambda x: -x[1],
        ):
            ftbl.add_row(ftype, str(count))
        console.print(ftbl)

        # ── Output files ──────────────────────────────────────────────────────
        if report.get("output_files"):
            console.print("\n[bold]Output files:[/bold]")
            for fpath in report["output_files"]:
                p = Path(fpath)
                size_kb = p.stat().st_size / 1024 if p.exists() else 0
                console.print(f"  [green]✓[/green] {p.name:<40} {size_kb:,.1f} KB")
        console.print()

    except ImportError:
        # Fallback: plain-text output if rich is not installed
        _plain_report(report)


# ── Private helpers ───────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    """Print a step separator to the log."""
    sep = "─" * 60
    logger.info(sep)
    logger.info(f"  {msg}")
    logger.info(sep)


def _log_saved(path: Path) -> None:
    """Log a saved file with its size in KB."""
    size_kb = path.stat().st_size / 1024
    logger.info(f"  Saved: {path.name:<45} {size_kb:,.1f} KB")


def _plain_report(report: Dict[str, Any]) -> None:
    """Fallback plain-text report when rich is unavailable."""
    print("\nRF-Sentinel — Layer 1 Report")
    print(f"Timestamp  : {report['run_timestamp']}")
    print(f"Runtime    : {report['total_runtime_seconds']:.1f}s")
    print()
    for ds_name, ds_info in report["datasets"].items():
        print(f"[{ds_name.upper()}]")
        for k, v in ds_info.items():
            print(f"  {k:<35}: {v}")
        print()
    print("[UNIFIED]")
    for k, v in report["unified"].items():
        print(f"  {k:<35}: {v}")
    print()


# ── Main block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF-Sentinel Layer 1 data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.data.layer1_data_ingestion.pipeline\n"
            "  python -m src.data.layer1_data_ingestion.pipeline --no-save\n"
            "  python -m src.data.layer1_data_ingestion.pipeline --report\n"
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry run: execute the full pipeline without writing any files.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the report from the last successful run and exit.",
    )
    args = parser.parse_args()

    if args.report:
        print_layer1_report()
        sys.exit(0)

    run_layer1_pipeline(save_outputs=not args.no_save)

    print(
        "\nLayer 1 complete. "
        "Run next:\n"
        "  python -m src.data.layer1_data_ingestion.pipeline --report"
    )
