"""
eda_runner.py — Master EDA runner for RF-Sentinel Layer 2.

Runs all 4 EDA modules in sequence and saves a summary report
to outputs/eda/eda_summary.json.

Usage:
    python -m src.data.layer2_eda.eda_runner

Or import:
    from layer2_eda.eda_runner import run_all_eda
    run_all_eda()
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from layer1_data_ingestion.config import ROOT_DIR
from layer2_eda.eda_ai4i import run_ai4i_eda
from layer2_eda.eda_cmapss import run_cmapss_eda
from layer2_eda.eda_secom import run_secom_eda
from layer2_eda.eda_unified import run_unified_eda

EDA_OUTPUT_DIR = ROOT_DIR / "outputs" / "eda"
SUMMARY_PATH   = EDA_OUTPUT_DIR / "eda_summary.json"


def run_all_eda() -> Dict[str, Any]:
    """
    Execute all four Layer 2 EDA modules in sequence.

    Runs CMAPSS → SECOM → AI4I → Unified EDA, collects per-module
    stats (plots saved, runtime, file paths, status), writes a JSON
    summary, and returns a consolidated result dict.

    Returns
    -------
    dict with keys:
        total_plots   int          total PNG files saved across all modules
        total_runtime float        wall-clock seconds for the full run
        summary_path  str          absolute path to eda_summary.json
        all_files     list[str]    flat list of every saved PNG path
    """
    SEP = "=" * 50

    # ── STEP 1: Header ────────────────────────────────────────────────────────
    logger.info(SEP)
    logger.info("RF-Sentinel — Layer 2 EDA Runner Starting")
    logger.info(SEP)
    run_start = time.time()

    # ── STEP 2: Run each module ───────────────────────────────────────────────
    modules = [
        ("cmapss",   "CMAPSS EDA",   lambda: run_cmapss_eda(datasets=["FD001", "FD002", "FD003", "FD004"])),
        ("secom",    "SECOM EDA",    run_secom_eda),
        ("ai4i",     "AI4I EDA",     run_ai4i_eda),
        ("unified",  "Unified EDA",  run_unified_eda),
    ]

    module_results: Dict[str, Dict[str, Any]] = {}

    for idx, (key, name, func) in enumerate(modules, start=1):
        logger.info(f"Running module {idx}/4: {name}...")
        mod_start = time.time()
        status    = "success"
        files: List[str] = []

        try:
            result = func()
            # CMAPSS returns a dict; others return a list
            if isinstance(result, dict):
                for v in result.values():
                    files.extend(v if isinstance(v, list) else [])
            elif isinstance(result, list):
                files = result
        except Exception as exc:
            logger.error(f"  {name} failed: {exc}")
            status = "failed"

        runtime = time.time() - mod_start
        module_results[key] = {
            "plots_saved":      len(files),
            "runtime_seconds":  round(runtime, 2),
            "files":            files,
            "status":           status,
        }
        logger.success(
            f"{name} complete — {len(files)} plots saved in {runtime:.1f}s"
        )

    total_runtime = time.time() - run_start
    total_plots   = sum(m["plots_saved"] for m in module_results.values())
    all_files     = [f for m in module_results.values() for f in m["files"]]

    # ── STEP 3: Save summary JSON ─────────────────────────────────────────────
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_timestamp":        datetime.now().isoformat(),
        "total_runtime_seconds": round(total_runtime, 2),
        "total_plots_saved":    total_plots,
        "modules":              module_results,
        "output_folders": {
            "cmapss":  "outputs/eda/cmapss/",
            "secom":   "outputs/eda/secom/",
            "ai4i":    "outputs/eda/ai4i/",
            "unified": "outputs/eda/unified/",
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── STEP 4: Final summary ─────────────────────────────────────────────────
    logger.info(SEP)
    logger.info("LAYER 2 EDA COMPLETE")
    logger.info(SEP)
    logger.info(f"Total runtime    : {total_runtime:.1f}s")
    logger.info(f"Total plots saved: {total_plots}")
    logger.info("Breakdown:")
    for key, name, _ in modules:
        m = module_results[key]
        logger.info(
            f"  {name:<14}: {m['plots_saved']} plots in {m['runtime_seconds']:.1f}s"
            + ("  [FAILED]" if m["status"] == "failed" else "")
        )
    logger.info("Summary saved  : outputs/eda/eda_summary.json")
    logger.info("All figures in : outputs/eda/")
    logger.info(SEP)

    # ── STEP 5: Return ────────────────────────────────────────────────────────
    return {
        "total_plots":   total_plots,
        "total_runtime": round(total_runtime, 2),
        "summary_path":  str(SUMMARY_PATH),
        "all_files":     all_files,
    }


if __name__ == "__main__":
    results = run_all_eda()
    print(f"\nDone. {results['total_plots']} plots saved.")
    print(f"Summary: {results['summary_path']}")
