"""
eda_cmapss.py — Exploratory Data Analysis for C-MAPSS turbofan degradation data.

Generates 8 visualisations per dataset subset (32 total across FD001–FD004):

    1. Sensor degradation waves    — how each sensor drifts as failure nears
    2. RUL distribution            — histogram and per-engine spread of remaining life
    3. Sensor correlation heatmap  — inter-sensor collinearity map
    4. Sensor vs RUL scatter       — individual sensor trends against remaining cycles
    5. Operating conditions        — cluster view of throttle / altitude regimes
    6. Degradation index           — composite 0–1 health score over engine lifetime
    7. Failure rate by cycle       — when in the life cycle failures tend to occur
    8. Sensor discriminative power — which sensors best separate healthy vs failing

All plots are saved to outputs/eda/cmapss/.
Insights feed directly into feature selection and model design for RF-Sentinel.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from layer1_data_ingestion.config import (
    CMAPSS_SENSOR_LABELS,
    CMAPSS_USEFUL_SENSORS,
    ROOT_DIR,
)
from layer1_data_ingestion.loaders import load_cmapss

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Output directory ──────────────────────────────────────────────────────────
EDA_DIR = ROOT_DIR / "outputs" / "eda" / "cmapss"
EDA_DIR.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H   = 14, 6
RUL_THRESHOLD  = 30
RANDOM_SEED    = 42

# Short axis labels derived from CMAPSS_SENSOR_LABELS
_SHORT = {s: CMAPSS_SENSOR_LABELS[s].split("(")[0].strip() for s in CMAPSS_USEFUL_SENSORS}


def _save(fig: plt.Figure, fname: str) -> Path:
    """Save figure and close it; return the saved path."""
    path = EDA_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 1: Sensor Degradation Waves ─────────────────────────────────────────

def plot_sensor_degradation_waves(
    df: pd.DataFrame,
    dataset_name: str,
    n_units: int = 5,
) -> Path:
    """
    Show how each of the 14 useful sensor signals evolve as the engine
    approaches failure.

    RF insight: RF component gain, noise figure, and output power all drift
    monotonically before failure — the same pattern we expect to see here.
    Sensors that show a clear late-life trend are the best RUL predictors.
    """
    rng   = np.random.default_rng(RANDOM_SEED)
    units = rng.choice(df["unit_id"].unique(), size=min(n_units, df["unit_id"].nunique()), replace=False)
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(units)))

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for idx, sensor in enumerate(CMAPSS_USEFUL_SENSORS):
        ax = axes[idx]
        for color, unit in zip(palette, units):
            eng = df[df["unit_id"] == unit].sort_values("cycle")
            ax.plot(eng["cycle"], eng[sensor], color=color, alpha=0.8,
                    linewidth=1.2, label=f"Eng {unit}")

            # Failure zone: last RUL_THRESHOLD cycles
            max_cycle = eng["cycle"].max()
            fail_start = max_cycle - RUL_THRESHOLD
            ax.axvline(fail_start, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvspan(fail_start, max_cycle, alpha=0.08, color="red")

        ax.set_title(_SHORT[sensor], fontsize=9, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel(sensor, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplot cells (14 sensors, 16 cells)
    for idx in range(len(CMAPSS_USEFUL_SENSORS), len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=palette[i], label=f"Eng {u}") for i, u in enumerate(units)]
    fig.legend(handles=handles, loc="lower right", ncol=len(units),
               fontsize=9, title="Engines", framealpha=0.9)

    fig.suptitle(f"Sensor Degradation Waves — CMAPSS {dataset_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, f"{dataset_name}_sensor_degradation_waves.png")


# ── Plot 2: RUL Distribution ──────────────────────────────────────────────────

def plot_rul_distribution(df: pd.DataFrame, dataset_name: str) -> Path:
    """
    Show the statistical distribution of Remaining Useful Life across all cycles
    and per engine.

    RF insight: A heavily right-skewed RUL distribution means the model will
    see very few near-failure samples — the core class-imbalance problem that
    SMOTE and RUL_THRESHOLD are designed to address.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    # ── Left: RUL histogram ───────────────────────────────────────────────────
    rul_fail = df.loc[df["RUL"] <= RUL_THRESHOLD, "RUL"]
    rul_safe = df.loc[df["RUL"] >  RUL_THRESHOLD, "RUL"]

    ax1.hist(rul_safe, bins=40, color="steelblue", alpha=0.75, label="Safe (RUL > 30)")
    ax1.hist(rul_fail, bins=20, color="crimson",   alpha=0.75, label="Failure zone (RUL ≤ 30)")
    ax1.axvline(RUL_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                label=f"Failure threshold (RUL={RUL_THRESHOLD})")
    ax1.set_title("RUL Histogram", fontweight="bold")
    ax1.set_xlabel("Remaining Useful Life (cycles)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)

    # ── Right: per-engine RUL box plot (sampled for readability) ─────────────
    sample_units = sorted(df["unit_id"].unique())[:30]
    box_data = [df.loc[df["unit_id"] == u, "RUL"].values for u in sample_units]
    medians   = [np.median(d) for d in box_data]
    colors    = ["crimson" if m <= RUL_THRESHOLD else "steelblue" for m in medians]

    bp = ax2.boxplot(box_data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(RUL_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
                label="Failure threshold")
    ax2.set_title("RUL per Engine (first 30)", fontweight="bold")
    ax2.set_xlabel("Engine unit_id")
    ax2.set_ylabel("RUL (cycles)")
    ax2.set_xticks(range(1, len(sample_units) + 1))
    ax2.set_xticklabels([str(u) for u in sample_units], rotation=90, fontsize=6)
    ax2.legend(fontsize=9)

    fig.suptitle(f"RUL Distribution — CMAPSS {dataset_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, f"{dataset_name}_rul_distribution.png")


# ── Plot 3: Sensor Correlation Heatmap ───────────────────────────────────────

def plot_sensor_correlation_heatmap(df: pd.DataFrame, dataset_name: str) -> Path:
    """
    Show pairwise Pearson correlation between all 14 useful sensors.

    RF insight: Highly correlated sensor pairs (|r| > 0.8) can be candidates
    for PCA compression or single-sensor substitution, reducing model complexity
    without losing diagnostic information.
    """
    corr = df[CMAPSS_USEFUL_SENSORS].corr()
    short_labels = [_SHORT[s] for s in CMAPSS_USEFUL_SENSORS]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        square=True, linewidths=0.5,
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
        annot_kws={"size": 7},
    )

    # Highlight cells where |correlation| > 0.8 with a gold border
    for i in range(len(CMAPSS_USEFUL_SENSORS)):
        for j in range(len(CMAPSS_USEFUL_SENSORS)):
            if i != j and abs(corr.iloc[i, j]) > 0.8:
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="gold", linewidth=2.5, zorder=3,
                ))

    ax.set_title(f"Sensor Correlation Matrix — CMAPSS {dataset_name}\n"
                 "(gold border = |r| > 0.8)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Sensor", fontsize=10)
    ax.set_ylabel("Sensor", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    return _save(fig, f"{dataset_name}_sensor_correlation.png")


# ── Plot 4: Degradation Index ────────────────────────────────────────────────

def plot_degradation_index(
    df: pd.DataFrame,
    dataset_name: str,
    n_units: int = 10,
) -> Path:
    """
    Collapse all 14 sensors into a single composite health score (0=healthy,
    1=degraded) and trace it over normalised engine lifetime.

    RF insight: The degradation index is the RF-Sentinel health gauge. Engines
    that degrade faster (steeper slope) are analogous to RF components with
    accelerated gain compression — they need more frequent maintenance cycles.
    """
    rng   = np.random.default_rng(RANDOM_SEED)
    units = rng.choice(df["unit_id"].unique(), size=min(n_units, df["unit_id"].nunique()), replace=False)

    # Use tab10 for ≤10 engines, tab20 for up to 20 — each engine gets a unique colour
    palette = (plt.cm.tab10 if len(units) <= 10 else plt.cm.tab20)(
        np.linspace(0, 1, len(units))
    )

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, unit in enumerate(units):
        eng = df[df["unit_id"] == unit].sort_values("cycle").copy()
        max_cycle = eng["cycle"].max()
        eng["cycle_pct"] = eng["cycle"] / max_cycle
        min_rul   = int(eng["RUL"].min())

        # Per-engine normalisation of each sensor
        norm_sensors = []
        for s in CMAPSS_USEFUL_SENSORS:
            smin, smax = eng[s].min(), eng[s].max()
            norm_sensors.append((eng[s] - smin) / (smax - smin + 1e-8))
        deg_index = pd.concat(norm_sensors, axis=1).mean(axis=1)

        # Rolling smooth
        smoothed = deg_index.rolling(5, min_periods=1).mean()

        ax.plot(
            eng["cycle_pct"].values, smoothed.values,
            color=palette[i], alpha=0.9, linewidth=2.0,
            label=f"Eng {unit}  (life={max_cycle} cyc, minRUL={min_rul})",
        )

    ax.axhline(0.7, color="darkred", linestyle="--", linewidth=1.5,
               label="Critical threshold (0.7)")
    ax.set_title(f"Engine Degradation Index — CMAPSS {dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalised Lifetime (0 = new, 1 = end of life)")
    ax.set_ylabel("Degradation Index (0 = healthy, 1 = degraded)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    return _save(fig, f"{dataset_name}_degradation_index.png")


# ── Plot 5: Sensor Variance Ranking ──────────────────────────────────────────

def plot_sensor_variance_ranking(df: pd.DataFrame, dataset_name: str) -> Path:
    """
    Rank each sensor by its ability to statistically separate failing engines
    from healthy ones using a Fisher-style separation score.

    RF insight: High-separation sensors are the most valuable features for
    classification. This plot directly guides feature selection — the top 5
    sensors here should always be included in the model's input regardless
    of dimensionality reduction applied to the rest.
    """
    records = []
    for s in CMAPSS_USEFUL_SENSORS:
        overall_std = df[s].std()
        fail_mean   = df.loc[df["fail_soon"] == 1, s].mean()
        pass_mean   = df.loc[df["fail_soon"] == 0, s].mean()
        separation  = abs(fail_mean - pass_mean) / (overall_std + 1e-8)
        records.append({
            "sensor":     s,
            "label":      _SHORT[s],
            "separation": separation,
            "fail_mean":  fail_mean,
            "pass_mean":  pass_mean,
            "overall_std": overall_std,
        })

    stats = (
        pd.DataFrame(records)
        .sort_values("separation", ascending=False)
        .reset_index(drop=True)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: separation score bar chart ─────────────────────────────────────
    colors = ["#2ca02c" if i < 5 else "#aec7e8" for i in range(len(stats))]
    ax1.barh(stats["label"], stats["separation"],
             color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Fisher Separation Score\n(green = top-5 most discriminative)",
                  fontweight="bold")
    ax1.set_xlabel("Separation Score  |μ_fail − μ_pass| / σ")
    ax1.set_ylabel("Sensor")
    ax1.invert_yaxis()
    ax1.tick_params(labelsize=8)

    for i, (_, row) in enumerate(stats.iterrows()):
        ax1.text(row["separation"] + 0.01, i,
                 f"{row['separation']:.2f}", va="center", fontsize=8)

    # ── Right: grouped bar — normalised fail vs pass means ────────────────────
    # Normalise each sensor's means to 0–1 across the two groups for comparison
    norm_fail = (stats["fail_mean"] - stats["fail_mean"].min()) / \
                (stats["fail_mean"].max() - stats["fail_mean"].min() + 1e-8)
    norm_pass = (stats["pass_mean"] - stats["pass_mean"].min()) / \
                (stats["pass_mean"].max() - stats["pass_mean"].min() + 1e-8)

    x    = np.arange(len(stats))
    w    = 0.38
    ax2.bar(x - w/2, norm_pass, w, color="steelblue", alpha=0.8, label="Pass (fail_soon=0)")
    ax2.bar(x + w/2, norm_fail, w, color="crimson",   alpha=0.8, label="Fail (fail_soon=1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats["label"], rotation=45, ha="right", fontsize=8)
    ax2.set_title("Normalised Mean Value\nFail vs Pass per Sensor", fontweight="bold")
    ax2.set_ylabel("Normalised Mean (0–1)")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=8)

    fig.suptitle(f"Sensor Discriminative Power — CMAPSS {dataset_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, f"{dataset_name}_sensor_variance_ranking.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def run_cmapss_eda(
    datasets: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Execute all 8 EDA plots for each specified C-MAPSS dataset subset.

    Loads each subset via load_cmapss(), runs plots 1–8 in order, and
    returns a dict mapping dataset name → list of saved PNG file paths.

    Parameters
    ----------
    datasets : list[str] or None
        Subsets to process. Default: ["FD001","FD002","FD003","FD004"].

    Returns
    -------
    dict[str, list[str]]
        { "FD001": ["/path/to/plot1.png", ...], ... }
    """
    if datasets is None:
        datasets = ["FD001", "FD002", "FD003", "FD004"]

    plot_funcs = [
        ("Sensor degradation waves",   plot_sensor_degradation_waves),
        ("RUL distribution",           plot_rul_distribution),
        ("Sensor correlation heatmap", plot_sensor_correlation_heatmap),
        ("Degradation index",          plot_degradation_index),
        ("Sensor variance ranking",    plot_sensor_variance_ranking),
    ]

    results: Dict[str, List[str]] = {}
    total_saved = 0

    for ds in datasets:
        logger.info(f"[EDA | C-MAPSS] Starting {ds} — {len(plot_funcs)} plots")
        data   = load_cmapss(ds)
        df     = data["train_raw"]
        saved: List[str] = []

        for i, (plot_name, func) in enumerate(plot_funcs, start=1):
            logger.info(f"  Generating plot {i}/{len(plot_funcs)}: {plot_name}")
            try:
                path = func(df, ds)
                saved.append(str(path))
            except Exception as exc:
                logger.error(f"  Plot '{plot_name}' failed for {ds}: {exc}")

        results[ds] = saved
        total_saved += len(saved)
        logger.success(f"[EDA | C-MAPSS] {ds} complete — {len(saved)} plots saved")

    logger.success(
        f"[EDA | C-MAPSS] All done — "
        f"total plots saved: {total_saved} | "
        f"output folder: {EDA_DIR}"
    )
    return results


if __name__ == "__main__":
    run_cmapss_eda()
