"""
eda_unified.py — EDA for the unified RF-Sentinel dataset.

Combines rows from CMAPSS, SECOM and AI4I into a single RF-parameter schema
and generates 5 visualisations covering:
    1. Source breakdown       — row counts and pass/fail split per dataset
    2. Failure rate comparison — failure rates and type distributions by source
    3. RF param distributions  — KDE of all 8 RF parameters per source
    4. RF params vs failure    — box/jitter plots of each param for pass vs fail
    5. Correlation heatmap     — inter-parameter and param-to-failure correlations

All plots saved to outputs/eda/unified/.
Run Layer 1 pipeline first to generate the parquet file:
    python -m src.data.layer1_data_ingestion.pipeline
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from layer1_data_ingestion.config import RANDOM_STATE, ROOT_DIR, UNIFIED_DATASET

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Output directory ──────────────────────────────────────────────────────────
EDA_DIR = ROOT_DIR / "outputs" / "eda" / "unified"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RF_PARAMS = [f"rf_param_{i}" for i in range(1, 9)]

RF_PARAM_LABELS = {
    "rf_param_1": "S21/gain proxy",
    "rf_param_2": "Noise figure proxy",
    "rf_param_3": "P1dB/power proxy",
    "rf_param_4": "IP3/linearity proxy",
    "rf_param_5": "Temperature",
    "rf_param_6": "Frequency proxy",
    "rf_param_7": "Pressure/bias proxy",
    "rf_param_8": "Secondary param",
}

RF_SHORT_LABELS = {
    "rf_param_1": "s21_proxy",
    "rf_param_2": "nf_proxy",
    "rf_param_3": "p1db_proxy",
    "rf_param_4": "ip3_proxy",
    "rf_param_5": "temp",
    "rf_param_6": "freq_proxy",
    "rf_param_7": "bias_proxy",
    "rf_param_8": "secondary",
}

SOURCE_COLORS = {
    "cmapss_FD001": "steelblue",
    "secom":        "orange",
    "ai4i":         "limegreen",
}

FAILURE_TYPE_COLORS = {
    "pass":                    "#a8c8e8",
    "sensor_degradation":      "steelblue",
    "manufacturing_defect":    "orange",
    "heat_dissipation_failure": "tomato",
    "power_failure":           "purple",
    "overstrain_failure":      "limegreen",
    "thermal_wear_failure":    "crimson",
    "random_failure":          "dimgray",
}


def _save(fig: plt.Figure, fname: str) -> Path:
    """Save figure, close it, return path."""
    path = EDA_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_unified() -> pd.DataFrame:
    """
    Load the unified RF-Sentinel parquet file produced by Layer 1 pipeline.

    Returns
    -------
    pd.DataFrame
        Unified dataset with columns: row_id, device_id, cycle_or_sample,
        rf_param_1…rf_param_8, failure_label, failure_type, dataset_source, rul.
    """
    df = pd.read_parquet(UNIFIED_DATASET)
    logger.info(
        f"[EDA | Unified] Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols | "
        f"sources: {df['dataset_source'].unique().tolist()}"
    )
    return df


# ── Plot 1: Dataset Source Breakdown ─────────────────────────────────────────

def plot_dataset_source_breakdown(df: pd.DataFrame) -> Path:
    """
    Show how many rows each dataset contributes to the unified dataset and
    the pass/fail balance within each source.

    Insight: CMAPSS dominates by row count (multi-cycle time-series), while
    SECOM and AI4I each contribute ~1,500 and ~10,000 single-sample rows.
    Class imbalance varies by source — critical for weighted loss functions.
    """
    total   = len(df)
    sources = df["dataset_source"].unique().tolist()
    counts  = df["dataset_source"].value_counts()
    colors  = [SOURCE_COLORS.get(s, "gray") for s in counts.index]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # ── Left: horizontal bar chart ────────────────────────────────────────────
    bars = ax1.barh(counts.index[::-1], counts.values[::-1],
                    color=colors[::-1], edgecolor="white", height=0.55)
    for bar, src in zip(bars, counts.index[::-1]):
        cnt = counts[src]
        pct = cnt / total * 100
        ax1.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
                 f"{cnt:,}  ({pct:.1f}%)", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, counts.max() * 1.35)
    ax1.set_title("Rows per Dataset Source", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Row Count", fontsize=11)
    ax1.set_ylabel("Dataset Source", fontsize=11)
    ax1.tick_params(labelsize=10)
    sns.despine(ax=ax1)

    # ── Middle: donut chart ───────────────────────────────────────────────────
    wedges, _ = ax2.pie(
        counts.values,
        colors=[SOURCE_COLORS.get(s, "gray") for s in counts.index],
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
        labels=None,
    )
    ax2.text(0, 0, f"{total:,}\ntotal rows",
             ha="center", va="center", fontsize=13, fontweight="bold", color="dimgray")

    legend_patches = [
        mpatches.Patch(
            color=SOURCE_COLORS.get(s, "gray"),
            label=f"{s}: {counts[s]:,} ({counts[s]/total*100:.1f}%)"
        )
        for s in counts.index
    ]
    ax2.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=False, ncol=1)
    ax2.set_title("Dataset Composition", fontweight="bold", fontsize=12)

    # ── Right: stacked pass/fail per source ───────────────────────────────────
    src_list = counts.index.tolist()
    x        = np.arange(len(src_list))
    pass_cnts = [df.loc[df["dataset_source"] == s, "failure_label"].eq(0).sum() for s in src_list]
    fail_cnts = [df.loc[df["dataset_source"] == s, "failure_label"].eq(1).sum() for s in src_list]

    ax3.bar(x, pass_cnts, color="steelblue", label="Pass", edgecolor="white")
    ax3.bar(x, fail_cnts, bottom=pass_cnts, color="crimson", label="Fail", edgecolor="white")

    for i, (p, f) in enumerate(zip(pass_cnts, fail_cnts)):
        rate = f / (p + f) * 100
        ax3.text(i, p + f + 50, f"{rate:.1f}%\nfailure",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="darkred")

    ax3.set_xticks(x)
    ax3.set_xticklabels(src_list, rotation=15, ha="right", fontsize=10)
    ax3.set_title("Pass / Fail per Dataset Source", fontweight="bold", fontsize=12)
    ax3.set_xlabel("Dataset Source", fontsize=11)
    ax3.set_ylabel("Sample Count", fontsize=11)
    ax3.legend(fontsize=10)
    sns.despine(ax=ax3)

    fig.suptitle("RF-Sentinel Unified Dataset — Source Breakdown",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "unified_dataset_source_breakdown.png")


# ── Plot 2: Failure Rate Comparison ──────────────────────────────────────────

def plot_failure_rate_comparison(df: pd.DataFrame) -> Path:
    """
    Compare failure rates and failure type distributions across all sources.

    Insight: CMAPSS has the highest failure rate by design (RUL_THRESHOLD
    creates ~15% label density). SECOM and AI4I reflect real-world rates
    (~6.6% and ~3.4%). This imbalance must be handled per-source in training.
    """
    sources      = df["dataset_source"].unique().tolist()
    overall_rate = df["failure_label"].mean() * 100

    src_stats = {}
    for s in sources:
        sub   = df[df["dataset_source"] == s]
        n_fail = sub["failure_label"].sum()
        rate   = sub["failure_label"].mean() * 100
        src_stats[s] = {"rate": rate, "n_fail": n_fail, "total": len(sub)}

    all_types = df["failure_type"].value_counts().index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # ── Left: grouped bar — failure rate per source ───────────────────────────
    x      = np.arange(len(sources))
    colors = [SOURCE_COLORS.get(s, "gray") for s in sources]
    rates  = [src_stats[s]["rate"] for s in sources]
    bars   = ax1.bar(x, rates, color=colors, edgecolor="white", width=0.5)

    ax1.axhline(overall_rate, color="black", linestyle="--", linewidth=1.3,
                label=f"Overall = {overall_rate:.2f}%")

    for bar, s in zip(bars, sources):
        st  = src_stats[s]
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f"{st['rate']:.1f}%\n(n={st['n_fail']})",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(sources, rotation=10, ha="right", fontsize=11)
    ax1.set_title("Failure Rate by Dataset Source", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Dataset Source", fontsize=11)
    ax1.set_ylabel("Failure Rate (%)", fontsize=11)
    ax1.legend(fontsize=10)
    sns.despine(ax=ax1)

    # ── Right: stacked horizontal bar — failure type per source ───────────────
    y = np.arange(len(sources))
    lefts = np.zeros(len(sources))

    for ftype in all_types:
        counts_by_src = [
            df.loc[df["dataset_source"] == s, "failure_type"].eq(ftype).sum()
            for s in sources
        ]
        color = FAILURE_TYPE_COLORS.get(ftype, "lightgray")
        ax2.barh(y, counts_by_src, left=lefts,
                 color=color, label=ftype, edgecolor="white", height=0.55)
        lefts += np.array(counts_by_src)

    ax2.set_yticks(y)
    ax2.set_yticklabels(sources, fontsize=11)
    ax2.set_title("Failure Type Distribution per Source", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Count", fontsize=11)
    ax2.set_ylabel("Dataset Source", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right", frameon=True, ncol=2)
    sns.despine(ax=ax2)

    fig.suptitle("RF-Sentinel — Cross-Dataset Failure Rate Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "unified_failure_rate_comparison.png")


# ── Plot 3: RF Parameter Distributions ───────────────────────────────────────

def plot_rf_param_distributions(df: pd.DataFrame) -> Path:
    """
    Show the distribution of all 8 unified RF parameters for each source.

    Insight: Because the three datasets measure different physical systems,
    each rf_param will have very different scales per source — confirming
    that per-source normalisation is required before joint modelling.
    """
    sources = df["dataset_source"].unique().tolist()
    rng     = np.random.default_rng(RANDOM_STATE)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, param in enumerate(RF_PARAMS):
        ax = axes[idx]
        for src in sources:
            vals = df.loc[df["dataset_source"] == src, param].dropna()
            if len(vals) < 10:
                continue
            color = SOURCE_COLORS.get(src, "gray")
            # Downsample large datasets for KDE speed
            if len(vals) > 5000:
                vals = vals.sample(5000, random_state=RANDOM_STATE)

            try:
                vals.plot.kde(ax=ax, color=color, linewidth=2, label=src)
                kde_x = np.linspace(vals.min(), vals.max(), 200)
                from scipy.stats import gaussian_kde
                kde_y = gaussian_kde(vals)(kde_x)
                ax.fill_between(kde_x, 0, kde_y, alpha=0.15, color=color)
                ax.axvline(vals.mean(), color=color, linestyle="--",
                           linewidth=1, alpha=0.8)
            except Exception:
                pass

        ax.set_title(f"{param}\n({RF_PARAM_LABELS[param]})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("RF-Sentinel — Unified RF Parameter Distributions by Source",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "unified_rf_param_distributions.png")


# ── Plot 4: RF Parameters vs Failure ─────────────────────────────────────────

def plot_rf_params_vs_failure(df: pd.DataFrame) -> Path:
    """
    Compare each RF parameter distribution between passing and failing samples.

    Insight: Parameters with large pass/fail separation are the strongest
    predictors for the binary failure classifier. Parameters with near-zero
    difference add noise and are candidates for dropping.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    box_colors = {0: "steelblue", 1: "crimson"}
    labels     = {0: "Pass", 1: "Fail"}

    for idx, param in enumerate(RF_PARAMS):
        ax = axes[idx]

        data_pass = df.loc[df["failure_label"] == 0, param].dropna()
        data_fail = df.loc[df["failure_label"] == 1, param].dropna()

        box_data = [data_pass.values, data_fail.values]
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False,
                        widths=0.45, positions=[0, 1])
        for patch, color in zip(bp["boxes"], [box_colors[0], box_colors[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Jitter scatter (max 200 points per class)
        for i, (vals, color) in enumerate(zip(box_data, [box_colors[0], box_colors[1]])):
            n      = min(200, len(vals))
            sample = rng.choice(vals, size=n, replace=False) if len(vals) > n else vals
            jitter = rng.uniform(-0.12, 0.12, size=len(sample))
            ax.scatter(i + jitter, sample, alpha=0.25, s=8, color=color, zorder=3)

        # Mean difference annotation
        if len(data_pass) > 0 and len(data_fail) > 0:
            mean_diff = data_fail.mean() - data_pass.mean()
            ax.text(0.5, 0.97, f"Δmean = {mean_diff:+.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="darkred",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="lightyellow", alpha=0.8))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pass", "Fail"], fontsize=9)
        ax.set_title(f"{param}: pass vs fail\n({RF_PARAM_LABELS[param]})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Label", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("RF-Sentinel — RF Parameters: Pass vs Fail Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "unified_rf_params_vs_failure.png")


# ── Plot 5: Unified Correlation Heatmap ──────────────────────────────────────

def plot_unified_correlation_heatmap(df: pd.DataFrame) -> Path:
    """
    Show inter-parameter correlations and each parameter's relationship
    to the binary failure label across the unified dataset.

    Insight: High inter-parameter correlation suggests the 8-param schema
    has redundancy — PCA or feature selection can compress further. The
    failure-correlation bar chart gives a quick ranking of diagnostic value.
    """
    param_df  = df[RF_PARAMS].copy()
    short_lbl = [RF_SHORT_LABELS[p] for p in RF_PARAMS]

    corr_matrix = param_df.corr()
    corr_matrix.index   = short_lbl
    corr_matrix.columns = short_lbl

    # Correlation of each param with failure_label (drop NaN rows first)
    valid = df[RF_PARAMS + ["failure_label"]].dropna()
    fail_corr = valid[RF_PARAMS].corrwith(valid["failure_label"])
    fail_corr.index = short_lbl
    fail_corr_sorted = fail_corr.reindex(
        fail_corr.abs().sort_values(ascending=False).index
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ── Left: inter-parameter heatmap ─────────────────────────────────────────
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        square=True, linewidths=0.5,
        annot_kws={"size": 9},
        ax=ax1,
    )
    ax1.set_title("RF Parameter Inter-correlation Matrix",
                  fontweight="bold", fontsize=12)
    ax1.set_xlabel("RF Parameter", fontsize=10)
    ax1.set_ylabel("RF Parameter", fontsize=10)
    ax1.tick_params(axis="x", rotation=45, labelsize=9)
    ax1.tick_params(axis="y", rotation=0,  labelsize=9)

    # ── Right: correlation with failure_label ─────────────────────────────────
    bar_colors = ["crimson" if v > 0 else "steelblue"
                  for v in fail_corr_sorted.values]
    bars = ax2.barh(fail_corr_sorted.index[::-1],
                    fail_corr_sorted.values[::-1],
                    color=bar_colors[::-1], edgecolor="white", height=0.6)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.2)

    for bar, val in zip(bars, fail_corr_sorted.values[::-1]):
        x_pos = bar.get_width() + (0.003 if val >= 0 else -0.003)
        ha    = "left" if val >= 0 else "right"
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", va="center", ha=ha,
                 fontsize=10, fontweight="bold")

    pos_patch = mpatches.Patch(color="crimson",   label="Positive correlation")
    neg_patch = mpatches.Patch(color="steelblue", label="Negative correlation")
    ax2.legend(handles=[pos_patch, neg_patch], fontsize=10)
    ax2.set_title("RF Parameter Correlation with Failure",
                  fontweight="bold", fontsize=12)
    ax2.set_xlabel("Pearson r with failure_label", fontsize=11)
    ax2.set_ylabel("RF Parameter", fontsize=11)
    ax2.tick_params(labelsize=10)
    sns.despine(ax=ax2)

    fig.suptitle("RF-Sentinel — Unified Parameter Correlation Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "unified_correlation_heatmap.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def run_unified_eda() -> List[str]:
    """
    Load the unified parquet file and execute all 5 EDA plots in order.

    Raises
    ------
    FileNotFoundError
        If the unified parquet has not been generated yet.

    Returns
    -------
    list[str]
        Paths to all saved PNG files.
    """
    if not UNIFIED_DATASET.exists():
        msg = (
            f"Unified dataset not found at {UNIFIED_DATASET}. "
            "Run Layer 1 pipeline first: "
            "python -m src.data.layer1_data_ingestion.pipeline"
        )
        logger.error(f"[EDA | Unified] {msg}")
        raise FileNotFoundError(msg)

    df = _load_unified()

    plot_funcs = [
        ("Dataset source breakdown",    lambda: plot_dataset_source_breakdown(df)),
        ("Failure rate comparison",     lambda: plot_failure_rate_comparison(df)),
        ("RF param distributions",      lambda: plot_rf_param_distributions(df)),
        ("RF params vs failure",        lambda: plot_rf_params_vs_failure(df)),
        ("Unified correlation heatmap", lambda: plot_unified_correlation_heatmap(df)),
    ]

    saved: List[str] = []

    for i, (name, func) in enumerate(plot_funcs, start=1):
        logger.info(f"  Generating unified plot {i}/{len(plot_funcs)}: {name}")
        try:
            path = func()
            saved.append(str(path))
        except Exception as exc:
            logger.error(f"  Plot '{name}' failed: {exc}")

    logger.success(
        f"[EDA | Unified] Done — "
        f"total plots saved: {len(saved)} | "
        f"output folder: {EDA_DIR}"
    )
    return saved


if __name__ == "__main__":
    run_unified_eda()
