"""
eda_ai4i.py — Exploratory Data Analysis for the AI4I 2020 predictive maintenance dataset.

Generates 5 visualisations covering:
    1. Failure type distribution    — donut chart + breakdown of all 6 failure modes
    2. Temperature vs failure        — how air/process temperature relates to failures
    3. Torque-speed operating space  — where each failure type occurs in the operating envelope
    4. Tool wear survival analysis   — accumulation of wear before each failure type
    5. Failure correlation matrix    — feature-to-failure and failure-to-failure correlations

All plots saved to outputs/eda/ai4i/.
Insights map directly to RF-Sentinel's root-cause classification task:
each failure type in AI4I corresponds to a distinct RF component failure mode.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import pointbiserialr

from layer1_data_ingestion.config import (
    AI4I_FAILURE_COLS,
    AI4I_TO_RF_MAP,
    RANDOM_STATE,
    ROOT_DIR,
)
from layer1_data_ingestion.loaders import load_ai4i

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Output directory ──────────────────────────────────────────────────────────
EDA_DIR = ROOT_DIR / "outputs" / "eda" / "ai4i"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color mapping across all plots
TYPE_COLORS = {
    "pass":                    "steelblue",
    "thermal_wear_failure":    "crimson",
    "heat_dissipation_failure": "orange",
    "power_failure":           "purple",
    "overstrain_failure":      "limegreen",
    "random_failure":          "dimgray",
}

# Short display names for axes
SHORT_NAMES = {
    "pass":                    "Pass",
    "thermal_wear_failure":    "TWF",
    "heat_dissipation_failure": "HDF",
    "power_failure":           "PWF",
    "overstrain_failure":      "OSF",
    "random_failure":          "RNF",
}


def _save(fig: plt.Figure, fname: str) -> Path:
    """Save figure, close it, return path."""
    path = EDA_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 1: Failure Type Donut ────────────────────────────────────────────────

def plot_failure_type_donut(df_raw, failure_type_counts):
    """
    Show the breakdown of all 6 failure types (including pass) as a donut
    chart, and provide a ranked bar chart of the failure-only cases.

    RF insight: Heat dissipation failure is the most common mode — directly
    analogous to RF amplifier thermal runaway. Knowing the failure mix
    determines the class weights and SMOTE strategy.
    """
    total = len(df_raw)
    fail_df = df_raw[df_raw["Machine failure"] == 1]
    total_failures = len(fail_df)

    type_counts = {
        "HDF": int(df_raw["HDF"].sum()),
        "PWF": int(df_raw["PWF"].sum()),
        "OSF": int(df_raw["OSF"].sum()),
        "TWF": int(df_raw["TWF"].sum()),
        "RNF": int(df_raw["RNF"].sum()),
    }
    pass_count = total - df_raw["Machine failure"].sum()

    colors_map = {
        "Pass": "steelblue",
        "HDF":  "orange",
        "PWF":  "purple",
        "OSF":  "green",
        "TWF":  "crimson",
        "RNF":  "gray",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(wspace=0.4)

    # ── Left: clean donut ────────────────────────────────────
    fail_total   = total - pass_count
    donut_sizes  = [pass_count, fail_total]
    donut_colors = ["steelblue", "crimson"]

    wedges, _ = ax1.pie(
        donut_sizes,
        colors=donut_colors,
        explode=(0, 0.06),
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        labels=None,
    )
    ax1.text(0, 0, f"{total:,}\nsamples",
             ha="center", va="center",
             fontsize=14, fontweight="bold", color="black")

    legend_labels = [f"Pass: {pass_count:,} ({pass_count/total*100:.1f}%)"]
    legend_colors = ["steelblue"]
    for name, cnt in type_counts.items():
        legend_labels.append(f"{name}: {cnt} ({cnt/total*100:.1f}%)")
        legend_colors.append(colors_map[name])

    legend_patches = [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in zip(legend_colors, legend_labels)
    ]
    ax1.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=11,
        frameon=False,
    )
    ax1.set_title("Pass vs All Failures", fontweight="bold", fontsize=13, pad=15)

    # ── Right: horizontal bar chart failures only ─────────────
    names      = list(type_counts.keys())
    counts_raw = list(type_counts.values())
    bar_colors = [colors_map[n] for n in names]

    sorted_pairs = sorted(zip(counts_raw, names, bar_colors), reverse=True)
    counts_s, names_s, colors_s = zip(*sorted_pairs)

    bars = ax2.barh(names_s, counts_s, color=colors_s, edgecolor="white", height=0.55)
    for bar, cnt in zip(bars, counts_s):
        pct_of_failures = cnt / total_failures * 100
        ax2.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt}  ({pct_of_failures:.1f}%)",
            va="center", ha="left",
            fontsize=12, fontweight="bold",
        )

    mean_count = sum(counts_s) / len(counts_s)
    ax2.axvline(mean_count, color="black", linestyle="--",
                linewidth=1.2, label=f"Mean = {mean_count:.0f}")
    ax2.set_xlim(0, max(counts_s) * 1.35)
    ax2.set_title(
        f"Failure Type Breakdown ({total_failures} total failures)",
        fontweight="bold", fontsize=13,
    )
    ax2.set_xlabel("Count", fontsize=12)
    ax2.set_ylabel("Failure Type", fontsize=12)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=11)
    sns.despine(ax=ax2)

    fig.suptitle("AI4I — Failure Type Distribution", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "ai4i_failure_type_donut.png")


# ── Plot 2: Temperature vs Failure ────────────────────────────────────────────

def plot_temperature_vs_failure(df_raw: pd.DataFrame) -> Path:
    """
    Analyse how air and process temperature relate to machine failures.

    RF insight: RF transistors are thermally limited — elevated junction
    temperature is the leading cause of gain compression and noise figure
    degradation. This plot establishes the thermal operating boundary
    beyond which failure risk rises sharply.
    """
    air_col     = "Air temperature [K]"
    proc_col    = "Process temperature [K]"
    fail_col    = "Machine failure"
    type_col    = "failure_type"

    df_pass = df_raw[df_raw[fail_col] == 0]
    df_fail = df_raw[df_raw[fail_col] == 1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # ── Left: scatter air vs process temp ────────────────────────────────────
    ax1.scatter(df_pass[air_col], df_pass[proc_col],
                c="steelblue", alpha=0.2, s=6, label="Pass", rasterized=True)
    ax1.scatter(df_fail[air_col], df_fail[proc_col],
                c="crimson", alpha=0.8, s=40, label="Fail", zorder=5)

    # Diagonal: process temp ≈ air temp + 10 K
    x_range = np.linspace(df_raw[air_col].min(), df_raw[air_col].max(), 100)
    ax1.plot(x_range, x_range + 10, color="gray", linestyle="--", linewidth=1,
             label="+10 K reference")

    ax1.set_title("Air vs Process Temperature", fontweight="bold", fontsize=11)
    ax1.set_xlabel(air_col, fontsize=10)
    ax1.set_ylabel(proc_col, fontsize=10)
    ax1.legend(fontsize=9)

    # ── Middle: box plot — air temp per failure type ──────────────────────────
    order  = df_raw[type_col].value_counts().index.tolist()
    colors = [TYPE_COLORS.get(t, "lightgray") for t in order]

    bp = ax2.boxplot(
        [df_raw.loc[df_raw[type_col] == t, air_col].values for t in order],
        patch_artist=True, showfliers=False, widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    mean_temp = df_raw[air_col].mean()
    ax2.axhline(mean_temp, color="black", linestyle="--", linewidth=1,
                label=f"Overall mean = {mean_temp:.1f} K")
    ax2.set_xticks(range(1, len(order) + 1))
    ax2.set_xticklabels([SHORT_NAMES.get(t, t) for t in order],
                        rotation=45, ha="right", fontsize=9)
    ax2.set_title("Air Temperature by Failure Type", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Failure Type", fontsize=10)
    ax2.set_ylabel(air_col, fontsize=10)
    ax2.legend(fontsize=9)

    # ── Right: 2D histogram + failure overlay ─────────────────────────────────
    h, xedges, yedges, img = ax3.hist2d(
        df_raw[air_col], df_raw[proc_col],
        bins=40, cmap="YlOrRd",
    )
    fig.colorbar(img, ax=ax3, label="Sample density")
    ax3.scatter(df_fail[air_col], df_fail[proc_col],
                c="blue", s=15, alpha=0.6, label="Failures", zorder=5)
    ax3.set_title("Temperature Density + Failure Locations", fontweight="bold", fontsize=11)
    ax3.set_xlabel(air_col, fontsize=10)
    ax3.set_ylabel(proc_col, fontsize=10)
    ax3.legend(fontsize=9)

    fig.suptitle("AI4I — Temperature vs Failure Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "ai4i_temperature_vs_failure.png")


# ── Plot 3: Torque-Speed Contour ──────────────────────────────────────────────

def plot_torque_speed_contour(df_raw: pd.DataFrame) -> Path:
    """
    Map where each failure type occurs in the torque-speed operating space.

    RF insight: In RF power amplifiers, operating too close to the maximum
    power or saturated output creates analogous stress. This plot identifies
    which combinations of 'speed' (frequency) and 'torque' (drive level)
    push the system into each failure regime.
    """
    speed_col = "Rotational speed [rpm]"
    torque_col = "Torque [Nm]"
    type_col   = "failure_type"
    fail_col   = "Machine failure"

    df_pass = df_raw[df_raw[fail_col] == 0]
    df_fail = df_raw[df_raw[fail_col] == 1]

    key_types = ["heat_dissipation_failure", "power_failure", "overstrain_failure"]

    fig = plt.figure(figsize=(20, 7))
    gs  = gridspec.GridSpec(3, 3, figure=fig, wspace=0.38, hspace=0.55)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax_right = [fig.add_subplot(gs[i, 2]) for i in range(3)]

    # ── Left: all failure types scatter ──────────────────────────────────────
    ax1.scatter(df_pass[speed_col], df_pass[torque_col],
                c="steelblue", alpha=0.15, s=5, label="Pass", rasterized=True)

    for ft in [t for t in df_fail[type_col].unique() if t != "pass"]:
        sub = df_fail[df_fail[type_col] == ft]
        ax1.scatter(sub[speed_col], sub[torque_col],
                    c=TYPE_COLORS.get(ft, "black"),
                    s=50, alpha=0.9, label=SHORT_NAMES.get(ft, ft), zorder=5)

    ax1.set_title("Torque vs Speed — All Failure Types", fontweight="bold", fontsize=11)
    ax1.set_xlabel(speed_col, fontsize=9)
    ax1.set_ylabel(torque_col, fontsize=9)
    ax1.legend(fontsize=8, markerscale=1.2)

    # ── Middle: KDE contour of pass + failure overlay ─────────────────────────
    sns.kdeplot(
        x=df_pass[speed_col], y=df_pass[torque_col],
        ax=ax2, cmap="Blues", fill=True, levels=12, thresh=0.05,
    )
    for ft in [t for t in df_fail[type_col].unique() if t != "pass"]:
        sub = df_fail[df_fail[type_col] == ft]
        ax2.scatter(sub[speed_col], sub[torque_col],
                    c=TYPE_COLORS.get(ft, "black"),
                    s=40, alpha=0.85, label=SHORT_NAMES.get(ft, ft), zorder=5)

    ax2.set_title("Operating Envelope — Pass Region", fontweight="bold", fontsize=11)
    ax2.set_xlabel(speed_col, fontsize=9)
    ax2.set_ylabel(torque_col, fontsize=9)
    ax2.legend(fontsize=8)

    # ── Right: 3 small subplots — HDF, PWF, OSF ──────────────────────────────
    for ax_sub, ft in zip(ax_right, key_types):
        sub = df_raw[df_raw[type_col] == ft]
        ax_sub.scatter(df_pass[speed_col], df_pass[torque_col],
                       c="lightgray", alpha=0.3, s=4, rasterized=True)
        ax_sub.scatter(sub[speed_col], sub[torque_col],
                       c=TYPE_COLORS.get(ft, "black"),
                       s=35, alpha=0.9, zorder=5)
        ax_sub.set_title(f"{SHORT_NAMES.get(ft, ft)}  (n={len(sub)})",
                         fontweight="bold", fontsize=9)
        ax_sub.set_xlabel(speed_col, fontsize=7)
        ax_sub.set_ylabel(torque_col, fontsize=7)
        ax_sub.tick_params(labelsize=7)

    fig.suptitle("AI4I — Torque vs Speed Operating Space", fontsize=14, fontweight="bold")
    return _save(fig, "ai4i_torque_speed_contour.png")


# ── Plot 4: Tool Wear Survival ────────────────────────────────────────────────

def plot_tool_wear_survival(df_raw: pd.DataFrame) -> Path:
    """
    Show how accumulated tool wear correlates with failure probability and
    which failure types emerge at different wear levels.

    RF insight: Tool wear is analogous to RF component ageing — insertion loss
    and VSWR degrade gradually with operational hours. The 'high risk zone'
    threshold here maps directly to the maintenance interval at which RF
    component replacement should be scheduled.
    """
    rng      = np.random.default_rng(RANDOM_STATE)
    wear_col = "Tool wear [min]"
    fail_col = "Machine failure"
    type_col = "failure_type"

    df_pass = df_raw[df_raw[fail_col] == 0]
    df_fail = df_raw[df_raw[fail_col] == 1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # ── Left: wear histogram pass vs fail ────────────────────────────────────
    ax1.hist(df_pass[wear_col], bins=40, color="steelblue", alpha=0.6,
             label="Pass", density=False)
    ax1.hist(df_fail[wear_col], bins=30, color="crimson", alpha=0.7,
             label="Fail", density=False)

    pass_mean = df_pass[wear_col].mean()
    fail_mean = df_fail[wear_col].mean()
    ax1.axvline(pass_mean, color="steelblue", linestyle="--", linewidth=1.5,
                label=f"Pass mean = {pass_mean:.0f}")
    ax1.axvline(fail_mean, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Fail mean = {fail_mean:.0f}")

    risk_thresh = float(np.percentile(df_fail[wear_col], 75))
    ax1.axvline(risk_thresh, color="darkred", linestyle="-", linewidth=2,
                label=f"High risk zone ≥{risk_thresh:.0f}")
    ax1.text(risk_thresh + 5, ax1.get_ylim()[1] * 0.85,
             "High risk\nzone", color="darkred", fontsize=9, fontweight="bold")

    ax1.set_title("Tool Wear Distribution: Pass vs Fail", fontweight="bold", fontsize=11)
    ax1.set_xlabel(wear_col, fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend(fontsize=8)

    # ── Middle: failure rate vs tool wear bins ────────────────────────────────
    n_bins = 20
    df_raw_c = df_raw.copy()
    df_raw_c["wear_bin"] = pd.cut(df_raw_c[wear_col], bins=n_bins, labels=False)

    bin_stats = (
        df_raw_c.groupby("wear_bin")[fail_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "failures", "count": "total"})
    )
    bin_stats["fail_rate"] = (bin_stats["failures"] / bin_stats["total"] * 100).fillna(0)

    overall_rate = df_raw[fail_col].mean() * 100
    bin_mids     = [df_raw_c.loc[df_raw_c["wear_bin"] == i, wear_col].mean()
                    for i in bin_stats.index]

    ax2.plot(bin_mids, bin_stats["fail_rate"], color="crimson",
             linewidth=2, marker="o", markersize=5)
    ax2.fill_between(bin_mids, 0, bin_stats["fail_rate"], alpha=0.2, color="crimson")
    ax2.axhline(overall_rate, color="gray", linestyle="--", linewidth=1.2,
                label=f"Overall rate = {overall_rate:.2f}%")

    exceed_idx = bin_stats[bin_stats["fail_rate"] > 10].index
    if not exceed_idx.empty:
        first_exceed = exceed_idx[0]
        x_exceed = bin_mids[first_exceed] if first_exceed < len(bin_mids) else None
        if x_exceed and not np.isnan(x_exceed):
            ax2.axvline(x_exceed, color="darkred", linestyle=":", linewidth=1.5,
                        label=f">10% risk at {x_exceed:.0f} min")

    ax2.set_title("Failure Rate vs Tool Wear", fontweight="bold", fontsize=11)
    ax2.set_xlabel(wear_col, fontsize=10)
    ax2.set_ylabel("Failure Rate (%)", fontsize=10)
    ax2.legend(fontsize=9)

    # ── Right: box plot + jitter per failure type ─────────────────────────────
    type_order = df_raw[type_col].value_counts().index.tolist()
    box_data   = [df_raw.loc[df_raw[type_col] == t, wear_col].values for t in type_order]
    box_colors = [TYPE_COLORS.get(t, "lightgray") for t in type_order]

    bp = ax3.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, (vals, color) in enumerate(zip(box_data, box_colors), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=min(len(vals), 200))
        sample = vals[:200]
        ax3.scatter(i + jitter, sample, alpha=0.3, s=8, color=color, zorder=3)
        med = float(np.median(vals))
        ax3.text(i, med + 2, f"{med:.0f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    ax3.set_xticks(range(1, len(type_order) + 1))
    ax3.set_xticklabels([SHORT_NAMES.get(t, t) for t in type_order],
                        rotation=45, ha="right", fontsize=9)
    ax3.set_title("Tool Wear Distribution per Failure Type", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Failure Type", fontsize=10)
    ax3.set_ylabel(wear_col, fontsize=10)

    fig.suptitle("AI4I — Tool Wear & Survival Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "ai4i_tool_wear_survival.png")


# ── Plot 5: Failure Correlation Matrix ───────────────────────────────────────

def plot_failure_correlation_matrix(df_raw: pd.DataFrame) -> Path:
    """
    Show how failure types co-occur with each other and how well each feature
    predicts each individual failure mode.

    RF insight: The feature-to-failure heatmap is the root-cause analysis
    fingerprint — it tells the model which sensor reading most strongly
    signals which failure mode, directly informing the multiclass classifier's
    feature importance expectations.
    """
    # ── Left: failure flag co-occurrence correlation ──────────────────────────
    fail_cols_plus = AI4I_FAILURE_COLS + ["Machine failure"]
    corr_fail = df_raw[fail_cols_plus].corr()

    # ── Right: feature → failure type correlation (point-biserial) ────────────
    feat_cols_numeric = [
        "Type_encoded",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    feat_labels = [
        "Type",
        "Air Temp",
        "Proc Temp",
        "Speed",
        "Torque",
        "Tool Wear",
    ]

    corr_data = np.zeros((len(feat_cols_numeric), len(AI4I_FAILURE_COLS)))
    for i, feat in enumerate(feat_cols_numeric):
        for j, fail in enumerate(AI4I_FAILURE_COLS):
            r, _ = pointbiserialr(df_raw[fail], df_raw[feat])
            corr_data[i, j] = r if not np.isnan(r) else 0.0

    short_rf_labels = [SHORT_NAMES.get(AI4I_TO_RF_MAP.get(fc, fc), fc)
                       for fc in AI4I_FAILURE_COLS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        corr_fail,
        annot=True, fmt=".3f",
        cmap="Reds", square=True,
        linewidths=0.5,
        annot_kws={"size": 9},
        ax=ax1,
    )
    ax1.set_title(
        "Failure Type Co-occurrence Correlation\n"
        "(high value = two failure types often occur together)",
        fontweight="bold", fontsize=11,
    )
    ax1.set_xlabel("Failure Flag", fontsize=10)
    ax1.set_ylabel("Failure Flag", fontsize=10)
    ax1.tick_params(axis="x", rotation=45, labelsize=9)
    ax1.tick_params(axis="y", rotation=0,  labelsize=9)

    corr_df = pd.DataFrame(corr_data,
                           index=feat_labels,
                           columns=short_rf_labels)
    sns.heatmap(
        corr_df,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, square=False,
        annot_kws={"size": 10},
        ax=ax2,
    )
    ax2.set_title(
        "Feature → Failure Type Correlation\n"
        "(point-biserial r with each failure flag)",
        fontweight="bold", fontsize=11,
    )
    ax2.set_xlabel("Failure Type (RF equivalent)", fontsize=10)
    ax2.set_ylabel("Feature", fontsize=10)
    ax2.tick_params(axis="x", rotation=45, labelsize=9)
    ax2.tick_params(axis="y", rotation=0,  labelsize=10)

    fig.suptitle("AI4I — Failure Correlation Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "ai4i_failure_correlation.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def run_ai4i_eda() -> List[str]:
    """
    Load AI4I data and execute all 5 EDA plots in order.

    Returns
    -------
    list[str]
        Paths to all saved PNG files.
    """
    logger.info("[EDA | AI4I] Loading dataset...")
    data = load_ai4i()

    df_raw              = data["df_raw"]
    failure_type_counts = data["failure_type_counts"]

    plot_funcs = [
        ("Failure type donut",         lambda: plot_failure_type_donut(df_raw, failure_type_counts)),
        ("Temperature vs failure",     lambda: plot_temperature_vs_failure(df_raw)),
        ("Torque-speed contour",       lambda: plot_torque_speed_contour(df_raw)),
        ("Tool wear survival",         lambda: plot_tool_wear_survival(df_raw)),
        ("Failure correlation matrix", lambda: plot_failure_correlation_matrix(df_raw)),
    ]

    saved: List[str] = []

    for i, (name, func) in enumerate(plot_funcs, start=1):
        logger.info(f"  Generating AI4I plot {i}/{len(plot_funcs)}: {name}")
        try:
            path = func()
            saved.append(str(path))
        except Exception as exc:
            logger.error(f"  Plot '{name}' failed: {exc}")

    logger.success(
        f"[EDA | AI4I] Done — "
        f"total plots saved: {len(saved)} | "
        f"output folder: {EDA_DIR}"
    )
    return saved


if __name__ == "__main__":
    run_ai4i_eda()
