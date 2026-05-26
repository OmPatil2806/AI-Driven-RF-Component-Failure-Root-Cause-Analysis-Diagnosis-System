"""
eda_secom.py — Exploratory Data Analysis for the SECOM semiconductor dataset.

Generates 6 visualisations covering the core challenges of SECOM:
    1. Class imbalance analysis      — 6.6 % failure rate + SMOTE preview
    2. PCA dimensionality reduction  — variance explained, PC1/PC2 scatter
    3. Feature-failure correlation   — which features correlate with failure
    4. Top feature distributions     — KDE pass vs fail per top feature
    5. Missing values vs failure     — is missingness itself a failure signal?
    6. Feature variance overview     — near-zero vs informative features

All plots saved to outputs/eda/secom/.
Insights feed directly into preprocessing choices (SMOTE, PCA, imputation)
and feature selection for RF-Sentinel.
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from layer1_data_ingestion.config import RANDOM_STATE, ROOT_DIR
from layer1_data_ingestion.loaders import load_secom

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Output directory ──────────────────────────────────────────────────────────
EDA_DIR = ROOT_DIR / "outputs" / "eda" / "secom"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, fname: str) -> Path:
    """Save figure, close it, return path."""
    path = EDA_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _median_impute(X: pd.DataFrame) -> np.ndarray:
    """Fill NaN with column median and return numpy array."""
    return X.fillna(X.median()).values


# ── Plot 1: Class Imbalance ───────────────────────────────────────────────────

def plot_class_imbalance(y: pd.Series) -> Path:
    """
    Visualise how severely imbalanced the SECOM dataset is and preview
    what SMOTE oversampling will do to the training distribution.

    RF insight: A 93/7 pass/fail split means a naive classifier that always
    predicts 'pass' achieves 93 % accuracy — useless for failure detection.
    SMOTE is essential; this plot shows why and by how much.
    """
    pass_count = (y == 0).sum()
    fail_count = (y == 1).sum()
    total      = len(y)
    ratio      = pass_count / fail_count
    synthetic  = pass_count - fail_count

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(top=0.82, wspace=0.35)

    # ── Left: bar chart ───────────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar(
        ["Pass", "Fail"],
        [pass_count, fail_count],
        color=["steelblue", "crimson"],
        edgecolor="white",
        width=0.45
    )
    for bar, count in zip(bars, [pass_count, fail_count]):
        pct = count / total * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom",
            fontweight="bold", fontsize=13
        )
    ax1.text(
        0.5, 0.92,
        f"Imbalance ratio  {ratio:.0f} : 1",
        transform=ax1.transAxes,
        ha="center", va="top",
        fontsize=11, color="darkred",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow", alpha=0.9)
    )
    ax1.set_title("Class Distribution", fontweight="bold", fontsize=13, pad=12)
    ax1.set_xlabel("Class", fontsize=11)
    ax1.set_ylabel("Sample Count", fontsize=11)
    ax1.set_ylim(0, pass_count * 1.25)
    ax1.tick_params(labelsize=11)
    sns.despine(ax=ax1)

    # ── Middle: pie chart ─────────────────────────────────
    ax2 = axes[1]
    sizes  = [pass_count, fail_count]
    colors = ["steelblue", "crimson"]
    wedge_props = dict(width=0.6, edgecolor="white", linewidth=2)
    wedges, texts, autotexts = ax2.pie(
        sizes,
        colors=colors,
        explode=(0, 0.08),
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=wedge_props,
        textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.legend(
        wedges,
        [f"Pass  ({pass_count})", f"Fail  ({fail_count})"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=11,
        frameon=False
    )
    ax2.set_title("Pass / Fail Split", fontweight="bold", fontsize=13, pad=12)

    # ── Right: before vs after SMOTE grouped bar ──────────
    ax3 = axes[2]
    x      = [0, 1]
    width  = 0.35

    ax3.bar(
        [xi - width/2 for xi in x],
        [pass_count, pass_count],
        width=width,
        color="steelblue",
        label="Pass",
        edgecolor="white"
    )
    ax3.bar(
        [xi + width/2 for xi in x],
        [fail_count, fail_count],
        width=width,
        color="crimson",
        label="Original Fail",
        edgecolor="white"
    )
    ax3.bar(
        [1 + width/2],
        [synthetic],
        width=width,
        bottom=[fail_count],
        color="orange",
        label=f"Synthetic SMOTE (+{synthetic})",
        edgecolor="white"
    )

    ax3.text(0, pass_count + fail_count + 40,
             f"Total\n{total}", ha="center", fontsize=10, fontweight="bold")
    ax3.text(1, pass_count + fail_count + synthetic + 40,
             f"Total\n{pass_count + fail_count + synthetic}",
             ha="center", fontsize=10, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(["Original", "After SMOTE"], fontsize=12)
    ax3.set_title("Before vs After SMOTE", fontweight="bold", fontsize=13, pad=12)
    ax3.set_xlabel("Dataset State", fontsize=11)
    ax3.set_ylabel("Sample Count", fontsize=11)
    ax3.set_ylim(0, pass_count * 1.35)
    ax3.legend(fontsize=10, loc="upper right")
    sns.despine(ax=ax3)

    fig.suptitle(
        "SECOM — Class Imbalance Analysis  (6.6% failure rate)",
        fontsize=16, fontweight="bold", y=0.97
    )
    return _save(fig, "secom_class_imbalance.png")


# ── Plot 2: PCA Explained Variance ───────────────────────────────────────────

def plot_pca_explained_variance(X: pd.DataFrame, y: pd.Series) -> Path:
    """
    Show how many PCA components are needed to capture 95 % of variance
    in SECOM's 562-feature space, and whether the first two components
    separate pass from fail.

    RF insight: If 95 % variance is captured in ~150 components (vs 562),
    we can reduce model input size by 73 % without losing predictive signal —
    critical for avoiding overfitting on 1,567 samples.
    """
    X_imp    = _median_impute(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_comp = min(100, X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_scaled)

    evr     = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ── Left: per-component explained variance ────────────────────────────────
    components = np.arange(1, n_comp + 1)
    ax1.plot(components, evr * 100, color="steelblue", linewidth=1.5)
    ax1.fill_between(components, 0, evr * 100, alpha=0.2, color="steelblue")

    # Mark elbow: largest second derivative
    d2    = np.diff(evr, n=2)
    elbow = int(np.argmax(d2) + 2)
    ax1.scatter(elbow, evr[elbow - 1] * 100, color="red", s=80, zorder=5)
    ax1.annotate(f"Elbow ~PC{elbow}",
                 xy=(elbow, evr[elbow - 1] * 100),
                 xytext=(elbow + 4, evr[elbow - 1] * 100 + 0.3),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 color="red", fontsize=9)
    ax1.set_title("Explained Variance per Component", fontweight="bold")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")

    # ── Middle: cumulative variance ───────────────────────────────────────────
    ax2.plot(components, cum_evr, color="steelblue", linewidth=2)
    ax2.fill_between(components, 0, cum_evr, alpha=0.2, color="steelblue")

    for threshold, color in [(80, "gray"), (90, "orange"), (95, "red")]:
        ax2.axhline(threshold, color=color, linestyle="--", linewidth=1,
                    label=f"{threshold}% variance")
        idx_cross = np.searchsorted(cum_evr, threshold)
        if idx_cross < len(components):
            ax2.scatter(components[idx_cross], cum_evr[idx_cross],
                        color=color, s=60, zorder=5)
            if threshold == 95:
                ax2.annotate(f"{components[idx_cross]} components\nfor 95%",
                             xy=(components[idx_cross], cum_evr[idx_cross]),
                             xytext=(components[idx_cross] + 5, cum_evr[idx_cross] - 8),
                             arrowprops=dict(arrowstyle="->", color="red"),
                             color="red", fontsize=9)

    ax2.set_title("Cumulative Explained Variance", fontweight="bold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)

    # ── Right: PC1 vs PC2 scatter ─────────────────────────────────────────────
    ax3.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
                c="steelblue", alpha=0.4, s=15, label="Pass", rasterized=True)
    ax3.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
                c="crimson", alpha=0.8, s=30, label="Fail", zorder=5)

    # Ellipses for each class
    for cls, color in [(0, "steelblue"), (1, "crimson")]:
        pts = X_pca[y == cls, :2]
        if len(pts) < 3:
            continue
        mean = pts.mean(axis=0)
        cov  = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(vals)
        ellipse = mpatches.Ellipse(mean, width, height, angle=angle,
                                   edgecolor=color, facecolor="none",
                                   linewidth=2, linestyle="--")
        ax3.add_patch(ellipse)

    ax3.set_title("PCA — First 2 Components (Pass vs Fail)", fontweight="bold")
    ax3.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax3.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax3.legend(fontsize=9)

    fig.suptitle("SECOM — PCA Dimensionality Reduction Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "secom_pca_analysis.png")


# ── Plot 3: Feature-Failure Correlation ──────────────────────────────────────

def plot_feature_failure_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 50,
) -> Path:
    """
    Rank every feature by its point-biserial correlation with the binary
    failure label and show inter-correlation among the top candidates.

    RF insight: Point-biserial correlation is the gold-standard measure of
    how strongly a continuous feature discriminates a binary outcome. Features
    at the top of this ranking are the first candidates for the final model
    feature set — before any PCA or regularisation is applied.
    """
    X_imp = X.fillna(X.median())

    corr_vals, feat_names = [], []
    for col in X_imp.columns:
        r, _ = stats.pointbiserialr(y, X_imp[col])
        if not np.isnan(r):
            corr_vals.append(r)
            feat_names.append(col)

    corr_df = (
        pd.DataFrame({"feature": feat_names, "correlation": corr_vals})
        .assign(abs_corr=lambda d: d["correlation"].abs())
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    top_df   = corr_df.head(top_n)
    top20_df = corr_df.head(20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ── Left: horizontal bar chart ────────────────────────────────────────────
    colors = ["crimson" if c > 0 else "steelblue" for c in top_df["correlation"]]
    ax1.barh(top_df["feature"][::-1], top_df["correlation"][::-1],
             color=colors[::-1], edgecolor="none", height=0.7)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Annotate top 10
    for i, (_, row) in enumerate(top_df.head(10).iterrows()):
        x_pos = row["correlation"] + (0.002 if row["correlation"] >= 0 else -0.002)
        ha    = "left" if row["correlation"] >= 0 else "right"
        ax1.text(x_pos, top_n - 1 - i, f"{row['correlation']:.3f}",
                 va="center", ha=ha, fontsize=7, fontweight="bold")

    ax1.set_title(f"Top {top_n} Features Correlated with Failure",
                  fontweight="bold", fontsize=11)
    ax1.set_xlabel("Point-biserial correlation with failure")
    ax1.set_ylabel("Feature")
    ax1.tick_params(axis="y", labelsize=7)

    pos_patch = mpatches.Patch(color="crimson",   label="Positive correlation")
    neg_patch = mpatches.Patch(color="steelblue", label="Negative correlation")
    ax1.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

    # ── Right: top-20 inter-correlation heatmap ───────────────────────────────
    top20_cols  = top20_df["feature"].tolist()
    corr_matrix = X_imp[top20_cols].corr()

    sns.heatmap(corr_matrix,
                annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.5, square=True,
                annot_kws={"size": 6},
                ax=ax2)
    ax2.set_title("Top 20 Features — Inter-correlation Matrix",
                  fontweight="bold", fontsize=11)
    ax2.set_xlabel("Feature")
    ax2.set_ylabel("Feature")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2.tick_params(axis="y", rotation=0,  labelsize=7)

    fig.suptitle("SECOM — Feature-Failure Correlation Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "secom_feature_correlation.png")


# ── Plot 4: Top Feature Distributions ────────────────────────────────────────

def plot_top_feature_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 12,
) -> Path:
    """
    Compare the distribution of the most failure-correlated features
    between passing and failing wafers using kernel density estimation.

    RF insight: A feature with clearly separated pass/fail KDE peaks is a
    strong discriminator and should be prioritised in the model. Wide
    overlap means the feature adds noise, not signal.
    """
    X_imp = X.fillna(X.median())

    # Rank features by point-biserial |correlation|
    records = []
    for col in X_imp.columns:
        r, _ = stats.pointbiserialr(y, X_imp[col])
        if not np.isnan(r):
            records.append({"feature": col, "abs_corr": abs(r)})

    top_feats = (
        pd.DataFrame(records)
        .sort_values("abs_corr", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )

    rows, cols = 3, 4
    fig, axes  = plt.subplots(rows, cols, figsize=(20, 12))
    axes       = axes.flatten()

    X_pass = X_imp[y == 0]
    X_fail = X_imp[y == 1]

    for idx, feat in enumerate(top_feats):
        ax = axes[idx]

        pass_vals = X_pass[feat].dropna()
        fail_vals = X_fail[feat].dropna()

        # KDE curves
        pass_vals.plot.kde(ax=ax, color="steelblue", linewidth=2, label="Pass")
        fail_vals.plot.kde(ax=ax, color="crimson",   linewidth=2, label="Fail")

        # Fill under curves
        kde_x = np.linspace(min(pass_vals.min(), fail_vals.min()),
                            max(pass_vals.max(), fail_vals.max()), 300)
        from scipy.stats import gaussian_kde
        if len(pass_vals) > 1:
            ax.fill_between(kde_x,
                            gaussian_kde(pass_vals)(kde_x),
                            alpha=0.25, color="steelblue")
        if len(fail_vals) > 1:
            ax.fill_between(kde_x,
                            gaussian_kde(fail_vals)(kde_x),
                            alpha=0.35, color="crimson")

        # Mean lines
        pass_mean = pass_vals.mean()
        fail_mean = fail_vals.mean()
        ax.axvline(pass_mean, color="steelblue", linestyle="--", linewidth=1.2,
                   label=f"Pass μ={pass_mean:.2f}")
        ax.axvline(fail_mean, color="crimson",   linestyle="--", linewidth=1.2,
                   label=f"Fail μ={fail_mean:.2f}")

        # Separation score
        pooled_std = np.sqrt((pass_vals.std() ** 2 + fail_vals.std() ** 2) / 2 + 1e-8)
        sep        = abs(fail_mean - pass_mean) / pooled_std
        ax.set_title(f"{feat}\n(sep={sep:.2f})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for idx in range(top_n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("SECOM — Top Feature Distributions: Pass vs Fail",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "secom_feature_distributions.png")


# ── Plot 5: Missing Values vs Failure ────────────────────────────────────────

def plot_missing_vs_failure(X: pd.DataFrame, y: pd.Series) -> Path:
    """
    Test whether samples with more missing values are more likely to fail.

    RF insight: If failing wafers have systematically higher missing rates,
    missingness is itself a failure signal and should be encoded as a feature
    (e.g. a 'missing_pct' column) rather than just imputed away.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    miss_pct = X.isna().mean(axis=1) * 100
    df_plot  = pd.DataFrame({
        "missing_pct": miss_pct.values,
        "label":       y.values,
        "class_name":  y.map({0: "Pass", 1: "Fail"}).values,
    })

    pass_miss = df_plot.loc[df_plot["label"] == 0, "missing_pct"]
    fail_miss = df_plot.loc[df_plot["label"] == 1, "missing_pct"]
    overall_mean = miss_pct.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: box plot + jitter ───────────────────────────────────────────────
    box_data   = [pass_miss.values, fail_miss.values]
    box_colors = ["steelblue", "crimson"]
    bp = ax1.boxplot(box_data, patch_artist=True, widths=0.4,
                     showfliers=False, positions=[0, 1])
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Jittered scatter
    for i, (vals, color) in enumerate(zip(box_data, box_colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax1.scatter(i + jitter, vals, alpha=0.3, s=12, color=color, zorder=3)

    # Median annotations
    for i, vals in enumerate(box_data):
        med = np.median(vals)
        ax1.text(i, med + 0.4, f"median={med:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Pass", "Fail"], fontsize=11)
    ax1.set_title("Missing Values per Sample: Pass vs Fail", fontweight="bold")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Missing Features (%)")

    # ── Right: scatter — missing % vs sample index ────────────────────────────
    idx_arr    = np.arange(len(df_plot))
    color_pts  = df_plot["label"].map({0: "steelblue", 1: "crimson"}).values

    ax2.scatter(df_plot["missing_pct"], idx_arr,
                c=color_pts, alpha=0.4, s=10, rasterized=True)
    ax2.axvline(overall_mean, color="black", linestyle="--", linewidth=1.2,
                label=f"Mean missing = {overall_mean:.1f}%")

    pass_patch = mpatches.Patch(color="steelblue", label="Pass")
    fail_patch = mpatches.Patch(color="crimson",   label="Fail")
    ax2.legend(handles=[pass_patch, fail_patch, ax2.lines[0]], fontsize=9)
    ax2.set_title("Sample Missing % Distribution by Class", fontweight="bold")
    ax2.set_xlabel("Missing Features (%)")
    ax2.set_ylabel("Sample Index")

    fig.suptitle("SECOM — Are Missing Values Related to Failures?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "secom_missing_vs_failure.png")


# ── Plot 6: Feature Variance Overview ────────────────────────────────────────

def plot_feature_variance_overview(X: pd.DataFrame, y: pd.Series) -> Path:
    """
    Show the variance distribution across all ~562 features to identify
    near-zero-variance features that contribute noise rather than signal.

    RF insight: VarianceThreshold in the preprocessing pipeline drops
    constant features. This plot reveals how many features are affected
    and which feature indices carry the most dynamic range — guiding both
    the threshold choice and initial feature selection.
    """
    X_imp = X.fillna(X.median())
    variances  = X_imp.var()
    low_thresh = 0.01
    n_low      = (variances < low_thresh).sum()
    top50_idx  = variances.nlargest(50).index

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: histogram of variances (log scale) ──────────────────────────────
    var_vals = variances.values
    low_mask = var_vals < low_thresh

    ax1.hist(var_vals[~low_mask], bins=60, color="steelblue",
             alpha=0.75, log=False, label="Informative")
    ax1.hist(var_vals[low_mask],  bins=20, color="crimson",
             alpha=0.85, log=False, label=f"Near-zero (<{low_thresh})")
    ax1.axvline(low_thresh, color="darkred", linestyle="--", linewidth=1.5,
                label="Low variance threshold")
    ax1.set_xscale("log")
    ax1.text(0.97, 0.95, f"{n_low} near-zero\nvariance features",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=10, color="darkred",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax1.set_title("Distribution of Feature Variances (log scale)", fontweight="bold")
    ax1.set_xlabel("Variance (log scale)")
    ax1.set_ylabel("Number of Features")
    ax1.legend(fontsize=9)

    # ── Right: variance by feature index ─────────────────────────────────────
    feat_idx = np.arange(len(variances))

    # Gray for regular, green for top-50, red for near-zero
    colors_var = []
    for i, (fname, v) in enumerate(variances.items()):
        if v < low_thresh:
            colors_var.append("crimson")
        elif fname in top50_idx:
            colors_var.append("#2ca02c")
        else:
            colors_var.append("lightgray")

    ax2.scatter(feat_idx, variances.values,
                c=colors_var, s=8, alpha=0.7, rasterized=True)
    ax2.axhline(low_thresh, color="darkred", linestyle="--", linewidth=1.2,
                label=f"Threshold = {low_thresh}")
    ax2.set_yscale("log")

    gray_patch  = mpatches.Patch(color="lightgray",  label="Regular features")
    green_patch = mpatches.Patch(color="#2ca02c",     label="Top-50 by variance")
    red_patch   = mpatches.Patch(color="crimson",     label=f"Near-zero (<{low_thresh})")
    ax2.legend(handles=[gray_patch, green_patch, red_patch], fontsize=8)

    ax2.set_title("Feature Variance by Index", fontweight="bold")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Variance (log scale)")

    fig.suptitle(f"SECOM — Feature Variance Overview ({len(variances)} features)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "secom_feature_variance.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def run_secom_eda() -> List[str]:
    """
    Load SECOM data and execute all 6 EDA plots in order.

    Returns
    -------
    list[str]
        Paths to all saved PNG files.
    """
    logger.info("[EDA | SECOM] Loading dataset...")
    data = load_secom()
    X: pd.DataFrame = data["X"]
    y: pd.Series    = data["y"]

    plot_funcs = [
        ("Class imbalance analysis",       lambda: plot_class_imbalance(y)),
        ("PCA explained variance",         lambda: plot_pca_explained_variance(X, y)),
        ("Feature-failure correlation",    lambda: plot_feature_failure_correlation(X, y)),
        ("Top feature distributions",      lambda: plot_top_feature_distributions(X, y)),
        ("Missing values vs failure",      lambda: plot_missing_vs_failure(X, y)),
        ("Feature variance overview",      lambda: plot_feature_variance_overview(X, y)),
    ]

    saved: List[str] = []

    for i, (name, func) in enumerate(plot_funcs, start=1):
        logger.info(f"  Generating SECOM plot {i}/{len(plot_funcs)}: {name}")
        try:
            path = func()
            saved.append(str(path))
        except Exception as exc:
            logger.error(f"  Plot '{name}' failed: {exc}")

    logger.success(
        f"[EDA | SECOM] Done — "
        f"total plots saved: {len(saved)} | "
        f"output folder: {EDA_DIR}"
    )
    return saved


if __name__ == "__main__":
    run_secom_eda()
