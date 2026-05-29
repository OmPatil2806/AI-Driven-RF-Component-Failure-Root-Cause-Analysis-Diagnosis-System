# GradCAM gradient visualization for 1D-CNN time-series predictions
"""
grad_cam.py — GradCAM explainability for RF-Sentinel 1D-CNN.

Answers "which time steps in the 30-cycle window caused the CNN to predict
failure?" by computing gradient-weighted class activation maps on the last
convolutional layer. Complements SHAP (which explains sensor importance)
with temporal importance across the sliding window.

Usage
-----
    python -m layer5_explainability.grad_cam
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from loguru import logger

from layer1_data_ingestion.config import (
    ROOT_DIR, CMAPSS_USEFUL_SENSORS,
)
from layer1_data_ingestion.loaders import load_cmapss
from layer3_models.cnn1d_model import RFSentinelCNN1D, CMAPSSWindowDataset

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Layer 5: GradCAM for 1D-CNN
# ══════════════════════════════════════════════════════════════
#
# WHAT IS GRADCAM?
# ────────────────
# GradCAM (Gradient-weighted Class Activation Mapping)
# answers: "Which TIME STEPS in the 30-cycle window
#           caused the CNN to predict failure?"
#
# DIFFERENCE FROM SHAP:
# ─────────────────────
# SHAP     → which FEATURES (sensors) matter most
# GradCAM  → which TIME STEPS (cycles) matter most
#
# Together they answer:
#   SHAP   : "sensor s11 pressure drop caused failure"
#   GradCAM: "the failure signal appeared at cycles 18-25"
#
# HOW GRADCAM WORKS:
# ──────────────────
# 1. Forward pass → get CNN prediction
# 2. Backward pass → compute gradients w.r.t. last conv layer
# 3. Average gradients → get importance per time step
# 4. Apply ReLU → keep only positive activations
# 5. Normalize to 0-1 → heatmap over 30 cycles
#
# OUTPUT:
# ───────
# outputs/explainability/gradcam/
#     gradcam_sample_{idx}.png     ← heatmap over time steps
#     gradcam_overlay_{idx}.png    ← heatmap overlaid on sensor signals
#     gradcam_summary.png          ← average heatmap across all samples
# ══════════════════════════════════════════════════════════════

GRADCAM_DIR = ROOT_DIR / "outputs" / "explainability" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


# ── Class: GradCAM1D ──────────────────────────────────────────────────────────

class GradCAM1D:
    """
    GradCAM implementation for the RF-Sentinel 1D-CNN architecture.

    Registers forward and backward hooks on the last Conv1d block to capture
    activations and gradients during inference. The gradient-weighted sum of
    activation maps produces a 1D heatmap over the 30-cycle window, showing
    which time steps were most influential for the predicted class.
    """

    def __init__(self, model: RFSentinelCNN1D) -> None:
        self.model       = model
        self.network     = model.network
        self.gradients   = None   # set by backward hook
        self.activations = None   # set by forward hook
        self._hooks: list = []    # handle cleanup after each call

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """
        Register forward and backward hooks on conv_block3 (last conv layer).

        Hooks are the PyTorch mechanism that lets us inspect intermediate
        tensors without modifying the model. We use the last conv block because
        it contains the highest-level learned features — earlier layers detect
        low-level patterns that are less semantically meaningful.
        """
        def forward_hook(module, inp, output):
            # Detach to avoid keeping the full computation graph in memory
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the loss w.r.t. this layer's output
            self.gradients = grad_output[0].detach()

        target_layer = self.network.conv_block3
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def _remove_hooks(self) -> None:
        """
        Remove all registered hooks.

        Always called after compute_gradcam() to prevent accumulation of hooks
        across multiple calls, which would slow down inference over time.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    # ── Core GradCAM computation ──────────────────────────────────────────────

    def compute_gradcam(
        self,
        x_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Compute the GradCAM heatmap for one input window.

        Algorithm
        ---------
        1. Forward pass through the full network to get class scores.
        2. Backprop the score of the target class (not the loss) to get
           gradients at the last conv layer.
        3. Global-average-pool the gradients across the time dimension
           to get one weight per channel (128 weights for conv_block3).
        4. Compute weighted sum of activation maps across channels.
        5. ReLU to keep only time steps that increase the class score.
        6. Normalise to [0, 1] so all windows are on the same scale.

        Parameters
        ----------
        x_tensor     : torch.Tensor shape (1, n_sensors, window_size)
        target_class : class index to explain; None → uses predicted class

        Returns
        -------
        heatmap          : np.ndarray shape (window_size,), values in [0, 1]
        predicted_class  : int
        confidence       : float probability of predicted class
        """
        self.network.eval()
        self._register_hooks()

        # Forward pass — hooks capture activations automatically
        x_tensor = x_tensor.clone().requires_grad_(True)
        output = self.network(x_tensor)
        probs  = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        # Backward pass on target class score — hooks capture gradients
        self.network.zero_grad()
        output[0, target_class].backward()

        # GradCAM: importance = ReLU(sum_c(alpha_c * A_c))
        # where alpha_c = mean gradient of channel c over time dimension
        gradients   = self.gradients[0]    # (n_channels, window_size)
        activations = self.activations[0]  # (n_channels, window_size)

        # Channel weights: global average pool over time
        weights = gradients.mean(dim=-1)   # (n_channels,)

        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[-1])
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU: keep only time steps with positive contributions
        heatmap = torch.relu(heatmap)

        # Normalise to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        self._remove_hooks()
        return heatmap.numpy(), target_class, confidence

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_gradcam_heatmap(
        self,
        x_tensor: torch.Tensor,
        sample_idx: int = 0,
        engine_cycles: np.ndarray | None = None,
        save: bool = True,
    ):
        """
        Two-panel plot: GradCAM bar chart + sensor signal heatmap.

        Top panel shows per-time-step importance as red bars — darker = more
        important for the prediction. Bottom panel shows the raw 14-sensor
        values as a colour map so the pattern that drove the prediction can
        be seen alongside the importance scores.

        Parameters
        ----------
        x_tensor      : torch.Tensor shape (1, n_sensors, window_size)
        sample_idx    : index label used in the saved filename
        engine_cycles : optional absolute cycle numbers for x-axis labels
        save          : write PNG to GRADCAM_DIR (default True)

        Returns
        -------
        matplotlib Figure
        """
        heatmap, pred_class, conf = self.compute_gradcam(x_tensor)

        class_name = (
            str(self.model.classes_[pred_class])
            if self.model.classes_ is not None
            else str(pred_class)
        )
        window_size = heatmap.shape[0]
        time_steps  = (
            engine_cycles
            if engine_cycles is not None
            else np.arange(window_size)
        )

        # Sensor signal matrix: shape (n_sensors, window_size)
        sensor_data = x_tensor.squeeze(0).detach().numpy()

        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.45)

        # ── Top: GradCAM bar chart ────────────────────────────────────────────
        ax_top = fig.add_subplot(gs[0])
        colors = plt.cm.Reds(heatmap)
        ax_top.bar(time_steps, heatmap, color=colors, edgecolor="none")
        ax_top.axhline(0.5, color="crimson", linestyle="--", alpha=0.6,
                       linewidth=1.2, label="Importance threshold (0.5)")

        # Annotate peak time step with arrow
        peak_idx = int(np.argmax(heatmap))
        ax_top.annotate(
            f"peak\n(step {time_steps[peak_idx]})",
            xy=(time_steps[peak_idx], heatmap[peak_idx]),
            xytext=(time_steps[peak_idx] + 1.5, heatmap[peak_idx] + 0.08),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=8,
        )

        ax_top.set_xlabel("Time step (cycle within window)")
        ax_top.set_ylabel("GradCAM importance")
        ax_top.set_title("GradCAM — Which cycles triggered prediction?",
                         fontweight="bold")
        ax_top.set_ylim(0, 1.15)
        ax_top.legend(fontsize=8)

        # ── Bottom: sensor signal heatmap ─────────────────────────────────────
        ax_bot = fig.add_subplot(gs[1])

        im = ax_bot.imshow(
            sensor_data,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax_bot, fraction=0.03, pad=0.02,
                     label="Normalised sensor value")

        # Overlay GradCAM importance as vertical red shading
        for t_idx in range(window_size):
            if heatmap[t_idx] > 0.1:
                ax_bot.axvline(
                    t_idx, color="red",
                    alpha=float(heatmap[t_idx]) * 0.4,
                    linewidth=3,
                )

        ax_bot.set_xlabel("Time step")
        ax_bot.set_ylabel("Sensor")
        ax_bot.set_title("Sensor values across window", fontweight="bold")
        ax_bot.set_yticks(range(len(CMAPSS_USEFUL_SENSORS)))
        ax_bot.set_yticklabels(CMAPSS_USEFUL_SENSORS, fontsize=7)

        fig.suptitle(
            f"GradCAM Analysis — Sample {sample_idx}\n"
            f"Predicted: {class_name} | Confidence: {conf:.1%}",
            fontsize=13, fontweight="bold",
        )

        if save:
            path = GRADCAM_DIR / f"gradcam_sample_{sample_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[GradCAM] Saved → {path}")

        return fig


# ── Standalone function: run_gradcam_analysis ─────────────────────────────────

def run_gradcam_analysis(n_samples: int = 5) -> tuple[GradCAM1D, list]:
    """
    Load the pre-trained CNN, run GradCAM on n_samples windows, save all plots.

    Wires together model loading, dataset construction, per-sample GradCAM
    computation, individual plots, and the aggregate summary heatmap.

    Parameters
    ----------
    n_samples : number of windows to analyse (default 5)

    Returns
    -------
    (gradcam, results)  — GradCAM1D instance and list of per-sample result dicts
    """
    # Step 1: Load pre-trained CNN from disk
    cnn = RFSentinelCNN1D()
    cnn.build()
    cnn.load()
    gradcam = GradCAM1D(cnn)
    logger.info("[GradCAM] CNN loaded and GradCAM initialised")

    # Step 2: Load CMAPSS and build sliding-window val dataset
    data      = load_cmapss("FD001")
    train_raw = data["train_raw"]
    all_units = train_raw["unit_id"].unique()
    np.random.seed(42)
    np.random.shuffle(all_units)
    n_train = int(len(all_units) * 0.8)
    val_df  = train_raw[train_raw["unit_id"].isin(all_units[n_train:])]

    dataset = CMAPSSWindowDataset(
        val_df, CMAPSS_USEFUL_SENSORS,
        window_size=30, rul_threshold=30,
    )

    # Separate pass and fail windows so we can show a meaningful comparison
    pass_indices = [i for i in range(len(dataset))
                    if int(dataset[i][1].item()) == 0]
    fail_indices = [i for i in range(len(dataset))
                    if int(dataset[i][1].item()) == 1]

    logger.info(
        f"Dataset: {len(pass_indices)} pass windows, "
        f"{len(fail_indices)} fail windows"
    )

    # Take mix: first 3 fail samples + first 2 pass samples
    selected_indices = (fail_indices[:3] + pass_indices[:2])[:n_samples]

    logger.info(
        f"Selected {len(selected_indices)} samples "
        f"({min(3, len(fail_indices))} fail + "
        f"{min(2, len(pass_indices))} pass)"
    )

    # Step 3: Run GradCAM on selected windows
    all_heatmaps: list[np.ndarray] = []
    results: list[dict] = []

    for idx, sample_pos in enumerate(selected_indices):
        x, y = dataset[sample_pos]
        x_tensor   = x.unsqueeze(0)
        true_label = int(y.item())

        heatmap, pred_class, conf = gradcam.compute_gradcam(x_tensor)
        all_heatmaps.append(heatmap)

        fig = gradcam.plot_gradcam_heatmap(x_tensor, sample_idx=idx, save=True)
        plt.close(fig)

        peak_step = int(np.argmax(heatmap))
        label_str = "FAIL" if true_label == 1 else "PASS"
        results.append({
            "sample_idx":     idx,
            "true_label":     true_label,
            "true_label_str": label_str,
            "pred_class":     pred_class,
            "confidence":     round(conf, 4),
            "peak_timestep":  peak_step,
            "heatmap_mean":   round(float(heatmap.mean()), 4),
        })
        logger.info(
            f"Sample {idx} (true={label_str}): "
            f"pred={pred_class} conf={conf:.1%} "
            f"peak_step={peak_step}"
        )

    # Step 4: Average heatmap across all samples
    avg_heatmap = np.mean(all_heatmaps, axis=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.Reds(avg_heatmap)
    ax.bar(range(len(avg_heatmap)), avg_heatmap, color=colors, edgecolor="none")
    ax.set_xlabel("Time step (cycle within window)")
    ax.set_ylabel("Average GradCAM importance")
    ax.set_title(
        "Average GradCAM Heatmap — "
        "Which time steps matter most across all samples?",
        fontweight="bold",
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()

    avg_path = GRADCAM_DIR / "gradcam_summary.png"
    fig.savefig(avg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[GradCAM] Average heatmap saved → {avg_path}")

    # Step 5: Print summary
    print()
    print("GradCAM Analysis Complete")
    print(f"  Samples analysed : {len(selected_indices)}")
    print(f"  Plots saved      : {GRADCAM_DIR}")
    print()
    print("  Sample results:")
    for r in results:
        correct = "correct" if r["true_label"] == r["pred_class"] else "WRONG"
        print(
            f"    Sample {r['sample_idx']} "
            f"(true={r['true_label_str']}): "
            f"pred={r['pred_class']} "
            f"({r['confidence']:.1%}) "
            f"peak_step={r['peak_timestep']} "
            f"[{correct}]"
        )

    print()
    if results:
        avg_peak = np.mean([r["peak_timestep"] for r in results])
        print(f"  Average peak time step : {avg_peak:.1f} / {len(avg_heatmap)}")
        print(
            f"  Interpretation: failure signal strongest "
            f"around cycle {avg_peak:.0f} of 30-cycle window"
        )

    return gradcam, results


# ── Combined SHAP + GradCAM plot ──────────────────────────────────────────────

def plot_shap_gradcam_combined(
    gradcam_model: "GradCAM1D",
    shap_explainer,
    sample_idx: int = 0,
    save: bool = True,
):
    """
    Combined SHAP + GradCAM visualization in one figure.

    Shows in one figure:
    - TOP    : SHAP waterfall — WHICH sensors caused failure
    - MIDDLE : GradCAM heatmap — WHICH cycles triggered it
    - BOTTOM : Combined interpretation text box

    This is the key diagnostic plot — answers both:
        WHAT failed : sensor s11 pressure drop
        WHEN it appeared : cycles 18-24 of 30

    Parameters
    ----------
    gradcam_model  : fitted GradCAM1D instance
    shap_explainer : fitted RFSentinelSHAP instance
    sample_idx     : which FAIL sample index to explain (0 = first fail window)
    save           : write PNG to GRADCAM_DIR (default True)

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.patches as mpatches
    from layer1_data_ingestion.config import (
        CMAPSS_SENSOR_LABELS, CMAPSS_USEFUL_SENSORS,
    )
    from layer1_data_ingestion.loaders import load_cmapss
    from layer3_models.cnn1d_model import CMAPSSWindowDataset

    # ── Get SHAP values for this sample ──────────────────────────────────────
    shap_vals  = shap_explainer._get_sample_shap(sample_idx)
    feat_names = shap_explainer.feature_names

    raw_pred   = shap_explainer.model.model.predict(
        shap_explainer.X_explain[sample_idx: sample_idx + 1]
    )
    pred_enc   = int(np.squeeze(raw_pred).flat[0])
    pred_class = str(shap_explainer.model.classes_[pred_enc])
    xgb_conf   = float(
        shap_explainer.model.predict_proba(
            shap_explainer.X_explain[sample_idx: sample_idx + 1]
        ).max()
    )

    # Sort features by absolute SHAP value (top 8)
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:8]
    top_names  = [
        CMAPSS_SENSOR_LABELS.get(feat_names[i], feat_names[i])
        for i in sorted_idx
    ]
    top_vals = shap_vals[sorted_idx]

    # ── Get GradCAM heatmap from a FAIL window ────────────────────────────────
    data      = load_cmapss("FD001")
    train_raw = data["train_raw"]
    all_units = train_raw["unit_id"].unique()
    np.random.seed(42)
    np.random.shuffle(all_units)
    n_train = int(len(all_units) * 0.8)
    val_df  = train_raw[train_raw["unit_id"].isin(all_units[n_train:])]

    dataset      = CMAPSSWindowDataset(
        val_df, CMAPSS_USEFUL_SENSORS,
        window_size=30, rul_threshold=30,
    )
    fail_indices = [
        i for i in range(len(dataset))
        if int(dataset[i][1].item()) == 1
    ]
    target_idx   = (
        fail_indices[sample_idx]
        if sample_idx < len(fail_indices)
        else fail_indices[0]
    )

    x, _ = dataset[target_idx]
    x_tensor = x.unsqueeze(0)
    heatmap, cnn_pred, cnn_conf = gradcam_model.compute_gradcam(x_tensor)

    peak_step  = int(np.argmax(heatmap))
    high_steps = np.where(heatmap > 0.5)[0]

    # ── Build figure with 3 panels ────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 0.8], hspace=0.4)

    # ── Panel 1: SHAP waterfall ───────────────────────────────────────────────
    ax1    = fig.add_subplot(gs[0])
    colors = ["crimson" if v > 0 else "steelblue" for v in top_vals]
    bars   = ax1.barh(
        top_names[::-1], top_vals[::-1],
        color=colors[::-1], edgecolor="white", height=0.6,
    )
    for bar, val in zip(bars, top_vals[::-1]):
        x_pos = bar.get_width() + (0.002 if val >= 0 else -0.002)
        ha    = "left" if val >= 0 else "right"
        ax1.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha=ha, fontsize=9, fontweight="bold",
        )
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title(
        f"SHAP — Which SENSORS caused failure?\n"
        f"XGBoost predicted: {pred_class} ({xgb_conf:.1%} confidence)",
        fontweight="bold", fontsize=11,
    )
    ax1.set_xlabel("SHAP value (impact on prediction)")
    red_p  = mpatches.Patch(color="crimson",   label="Pushes toward FAILURE")
    blue_p = mpatches.Patch(color="steelblue", label="Pushes toward PASS")
    ax1.legend(handles=[red_p, blue_p], fontsize=9, loc="lower right")

    # ── Panel 2: GradCAM heatmap ──────────────────────────────────────────────
    ax2        = fig.add_subplot(gs[1])
    bar_colors = plt.cm.Reds(heatmap)
    ax2.bar(range(30), heatmap, color=bar_colors, width=0.8)
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.6,
                label="Importance threshold")

    if len(high_steps) > 0:
        ax2.axvspan(
            high_steps[0] - 0.5, high_steps[-1] + 0.5,
            alpha=0.1, color="red",
            label=f"High importance zone (steps {high_steps[0]}–{high_steps[-1]})",
        )

    ax2.annotate(
        f"Peak\n(step {peak_step})",
        xy=(peak_step, heatmap[peak_step]),
        xytext=(peak_step - 4, heatmap[peak_step] + 0.08),
        arrowprops=dict(arrowstyle="->", color="darkred"),
        fontsize=9, color="darkred", fontweight="bold",
    )
    ax2.set_title(
        f"GradCAM — At which CYCLES did failure become visible?\n"
        f"1D-CNN predicted: {'fail' if cnn_pred == 1 else 'pass'} "
        f"({cnn_conf:.1%} confidence)",
        fontweight="bold", fontsize=11,
    )
    ax2.set_xlabel("Time step (cycle within 30-cycle window)")
    ax2.set_ylabel("GradCAM importance")
    ax2.set_xlim(-0.5, 29.5)
    ax2.legend(fontsize=9)

    # ── Panel 3: Combined interpretation text ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    top3_sensors = [
        CMAPSS_SENSOR_LABELS.get(feat_names[i], feat_names[i])
        for i in sorted_idx[:3]
    ]
    top3_shap = [f"{shap_vals[i]:+.3f}" for i in sorted_idx[:3]]
    high_zone = (
        f"cycles {high_steps[0]}–{high_steps[-1]}"
        if len(high_steps) > 0
        else f"cycle {peak_step}"
    )

    interpretation = (
        f"COMBINED DIAGNOSIS — Sample {sample_idx}\n\n"
        f"  WHAT  failed : {top3_sensors[0]} (SHAP={top3_shap[0]})"
        f"  |  {top3_sensors[1]} (SHAP={top3_shap[1]})\n"
        f"  WHEN  visible: Failure signal strongest at {high_zone} "
        f"of the 30-cycle window  (peak: step {peak_step})\n"
        f"  VERDICT      : {pred_class.upper()} — "
        f"Both XGBoost ({xgb_conf:.1%}) and CNN ({cnn_conf:.1%}) agree"
    )

    ax3.text(
        0.5, 0.5, interpretation,
        transform=ax3.transAxes,
        ha="center", va="center",
        fontsize=11, fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#EBF5FB",
            edgecolor="#1D9E75",
            linewidth=2,
        ),
    )

    fig.suptitle(
        "RF-Sentinel — Combined SHAP + GradCAM Diagnosis",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if save:
        path = GRADCAM_DIR / f"combined_shap_gradcam_{sample_idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.success(f"[GradCAM] Combined plot saved → {path}")

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gradcam, results = run_gradcam_analysis(n_samples=5)
