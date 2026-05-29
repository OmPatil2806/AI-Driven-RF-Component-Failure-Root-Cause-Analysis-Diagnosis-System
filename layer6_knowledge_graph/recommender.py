# Ranked P1/P2/P3 repair actions with confidence scores and pyvis HTML graph
"""
recommender.py — Final repair action ranker for RF-Sentinel.

Takes repair paths from shap_to_graph.py and produces clean P1/P2/P3 ranked
diagnosis reports for the engineer. Combines graph-based confidence with SHAP
evidence to score and deduplicate repair actions.

Usage
-----
    python -m layer6_knowledge_graph.recommender
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from loguru import logger

from layer1_data_ingestion.config import ROOT_DIR
from layer6_knowledge_graph.build_graph import build_rf_knowledge_graph
from layer6_knowledge_graph.shap_to_graph import SHAPToGraphLinker

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Layer 6: Repair Recommender
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# Final step of the RF-Sentinel pipeline.
# Takes repair paths from shap_to_graph.py and produces
# a clean ranked diagnosis report for the engineer.
#
# OUTPUT FORMAT:
# ──────────────
# P1 (Priority 1) → do this FIRST — highest confidence
# P2 (Priority 2) → do this SECOND if P1 not conclusive
# P3 (Priority 3) → do this LAST — lowest confidence
#
# FULL PIPELINE:
# ──────────────
# Raw sensor data
#     ↓ Layer 1
# Clean features
#     ↓ Layer 3
# XGBoost prediction + probabilities
#     ↓ Layer 5
# SHAP top-3 features
#     ↓ Layer 6 shap_to_graph.py
# Repair paths
#     ↓ Layer 6 recommender.py (THIS FILE)
# Final diagnosis report → engineer
# ══════════════════════════════════════════════════════════════

REPORT_DIR = ROOT_DIR / "outputs" / "knowledge_graph"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Class: RFSentinelRecommender ──────────────────────────────────────────────

class RFSentinelRecommender:
    """
    End-to-end repair recommender for RF-Sentinel diagnostics.

    Combines the knowledge graph (Layer 6) with SHAP evidence (Layer 5)
    to produce ranked, human-readable repair actions. The recommender
    deduplicates repair nodes that appear across multiple paths and assigns
    a final P1/P2/P3 label based on combined confidence score.
    """

    def __init__(self) -> None:
        self.G      = build_rf_knowledge_graph()
        self.linker = SHAPToGraphLinker(self.G)
        logger.info("[Recommender] Initialized with knowledge graph")

    # ── Method 1: recommend ───────────────────────────────────────────────────

    def recommend(
        self,
        top_features:    list[str],
        shap_values:     list[float],
        predicted_class: str,
        confidence:      float,
        device_id:       str = "unknown",
        save_report:     bool = True,
    ) -> dict:
        """
        Main method — produces a complete P1/P2/P3 diagnosis report.

        Pipeline:
            1. Traverse graph to get raw repair paths (via SHAPToGraphLinker)
            2. Deduplicate repair nodes; aggregate scores across paths
            3. Re-rank unique repairs by combined score
            4. Assign P1 / P2 / P3 labels to top-5 repairs
            5. Build structured report dict and optionally save JSON

        Parameters
        ----------
        top_features    : SHAP feature names (e.g. ["s11", "s9"])
        shap_values     : corresponding signed SHAP values
        predicted_class : model-predicted failure class label
        confidence      : predicted class probability (0–1)
        device_id       : identifier of the device under test
        save_report     : write JSON to REPORT_DIR (default True)

        Returns
        -------
        dict  structured diagnosis report
        """
        # Step 1: Get repair paths from graph
        result = self.linker.get_repair_paths(
            top_features, shap_values, max_paths=15,
        )
        paths = result["repair_paths"]

        # Step 2: Deduplicate repair nodes — take max score per unique repair
        repair_agg: dict[str, dict] = {}
        for path in paths:
            node = path["repair_node"]
            if node not in repair_agg or path["final_score"] > repair_agg[node]["score"]:
                repair_agg[node] = {
                    "repair_node":   node,
                    "repair_label":  path["repair_label"],
                    "cause_label":   path["cause_label"],
                    "priority":      self.G.nodes[node].get("priority", "P3"),
                    "estimated_time":self.G.nodes[node].get("estimated_time", "unknown"),
                    "score":         path["final_score"],
                }

        ranked_repairs = sorted(repair_agg.values(), key=lambda r: r["score"], reverse=True)[:5]

        # Step 3: Assign display priority labels (P1 → P2 → P3 for remaining)
        priority_labels = ["P1", "P2", "P3"] + ["P3"] * (len(ranked_repairs) - 3)
        for action, label in zip(ranked_repairs, priority_labels):
            action["priority"] = label

        primary_cause = paths[0]["cause_label"] if paths else "Unknown"
        timestamp     = datetime.now().isoformat()

        # Step 4: Build SHAP evidence list
        mapped_symptoms = result.get("mapped_symptoms", [])
        shap_evidence = [
            {
                "feature":   feat,
                "shap_value": round(float(val), 3),
                "symptom":   (
                    mapped_symptoms[i]["symptom_label"]
                    if i < len(mapped_symptoms)
                    else feat
                ),
                "direction": "toward_failure" if float(val) < 0 else "toward_pass",
            }
            for i, (feat, val) in enumerate(zip(top_features, shap_values))
        ]

        # Step 5: Assemble report
        report = {
            "timestamp":        timestamp,
            "device_id":        device_id,
            "predicted_failure": predicted_class,
            "model_confidence": round(confidence * 100, 1),
            "primary_cause":    primary_cause,
            "shap_evidence":    shap_evidence,
            "repair_actions":   [
                {
                    "rank":           i + 1,
                    "priority":       action["priority"],
                    "action":         action["repair_label"],
                    "cause":          action["cause_label"],
                    "confidence_pct": round(action["score"] * 100, 1),
                    "estimated_time": action["estimated_time"],
                }
                for i, action in enumerate(ranked_repairs)
            ],
            "knowledge_paths": paths[:5],
            "graph_stats": {
                "symptom_nodes_matched": len(result["mapped_symptoms"]),
                "total_paths_found":     len(paths),
            },
        }

        # Step 6: Save JSON report
        if save_report:
            ts_slug   = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = REPORT_DIR / f"diagnosis_{device_id}_{ts_slug}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"[Recommender] Report saved → {json_path}")

        # Step 7: Log formatted summary
        sep   = "=" * 55
        acts  = report["repair_actions"]
        p1    = acts[0] if len(acts) > 0 else {"action": "—", "estimated_time": "—"}
        p2    = acts[1] if len(acts) > 1 else {"action": "—", "estimated_time": "—"}
        p3    = acts[2] if len(acts) > 2 else {"action": "—", "estimated_time": "—"}

        logger.success(sep)
        logger.success("  RF-SENTINEL DIAGNOSIS REPORT")
        logger.success(sep)
        logger.success(f"  Device        : {device_id}")
        logger.success(f"  Predicted     : {predicted_class} ({confidence:.1%})")
        logger.success(f"  Primary cause : {primary_cause}")
        logger.success("")
        logger.success("  Repair Actions:")
        logger.success(f"    P1: {p1['action']} ({p1['estimated_time']})")
        logger.success(f"    P2: {p2['action']} ({p2['estimated_time']})")
        logger.success(f"    P3: {p3['action']} ({p3['estimated_time']})")
        logger.success(sep)

        return report

    # ── Method 2: plot_diagnosis_card ─────────────────────────────────────────

    def plot_diagnosis_card(
        self,
        report: dict,
        save:   bool = True,
    ) -> plt.Figure:
        """
        Generate a visual diagnosis card for the engineer.

        Five panels arranged in a grid:
            Top-left   : prediction header (device, class, confidence)
            Top-right  : SHAP evidence bar chart
            Mid-left   : primary cause description
            Mid-right  : P1 → P2 → P3 repair timeline
            Bottom     : confidence breakdown bar across repairs

        Parameters
        ----------
        report : diagnosis report dict from recommend()
        save   : write PNG to REPORT_DIR (default True)

        Returns
        -------
        matplotlib Figure
        """
        from matplotlib.gridspec import GridSpec

        device_id       = report["device_id"]
        pred_failure    = report["predicted_failure"]
        confidence_pct  = report["model_confidence"]
        primary_cause   = report["primary_cause"]
        shap_ev         = report["shap_evidence"]
        repairs         = report["repair_actions"]
        timestamp       = report["timestamp"][:19].replace("T", " ")

        is_failure = pred_failure.lower() not in ("pass", "no_failure")
        header_col = "#C0392B" if is_failure else "#1D9E75"

        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

        # ── Panel 1: Prediction header ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(header_col)
        ax1.axis("off")
        ax1.text(0.5, 0.75, pred_failure.upper().replace("_", " "),
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=16, fontweight="bold", color="white")
        ax1.text(0.5, 0.45, f"{confidence_pct}% confidence",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=13, color="white")
        ax1.text(0.5, 0.15, f"Device: {device_id}",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=10, color="white", alpha=0.9)
        ax1.set_title("Prediction", fontweight="bold", pad=6)

        # ── Panel 2: SHAP evidence ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        if shap_ev:
            feat_labels = [e["symptom"][:30] for e in shap_ev]
            shap_vals   = [e["shap_value"] for e in shap_ev]
            colors      = ["crimson" if v < 0 else "steelblue" for v in shap_vals]
            ax2.barh(feat_labels[::-1], shap_vals[::-1], color=colors[::-1], edgecolor="white")
            ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("SHAP value")
        ax2.set_title("SHAP Evidence — Why this prediction?", fontweight="bold", pad=6)
        red_p  = mpatches.Patch(color="crimson",   label="→ failure")
        blue_p = mpatches.Patch(color="steelblue", label="→ pass")
        ax2.legend(handles=[red_p, blue_p], fontsize=8, loc="lower right")

        # ── Panel 3: Primary cause ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor("#F4F6F7")
        ax3.axis("off")
        cause_desc = ""
        if primary_cause != "Unknown":
            for node, data in self.G.nodes(data=True):
                if data.get("label") == primary_cause:
                    cause_desc = data.get("description", "")
                    break
        ax3.text(0.5, 0.65, "Root Cause:", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=11, color="#555555")
        ax3.text(0.5, 0.42, primary_cause, transform=ax3.transAxes,
                 ha="center", va="center", fontsize=13, fontweight="bold", color="#C0392B")
        ax3.text(0.5, 0.18, cause_desc[:70], transform=ax3.transAxes,
                 ha="center", va="center", fontsize=8, color="#666666",
                 style="italic", wrap=True)
        ax3.set_title("Identified Root Cause", fontweight="bold", pad=6)

        # ── Panel 4: Repair timeline ──────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        p_colors = {"P1": "#C0392B", "P2": "#E67E22", "P3": "#2980B9"}
        x_positions = np.linspace(0.1, 0.9, max(len(repairs), 1))

        for xi, action in zip(x_positions, repairs[:3]):
            col = p_colors.get(action["priority"], "#888888")
            ax4.add_patch(mpatches.FancyBboxPatch(
                (xi - 0.12, 0.3), 0.24, 0.35,
                boxstyle="round,pad=0.02",
                facecolor=col, edgecolor="white", linewidth=1.5,
                transform=ax4.transAxes,
            ))
            ax4.text(xi, 0.62, action["priority"],
                     transform=ax4.transAxes, ha="center", va="center",
                     fontsize=14, fontweight="bold", color="white")
            ax4.text(xi, 0.46, action["action"][:22],
                     transform=ax4.transAxes, ha="center", va="center",
                     fontsize=7, color="white", wrap=True)
            ax4.text(xi, 0.32, action["estimated_time"],
                     transform=ax4.transAxes, ha="center", va="center",
                     fontsize=7, color="white", alpha=0.9)
            # Connector arrows between boxes
            if xi < x_positions[min(len(repairs) - 1, 2)]:
                ax4.annotate("", xy=(xi + 0.15, 0.48), xytext=(xi + 0.12, 0.48),
                             xycoords="axes fraction", textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="->", color="grey", lw=1.5))

        ax4.set_title("Repair Action Timeline", fontweight="bold", pad=6)

        # ── Panel 5: Confidence breakdown bar ─────────────────────────────────
        ax5 = fig.add_subplot(gs[2, :])
        if repairs:
            labels  = [f"[{r['priority']}] {r['action'][:28]}" for r in repairs]
            scores  = [r["confidence_pct"] for r in repairs]
            cols    = [p_colors.get(r["priority"], "#888888") for r in repairs]
            bars    = ax5.barh(labels[::-1], scores[::-1], color=cols[::-1], edgecolor="white")
            for bar, score in zip(bars, scores[::-1]):
                ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{score:.1f}%", va="center", ha="left", fontsize=8)
            ax5.set_xlabel("Confidence score (%)")
            ax5.set_xlim(0, max(scores) * 1.15)
        ax5.set_title("Repair Confidence Breakdown", fontweight="bold", pad=6)

        fig.suptitle(
            f"RF-Sentinel — Automated Diagnosis Report\nGenerated: {timestamp}",
            fontsize=13, fontweight="bold",
        )

        if save:
            path = REPORT_DIR / f"diagnosis_card_{device_id}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"[Recommender] Diagnosis card saved → {path}")

        return fig


# ── Standalone function: run_full_pipeline_demo ───────────────────────────────

def run_full_pipeline_demo() -> dict:
    """
    End-to-end demo: load saved SHAP report → run recommender → save outputs.

    Tries to load a real SHAP diagnosis report from Layer 5. Falls back to
    a hardcoded CMAPSS example if the file doesn't exist yet.

    Returns
    -------
    dict  full diagnosis report
    """
    # Step 1: Load SHAP report from Layer 5 (if available)
    shap_report_path = (
        ROOT_DIR / "outputs" / "explainability" / "shap" / "diagnosis_report_0.json"
    )

    if shap_report_path.exists():
        with open(shap_report_path, "r", encoding="utf-8") as f:
            shap_report = json.load(f)
        top_features = [feat["feature"]    for feat in shap_report["top_features"]]
        shap_values  = [feat["shap_value"] for feat in shap_report["top_features"]]
        predicted_raw = shap_report["predicted_class"]
        # Convert integer label to string name
        # "1" means sensor_degradation in CMAPSS binary classification
        CLASS_MAP = {
            "0": "pass",
            "1": "sensor_degradation",
            0  : "pass",
            1  : "sensor_degradation",
        }
        predicted = CLASS_MAP.get(predicted_raw, str(predicted_raw))
        confidence   = shap_report["confidence_pct"] / 100
        logger.info(f"[Recommender] Loaded SHAP report from {shap_report_path}")
    else:
        logger.info("[Recommender] No SHAP report found — using hardcoded example")
        top_features = ["s11", "s9", "s4"]
        shap_values  = [-1.235, -0.439, -0.325]
        predicted    = "sensor_degradation"
        confidence   = 0.999

    # Step 2: Run recommender
    rec    = RFSentinelRecommender()
    report = rec.recommend(
        top_features    = top_features,
        shap_values     = shap_values,
        predicted_class = predicted,
        confidence      = confidence,
        device_id       = "DUT_FD001_sample0",
        save_report     = True,
    )

    # Step 3: Generate diagnosis card
    fig = rec.plot_diagnosis_card(report, save=True)
    plt.close(fig)

    # Step 4: Print summary
    print()
    print("FULL PIPELINE COMPLETE")
    print("=" * 55)
    print(f"Device        : {report['device_id']}")
    print(f"Predicted     : {report['predicted_failure']}")
    print(f"Confidence    : {report['model_confidence']}%")
    print(f"Primary cause : {report['primary_cause']}")
    print()
    print("Repair Actions:")
    for action in report["repair_actions"]:
        print(
            f"  [{action['priority']}] "
            f"{action['action']}"
            f" — {action['estimated_time']}"
            f" (confidence: {action['confidence_pct']}%)"
        )
    print()
    print(f"Report saved  : {REPORT_DIR}")
    print("=" * 55)

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_pipeline_demo()
