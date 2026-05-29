# Links SHAP top-3 features to knowledge graph for root cause traversal
"""
shap_to_graph.py — Links SHAP feature importance to the RF-Sentinel knowledge graph.

Maps SHAP top-N sensor names to symptom nodes in the knowledge graph, then
traverses the graph to find all reachable cause → repair paths. Path confidence
is scored by combining the SHAP weight with edge weights along the path.

Usage
-----
    python -m layer6_knowledge_graph.shap_to_graph
"""

import networkx as nx
from loguru import logger

from layer6_knowledge_graph.build_graph import (
    build_rf_knowledge_graph, load_graph, GRAPH_SAVE_PATH,
)

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Layer 6: SHAP to Knowledge Graph Linker
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# Links SHAP explainability output (Layer 5) to the
# knowledge graph (Layer 6) to find root causes and repairs.
#
# PIPELINE:
# ─────────
# 1. SHAP gives top features: ["s11", "s9", "s4"]
#    with values: [−1.235, −0.439, −0.325]
#
# 2. This file maps sensor names to graph symptom nodes:
#    "s11" → "s11_pressure_drop"
#    "s9"  → "s9_speed_reduction"
#    "s4"  → "s4_temp_rise"
#
# 3. Graph traversal finds all paths:
#    s11_pressure_drop → bearing_wear → inspect_bearing
#    s11_pressure_drop → seal_degradation → replace_seals
#    s9_speed_reduction → bearing_wear → inspect_bearing
#
# 4. Paths ranked by combined confidence:
#    confidence = abs(shap_value) × edge1_weight × edge2_weight
#
# 5. Top paths passed to recommender.py
# ══════════════════════════════════════════════════════════════

# Maps CMAPSS sensor column names and AI4I failure type strings to the
# symptom node IDs used in the knowledge graph. Some sensors share a
# symptom node (e.g. s2 and s4 both map to thermal symptoms) because they
# measure correlated physical quantities.
SENSOR_TO_SYMPTOM: dict[str, str] = {
    # CMAPSS sensors
    "s11": "s11_pressure_drop",
    "s9":  "s9_speed_reduction",
    "s4":  "s4_temp_rise",
    "s3":  "s3_temp_rise",
    "s14": "s14_speed_drift",
    "s8":  "s8_fan_speed_drop",
    "s13": "s13_fan_corrected_drop",
    "s21": "s21_coolant_drop",
    "s15": "s15_bypass_drop",
    "s2":  "s4_temp_rise",        # LPC temp → thermal symptom
    "s7":  "s11_pressure_drop",   # HPC pressure → pressure symptom
    "s12": "s9_speed_reduction",  # fuel flow → speed symptom
    "s17": "s4_temp_rise",        # bleed enthalpy → thermal
    "s20": "s21_coolant_drop",    # HPT coolant → coolant symptom
    # AI4I failure types
    "heat_dissipation_failure": "hdf_heat_dissipation",
    "power_failure":            "pwf_power_failure",
    "overstrain_failure":       "osf_overstrain",
    "thermal_wear_failure":     "twf_tool_wear",
    "random_failure":           "rnf_random",
    "manufacturing_defect":     "secom_process_drift",
    "sensor_degradation":       "s11_pressure_drop",
}


# ── Class: SHAPToGraphLinker ──────────────────────────────────────────────────

class SHAPToGraphLinker:
    """
    Links SHAP feature importance scores to the RF-Sentinel knowledge graph.

    The linker bridges two layers: Layer 5 (SHAP explainability) and Layer 6
    (knowledge graph traversal). It takes the output of RFSentinelSHAP
    (top feature names + SHAP values) and returns ranked repair paths that
    combine data-driven confidence (SHAP weights) with expert knowledge
    (graph edge weights).
    """

    def __init__(self, G: nx.DiGraph | None = None) -> None:
        if G is not None:
            self.G = G
        elif GRAPH_SAVE_PATH.exists():
            self.G = load_graph()
        else:
            logger.info("[KG Linker] No saved graph found — building fresh")
            self.G = build_rf_knowledge_graph()

        self.sensor_map = SENSOR_TO_SYMPTOM

    # ── Method 1: map_features_to_symptoms ────────────────────────────────────

    def map_features_to_symptoms(
        self,
        top_features: list[str],
        shap_values:  list[float],
    ) -> list[dict]:
        """
        Map SHAP top features to knowledge graph symptom node IDs.

        Uses SENSOR_TO_SYMPTOM to translate raw column names (e.g. "s11") to
        graph node IDs (e.g. "s11_pressure_drop"). Features not in the map, or
        whose target node does not exist in the graph, are skipped with a warning.

        Parameters
        ----------
        top_features : list of feature names from SHAP (e.g. ["s11", "s9"])
        shap_values  : corresponding SHAP values (signed floats)

        Returns
        -------
        list of dicts, each containing:
            feature, symptom_node, symptom_label, shap_value, shap_weight
        """
        mapped: list[dict] = []

        for feature, shap_val in zip(top_features, shap_values):
            symptom_node = self.sensor_map.get(feature)
            if symptom_node is None:
                logger.warning(f"[KG Linker] Feature '{feature}' not in sensor map")
                continue
            if symptom_node not in self.G:
                logger.warning(f"[KG Linker] Symptom node '{symptom_node}' not in graph")
                continue

            mapped.append({
                "feature":       feature,
                "symptom_node":  symptom_node,
                "symptom_label": self.G.nodes[symptom_node].get("label", symptom_node),
                "shap_value":    float(shap_val),
                "shap_weight":   float(abs(shap_val)),
            })

        logger.info(f"[KG Linker] Mapped {len(mapped)} features to symptom nodes")
        return mapped

    # ── Method 2: traverse_graph ──────────────────────────────────────────────

    def traverse_graph(
        self,
        symptom_nodes: list[str],
        max_paths: int = 10,
    ) -> list[dict]:
        """
        Traverse the graph from symptom nodes to find all cause → repair paths.

        Walks exactly two hops: symptom → cause → repair. Edge weights along
        each hop are multiplied to form the path confidence. Only paths that
        reach a REPAIR node are included in the results.

        Parameters
        ----------
        symptom_nodes : list of symptom node IDs to start from
        max_paths     : maximum number of paths to return (sorted by confidence)

        Returns
        -------
        list of path dicts sorted by path_confidence descending
        """
        all_paths: list[dict] = []

        for symptom_node in symptom_nodes:
            if symptom_node not in self.G:
                continue

            # Hop 1: symptom → cause
            for cause_node in self.G.successors(symptom_node):
                if self.G.nodes[cause_node].get("type") != "cause":
                    continue
                symptom_cause_w = self.G[symptom_node][cause_node].get("weight", 0.5)

                # Hop 2: cause → repair
                for repair_node in self.G.successors(cause_node):
                    if self.G.nodes[repair_node].get("type") != "repair":
                        continue
                    cause_repair_w = self.G[cause_node][repair_node].get("weight", 0.5)

                    all_paths.append({
                        "symptom_node":          symptom_node,
                        "symptom_label":         self.G.nodes[symptom_node].get("label", symptom_node),
                        "cause_node":            cause_node,
                        "cause_label":           self.G.nodes[cause_node].get("label", cause_node),
                        "repair_node":           repair_node,
                        "repair_label":          self.G.nodes[repair_node].get("label", repair_node),
                        "repair_priority":       self.G.nodes[repair_node].get("priority", "P3"),
                        "repair_time":           self.G.nodes[repair_node].get("estimated_time", "unknown"),
                        "path_confidence":       round(symptom_cause_w * cause_repair_w, 4),
                        "symptom_cause_weight":  symptom_cause_w,
                        "cause_repair_weight":   cause_repair_w,
                    })

        all_paths.sort(key=lambda p: p["path_confidence"], reverse=True)
        result = all_paths[:max_paths]

        logger.info(
            f"[KG Linker] Found {len(all_paths)} paths from "
            f"{len(symptom_nodes)} symptom node(s) — returning top {len(result)}"
        )
        return result

    # ── Method 3: get_repair_paths ────────────────────────────────────────────

    def get_repair_paths(
        self,
        top_features: list[str],
        shap_values:  list[float],
        max_paths:    int = 10,
    ) -> dict:
        """
        Main entry point: maps SHAP features to graph, traverses, and scores paths.

        Combines data-driven confidence (SHAP absolute value) with expert
        knowledge (graph edge weights) into a single final_score per path:
            final_score = shap_weight × path_confidence

        This is called by recommender.py to produce the ranked action list.

        Parameters
        ----------
        top_features : list of SHAP feature names (e.g. ["s11", "s9", "s4"])
        shap_values  : corresponding SHAP values
        max_paths    : maximum repair paths to return

        Returns
        -------
        dict with keys:
            input_features, shap_values, mapped_symptoms,
            repair_paths, top_cause, top_repair
        """
        # Step 1: Map features → symptom nodes
        mapped = self.map_features_to_symptoms(top_features, shap_values)

        # Step 2: Deduplicate symptom nodes (multiple sensors can map to same symptom)
        seen: set = set()
        symptom_nodes: list[str] = []
        for m in mapped:
            if m["symptom_node"] not in seen:
                seen.add(m["symptom_node"])
                symptom_nodes.append(m["symptom_node"])

        # Step 3: Traverse graph for all cause → repair paths
        paths = self.traverse_graph(symptom_nodes, max_paths=max_paths * 3)

        # Step 4: Score each path using SHAP weight of the triggering symptom
        shap_lookup: dict[str, float] = {
            m["symptom_node"]: m["shap_weight"] for m in mapped
        }
        for path in paths:
            shap_w = shap_lookup.get(path["symptom_node"], 1.0)
            path["final_score"] = round(shap_w * path["path_confidence"], 4)

        paths.sort(key=lambda p: p["final_score"], reverse=True)
        paths = paths[:max_paths]

        top_cause  = paths[0]["cause_label"]  if paths else None
        top_repair = paths[0]["repair_label"] if paths else None

        logger.success("[KG Linker] Graph traversal complete")
        logger.info(f"  Input features   : {top_features}")
        logger.info(f"  Symptom nodes    : {symptom_nodes}")
        logger.info(f"  Paths found      : {len(paths)}")
        logger.info(f"  Top cause        : {top_cause}")
        logger.info(f"  Top repair       : {top_repair}")

        return {
            "input_features":  top_features,
            "shap_values":     [float(v) for v in shap_values],
            "mapped_symptoms": mapped,
            "repair_paths":    paths,
            "top_cause":       top_cause,
            "top_repair":      top_repair,
        }


# ── Standalone function: run_shap_to_graph_demo ───────────────────────────────

def run_shap_to_graph_demo() -> tuple[dict, dict]:
    """
    Demonstrate the linker with two realistic SHAP output examples.

    Example 1 uses CMAPSS sensor names (as returned by RFSentinelSHAP).
    Example 2 uses AI4I failure type strings (as returned by multiclass XGBoost).

    Returns
    -------
    (result1, result2)  — full repair path dicts for both examples
    """
    G      = build_rf_knowledge_graph()
    linker = SHAPToGraphLinker(G)

    # Example 1: CMAPSS sensor failure
    print("Example 1 — CMAPSS sensor failure:")
    result1 = linker.get_repair_paths(
        top_features=["s11", "s9", "s4"],
        shap_values= [-1.235, -0.439, -0.325],
    )
    print(f"  Top cause  : {result1['top_cause']}")
    print(f"  Top repair : {result1['top_repair']}")
    print(f"  Paths found: {len(result1['repair_paths'])}")
    print()
    print("  Top 5 repair paths:")
    for i, path in enumerate(result1["repair_paths"][:5]):
        print(
            f"  {i+1}. [{path['repair_priority']}] "
            f"{path['repair_label']} "
            f"(score={path['final_score']:.3f})"
        )

    print()

    # Example 2: AI4I heat dissipation failure
    print("Example 2 — AI4I heat dissipation failure:")
    result2 = linker.get_repair_paths(
        top_features=["heat_dissipation_failure", "power_failure"],
        shap_values= [0.88, 0.45],
    )
    print(f"  Top cause  : {result2['top_cause']}")
    print(f"  Top repair : {result2['top_repair']}")
    print()
    print("  Top 5 repair paths:")
    for i, path in enumerate(result2["repair_paths"][:5]):
        print(
            f"  {i+1}. [{path['repair_priority']}] "
            f"{path['repair_label']} "
            f"(score={path['final_score']:.3f})"
        )

    return result1, result2


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_shap_to_graph_demo()
