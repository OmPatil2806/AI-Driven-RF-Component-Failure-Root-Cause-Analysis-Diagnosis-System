# NetworkX knowledge graph: symptom → cause → repair nodes
"""
build_graph.py — Constructs the RF-Sentinel expert knowledge graph.

Encodes domain knowledge about RF component failures as a directed graph:
    SYMPTOM → CAUSE → REPAIR

Edge weights represent confidence that a symptom indicates a cause, or that
a cause requires a specific repair. This graph is traversed by shap_to_graph.py
using SHAP top-3 features to produce ranked repair recommendations.

Usage
-----
    python -m layer6_knowledge_graph.build_graph
"""

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

from layer1_data_ingestion.config import ROOT_DIR

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Layer 6: Knowledge Graph Builder
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# Builds a NetworkX directed graph encoding expert knowledge
# about RF component failures and their root causes.
#
# THREE NODE TYPES:
# ─────────────────
# SYMPTOM → what the sensor/measurement shows
#            (e.g. s11_pressure_drop, s9_speed_reduction)
# CAUSE   → the physical root cause
#            (e.g. bearing_wear, thermal_stress, connector_fault)
# REPAIR  → what the engineer must do
#            (e.g. replace_bearing, check_cooling, re_torque_connector)
#
# GRAPH STRUCTURE:
# ────────────────
# SYMPTOM → CAUSE  (edge: "indicates")
# CAUSE   → REPAIR (edge: "requires")
#
# This encodes domain knowledge:
#   "When sensor s11 drops → it indicates impedance mismatch
#    → which requires checking connector torque"
#
# HOW IT CONNECTS TO SHAP:
# ────────────────────────
# SHAP gives top-3 features: ["s11", "s9", "s4"]
# shap_to_graph.py maps these to SYMPTOM nodes
# Graph traversal finds CAUSE → REPAIR path
# recommender.py ranks and returns repair actions
# ══════════════════════════════════════════════════════════════

GRAPH_DIR       = ROOT_DIR / "outputs" / "knowledge_graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_SAVE_PATH = GRAPH_DIR / "rf_sentinel_knowledge_graph.gml"


# ── Function 1: build_rf_knowledge_graph ──────────────────────────────────────

def build_rf_knowledge_graph() -> nx.DiGraph:
    """
    Build and return the complete RF-Sentinel knowledge graph.

    Nodes encode symptoms (sensor readings), physical causes, and repair
    actions. Directed edges encode the domain relationships:
        - symptom → cause  (edge weight = confidence of indication)
        - cause   → repair (edge weight = priority of repair action)

    The graph covers all three RF-Sentinel datasets:
        - CMAPSS : 9 sensor symptoms from turbofan degradation signals
        - AI4I   : 5 failure-type symptoms from predictive maintenance labels
        - SECOM  : 2 process-level symptoms from semiconductor manufacturing

    Returns
    -------
    nx.DiGraph  fully populated knowledge graph
    """
    G = nx.DiGraph()

    # ── SYMPTOM nodes ─────────────────────────────────────────────────────────

    # CMAPSS sensor symptoms
    G.add_node("s11_pressure_drop",
               type="symptom", severity="high", dataset="cmapss",
               label="HPC Pressure Drop",
               description="HPC outlet static pressure below normal range")
    G.add_node("s9_speed_reduction",
               type="symptom", severity="high", dataset="cmapss",
               label="Core Speed Reduction",
               description="Core speed dropping below design point")
    G.add_node("s4_temp_rise",
               type="symptom", severity="high", dataset="cmapss",
               label="LPT Temperature Rise",
               description="LPT outlet temperature above normal")
    G.add_node("s3_temp_rise",
               type="symptom", severity="medium", dataset="cmapss",
               label="HPC Temperature Rise",
               description="HPC outlet temperature increasing")
    G.add_node("s14_speed_drift",
               type="symptom", severity="medium", dataset="cmapss",
               label="Corrected Core Speed Drift",
               description="Corrected core speed deviating from baseline")
    G.add_node("s8_fan_speed_drop",
               type="symptom", severity="medium", dataset="cmapss",
               label="Fan Speed Reduction",
               description="Physical fan speed below normal")
    G.add_node("s13_fan_corrected_drop",
               type="symptom", severity="low", dataset="cmapss",
               label="Corrected Fan Speed Drop",
               description="Corrected fan speed drifting")
    G.add_node("s21_coolant_drop",
               type="symptom", severity="medium", dataset="cmapss",
               label="LPT Coolant Bleed Reduction",
               description="LPT coolant bleed flow decreasing")
    G.add_node("s15_bypass_drop",
               type="symptom", severity="medium", dataset="cmapss",
               label="Bypass Ratio Reduction",
               description="Engine bypass ratio below specification")

    # AI4I failure-type symptoms
    G.add_node("hdf_heat_dissipation",
               type="symptom", severity="high", dataset="ai4i",
               label="Heat Dissipation Failure",
               description="Machine overheating — cannot cool fast enough")
    G.add_node("pwf_power_failure",
               type="symptom", severity="high", dataset="ai4i",
               label="Power Failure",
               description="Machine receives wrong power level for speed")
    G.add_node("osf_overstrain",
               type="symptom", severity="high", dataset="ai4i",
               label="Overstrain Failure",
               description="Machine pushed beyond physical limit")
    G.add_node("twf_tool_wear",
               type="symptom", severity="medium", dataset="ai4i",
               label="Tool Wear Failure",
               description="Tool used beyond safe wear limit")
    G.add_node("rnf_random",
               type="symptom", severity="low", dataset="ai4i",
               label="Random Failure",
               description="Unexplained random failure")

    # SECOM process symptoms
    G.add_node("secom_process_drift",
               type="symptom", severity="high", dataset="secom",
               label="Process Parameter Drift",
               description="Semiconductor manufacturing parameter out of control")
    G.add_node("secom_contamination",
               type="symptom", severity="high", dataset="secom",
               label="Contamination Event",
               description="Process contamination detected in sensor cluster")

    # ── CAUSE nodes ───────────────────────────────────────────────────────────

    G.add_node("bearing_wear",
               type="cause", severity="high",
               label="Bearing Wear",
               description="Engine bearing degrading — increases friction and heat")
    G.add_node("blade_fouling",
               type="cause", severity="medium",
               label="Compressor Blade Fouling",
               description="Deposits on compressor blades reducing efficiency")
    G.add_node("thermal_stress",
               type="cause", severity="high",
               label="Thermal Stress",
               description="Excessive heat causing component deformation")
    G.add_node("cooling_degradation",
               type="cause", severity="high",
               label="Cooling System Degradation",
               description="Coolant flow reduced — thermal protection failing")
    G.add_node("fuel_system_fault",
               type="cause", severity="high",
               label="Fuel System Fault",
               description="Fuel delivery irregularity affecting combustion")
    G.add_node("seal_degradation",
               type="cause", severity="medium",
               label="Seal Degradation",
               description="Internal seals worn — causing pressure leakage")
    G.add_node("connector_fault",
               type="cause", severity="high",
               label="RF Connector Fault",
               description="Connector impedance mismatch or physical damage")
    G.add_node("overheating",
               type="cause", severity="high",
               label="System Overheating",
               description="Thermal management failure — temperature exceeding limits")
    G.add_node("power_supply_fault",
               type="cause", severity="high",
               label="Power Supply Fault",
               description="Voltage or current outside specification")
    G.add_node("mechanical_overload",
               type="cause", severity="high",
               label="Mechanical Overload",
               description="Forces exceeding component design limits")
    G.add_node("tool_end_of_life",
               type="cause", severity="medium",
               label="Tool End of Life",
               description="Cutting tool has exceeded useful operating hours")
    G.add_node("process_contamination",
               type="cause", severity="high",
               label="Process Contamination",
               description="Foreign material in manufacturing process")

    # ── REPAIR nodes ──────────────────────────────────────────────────────────

    G.add_node("inspect_bearing",
               type="repair", priority="P1", estimated_time="4 hours",
               label="Inspect and Replace Bearing Assembly",
               description="Remove and inspect bearing — replace if worn beyond limit")
    G.add_node("clean_compressor_blades",
               type="repair", priority="P2", estimated_time="2 hours",
               label="Clean Compressor Blades",
               description="Water wash or chemical clean compressor stage")
    G.add_node("check_cooling_system",
               type="repair", priority="P1", estimated_time="1 hour",
               label="Check Cooling System",
               description="Inspect coolant flow, check for blockages, verify pump operation")
    G.add_node("replace_seals",
               type="repair", priority="P2", estimated_time="6 hours",
               label="Replace Internal Seals",
               description="Replace worn seals in affected stage")
    G.add_node("check_fuel_system",
               type="repair", priority="P1", estimated_time="2 hours",
               label="Inspect Fuel Delivery System",
               description="Check fuel nozzles, filters, and pump pressure")
    G.add_node("thermal_inspection",
               type="repair", priority="P1", estimated_time="30 minutes",
               label="Thermal Inspection",
               description="Infrared scan to identify hot spots and thermal anomalies")
    G.add_node("retorque_connector",
               type="repair", priority="P1", estimated_time="15 minutes",
               label="Re-torque RF Connector",
               description="Re-torque SMA/N-type connector to spec (0.9 N·m)")
    G.add_node("replace_connector",
               type="repair", priority="P2", estimated_time="30 minutes",
               label="Replace RF Connector",
               description="Replace damaged connector — clean mating surfaces")
    G.add_node("check_power_supply",
               type="repair", priority="P1", estimated_time="30 minutes",
               label="Check Power Supply Rails",
               description="Verify DC supply voltages with DMM — check for ripple")
    G.add_node("reduce_load",
               type="repair", priority="P1", estimated_time="immediate",
               label="Reduce Operating Load",
               description="Reduce speed or torque to within design envelope")
    G.add_node("replace_tool",
               type="repair", priority="P1", estimated_time="20 minutes",
               label="Replace Cutting Tool",
               description="Install new cutting tool — verify torque and alignment")
    G.add_node("process_audit",
               type="repair", priority="P2", estimated_time="2 hours",
               label="Process Environment Audit",
               description="Full audit of process parameters — check for contamination")
    G.add_node("retest_at_ambient",
               type="repair", priority="P2", estimated_time="1 hour",
               label="Retest at 25°C Ambient",
               description="Retest component at controlled ambient — isolate thermal drift")
    G.add_node("escalate_to_engineering",
               type="repair", priority="P3", estimated_time="varies",
               label="Escalate to Engineering Team",
               description="Random failure — log incident and escalate for root cause analysis")

    # ── EDGES: SYMPTOM → CAUSE ────────────────────────────────────────────────

    # CMAPSS sensor → cause mappings
    G.add_edge("s11_pressure_drop",     "bearing_wear",          weight=0.85, relation="indicates")
    G.add_edge("s11_pressure_drop",     "seal_degradation",      weight=0.75, relation="indicates")
    G.add_edge("s11_pressure_drop",     "connector_fault",       weight=0.70, relation="indicates")
    G.add_edge("s9_speed_reduction",    "bearing_wear",          weight=0.80, relation="indicates")
    G.add_edge("s9_speed_reduction",    "blade_fouling",         weight=0.65, relation="indicates")
    G.add_edge("s9_speed_reduction",    "fuel_system_fault",     weight=0.60, relation="indicates")
    G.add_edge("s4_temp_rise",          "thermal_stress",        weight=0.85, relation="indicates")
    G.add_edge("s4_temp_rise",          "cooling_degradation",   weight=0.80, relation="indicates")
    G.add_edge("s3_temp_rise",          "thermal_stress",        weight=0.75, relation="indicates")
    G.add_edge("s3_temp_rise",          "blade_fouling",         weight=0.60, relation="indicates")
    G.add_edge("s14_speed_drift",       "bearing_wear",          weight=0.70, relation="indicates")
    G.add_edge("s14_speed_drift",       "seal_degradation",      weight=0.55, relation="indicates")
    G.add_edge("s8_fan_speed_drop",     "bearing_wear",          weight=0.75, relation="indicates")
    G.add_edge("s8_fan_speed_drop",     "blade_fouling",         weight=0.65, relation="indicates")
    G.add_edge("s13_fan_corrected_drop","blade_fouling",         weight=0.65, relation="indicates")
    G.add_edge("s13_fan_corrected_drop","bearing_wear",          weight=0.55, relation="indicates")
    G.add_edge("s21_coolant_drop",      "cooling_degradation",   weight=0.85, relation="indicates")
    G.add_edge("s21_coolant_drop",      "thermal_stress",        weight=0.70, relation="indicates")
    G.add_edge("s15_bypass_drop",       "seal_degradation",      weight=0.80, relation="indicates")
    G.add_edge("s15_bypass_drop",       "blade_fouling",         weight=0.60, relation="indicates")

    # AI4I failure types → causes
    G.add_edge("hdf_heat_dissipation",  "overheating",           weight=0.95, relation="indicates")
    G.add_edge("hdf_heat_dissipation",  "cooling_degradation",   weight=0.80, relation="indicates")
    G.add_edge("pwf_power_failure",     "power_supply_fault",    weight=0.90, relation="indicates")
    G.add_edge("pwf_power_failure",     "fuel_system_fault",     weight=0.70, relation="indicates")
    G.add_edge("osf_overstrain",        "mechanical_overload",   weight=0.90, relation="indicates")
    G.add_edge("osf_overstrain",        "bearing_wear",          weight=0.65, relation="indicates")
    G.add_edge("twf_tool_wear",         "tool_end_of_life",      weight=0.95, relation="indicates")
    G.add_edge("rnf_random",            "process_contamination", weight=0.50, relation="indicates")

    # SECOM → causes
    G.add_edge("secom_process_drift",   "process_contamination", weight=0.80, relation="indicates")
    G.add_edge("secom_contamination",   "process_contamination", weight=0.90, relation="indicates")

    # ── EDGES: CAUSE → REPAIR ─────────────────────────────────────────────────

    G.add_edge("bearing_wear",          "inspect_bearing",         weight=0.95, relation="requires")
    G.add_edge("bearing_wear",          "retest_at_ambient",       weight=0.60, relation="requires")
    G.add_edge("blade_fouling",         "clean_compressor_blades", weight=0.90, relation="requires")
    G.add_edge("blade_fouling",         "inspect_bearing",         weight=0.50, relation="requires")
    G.add_edge("thermal_stress",        "thermal_inspection",      weight=0.90, relation="requires")
    G.add_edge("thermal_stress",        "check_cooling_system",    weight=0.85, relation="requires")
    G.add_edge("thermal_stress",        "retest_at_ambient",       weight=0.75, relation="requires")
    G.add_edge("cooling_degradation",   "check_cooling_system",    weight=0.95, relation="requires")
    G.add_edge("cooling_degradation",   "thermal_inspection",      weight=0.80, relation="requires")
    G.add_edge("fuel_system_fault",     "check_fuel_system",       weight=0.90, relation="requires")
    G.add_edge("seal_degradation",      "replace_seals",           weight=0.85, relation="requires")
    G.add_edge("seal_degradation",      "retest_at_ambient",       weight=0.65, relation="requires")
    G.add_edge("connector_fault",       "retorque_connector",      weight=0.90, relation="requires")
    G.add_edge("connector_fault",       "replace_connector",       weight=0.75, relation="requires")
    G.add_edge("overheating",           "check_cooling_system",    weight=0.95, relation="requires")
    G.add_edge("overheating",           "thermal_inspection",      weight=0.90, relation="requires")
    G.add_edge("overheating",           "reduce_load",             weight=0.80, relation="requires")
    G.add_edge("power_supply_fault",    "check_power_supply",      weight=0.95, relation="requires")
    G.add_edge("mechanical_overload",   "reduce_load",             weight=0.95, relation="requires")
    G.add_edge("mechanical_overload",   "inspect_bearing",         weight=0.70, relation="requires")
    G.add_edge("tool_end_of_life",      "replace_tool",            weight=0.95, relation="requires")
    G.add_edge("process_contamination", "process_audit",           weight=0.90, relation="requires")
    G.add_edge("process_contamination", "escalate_to_engineering", weight=0.70, relation="requires")

    # ── Summary log ───────────────────────────────────────────────────────────
    n_symptoms = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "symptom")
    n_causes   = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "cause")
    n_repairs  = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "repair")

    logger.success("[KG] Knowledge graph built")
    logger.info(f"  Nodes    : {G.number_of_nodes()}")
    logger.info(f"  Edges    : {G.number_of_edges()}")
    logger.info(f"  Symptoms : {n_symptoms}")
    logger.info(f"  Causes   : {n_causes}")
    logger.info(f"  Repairs  : {n_repairs}")

    return G


# ── Function 2: save_graph ────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, path=None) -> str:
    """
    Save the graph to GML format for reloading without rebuilding.

    GML is chosen over GraphML or pickle because it is human-readable,
    preserves all node/edge attributes, and is natively supported by NetworkX
    with no extra dependencies.

    Parameters
    ----------
    G    : built knowledge graph
    path : optional explicit output path; defaults to GRAPH_SAVE_PATH

    Returns
    -------
    str  absolute path of the saved .gml file
    """
    out_path = path or GRAPH_SAVE_PATH
    nx.write_gml(G, str(out_path))
    logger.info(f"[KG] Graph saved → {out_path}")
    return str(out_path)


# ── Function 3: load_graph ────────────────────────────────────────────────────

def load_graph(path=None) -> nx.DiGraph:
    """
    Load a previously saved knowledge graph from GML format.

    Parameters
    ----------
    path : optional explicit .gml path; defaults to GRAPH_SAVE_PATH

    Returns
    -------
    nx.DiGraph  restored knowledge graph
    """
    in_path = path or GRAPH_SAVE_PATH
    G = nx.read_gml(str(in_path))
    logger.info(
        f"[KG] Graph loaded ← {in_path} "
        f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
    )
    return G


# ── Function 4: plot_graph_static ─────────────────────────────────────────────

def plot_graph_static(G: nx.DiGraph, save: bool = True):
    """
    Plot a static matplotlib overview of the knowledge graph.

    Nodes are coloured by type and sized by degree. Spring layout is used
    for a readable spread. This plot is used in the notebook and in reports;
    the interactive pyvis version is embedded in the Streamlit app.

    Parameters
    ----------
    G    : knowledge graph
    save : write PNG to GRAPH_DIR (default True)

    Returns
    -------
    matplotlib Figure
    """
    colour_map = {"symptom": "steelblue", "cause": "crimson", "repair": "mediumseagreen"}
    node_colours = [colour_map.get(G.nodes[n].get("type", ""), "grey") for n in G.nodes()]
    node_sizes   = [300 + G.degree(n) * 120 for n in G.nodes()]
    node_labels  = {n: G.nodes[n].get("label", n) for n in G.nodes()}

    pos = nx.spring_layout(G, seed=42, k=2.5)

    n_symptoms = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "symptom")
    n_causes   = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "cause")
    n_repairs  = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "repair")

    fig, ax = plt.subplots(figsize=(20, 12))

    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=15,
                           edge_color="grey", alpha=0.5, width=1.2)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=6, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="steelblue",     label=f"Symptom ({n_symptoms})"),
        mpatches.Patch(color="crimson",        label=f"Cause ({n_causes})"),
        mpatches.Patch(color="mediumseagreen", label=f"Repair Action ({n_repairs})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

    ax.set_title(
        f"RF-Sentinel Knowledge Graph\n"
        f"{n_symptoms} symptoms → {n_causes} causes → {n_repairs} repairs",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    if save:
        path = GRAPH_DIR / "knowledge_graph_static.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"[KG] Static graph saved → {path}")

    return fig


# ── Function 5: plot_graph_pyvis ──────────────────────────────────────────────

def plot_graph_pyvis(G: nx.DiGraph, save: bool = True):
    """
    Create an interactive HTML knowledge graph using pyvis.

    Dark theme with colour-coded nodes. Edge width encodes confidence weight.
    Barnes-Hut physics gives a natural cluster separation between node types.
    The HTML file is embedded directly in the Streamlit app via st.components.

    Parameters
    ----------
    G    : knowledge graph
    save : write HTML to GRAPH_DIR (default True)

    Returns
    -------
    str (html_path) if save=True, else pyvis Network object
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("[KG] pyvis not installed — skipping interactive graph. Run: pip install pyvis")
        return None

    net = Network(
        height="700px", width="100%",
        bgcolor="#0d0d0d", font_color="white",
        directed=True,
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 150
        }
      },
      "nodes": {
        "font": {"size": 12}
      },
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "smooth": {"type": "curvedCW"}
      }
    }
    """)

    colour_map = {"symptom": "#378ADD", "cause": "#D85A30", "repair": "#1D9E75"}

    for node in G.nodes():
        node_type = G.nodes[node].get("type", "")
        colour    = colour_map.get(node_type, "#888888")
        size      = 20 + G.degree(node) * 3
        net.add_node(
            node,
            label=G.nodes[node].get("label", node),
            title=G.nodes[node].get("description", ""),
            color=colour,
            size=size,
        )

    for src, tgt, data in G.edges(data=True):
        weight = data.get("weight", 0.5)
        net.add_edge(
            src, tgt,
            title=f"weight: {weight:.2f}",
            width=weight * 3,
        )

    if save:
        html_path = GRAPH_DIR / "knowledge_graph_interactive.html"
        net.save_graph(str(html_path))
        logger.success(f"[KG] Interactive graph saved → {html_path}")
        return str(html_path)

    return net


# ── Function 6: get_graph_stats ───────────────────────────────────────────────

def get_graph_stats(G: nx.DiGraph) -> dict:
    """
    Return summary statistics about the knowledge graph.

    Useful for verifying graph integrity after loading and for displaying
    a health summary in the Streamlit app sidebar.

    Parameters
    ----------
    G : knowledge graph

    Returns
    -------
    dict  with keys: total_nodes, total_edges, n_symptoms, n_causes,
          n_repairs, avg_degree, most_connected_symptom, most_common_repair
    """
    n_symptoms = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "symptom")
    n_causes   = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "cause")
    n_repairs  = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "repair")

    degrees    = dict(G.degree())
    avg_degree = round(sum(degrees.values()) / max(len(degrees), 1), 2)

    symptom_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "symptom"]
    repair_nodes  = [n for n, d in G.nodes(data=True) if d.get("type") == "repair"]

    most_connected_symptom = (
        max(symptom_nodes, key=lambda n: G.out_degree(n))
        if symptom_nodes else None
    )
    most_common_repair = (
        max(repair_nodes, key=lambda n: G.in_degree(n))
        if repair_nodes else None
    )

    return {
        "total_nodes":             G.number_of_nodes(),
        "total_edges":             G.number_of_edges(),
        "n_symptoms":              n_symptoms,
        "n_causes":                n_causes,
        "n_repairs":               n_repairs,
        "avg_degree":              avg_degree,
        "most_connected_symptom":  most_connected_symptom,
        "most_common_repair":      most_common_repair,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G = build_rf_knowledge_graph()
    save_graph(G)
    plot_graph_static(G, save=True)
    plot_graph_pyvis(G, save=True)

    stats = get_graph_stats(G)
    print()
    print("Knowledge Graph Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
