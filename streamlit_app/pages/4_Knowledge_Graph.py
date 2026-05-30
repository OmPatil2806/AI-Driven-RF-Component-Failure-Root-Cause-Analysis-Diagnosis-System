"""
4_Knowledge_Graph.py — RF-Sentinel Streamlit page: Knowledge Graph.

Visualises the RF-Sentinel failure knowledge graph (symptom → cause → repair).
Supports interactive PyVis HTML embedding, static overview, manual graph query
by sensor selection, and per-node detail inspection.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(str(Path(__file__).resolve().parents[2]))

from layer6_knowledge_graph.build_graph import build_rf_knowledge_graph, get_graph_stats
from layer6_knowledge_graph.shap_to_graph import SENSOR_TO_SYMPTOM, SHAPToGraphLinker

# ── Paths ─────────────────────────────────────────────────────────────────────
_KG_OUT = Path(__file__).resolve().parents[2] / "outputs" / "knowledge_graph"

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════

st.title("Knowledge Graph")
st.markdown(
    "Explore the RF-Sentinel failure knowledge graph. "
    "Symptom nodes → Cause nodes → Repair action nodes."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — Graph stats
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_graph():
    return build_rf_knowledge_graph()


G     = load_graph()
stats = get_graph_stats(G)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Nodes", stats["total_nodes"])
c2.metric("Total Edges", stats["total_edges"])
c3.metric("Symptoms",    stats["n_symptoms"])
c4.metric("Causes",      stats["n_causes"])

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — Interactive PyVis graph
# ══════════════════════════════════════════════════════════════

st.subheader("Interactive Knowledge Graph")
st.markdown(
    "Click and drag nodes. "
    "**Blue** = Symptom &nbsp;·&nbsp; **Red** = Cause &nbsp;·&nbsp; **Green** = Repair"
)

html_path = _KG_OUT / "knowledge_graph_interactive.html"

if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.warning("Interactive graph HTML not found.")
    st.info("Run: python -m layer6_knowledge_graph.build_graph")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — Static graph image
# ══════════════════════════════════════════════════════════════

with st.expander("View Static Graph Overview"):
    static_path = _KG_OUT / "knowledge_graph_static.png"
    if static_path.exists():
        st.image(str(static_path), use_container_width=True)
    else:
        st.warning("Static graph image not found.")
        st.info("Run: python -m layer6_knowledge_graph.build_graph")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — Query the graph
# ══════════════════════════════════════════════════════════════

st.divider()
st.subheader("Query — Find Repair for a Symptom")

linker = SHAPToGraphLinker(G)

q_left, q_right = st.columns(2)

q_left.markdown("**Select sensors showing anomaly:**")
sensor_options   = list(SENSOR_TO_SYMPTOM.keys())
selected_sensors = q_left.multiselect(
    "Select sensors",
    options=sensor_options[:14],
    default=["s11", "s9"],
    help="Select sensors that are showing abnormal readings",
)

q_right.markdown("**Assign SHAP-style weights:**")
shap_values: list[float] = []
for sensor in selected_sensors:
    val = q_right.slider(
        f"{sensor} severity",
        -2.0, 2.0, -0.8,
        step=0.1,
        key=f"shap_{sensor}",
    )
    shap_values.append(val)

if selected_sensors:
    if st.button("Find Root Cause", type="primary"):
        with st.spinner("Traversing knowledge graph..."):
            result = linker.get_repair_paths(
                top_features=selected_sensors,
                shap_values=shap_values,
                max_paths=10,
            )

        st.success(
            f"Top cause: **{result['top_cause']}** | "
            f"Top repair: **{result['top_repair']}**"
        )

        st.markdown("**Repair paths found:**")
        paths = result["repair_paths"]
        if paths:
            paths_df = pd.DataFrame([
                {
                    "Priority":   p["repair_priority"],
                    "Repair":     p["repair_label"],
                    "Cause":      p["cause_label"],
                    "Symptom":    p["symptom_label"],
                    "Confidence": f"{min(p['final_score'] * 100, 99.9):.1f}%",
                    "Time":       p["repair_time"],
                }
                for p in paths[:8]
            ])
            st.dataframe(paths_df, use_container_width=True)
        else:
            st.info("No repair paths found for the selected sensors.")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — Node details lookup
# ══════════════════════════════════════════════════════════════

st.divider()
st.subheader("Node Details")

all_nodes     = list(G.nodes())
selected_node = st.selectbox(
    "Select a node to inspect",
    options=all_nodes,
    format_func=lambda n: G.nodes[n].get("label", n),
)

if selected_node:
    node_data = G.nodes[selected_node]
    node_type = node_data.get("type", "")

    d_left, d_right = st.columns(2)

    d_left.markdown(f"**Label:** {node_data.get('label', '')}")
    d_left.markdown(f"**Type:** {node_type.upper()}")
    d_left.markdown(f"**Severity:** {node_data.get('severity', '—')}")
    d_right.markdown(f"**Description:** {node_data.get('description', '—')}")

    if node_type == "repair":
        d_right.markdown(f"**Priority:** {node_data.get('priority', '—')}")
        d_right.markdown(f"**Est. Time:** {node_data.get('estimated_time', '—')}")

    successors   = list(G.successors(selected_node))
    predecessors = list(G.predecessors(selected_node))

    if successors:
        st.markdown("**Leads to:**")
        for s in successors:
            label  = G.nodes[s].get("label", s)
            weight = G.edges[selected_node, s].get("weight", 0.0)
            st.markdown(f"&nbsp;&nbsp;→ {label} &nbsp;*(confidence: {weight:.0%})*")

    if predecessors:
        st.markdown("**Triggered by:**")
        for p in predecessors:
            label = G.nodes[p].get("label", p)
            st.markdown(f"&nbsp;&nbsp;← {label}")
