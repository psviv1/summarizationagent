# app.py
import json
import os

import streamlit as st

from summarization_graph import graph  # import your LangGraph graph


# ---------- Helper to run the agent ----------
def run_portfolio_agent(user_id: str):
    initial_state = {"user_id": user_id}
    result = graph.invoke(initial_state)
    return result


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="AR Portfolio Summarization Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AR Portfolio Summarization Agent (Rule-Based)")
st.write(
    "This app runs the LangGraph agent on the synthetic AR data in `./synthetic_ar` "
    "and shows the KPIs, rule-based summary, and actions."
)

# Sidebar controls
st.sidebar.header("Run Settings")
default_user_id = "analyst_123"
user_id = st.sidebar.text_input("User ID", value=default_user_id)

if st.sidebar.button("Run Agent"):
    with st.spinner("Running agent on synthetic_ar data..."):
        try:
            result = run_portfolio_agent(user_id)
        except Exception as e:
            st.error(f"Error running agent: {e}")
            st.stop()

    # Save into session_state so we can re-display without re-running
    st.session_state["agent_result"] = result

# If we have a previous result, display it
result = st.session_state.get("agent_result")

if not result:
    st.info("Click **Run Agent** in the sidebar to generate a portfolio summary.")
    st.stop()

# ---------- Top-level run info ----------
st.subheader("Run Metadata")
cols_meta = st.columns(3)
with cols_meta[0]:
    st.metric("Run ID", result.get("run_id", "N/A"))
with cols_meta[1]:
    st.metric("User ID", result.get("user_id", user_id))
with cols_meta[2]:
    st.metric("Stage", result.get("stage", "N/A"))

if result.get("error"):
    st.warning(f"Agent reported an error: {result['error']}")

# ---------- KPIs from Node C ----------
kpis = result.get("kpis", {})
st.subheader("Key KPIs (Node C)")

if not kpis:
    st.write("No KPIs found in state.")
else:
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("DSO (90d)", kpis.get("dso_90", "N/A"))
        st.metric("Failed Promises (30d)", kpis.get("failed_promises_30d", 0))
    with k2:
        st.metric("Aging 30+", kpis.get("aging_30", 0))
        st.metric("Aging 60+", kpis.get("aging_60", 0))
    with k3:
        st.metric("Aging 90+", kpis.get("aging_90", 0))
        st.metric("Unapplied Cash (cents)", kpis.get("unapplied_cents", 0))

    st.write("**Dispute Rate (90d):** ", f"{kpis.get('dispute_rate_90', 0)}%")

    # Optional: show a few top high-util accounts
    high_util = kpis.get("high_util_accounts", [])
    if high_util:
        st.markdown("**High Utilization Accounts (≥95%) — top 5**")
        st.dataframe(high_util[:5])

# ---------- Summary (Nodes D/E) ----------
validated = result.get("validated") or {}
summary = result.get("summary") or {}

st.subheader("Rule-Based Summary (Validated)")

if not validated:
    st.write("No validated summary found.")
else:
    # Markdown narrative
    md = validated.get("narrative_markdown") or "# Summary\n\n(No narrative.)"
    st.markdown(md)

    # Actions bullets
    actions = validated.get("actions_bullets", [])
    if actions:
        st.markdown("**Recommended Actions:**")
        for a in actions:
            st.markdown(f"- {a}")

# ---------- Files written (if any) ----------
json_path = result.get("json_path")
md_path = result.get("md_path")

if json_path or md_path:
    st.subheader("Saved Artifacts")
    if json_path:
        st.write(f"JSON summary file: `{json_path}`")
    if md_path:
        st.write(f"Markdown summary file: `{md_path}`")

# ---------- Debug / developer view ----------
with st.expander("🔍 Developer View: Raw State", expanded=False):
    st.markdown("**Raw KPIs dict:**")
    st.json(kpis)

    st.markdown("**Raw summary (Node D output):**")
    st.json(summary)

    st.markdown("**Validated summary (Node E output):**")
    st.json(validated)
