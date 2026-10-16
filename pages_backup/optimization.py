"""
pages/optimization.py
------------------------
Purpose : Optimization page -- select a record, show its condition/wastage/
          anomaly status, candidate actions, and the selected action with
          full reasoning visible. Presentation only -- never silently
          modifies the dataset.
"""

import streamlit as st


def render(df, summary_df, pipeline_results: list):
    st.header("Optimization")
    st.caption(
        "Candidate actions and scores below are deterministic, explainable, ordinal rankings -- "
        "not quantitative kWh/cost savings estimates. See About for the full scoring-design notes."
    )

    record_ids = summary_df["record_id"].tolist()
    selected_id = st.selectbox("Select a record", record_ids)
    record = next(r for r in pipeline_results if r["record_id"] == selected_id)

    col1, col2, col3 = st.columns(3)
    col1.metric("Condition", record["condition_label"])
    col2.metric("Potential Wastage", "Yes" if record["potential_wastage"] else "No")
    col3.metric("Anomaly", record["anomaly_type"])

    st.markdown(f"**Tariff:** {record['electricity_tariff']} ({record['tariff_category']})  |  "
                f"**Energy Consumption:** {record['energy_consumption']:.2f}")

    st.divider()
    st.subheader("Candidate Actions")

    for action in record["optimization_actions"]:
        with st.container(border=True):
            st.markdown(f"**{action['appliance']}** -- selected action: `{action['selected_action']}`")
            st.caption(action["reason"])
            cand_rows = [
                {"Action": c["action"], "Feasible": c["feasible"], "Score": c["score"], "Reason": c["reason"]}
                for c in action["candidates"]
            ]
            st.dataframe(cand_rows, width='stretch', hide_index=True)
            st.caption(
                f"Score breakdown -- constraint_penalty: {action['constraint_penalty']}, "
                f"comfort_penalty: {action['comfort_penalty']}, operational_penalty: {action['operational_penalty']} "
                "(all currently 0 by documented design; see About)."
            )

    st.divider()
    st.subheader("Recommendations")
    for rec in record["recommendations"]:
        with st.container(border=True):
            st.markdown(f"**[{rec['priority']}] {rec['issue']}**")
            st.markdown(f"Action: {rec['recommended_action']}")
            st.caption(rec["reason"])
