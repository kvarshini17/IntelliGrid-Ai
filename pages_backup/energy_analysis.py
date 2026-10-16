"""
pages/energy_analysis.py
----------------------------
Purpose : Energy Analysis page -- select a record, inspect its full
          condition/wastage/anomaly context, with filtering across the
          dataset. Presentation only.
"""

import pandas as pd
import plotly.express as px
import streamlit as st


def render(df: pd.DataFrame, summary_df: pd.DataFrame, pipeline_results: list):
    st.header("Energy Analysis")

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        rooms = col1.multiselect("Room", sorted(summary_df["room_id"].unique()))
        hvac = col2.multiselect("HVAC Status", ["On", "Off"])
        lighting = col3.multiselect("Lighting Status", ["On", "Off"])

        col4, col5 = st.columns(2)
        occ_filter = col4.selectbox("Occupancy", ["All", "Occupied", "Unoccupied"])
        wastage_filter = col5.selectbox("Potential Wastage", ["All", "Yes", "No"])

    filtered = summary_df.copy()
    if rooms:
        filtered = filtered[filtered["room_id"].isin(rooms)]
    if hvac:
        filtered = filtered[filtered["hvac_status"].isin(hvac)]
    if lighting:
        filtered = filtered[filtered["lighting_status"].isin(lighting)]
    if occ_filter == "Occupied":
        filtered = filtered[filtered["occupancy"] > 0]
    elif occ_filter == "Unoccupied":
        filtered = filtered[filtered["occupancy"] == 0]
    if wastage_filter == "Yes":
        filtered = filtered[filtered["potential_wastage"]]
    elif wastage_filter == "No":
        filtered = filtered[~filtered["potential_wastage"]]

    st.caption(f"Showing {len(filtered)} of {len(summary_df)} records.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Energy Consumption vs Occupancy")
        fig = px.scatter(filtered, x="occupancy", y="energy_consumption", color="hvac_status", opacity=0.6)
        st.plotly_chart(fig, width='stretch')
    with col_b:
        st.subheader("Energy Consumption by Room")
        by_room = filtered.groupby("room_id")["energy_consumption"].sum().reset_index()
        fig2 = px.bar(by_room.sort_values("energy_consumption", ascending=False),
                       x="room_id", y="energy_consumption")
        st.plotly_chart(fig2, width='stretch')

    st.divider()
    st.subheader("Record Detail")
    if filtered.empty:
        st.warning("No records match the current filters.")
        return

    record_ids = filtered["record_id"].tolist()
    selected_id = st.selectbox("Select a record", record_ids)

    record = next(r for r in pipeline_results if r["record_id"] == selected_id)

    col1, col2, col3 = st.columns(3)
    col1.metric("Condition", record["condition_label"])
    col2.metric("Potential Wastage", "Yes" if record["potential_wastage"] else "No")
    col3.metric("Anomaly", record["anomaly_type"])

    st.markdown(f"**Room:** {record['room_id']} ({record['square_footage']} sq ft)")
    st.markdown(f"**Timestamp:** {record['timestamp']}")
    st.markdown(f"**Occupancy:** {record['occupancy']}  |  **HVAC:** {record['hvac_status']}  |  "
                f"**Lighting:** {record['lighting_status']}  |  **Appliance:** {record['appliance']} "
                f"({record['appliance_status']})")
    st.markdown(f"**Temperature:** {record['temperature']:.1f} ({record['temperature_context']})")
    st.markdown(f"**Electricity Tariff:** {record['electricity_tariff']} ({record['tariff_category']})")
    st.markdown(f"**Energy Consumption:** {record['energy_consumption']:.2f}")

    st.markdown("---")
    st.markdown(f"**Wastage reason:** {record['reason']}")
    st.markdown(f"**Anomaly reason:** {record['anomaly_reason']}")
    if record["anomaly_type"] != "INSUFFICIENT_BASELINE":
        st.caption(
            f"Context group: {record['context_group']} (n={record['context_count']}, "
            f"mean={record['context_mean']:.2f}, std={record['context_std']})  |  "
            f"z-score: {record['z_score'] if record['z_score'] is None else round(record['z_score'], 2)}"
        )
