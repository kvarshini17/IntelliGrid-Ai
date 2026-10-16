"""
pages/dashboard.py
---------------------
Purpose : Dashboard page -- overview KPIs and charts. Presentation only; all
          numbers come from the already-computed pipeline/summary results.
"""

import pandas as pd
import plotly.express as px
import streamlit as st


def render(df: pd.DataFrame, summary_df: pd.DataFrame, pipeline_results: list):
    st.header("Dashboard")

    total = len(summary_df)
    occupied = int((summary_df["occupancy"] > 0).sum())
    unoccupied = total - occupied
    potential_wastage = int(summary_df["potential_wastage"].sum())
    anomalies = int(summary_df["is_anomaly"].sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", total)
    col2.metric("Occupied", occupied)
    col3.metric("Unoccupied", unoccupied)
    col4.metric("Potential Wastage Records", potential_wastage)
    col5.metric("Anomalies", anomalies)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Energy Consumption Over Time")
        ts_df = summary_df.copy()
        ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
        daily = ts_df.groupby(ts_df["timestamp"].dt.date)["energy_consumption"].sum().reset_index()
        fig = px.line(daily, x="timestamp", y="energy_consumption",
                      labels={"timestamp": "Date", "energy_consumption": "Total Energy Consumption"})
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Electricity Tariff Distribution")
        tariff_counts = summary_df["tariff_category"].value_counts().reset_index()
        tariff_counts.columns = ["tariff_category", "count"]
        fig2 = px.pie(tariff_counts, names="tariff_category", values="count", hole=0.4)
        st.plotly_chart(fig2, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Condition Distribution")
        cond_counts = summary_df["condition_label"].value_counts().reset_index()
        cond_counts.columns = ["condition_label", "count"]
        fig3 = px.bar(cond_counts, x="condition_label", y="count")
        fig3.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig3, width='stretch')

    with col_d:
        st.subheader("Anomaly Type Distribution")
        anom_counts = summary_df["anomaly_type"].value_counts().reset_index()
        anom_counts.columns = ["anomaly_type", "count"]
        fig4 = px.bar(anom_counts, x="anomaly_type", y="count")
        st.plotly_chart(fig4, width='stretch')

    st.divider()
    st.subheader("Key Alerts")
    st.caption(
        "\"Potential wastage\" and \"anomaly\" are independent, condition-based / statistical "
        "indicators -- not confirmed ground-truth wastage or proof of a fault. See the About "
        "section for the full research-integrity notes."
    )
    alerts_df = summary_df[(summary_df["potential_wastage"]) | (summary_df["is_anomaly"])].copy()
    if alerts_df.empty:
        st.success("No potential wastage or anomaly conditions in the current selection.")
    else:
        display_cols = ["record_id", "timestamp", "room_id", "condition_label",
                         "potential_wastage", "wastage_type", "severity",
                         "anomaly_type", "z_score"]
        st.dataframe(alerts_df[display_cols].sort_values("timestamp"), width='stretch', height=300)
