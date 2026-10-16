"""
views/dashboard.py
---------------------
Purpose : Dashboard page -- overview KPIs, charts and alert tables.
          Presentation only; every number comes from the already-computed
          pipeline/summary results. No detection logic lives here.

Presentation rules this page follows (see the "About" page for the full
research-integrity notes):
    - "Potential wastage" (Phase 2, condition-based) and "anomaly" (Phase 3,
      statistical) are two independent signals. They are never merged into one
      count or one table.
    - A day with fewer than 24 hourly records is a PARTIAL day. It is shown but
      visibly marked, and never drawn as if it were comparable to full days.
    - energy_consumption has no documented unit, so axes say "dataset units".
"""

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

HOURS_PER_FULL_DAY = 24  # the dataset is hourly

ANOMALY_LABELS: Dict[str, str] = {
    "NORMAL": "Normal",
    "HIGH_ENERGY_ANOMALY": "High-energy anomaly",
    "LOW_ENERGY_ANOMALY": "Low-energy anomaly",
    "INSUFFICIENT_BASELINE": "Not classified (insufficient baseline)",
}
ANOMALY_COLORS: Dict[str, str] = {
    "Normal": "#4c78a8",
    "High-energy anomaly": "#e45756",
    "Low-energy anomaly": "#f58518",
    "Not classified (insufficient baseline)": "#9d9d9d",
}
WASTAGE_TYPE_LABELS: Dict[str, str] = {
    "HVAC": "HVAC",
    "LIGHTING": "Lighting",
    "HVAC_AND_LIGHTING": "HVAC + Lighting",
}
# Tariff bands as documented in TRANSFORM_LOG.md / config.py.
TARIFF_BAND_LABELS: Dict[str, str] = {
    "Off-Peak": "Off-Peak (hours 0-5, 23)",
    "Standard": "Standard (hours 6-17)",
    "Peak": "Peak (hours 18-22)",
}


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit calls) -- unit-tested in tests/test_app.py
# ---------------------------------------------------------------------------
def _daily_totals(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily energy totals, with each day's record count and a partial-day flag.

    A day is "partial" when it has fewer than HOURS_PER_FULL_DAY hourly
    records. Totals of partial days are NOT comparable with full days.
    """
    ts = pd.to_datetime(summary_df["timestamp"])
    daily = (
        summary_df.assign(date=ts.dt.normalize())
        .groupby("date")
        .agg(records=("energy_consumption", "size"), total_energy=("energy_consumption", "sum"))
        .reset_index()
    )
    daily["is_partial"] = daily["records"] < HOURS_PER_FULL_DAY
    return daily


def _signal_overlap_counts(summary_df: pd.DataFrame) -> Dict[str, int]:
    """How many records each independent signal flags, and how many both."""
    wastage = summary_df["potential_wastage"].astype(bool)
    anomaly = summary_df["is_anomaly"].astype(bool)
    return {
        "wastage_only": int((wastage & ~anomaly).sum()),
        "anomaly_only": int((~wastage & anomaly).sum()),
        "both": int((wastage & anomaly).sum()),
    }


def _wastage_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Readable table of potential-wastage records (condition-based)."""
    w = summary_df[summary_df["potential_wastage"].astype(bool)]
    table = pd.DataFrame({
        "Timestamp": pd.to_datetime(w["timestamp"]),
        "Room": w["room_id"],
        "Operating condition": w["condition_label"],
        "Wastage type": w["wastage_type"].map(WASTAGE_TYPE_LABELS),
        "Priority": w["severity"].str.title(),
        "Energy consumption (context only)": w["energy_consumption"].round(2),
        "Tariff band": w["tariff_category"],
    })
    return table.sort_values("Timestamp").reset_index(drop=True)


def _anomaly_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Readable table of statistical anomalies (context-based)."""
    a = summary_df[summary_df["is_anomaly"].astype(bool)]
    table = pd.DataFrame({
        "Timestamp": pd.to_datetime(a["timestamp"]),
        "Room": a["room_id"],
        "Direction": a["anomaly_type"].map(
            {"HIGH_ENERGY_ANOMALY": "High", "LOW_ENERGY_ANOMALY": "Low"}
        ),
        "Operating condition": a["condition_label"],
        "Energy consumption": a["energy_consumption"].round(2),
        "Comparable-context mean": a["context_mean"].round(2),
        "z-score": a["z_score"].round(2),
        "Comparable records (n)": a["context_count"],
        "Also flagged as potential wastage": a["potential_wastage"].map({True: "Yes", False: "No"}),
    })
    return table.sort_values("Timestamp").reset_index(drop=True)


def _energy_over_time_figure(daily: pd.DataFrame) -> go.Figure:
    """Daily totals: full days as a line, partial day(s) as a separate, labelled marker."""
    full = daily[~daily["is_partial"]]
    partial = daily[daily["is_partial"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full["date"], y=full["total_energy"], mode="lines+markers",
        name=f"Full day ({HOURS_PER_FULL_DAY} hourly records)", line=dict(color="#4c78a8"),
        hovertemplate="%{x|%Y-%m-%d}<br>Total: %{y:.1f}<extra></extra>",
    ))
    if not partial.empty:
        fig.add_trace(go.Scatter(
            x=partial["date"], y=partial["total_energy"], mode="markers",
            name="Partial day (not comparable)",
            marker=dict(symbol="diamond-open", size=13, color="#e45756", line=dict(width=2)),
            customdata=partial["records"],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Total: %{y:.1f}"
                f"<br>Only %{{customdata}} of {HOURS_PER_FULL_DAY} hourly records<extra></extra>"
            ),
        ))
        for _, row in partial.iterrows():
            fig.add_annotation(
                x=row["date"], y=row["total_energy"], ax=-60, ay=-40,
                text=f"Partial day: {int(row['records'])} of {HOURS_PER_FULL_DAY} hours",
                showarrow=True, arrowhead=2, font=dict(color="#e45756"),
            )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Total energy consumption per day (dataset units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame, summary_df: pd.DataFrame, pipeline_results: list):
    st.header("Dashboard")

    total = len(summary_df)
    occupied = int((summary_df["occupancy"] > 0).sum())
    unoccupied = total - occupied
    potential_wastage = int(summary_df["potential_wastage"].sum())
    anomalies = int(summary_df["is_anomaly"].sum())
    not_classified = int((summary_df["anomaly_type"] == "INSUFFICIENT_BASELINE").sum())
    overlap = _signal_overlap_counts(summary_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", total)
    c2.metric("Occupied records", occupied)
    c3.metric("Unoccupied records", unoccupied)

    c4, c5 = st.columns(2)
    c4.metric(
        "Potential wastage records (condition-based)", potential_wastage,
        help="Occupancy is 0 while HVAC and/or lighting is On. Decided only from occupancy and "
             "appliance state -- never from energy consumption. Not confirmed wastage.",
    )
    c5.metric(
        "Statistical anomalies (context-based)", anomalies,
        help="Energy consumption is unusually high or low compared with records that share the same "
             "occupancy, HVAC, lighting and hour. A statistical flag, not proof of a fault or of wastage.",
    )

    st.caption(
        "**These are two independent signals.** "
        f"Flagged as potential wastage only: **{overlap['wastage_only']}** | "
        f"anomaly only: **{overlap['anomaly_only']}** | both: **{overlap['both']}**. "
        f"Anomaly detection could assess {total - not_classified} of {total} records; "
        f"{not_classified} fall in context groups with fewer than {config.MIN_CONTEXT_GROUP_SIZE} "
        "comparable records and are not classified."
    )

    st.divider()

    daily = _daily_totals(summary_df)
    partial_days = daily[daily["is_partial"]]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Daily energy consumption")
        st.plotly_chart(_energy_over_time_figure(daily), width="stretch")
        if not partial_days.empty:
            labels = ", ".join(
                f"{r.date:%Y-%m-%d} ({int(r.records)} of {HOURS_PER_FULL_DAY} hours)"
                for r in partial_days.itertuples()
            )
            st.caption(
                f"Partial day(s): {labels}. Their totals are lower only because fewer hours were "
                "recorded; do not compare them with full days."
            )

    with col_b:
        st.subheader("Records by electricity tariff band")
        tariff_counts = summary_df["tariff_category"].value_counts().reset_index()
        tariff_counts.columns = ["Tariff band", "Records"]
        tariff_counts["Tariff band"] = tariff_counts["Tariff band"].map(TARIFF_BAND_LABELS)
        fig2 = px.pie(tariff_counts, names="Tariff band", values="Records", hole=0.4)
        st.plotly_chart(fig2, width="stretch")
        st.caption("Tariff bands are a documented project assumption (see TRANSFORM_LOG.md), not a real utility schedule.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Records by operating condition")
        cond_counts = (
            summary_df.groupby(["condition_label", "occupancy_state"]).size().reset_index(name="Records")
            .rename(columns={"condition_label": "Operating condition", "occupancy_state": "Occupancy"})
            .sort_values("Records")
        )
        fig3 = px.bar(cond_counts, x="Records", y="Operating condition", color="Occupancy", orientation="h")
        st.plotly_chart(fig3, width="stretch")

    with col_d:
        st.subheader("Records by anomaly status")
        anom_counts = summary_df["anomaly_type"].map(ANOMALY_LABELS).value_counts().reset_index()
        anom_counts.columns = ["Anomaly status", "Records"]
        fig4 = px.bar(anom_counts, x="Records", y="Anomaly status", orientation="h",
                      color="Anomaly status", color_discrete_map=ANOMALY_COLORS)
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, width="stretch")
        st.caption("\"Not classified\" means too few comparable records to build a baseline -- it is not the same as \"Normal\".")

    st.divider()
    st.subheader("Alerts")
    tab_wastage, tab_anomaly = st.tabs([
        f"Potential wastage ({potential_wastage})", f"Statistical anomalies ({anomalies})",
    ])

    with tab_wastage:
        st.caption(
            "Condition-based: the room is unoccupied while HVAC and/or lighting is On. "
            "**Priority** ranks how many appliance categories are involved; it is not an amount of "
            "energy lost. Energy consumption is shown for context only and is not used to decide this flag."
        )
        wastage_table = _wastage_table(summary_df)
        if wastage_table.empty:
            st.info("No potential-wastage records in the current data.")
        else:
            st.dataframe(
                wastage_table, hide_index=True, width="stretch", height=320,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "Energy consumption (context only)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

    with tab_anomaly:
        st.caption(
            f"Statistical: energy consumption is at least {config.ANOMALY_Z_THRESHOLD:g} standard deviations "
            "from the mean of comparable records (same occupancy, HVAC, lighting and hour). "
            "This flags unusual values; it does not say why, and it does not mean wastage."
        )
        anomaly_table = _anomaly_table(summary_df)
        if anomaly_table.empty:
            st.info("No statistical anomalies in the current data.")
        else:
            st.dataframe(
                anomaly_table, hide_index=True, width="stretch", height=320,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "Energy consumption": st.column_config.NumberColumn(format="%.2f"),
                    "Comparable-context mean": st.column_config.NumberColumn(format="%.2f"),
                    "z-score": st.column_config.NumberColumn(format="%.2f"),
                },
            )
