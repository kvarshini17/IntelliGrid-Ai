"""
modules/pipeline.py
----------------------
Purpose : Thin orchestration helper that runs the full deterministic chain
          (condition -> wastage -> anomaly -> optimization -> recommendation)
          for every record and returns one merged list of dicts. Used by the
          Streamlit app (Phase 9) and the research experiments (Phase 10) so
          neither has to re-assemble the chain itself.

This module adds NO new detection/decision logic -- it only calls the
existing Phase 1-5 modules in order and merges their outputs.
"""

from typing import Dict, List

import pandas as pd

from modules.optimization import optimize_many
from modules.recommendation import generate_recommendations


def run_full_pipeline(df: pd.DataFrame) -> List[Dict]:
    """
    Run condition -> wastage -> anomaly -> optimization -> recommendation
    for every row of df.

    Output: list of dicts, each containing every field produced by Phases
    1-4 (via optimize_many) plus a "recommendations" key (Phase 5's output
    for that record).
    """
    optimization_results = optimize_many(df)
    for record in optimization_results:
        record["recommendations"] = generate_recommendations(record)
    return optimization_results


def pipeline_to_summary_dataframe(pipeline_results: List[Dict]) -> pd.DataFrame:
    """
    Build a flat, scalar-only DataFrame from pipeline results, suitable for
    filtering and charting in the UI (nested fields like
    `optimization_actions` and `recommendations` are dropped here -- look
    them up on the original pipeline_results list by record_id when needed).
    """
    scalar_fields = [
        "record_id", "timestamp", "room_id", "square_footage", "occupancy",
        "occupancy_state", "hour", "temperature", "temperature_context",
        "hvac_status", "lighting_status", "appliance", "appliance_status",
        "electricity_tariff", "tariff_category", "energy_consumption",
        "condition_code", "condition_label",
        "potential_wastage", "wastage_type", "severity", "reason",
        "context_group", "context_count", "context_mean", "context_std",
        "z_score", "anomaly_type", "is_anomaly", "anomaly_reason",
        "has_turn_off_recommendation",
    ]
    rows = [{k: r.get(k) for k in scalar_fields} for r in pipeline_results]
    return pd.DataFrame(rows)
