"""
modules/recommendation.py
----------------------------
Purpose : Recommendation Engine (Phase 5).

Converts condition + wastage + anomaly + optimization (Phases 1-4) into
structured, explainable, per-issue recommendations. Pure synthesis/templating
layer -- no new detection logic, no LLM, no fabricated savings.

Design note -- reuse, not duplication:
    Consumes the dict produced by `optimization.optimize_record()` (which
    itself chains Phase 1 -> 2 -> 3 -> 4). This module adds only
    presentation/synthesis: it never re-derives condition, wastage, anomaly,
    or action-selection logic.

AI technique used : Deterministic templating over already-computed
                     structured results. No ML, no LLM.
"""

from typing import Dict, List, Union

import pandas as pd

from modules.optimization import optimize_many, optimize_record

Record = Union[pd.Series, dict]

PRIORITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
_PRIORITY_RANK = {level: i for i, level in enumerate(PRIORITY_LEVELS)}


def _combine_priority(wastage_severity: str, is_anomalous: bool) -> str:
    """
    Deterministic priority rule (documented, not learned):
      - wastage + anomaly together           -> HIGH
      - anomaly alone (occupied or otherwise) -> MEDIUM (worth investigating,
        regardless of wastage status, since it may indicate an equipment issue)
      - wastage alone                          -> inherits Phase 2's own
        severity (MEDIUM or HIGH)
      - neither                                -> NONE
    """
    if wastage_severity != "NONE" and is_anomalous:
        return "HIGH"
    if is_anomalous:
        return "MEDIUM"
    if wastage_severity != "NONE":
        return wastage_severity
    return "NONE"


def generate_recommendations(record_result: Dict) -> List[Dict]:
    """
    Produce the list of recommendations for a single record.

    Input:  the dict returned by optimization.optimize_record() (already
            carries Phase 1-4 fields).
    Output: list of recommendation dicts, each with:
            issue, evidence, recommended_action, reason, priority, supporting_metrics
    """
    recs: List[Dict] = []
    is_anomalous = record_result["anomaly_type"] in ("HIGH_ENERGY_ANOMALY", "LOW_ENERGY_ANOMALY")
    wastage_severity = record_result.get("severity", "NONE")

    # --- Wastage/optimization-driven recommendations (one per appliance with a TURN_OFF action) ---
    for action in record_result["optimization_actions"]:
        if action["selected_action"] != "TURN_OFF":
            continue
        recs.append({
            "issue": f"Potential {action['appliance']} Wastage",
            "evidence": {
                "condition_label": record_result["condition_label"],
                "occupancy": record_result["occupancy"],
                "hvac_status": record_result["hvac_status"],
                "lighting_status": record_result["lighting_status"],
                "wastage_reason": record_result["reason"],
            },
            "recommended_action": f"Turn off {action['appliance']}.",
            "reason": f"Consider turning off the {action['appliance']} because the room is currently unoccupied.",
            "priority": _combine_priority(wastage_severity, is_anomalous),
            "supporting_metrics": {
                "energy_consumption": record_result["energy_consumption"],
                "electricity_tariff": record_result["electricity_tariff"],
                "tariff_category": record_result["tariff_category"],
            },
        })

    # --- Anomaly-driven recommendation (independent of wastage) ---
    if is_anomalous:
        direction = "high" if record_result["anomaly_type"] == "HIGH_ENERGY_ANOMALY" else "low"
        active_appliances = [
            name for name, status in (("HVAC", record_result["hvac_status"]),
                                       ("lighting", record_result["lighting_status"]))
            if status == "On"
        ]
        appliance_phrase = " and ".join(active_appliances) if active_appliances else "equipment"
        recs.append({
            "issue": f"{'High' if direction == 'high' else 'Low'}-Energy Anomaly",
            "evidence": {
                "condition_label": record_result["condition_label"],
                "context_group": record_result["context_group"],
                "context_count": record_result["context_count"],
                "anomaly_reason": record_result["anomaly_reason"],
            },
            "recommended_action": f"Review the {appliance_phrase} setting and room conditions.",
            "reason": (
                f"Energy consumption is unusually {direction} for the current operating context "
                f"(z-score={record_result['z_score']:.2f} relative to a contextual baseline of "
                f"{record_result['context_count']} comparable observation(s)). Review the equipment "
                f"settings and room conditions. This reflects a statistical deviation, not a confirmed cause."
            ),
            "priority": _combine_priority(wastage_severity, is_anomalous),
            "supporting_metrics": {
                "z_score": record_result["z_score"],
                "context_mean": record_result["context_mean"],
                "context_std": record_result["context_std"],
                "context_count": record_result["context_count"],
                "energy_consumption": record_result["energy_consumption"],
            },
        })

    # --- Nothing to report ---
    if not recs:
        recs.append({
            "issue": "No Issue Detected",
            "evidence": {"condition_label": record_result["condition_label"]},
            "recommended_action": "No action needed.",
            "reason": "No potential wastage or anomalous energy consumption was detected for this record.",
            "priority": "NONE",
            "supporting_metrics": {},
        })

    return recs


def generate_recommendations_many(df: pd.DataFrame) -> List[Dict]:
    """
    Run generate_recommendations() over every row of df.

    Output: list of {"record_id": ..., "recommendations": [...]}, one entry
    per input row, same order.
    """
    optimization_results = optimize_many(df)
    return [
        {"record_id": r["record_id"], "recommendations": generate_recommendations(r)}
        for r in optimization_results
    ]


def summarize_recommendations(df: pd.DataFrame) -> Dict:
    """
    Aggregate recommendation statistics across a dataset.

    Output: total records, count of records with at least one actionable
    recommendation, counts by issue type, and counts by priority level.
    """
    if df is None or df.empty:
        return {"total_records": 0}

    per_record = generate_recommendations_many(df)
    total = len(per_record)

    issue_counts: Dict[str, int] = {}
    priority_counts: Dict[str, int] = {level: 0 for level in PRIORITY_LEVELS}
    records_with_actionable_recommendation = 0

    for entry in per_record:
        has_actionable = False
        for rec in entry["recommendations"]:
            issue_counts[rec["issue"]] = issue_counts.get(rec["issue"], 0) + 1
            priority_counts[rec["priority"]] = priority_counts.get(rec["priority"], 0) + 1
            if rec["issue"] != "No Issue Detected":
                has_actionable = True
        if has_actionable:
            records_with_actionable_recommendation += 1

    return {
        "total_records": total,
        "records_with_actionable_recommendation": records_with_actionable_recommendation,
        "issue_counts": issue_counts,
        "priority_counts": priority_counts,
    }
