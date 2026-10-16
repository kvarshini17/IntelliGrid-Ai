"""
tests/test_recommendation.py
--------------------------------
Tests for modules/recommendation.py (Phase 5).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_manager import load_processed_dataset
from modules.anomaly_detection import analyze_anomalies
from modules.optimization import optimize_record
from modules.recommendation import (
    PRIORITY_LEVELS,
    _combine_priority,
    generate_recommendations,
    generate_recommendations_many,
    summarize_recommendations,
)


def _base_record(**overrides):
    record = {
        "timestamp": "2022-01-01 10:00:00", "hour": 10, "room_id": "Room 101", "square_footage": 1000,
        "occupancy": 0, "temperature": 24.0, "hvac_usage": "Off", "lighting_usage": "Off",
        "appliance": "None", "appliance_status": "Off", "electricity_tariff": 7.0, "energy_consumption": 70.0,
    }
    record.update(overrides)
    return record


def _opt_result_for(df: pd.DataFrame, row_idx: int = 0) -> dict:
    anomaly_results = analyze_anomalies(df)
    return optimize_record(anomaly_results[row_idx])


# ---------------------------------------------------------------------------
# Priority combination logic
# ---------------------------------------------------------------------------
def test_combine_priority_neither():
    assert _combine_priority("NONE", False) == "NONE"


def test_combine_priority_wastage_only_inherits_severity():
    assert _combine_priority("MEDIUM", False) == "MEDIUM"
    assert _combine_priority("HIGH", False) == "HIGH"


def test_combine_priority_anomaly_only_is_medium():
    assert _combine_priority("NONE", True) == "MEDIUM"


def test_combine_priority_both_is_high():
    assert _combine_priority("MEDIUM", True) == "HIGH"
    assert _combine_priority("HIGH", True) == "HIGH"


def test_priority_levels_are_well_formed():
    assert PRIORITY_LEVELS == ["NONE", "LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Worked examples (match project spec examples)
# ---------------------------------------------------------------------------
def test_unoccupied_hvac_on_normal_anomaly_produces_turn_off_recommendation():
    """Condition: Unoccupied + HVAC ON, Wastage: potential, Anomaly: normal
    -> 'Consider turning off the HVAC because the room is currently unoccupied.'"""
    df = pd.DataFrame([_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off")])
    result = _opt_result_for(df)
    recs = generate_recommendations(result)
    hvac_recs = [r for r in recs if r["issue"] == "Potential HVAC Wastage"]
    assert len(hvac_recs) == 1
    assert hvac_recs[0]["recommended_action"] == "Turn off HVAC."
    assert hvac_recs[0]["reason"] == "Consider turning off the HVAC because the room is currently unoccupied."


def test_no_issue_detected_for_clean_record():
    df = pd.DataFrame([_base_record(occupancy=3, hvac_usage="On", lighting_usage="Off")])
    result = _opt_result_for(df)
    recs = generate_recommendations(result)
    assert len(recs) == 1
    assert recs[0]["issue"] == "No Issue Detected"
    assert recs[0]["priority"] == "NONE"


def test_occupied_hvac_wastage_never_appears():
    """Occupied + HVAC ON must never produce a 'Potential HVAC Wastage' recommendation."""
    df = pd.DataFrame([_base_record(occupancy=5, hvac_usage="On", lighting_usage="On")])
    result = _opt_result_for(df)
    recs = generate_recommendations(result)
    assert all(r["issue"] != "Potential HVAC Wastage" for r in recs)
    assert all(r["issue"] != "Potential Lighting Wastage" for r in recs)


# ---------------------------------------------------------------------------
# Anomaly-driven recommendation (engineered synthetic outlier)
# ---------------------------------------------------------------------------
def test_high_anomaly_produces_review_recommendation_not_turn_off():
    """Occupied + HVAC ON with unusually high energy -> a review recommendation,
    never a turn-off action (occupied rooms are never auto-turned-off)."""
    rows = []
    typical = [60, 61, 59, 62, 58, 60, 61]
    for v in typical:
        rows.append(_base_record(occupancy=4, hvac_usage="On", lighting_usage="Off", hour=9, energy_consumption=v))
    rows.append(_base_record(occupancy=4, hvac_usage="On", lighting_usage="Off", hour=9, energy_consumption=200))
    df = pd.DataFrame(rows)
    anomaly_results = analyze_anomalies(df)
    outlier_result = [r for r in anomaly_results if r["energy_consumption"] == 200][0]
    opt_result = optimize_record(outlier_result)
    recs = generate_recommendations(opt_result)

    assert any(r["issue"] == "High-Energy Anomaly" for r in recs)
    assert all(r["issue"] != "Potential HVAC Wastage" for r in recs)  # occupied -> never wastage
    anomaly_rec = [r for r in recs if r["issue"] == "High-Energy Anomaly"][0]
    assert anomaly_rec["priority"] == "MEDIUM"  # anomaly alone, no wastage
    assert "turn off" not in anomaly_rec["recommended_action"].lower()


def test_both_wastage_and_anomaly_produces_high_priority_for_both_recs():
    """Unoccupied + HVAC ON with unusually high energy -> both a wastage
    recommendation AND an anomaly recommendation, both at HIGH priority."""
    rows = []
    typical = [40, 41, 39, 40, 41, 39, 40]
    for v in typical:
        rows.append(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=14, energy_consumption=v))
    rows.append(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=14, energy_consumption=300))
    df = pd.DataFrame(rows)
    anomaly_results = analyze_anomalies(df)
    outlier_result = [r for r in anomaly_results if r["energy_consumption"] == 300][0]
    opt_result = optimize_record(outlier_result)
    recs = generate_recommendations(opt_result)

    issues = {r["issue"] for r in recs}
    assert "Potential HVAC Wastage" in issues
    assert "High-Energy Anomaly" in issues
    assert all(r["priority"] == "HIGH" for r in recs)


# ---------------------------------------------------------------------------
# Structure / research-integrity checks
# ---------------------------------------------------------------------------
def test_every_recommendation_has_required_fields():
    df = pd.DataFrame([_base_record(occupancy=0, hvac_usage="On", lighting_usage="On")])
    result = _opt_result_for(df)
    recs = generate_recommendations(result)
    for r in recs:
        for key in ("issue", "evidence", "recommended_action", "reason", "priority", "supporting_metrics"):
            assert key in r
        assert r["priority"] in PRIORITY_LEVELS


def test_recommendation_reason_does_not_fabricate_savings_figure():
    df = pd.DataFrame([_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off")])
    result = _opt_result_for(df)
    recs = generate_recommendations(result)
    for r in recs:
        assert "kWh saved" not in r["reason"]
        assert "% reduction" not in r["reason"]


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------
def test_summarize_recommendations_empty_df():
    summary = summarize_recommendations(pd.DataFrame())
    assert summary["total_records"] == 0


def test_summarize_recommendations_synthetic():
    df = pd.DataFrame([
        _base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"),
        _base_record(occupancy=3, hvac_usage="On", lighting_usage="On"),
    ])
    summary = summarize_recommendations(df)
    assert summary["total_records"] == 2
    assert summary["records_with_actionable_recommendation"] == 1
    assert summary["issue_counts"]["Potential HVAC Wastage"] == 1
    assert summary["issue_counts"]["No Issue Detected"] == 1


# ---------------------------------------------------------------------------
# Real dataset integration
# ---------------------------------------------------------------------------
def test_real_dataset_summary_counts_are_internally_consistent():
    df = load_processed_dataset()
    summary = summarize_recommendations(df)
    assert summary["total_records"] == 1000
    assert summary["issue_counts"]["Potential HVAC Wastage"] == 44
    assert summary["issue_counts"]["Potential Lighting Wastage"] == 50
    assert summary["issue_counts"]["High-Energy Anomaly"] == 7
    assert summary["issue_counts"]["Low-Energy Anomaly"] == 6
    # Priority counts must sum to total recommendation-entries, not total records
    # (a record can generate more than one recommendation entry).
    total_entries = sum(summary["issue_counts"].values())
    assert sum(summary["priority_counts"].values()) == total_entries


def test_real_dataset_runs_without_error_for_every_record():
    df = load_processed_dataset()
    per_record = generate_recommendations_many(df)
    assert len(per_record) == 1000
    for entry in per_record:
        assert len(entry["recommendations"]) >= 1
