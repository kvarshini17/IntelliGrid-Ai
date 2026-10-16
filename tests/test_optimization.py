"""
tests/test_optimization.py
------------------------------
Tests for modules/optimization.py (Phase 4).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_manager import load_processed_dataset
from modules.anomaly_detection import analyze_anomalies
from modules.optimization import (
    generate_candidate_actions,
    optimize_dataset,
    optimize_many,
    optimize_record,
    select_best_action,
)


def _base_record(**overrides):
    record = {
        "timestamp": "2022-01-01 10:00:00", "hour": 10, "room_id": "Room 101", "square_footage": 1000,
        "occupancy": 0, "temperature": 24.0, "hvac_usage": "Off", "lighting_usage": "Off",
        "appliance": "None", "appliance_status": "Off", "electricity_tariff": 7.0, "energy_consumption": 70.0,
    }
    record.update(overrides)
    return record


def _result_for(**overrides):
    """Run a synthetic record through the full Phase1->3 pipeline to get a
    realistic record_result dict for optimization to consume."""
    df = pd.DataFrame([_base_record(**overrides)])
    return analyze_anomalies(df)[0]


# ---------------------------------------------------------------------------
# Candidate generation & hard constraints
# ---------------------------------------------------------------------------
def test_unoccupied_hvac_on_offers_turn_off():
    r = _result_for(occupancy=0, hvac_usage="On", lighting_usage="Off")
    candidates = generate_candidate_actions(r, "HVAC")
    actions = {c.action for c in candidates}
    assert "TURN_OFF" in actions
    assert "KEEP" in actions


def test_occupied_hvac_on_never_offers_turn_off():
    r = _result_for(occupancy=3, hvac_usage="On", lighting_usage="Off")
    candidates = generate_candidate_actions(r, "HVAC")
    actions = {c.action for c in candidates}
    assert "TURN_OFF" not in actions
    assert actions == {"KEEP"}


def test_occupied_lighting_on_never_offers_turn_off():
    r = _result_for(occupancy=2, hvac_usage="Off", lighting_usage="On")
    candidates = generate_candidate_actions(r, "Lighting")
    actions = {c.action for c in candidates}
    assert "TURN_OFF" not in actions


def test_appliance_already_off_only_offers_keep():
    r = _result_for(occupancy=0, hvac_usage="Off", lighting_usage="Off")
    candidates = generate_candidate_actions(r, "HVAC")
    assert {c.action for c in candidates} == {"KEEP"}


# ---------------------------------------------------------------------------
# Selection / scoring
# ---------------------------------------------------------------------------
def test_select_best_action_prefers_lower_score():
    r = _result_for(occupancy=0, hvac_usage="On", lighting_usage="Off")
    candidates = generate_candidate_actions(r, "HVAC")
    best = select_best_action(candidates)
    assert best.action == "TURN_OFF"
    assert best.score < 0


def test_select_best_action_keep_when_only_feasible():
    r = _result_for(occupancy=3, hvac_usage="On", lighting_usage="Off")
    candidates = generate_candidate_actions(r, "HVAC")
    best = select_best_action(candidates)
    assert best.action == "KEEP"


# ---------------------------------------------------------------------------
# Full record-level worked examples (match the project spec's examples)
# ---------------------------------------------------------------------------
def test_worked_example_unoccupied_hvac_on():
    r = _result_for(occupancy=0, hvac_usage="On", lighting_usage="Off")
    result = optimize_record(r)
    hvac_action = next(a for a in result["optimization_actions"] if a["appliance"] == "HVAC")
    assert hvac_action["selected_action"] == "TURN_OFF"
    assert result["has_turn_off_recommendation"] is True


def test_worked_example_unoccupied_lighting_on():
    r = _result_for(occupancy=0, hvac_usage="Off", lighting_usage="On")
    result = optimize_record(r)
    light_action = next(a for a in result["optimization_actions"] if a["appliance"] == "Lighting")
    assert light_action["selected_action"] == "TURN_OFF"


def test_worked_example_unoccupied_hvac_and_lighting():
    r = _result_for(occupancy=0, hvac_usage="On", lighting_usage="On")
    result = optimize_record(r)
    selected = {a["appliance"]: a["selected_action"] for a in result["optimization_actions"]}
    assert selected == {"HVAC": "TURN_OFF", "Lighting": "TURN_OFF"}


def test_worked_example_occupied_hvac_on_never_auto_off():
    r = _result_for(occupancy=4, hvac_usage="On", lighting_usage="Off")
    result = optimize_record(r)
    hvac_action = next(a for a in result["optimization_actions"] if a["appliance"] == "HVAC")
    assert hvac_action["selected_action"] == "KEEP"
    assert result["has_turn_off_recommendation"] is False


def test_worked_example_occupied_lighting_on_never_auto_off():
    r = _result_for(occupancy=4, hvac_usage="Off", lighting_usage="On")
    result = optimize_record(r)
    light_action = next(a for a in result["optimization_actions"] if a["appliance"] == "Lighting")
    assert light_action["selected_action"] == "KEEP"


# ---------------------------------------------------------------------------
# Score breakdown fields present and never fabricate kWh/cost
# ---------------------------------------------------------------------------
def test_score_breakdown_fields_present_and_zero_by_design():
    r = _result_for(occupancy=0, hvac_usage="On", lighting_usage="Off")
    result = optimize_record(r)
    hvac_action = next(a for a in result["optimization_actions"] if a["appliance"] == "HVAC")
    assert hvac_action["constraint_penalty"] == 0.0
    assert hvac_action["comfort_penalty"] == 0.0
    assert hvac_action["operational_penalty"] == 0.0


def test_reason_text_does_not_claim_a_savings_figure():
    r = _result_for(occupancy=0, hvac_usage="On", lighting_usage="Off")
    result = optimize_record(r)
    hvac_action = next(a for a in result["optimization_actions"] if a["appliance"] == "HVAC")
    # must not contain a fabricated kWh/currency figure
    assert "kWh" not in hvac_action["reason"]
    assert "$" not in hvac_action["reason"] and "Rs." not in hvac_action["reason"]


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------
def test_optimize_dataset_empty_df_returns_zero():
    summary = optimize_dataset(pd.DataFrame())
    assert summary["total_records"] == 0


def test_optimize_dataset_synthetic():
    df = pd.DataFrame([
        _base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"),
        _base_record(occupancy=0, hvac_usage="Off", lighting_usage="On"),
        _base_record(occupancy=3, hvac_usage="On", lighting_usage="On"),
    ])
    summary = optimize_dataset(df)
    assert summary["total_records"] == 3
    assert summary["hvac_turn_off_recommendations"] == 1
    assert summary["lighting_turn_off_recommendations"] == 1
    assert summary["records_with_any_turn_off_recommendation"] == 2


# ---------------------------------------------------------------------------
# Real dataset integration -- cross-check against known Phase 2 counts
# ---------------------------------------------------------------------------
def test_real_dataset_turn_off_counts_match_known_wastage_counts():
    df = load_processed_dataset()
    summary = optimize_dataset(df)
    assert summary["total_records"] == 1000
    assert summary["hvac_turn_off_recommendations"] == 44
    assert summary["lighting_turn_off_recommendations"] == 50
    assert summary["records_with_any_turn_off_recommendation"] == 70


def test_real_dataset_no_occupied_record_ever_gets_turn_off():
    df = load_processed_dataset()
    results = optimize_many(df)
    occupied_with_turn_off = [
        r for r in results
        if r["occupancy"] > 0 and r["has_turn_off_recommendation"]
    ]
    assert len(occupied_with_turn_off) == 0
