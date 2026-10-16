"""
tests/test_energy_analysis.py
---------------------------------
Tests for modules/energy_analysis.py (Phase 1 -- Energy Condition Analysis).

Covers: all 8 condition rules, tariff classification, missing/invalid
required fields, "None" appliance handling, multi-record analysis, and
dataset-level aggregation against the real processed dataset.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_manager import load_processed_dataset
from modules.energy_analysis import (
    EnergyAnalysisError,
    analyze_dataset,
    analyze_many,
    analyze_record,
    get_tariff_category,
    get_temperature_context,
)


def _base_record(**overrides):
    """A minimal, valid record; individual tests override only what they need."""
    record = {
        "timestamp": "2022-01-01 10:00:00",
        "hour": 10,
        "room_id": "Room 101",
        "square_footage": 1000,
        "occupancy": 0,
        "temperature": 24.0,
        "hvac_usage": "Off",
        "lighting_usage": "Off",
        "appliance": "None",
        "appliance_status": "Off",
        "electricity_tariff": 7.0,
        "energy_consumption": 70.0,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Condition rule tests (Tests 1-6 from the spec, plus the remaining 2 states)
# ---------------------------------------------------------------------------
def test_condition_1_unoccupied_no_appliances():
    r = analyze_record(_base_record(occupancy=0, hvac_usage="Off", lighting_usage="Off"))
    assert r["condition_code"] == "U_NONE"
    assert r["condition_label"] == "Unoccupied - No Active Appliances"
    assert r["potential_wastage_indicator"] is False


def test_condition_2_unoccupied_hvac():
    r = analyze_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"))
    assert r["condition_code"] == "U_HVAC"
    assert r["condition_label"] == "Unoccupied - HVAC Active"
    assert r["potential_wastage_indicator"] is True


def test_condition_3_unoccupied_lighting():
    r = analyze_record(_base_record(occupancy=0, hvac_usage="Off", lighting_usage="On"))
    assert r["condition_code"] == "U_LIGHT"
    assert r["condition_label"] == "Unoccupied - Lighting Active"
    assert r["potential_wastage_indicator"] is True


def test_condition_4_unoccupied_hvac_and_lighting():
    r = analyze_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="On"))
    assert r["condition_code"] == "U_HVAC_LIGHT"
    assert r["condition_label"] == "Unoccupied - HVAC and Lighting Active"
    assert r["potential_wastage_indicator"] is True


def test_condition_5_occupied_hvac_and_lighting():
    r = analyze_record(_base_record(occupancy=4, hvac_usage="On", lighting_usage="On"))
    assert r["condition_code"] == "O_HVAC_LIGHT"
    assert r["condition_label"] == "Occupied - HVAC and Lighting Active"
    assert r["potential_wastage_indicator"] is False


def test_condition_6_occupied_no_appliances():
    r = analyze_record(_base_record(occupancy=3, hvac_usage="Off", lighting_usage="Off"))
    assert r["condition_code"] == "O_NONE"
    assert r["condition_label"] == "Occupied - No Active Appliances"
    assert r["potential_wastage_indicator"] is False


def test_condition_occupied_hvac_only():
    r = analyze_record(_base_record(occupancy=2, hvac_usage="On", lighting_usage="Off"))
    assert r["condition_code"] == "O_HVAC"
    assert r["assessment"] == "Normal HVAC operation."
    assert r["potential_wastage_indicator"] is False


def test_condition_occupied_lighting_only():
    r = analyze_record(_base_record(occupancy=2, hvac_usage="Off", lighting_usage="On"))
    assert r["condition_code"] == "O_LIGHT"
    assert r["assessment"] == "Normal lighting operation."
    assert r["potential_wastage_indicator"] is False


def test_wording_never_claims_confirmed_wastage():
    """Research-integrity check: assessment text must use cautious wording,
    never assert confirmed wastage."""
    for key in ["U_HVAC", "U_LIGHT", "U_HVAC_LIGHT"]:
        pass
    r = analyze_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"))
    assert "potential" in r["assessment"].lower()
    assert "wastage" not in r["condition_label"].lower()  # label describes state, not a verdict


# ---------------------------------------------------------------------------
# Tariff classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hour,expected", [
    (0, "Off-Peak"), (5, "Off-Peak"), (23, "Off-Peak"),
    (6, "Standard"), (12, "Standard"), (17, "Standard"),
    (18, "Peak"), (20, "Peak"), (22, "Peak"),
])
def test_tariff_category_by_hour(hour, expected):
    assert get_tariff_category(hour) == expected


def test_tariff_category_invalid_hour_raises():
    with pytest.raises(EnergyAnalysisError):
        get_tariff_category(24)
    with pytest.raises(EnergyAnalysisError):
        get_tariff_category(-1)


def test_record_tariff_category_matches_hour_band():
    r = analyze_record(_base_record(hour=20, electricity_tariff=9.0))
    assert r["tariff_category"] == "Peak"
    assert r["data_quality_warnings"] == []  # rate (9.0) matches Peak, no mismatch warning


def test_record_tariff_rate_mismatch_produces_warning_not_crash():
    """If electricity_tariff disagrees with the hour-derived category, the
    record must still be classified (condition logic is independent of
    tariff), and the discrepancy must be surfaced, not silently ignored."""
    r = analyze_record(_base_record(hour=20, electricity_tariff=5.0))  # 5.0 => Off-Peak, hour=20 => Peak
    assert r["tariff_category"] == "Peak"
    assert any("does not match" in w for w in r["data_quality_warnings"])


# ---------------------------------------------------------------------------
# Temperature context
# ---------------------------------------------------------------------------
def test_temperature_context_bands():
    assert get_temperature_context(20.0) == "Cool"
    assert get_temperature_context(22.0) == "Moderate"
    assert get_temperature_context(24.0) == "Moderate"
    assert get_temperature_context(26.0) == "Moderate"
    assert get_temperature_context(27.0) == "Warm"


# ---------------------------------------------------------------------------
# Missing / invalid required fields
# ---------------------------------------------------------------------------
def test_missing_occupancy_raises():
    record = _base_record()
    del record["occupancy"]
    with pytest.raises(EnergyAnalysisError, match="occupancy"):
        analyze_record(record)


def test_missing_hvac_usage_raises():
    record = _base_record()
    del record["hvac_usage"]
    with pytest.raises(EnergyAnalysisError, match="hvac_usage"):
        analyze_record(record)


def test_invalid_hvac_usage_value_raises():
    record = _base_record(hvac_usage="Maybe")
    with pytest.raises(EnergyAnalysisError, match="hvac_usage"):
        analyze_record(record)


def test_negative_occupancy_raises():
    record = _base_record(occupancy=-1)
    with pytest.raises(EnergyAnalysisError, match="occupancy"):
        analyze_record(record)


def test_non_numeric_occupancy_raises():
    record = _base_record(occupancy="many")
    with pytest.raises(EnergyAnalysisError):
        analyze_record(record)


# ---------------------------------------------------------------------------
# "None" appliance handling
# ---------------------------------------------------------------------------
def test_none_string_appliance_passes_through_unchanged():
    record = _base_record(appliance="None", hvac_usage="Off", lighting_usage="Off")
    r = analyze_record(record)
    assert r["appliance"] == "None"
    assert isinstance(r["appliance"], str)


def test_missing_appliance_field_produces_warning_not_crash():
    record = _base_record()
    del record["appliance"]
    r = analyze_record(record)
    assert r["appliance"] is None
    assert any("appliance" in w for w in r["data_quality_warnings"])


# ---------------------------------------------------------------------------
# Multiple records / dataset-level aggregation (synthetic small dataset)
# ---------------------------------------------------------------------------
@pytest.fixture
def small_synthetic_df():
    rows = [
        _base_record(occupancy=0, hvac_usage="Off", lighting_usage="Off", hour=2, electricity_tariff=5.0),
        _base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=2, electricity_tariff=5.0),
        _base_record(occupancy=0, hvac_usage="Off", lighting_usage="On", hour=10, electricity_tariff=7.0),
        _base_record(occupancy=0, hvac_usage="On", lighting_usage="On", hour=10, electricity_tariff=7.0),
        _base_record(occupancy=5, hvac_usage="On", lighting_usage="Off", hour=19, electricity_tariff=9.0),
        _base_record(occupancy=5, hvac_usage="Off", lighting_usage="On", hour=19, electricity_tariff=9.0),
        _base_record(occupancy=5, hvac_usage="On", lighting_usage="On", hour=19, electricity_tariff=9.0),
        _base_record(occupancy=5, hvac_usage="Off", lighting_usage="Off", hour=19, electricity_tariff=9.0),
    ]
    return pd.DataFrame(rows)


def test_analyze_many_returns_one_result_per_row(small_synthetic_df):
    results = analyze_many(small_synthetic_df)
    assert len(results) == len(small_synthetic_df)
    codes = {r["condition_code"] for r in results}
    assert codes == {"U_NONE", "U_HVAC", "U_LIGHT", "U_HVAC_LIGHT",
                      "O_HVAC", "O_LIGHT", "O_HVAC_LIGHT", "O_NONE"}


def test_analyze_dataset_counts_on_synthetic_df(small_synthetic_df):
    summary = analyze_dataset(small_synthetic_df)
    assert summary["total_records"] == 8
    assert summary["occupied_records"] == 4
    assert summary["unoccupied_records"] == 4
    assert summary["potential_hvac_wastage_condition_records"] == 2   # U_HVAC + U_HVAC_LIGHT
    assert summary["potential_lighting_wastage_condition_records"] == 2  # U_LIGHT + U_HVAC_LIGHT
    assert summary["total_potential_wastage_condition_records"] == 3   # U_HVAC, U_LIGHT, U_HVAC_LIGHT (distinct rows)
    assert summary["peak_tariff_records"] == 4
    assert summary["off_peak_tariff_records"] == 2
    assert summary["standard_tariff_records"] == 2
    assert summary["condition_counts"]["Unoccupied - No Active Appliances"] == 1
    assert summary["condition_counts"]["Occupied - HVAC and Lighting Active"] == 1


def test_analyze_dataset_empty_df_returns_zero():
    empty = pd.DataFrame()
    summary = analyze_dataset(empty)
    assert summary["total_records"] == 0


# ---------------------------------------------------------------------------
# Real processed dataset (integration-style check, not just synthetic data)
# ---------------------------------------------------------------------------
def test_analyze_dataset_on_real_processed_dataset_runs_without_error():
    df = load_processed_dataset()
    summary = analyze_dataset(df)
    assert summary["total_records"] == 1000
    # Every condition count should be a non-negative integer and they should sum to the total.
    assert sum(summary["condition_counts"].values()) == 1000
    # No record should be missing a classification -- zero EnergyAnalysisError-worthy rows expected
    # since Phase 0 validation already confirmed hvac_usage/lighting_usage/occupancy are always valid.


def test_real_dataset_records_have_no_classification_blocking_warnings():
    """The real processed dataset should produce zero data_quality_warnings,
    since every required and contextual field is populated by Phase 0's
    pipeline. A non-zero count here would indicate a real data problem that
    must be reported, not silently accepted."""
    df = load_processed_dataset()
    results = analyze_many(df)
    warning_rows = [r for r in results if r["data_quality_warnings"]]
    assert len(warning_rows) == 0, f"{len(warning_rows)} rows had unexpected data quality warnings"
