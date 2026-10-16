"""
tests/test_wastage_detection.py
----------------------------------
Tests for modules/wastage_detection.py (Phase 2 -- Wastage Detection).

Covers all 8 occupancy/HVAC/lighting states, wastage-type classification,
severity assignment, explanation generation, appliance="None" handling,
missing/invalid value handling (reusing Phase 1's error behavior), dataset
aggregation, wastage percentages (with explicit denominators), and
integration against the real processed dataset.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_manager import load_processed_dataset
from modules.energy_analysis import EnergyAnalysisError
from modules.wastage_detection import (
    SEVERITY_LEVELS,
    WASTAGE_TYPES,
    analyze_wastage_dataset,
    analyze_wastage_many,
    analyze_wastage_record,
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
# 1-4: Unoccupied states
# ---------------------------------------------------------------------------
def test_unoccupied_hvac_on_is_potential_wastage():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"))
    assert r["potential_wastage"] is True
    assert r["wastage_type"] == "HVAC"
    assert r["severity"] == "MEDIUM"
    assert r["reason"] == "Potential HVAC wastage: the room is unoccupied while HVAC is active."


def test_unoccupied_lighting_on_is_potential_wastage():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="Off", lighting_usage="On"))
    assert r["potential_wastage"] is True
    assert r["wastage_type"] == "LIGHTING"
    assert r["severity"] == "MEDIUM"
    assert r["reason"] == "Potential lighting wastage: the room is unoccupied while lighting is active."


def test_unoccupied_hvac_and_lighting_on_is_high_severity():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="On"))
    assert r["potential_wastage"] is True
    assert r["wastage_type"] == "HVAC_AND_LIGHTING"
    assert r["severity"] == "HIGH"
    assert r["reason"] == (
        "Potential HVAC and lighting wastage: the room is unoccupied while both HVAC and lighting are active."
    )


def test_unoccupied_both_off_is_no_wastage():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="Off", lighting_usage="Off"))
    assert r["potential_wastage"] is False
    assert r["wastage_type"] == "NONE"
    assert r["severity"] == "NONE"
    assert r["reason"] == "No active HVAC or lighting was detected while the room was unoccupied."


# ---------------------------------------------------------------------------
# 5-8: Occupied states -- must NEVER be classified as wastage (Rule 5)
# ---------------------------------------------------------------------------
def test_occupied_hvac_on_is_not_wastage():
    r = analyze_wastage_record(_base_record(occupancy=3, hvac_usage="On", lighting_usage="Off"))
    assert r["potential_wastage"] is False
    assert r["wastage_type"] == "NONE"
    assert r["severity"] == "NONE"


def test_occupied_lighting_on_is_not_wastage():
    r = analyze_wastage_record(_base_record(occupancy=3, hvac_usage="Off", lighting_usage="On"))
    assert r["potential_wastage"] is False
    assert r["wastage_type"] == "NONE"


def test_occupied_hvac_and_lighting_on_is_not_wastage():
    r = analyze_wastage_record(_base_record(occupancy=3, hvac_usage="On", lighting_usage="On"))
    assert r["potential_wastage"] is False
    assert r["wastage_type"] == "NONE"
    assert r["severity"] == "NONE"


def test_occupied_both_off_is_not_wastage():
    r = analyze_wastage_record(_base_record(occupancy=3, hvac_usage="Off", lighting_usage="Off"))
    assert r["potential_wastage"] is False
    assert r["wastage_type"] == "NONE"


# ---------------------------------------------------------------------------
# 9-11: Wastage type classification
# ---------------------------------------------------------------------------
def test_wastage_type_values_are_within_declared_vocabulary():
    for occupancy, hvac, light in [(0, "Off", "Off"), (0, "On", "Off"), (0, "Off", "On"),
                                    (0, "On", "On"), (3, "On", "On")]:
        r = analyze_wastage_record(_base_record(occupancy=occupancy, hvac_usage=hvac, lighting_usage=light))
        assert r["wastage_type"] in WASTAGE_TYPES


def test_hvac_wastage_classification_excludes_lighting_cases():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"))
    assert r["wastage_type"] == "HVAC"
    r2 = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="On"))
    assert r2["wastage_type"] != "HVAC"  # must be HVAC_AND_LIGHTING, not plain HVAC


def test_combined_classification_only_when_both_active_and_unoccupied():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="On"))
    assert r["wastage_type"] == "HVAC_AND_LIGHTING"
    # occupied + both on must NOT produce HVAC_AND_LIGHTING wastage
    r2 = analyze_wastage_record(_base_record(occupancy=3, hvac_usage="On", lighting_usage="On"))
    assert r2["wastage_type"] == "NONE"


# ---------------------------------------------------------------------------
# 12: Severity assignment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("occupancy,hvac,light,expected_severity", [
    (0, "Off", "Off", "NONE"),
    (0, "On", "Off", "MEDIUM"),
    (0, "Off", "On", "MEDIUM"),
    (0, "On", "On", "HIGH"),
    (3, "On", "On", "NONE"),
    (3, "Off", "Off", "NONE"),
])
def test_severity_assignment(occupancy, hvac, light, expected_severity):
    r = analyze_wastage_record(_base_record(occupancy=occupancy, hvac_usage=hvac, lighting_usage=light))
    assert r["severity"] == expected_severity
    assert r["severity"] in SEVERITY_LEVELS


def test_severity_is_not_derived_from_energy_consumption():
    """Research-integrity check: two records with identical occupancy/HVAC/
    lighting state but very different energy_consumption must get the exact
    same severity -- severity must never be a function of consumption."""
    low = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off",
                                                energy_consumption=1.0))
    high = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off",
                                                 energy_consumption=9999.0))
    assert low["severity"] == high["severity"] == "MEDIUM"
    assert low["wastage_type"] == high["wastage_type"] == "HVAC"
    assert low["potential_wastage"] == high["potential_wastage"] is True


# ---------------------------------------------------------------------------
# 13: Explanation generation
# ---------------------------------------------------------------------------
def test_every_positive_detection_has_a_reason_matching_the_rule():
    cases = {
        (0, "On", "Off"): "Potential HVAC wastage",
        (0, "Off", "On"): "Potential lighting wastage",
        (0, "On", "On"): "Potential HVAC and lighting wastage",
    }
    for (occupancy, hvac, light), expected_phrase in cases.items():
        r = analyze_wastage_record(_base_record(occupancy=occupancy, hvac_usage=hvac, lighting_usage=light))
        assert r["reason"].startswith(expected_phrase)


def test_no_wastage_reason_does_not_claim_wastage():
    r = analyze_wastage_record(_base_record(occupancy=3, hvac_usage="On", lighting_usage="On"))
    assert "potential wastage" not in r["reason"].lower() or "not potential wastage" in r["reason"].lower()


# ---------------------------------------------------------------------------
# 14: Appliance = "None" handling
# ---------------------------------------------------------------------------
def test_appliance_none_string_handled_correctly():
    r = analyze_wastage_record(_base_record(occupancy=0, hvac_usage="Off", lighting_usage="Off",
                                              appliance="None"))
    assert r["appliance"] == "None"
    assert r["potential_wastage"] is False


# ---------------------------------------------------------------------------
# 15: Missing / invalid values -- must reuse Phase 1's error behavior exactly
# ---------------------------------------------------------------------------
def test_missing_occupancy_raises_energy_analysis_error():
    record = _base_record()
    del record["occupancy"]
    with pytest.raises(EnergyAnalysisError, match="occupancy"):
        analyze_wastage_record(record)


def test_missing_hvac_usage_raises_energy_analysis_error():
    record = _base_record()
    del record["hvac_usage"]
    with pytest.raises(EnergyAnalysisError, match="hvac_usage"):
        analyze_wastage_record(record)


def test_missing_lighting_usage_raises_energy_analysis_error():
    record = _base_record()
    del record["lighting_usage"]
    with pytest.raises(EnergyAnalysisError, match="lighting_usage"):
        analyze_wastage_record(record)


def test_invalid_occupancy_value_raises():
    record = _base_record(occupancy=-5)
    with pytest.raises(EnergyAnalysisError):
        analyze_wastage_record(record)


def test_invalid_hvac_usage_value_raises():
    record = _base_record(hvac_usage="Maybe")
    with pytest.raises(EnergyAnalysisError):
        analyze_wastage_record(record)


# ---------------------------------------------------------------------------
# 16-17: Dataset aggregation and wastage percentages (synthetic small dataset)
# ---------------------------------------------------------------------------
@pytest.fixture
def small_synthetic_df():
    rows = [
        _base_record(occupancy=0, hvac_usage="Off", lighting_usage="Off"),  # U_NONE
        _base_record(occupancy=0, hvac_usage="On", lighting_usage="Off"),   # U_HVAC
        _base_record(occupancy=0, hvac_usage="Off", lighting_usage="On"),   # U_LIGHT
        _base_record(occupancy=0, hvac_usage="On", lighting_usage="On"),    # U_HVAC_LIGHT
        _base_record(occupancy=5, hvac_usage="On", lighting_usage="Off"),   # O_HVAC
        _base_record(occupancy=5, hvac_usage="Off", lighting_usage="On"),   # O_LIGHT
        _base_record(occupancy=5, hvac_usage="On", lighting_usage="On"),    # O_HVAC_LIGHT
        _base_record(occupancy=5, hvac_usage="Off", lighting_usage="Off"),  # O_NONE
    ]
    return pd.DataFrame(rows)


def test_analyze_wastage_many_returns_one_result_per_row(small_synthetic_df):
    results = analyze_wastage_many(small_synthetic_df)
    assert len(results) == len(small_synthetic_df)
    # record_id should track the DataFrame's own index
    assert [r["record_id"] for r in results] == list(small_synthetic_df.index)


def test_analyze_wastage_dataset_counts(small_synthetic_df):
    summary = analyze_wastage_dataset(small_synthetic_df)
    assert summary["total_records"] == 8
    assert summary["occupied_records"] == 4
    assert summary["unoccupied_records"] == 4
    assert summary["potential_wastage_records"] == 3          # U_HVAC, U_LIGHT, U_HVAC_LIGHT
    assert summary["no_potential_wastage_records"] == 5
    # Rule-level (overlapping) counts: include both the "HVAC-only" row AND
    # the "HVAC_AND_LIGHTING" row, per Rule 1 / Rule 2's documented "includes
    # both X and Y conditions" definition -- matches Phase 1's convention.
    assert summary["potential_hvac_wastage_records"] == 2       # U_HVAC + U_HVAC_LIGHT
    assert summary["potential_lighting_wastage_records"] == 2   # U_LIGHT + U_HVAC_LIGHT
    # Mutually-exclusive "combined" count: only the row where BOTH fired together.
    assert summary["potential_hvac_and_lighting_wastage_records"] == 1
    assert summary["severity_counts"]["HIGH"] == 1
    assert summary["severity_counts"]["MEDIUM"] == 2
    assert summary["severity_counts"]["NONE"] == 5
    assert summary["severity_counts"]["LOW"] == 0


def test_wastage_percentages_have_correct_explicit_denominators(small_synthetic_df):
    summary = analyze_wastage_dataset(small_synthetic_df)
    # 3 potential-wastage records out of 8 total
    assert summary["potential_wastage_rate_of_all_records_pct"] == pytest.approx(3 / 8 * 100, abs=0.01)
    # 3 potential-wastage records out of 4 unoccupied records
    assert summary["potential_wastage_rate_of_unoccupied_records_pct"] == pytest.approx(3 / 4 * 100, abs=0.01)


def test_wastage_rate_of_unoccupied_is_none_when_no_unoccupied_records():
    all_occupied_df = pd.DataFrame([
        _base_record(occupancy=5, hvac_usage="On", lighting_usage="Off"),
        _base_record(occupancy=2, hvac_usage="Off", lighting_usage="On"),
    ])
    summary = analyze_wastage_dataset(all_occupied_df)
    assert summary["unoccupied_records"] == 0
    assert summary["potential_wastage_rate_of_unoccupied_records_pct"] is None


def test_analyze_wastage_dataset_empty_df_returns_zero():
    summary = analyze_wastage_dataset(pd.DataFrame())
    assert summary["total_records"] == 0


# ---------------------------------------------------------------------------
# 18: Real processed dataset integration
# ---------------------------------------------------------------------------
def test_real_dataset_wastage_counts_are_sane():
    """
    Integration check against the real 1000-row processed dataset.
    These figures are a SANITY CHECK against Phase 1's independently-computed
    condition counts, not a target the implementation was built to hit.
    """
    df = load_processed_dataset()
    summary = analyze_wastage_dataset(df)

    assert summary["total_records"] == 1000
    assert summary["occupied_records"] == 902
    assert summary["unoccupied_records"] == 98

    # Cross-check against Phase 1's independently-reported figures.
    assert summary["potential_hvac_wastage_records"] == 44
    assert summary["potential_lighting_wastage_records"] == 50
    assert summary["potential_wastage_records"] == 70

    # Internal consistency: counts must sum correctly.
    assert summary["potential_wastage_records"] + summary["no_potential_wastage_records"] == 1000
    assert (
        summary["potential_hvac_wastage_records"]
        + summary["potential_lighting_wastage_records"]
        - summary["potential_hvac_and_lighting_wastage_records"]  # counted in both HVAC and LIGHTING above
    ) == summary["potential_wastage_records"]


def test_real_dataset_no_record_raises_and_potential_wastage_matches_phase1_indicator():
    """Every real row must classify without error, and Phase 2's
    `potential_wastage` must always agree with Phase 1's
    `potential_wastage_indicator` -- they are derived from the same
    condition_code by construction, and this test guards against the two
    ever silently drifting apart in a future edit."""
    df = load_processed_dataset()
    results = analyze_wastage_many(df)
    assert len(results) == 1000
    mismatches = [r for r in results if r["potential_wastage"] != r["potential_wastage_indicator"]]
    assert len(mismatches) == 0


def test_real_dataset_all_energy_consumption_values_are_passthrough_only():
    """Confirm energy_consumption in the output is identical to the source
    dataset's values (never modified, never used to gate potential_wastage)."""
    df = load_processed_dataset()
    results = analyze_wastage_many(df.head(50))
    for (_, row), result in zip(df.head(50).iterrows(), results):
        assert result["energy_consumption"] == row["energy_consumption"]
