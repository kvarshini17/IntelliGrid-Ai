"""
tests/test_anomaly_detection.py
-----------------------------------
Tests for modules/anomaly_detection.py (Phase 3 -- Context-Aware Anomaly
Detection).

Covers the pure classify_anomaly() boundary logic in isolation, context
grouping correctness (occupancy/HVAC/lighting/hour separation), small-group
and zero-variance handling, wastage/anomaly independence, dataset
aggregation, and integration against the real 1000-row processed dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from modules.data_manager import load_processed_dataset
from modules.anomaly_detection import (
    ANOMALY_TYPES,
    analyze_anomalies,
    analyze_anomaly_dataset,
    build_context_key,
    classify_anomaly,
    format_context_label,
    get_occupancy_state,
)


def _base_record(**overrides):
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
# 1-3: Normal / high / low, via the pure classify_anomaly() function
# ---------------------------------------------------------------------------
def test_normal_observation():
    r = classify_anomaly(observed_energy=51, context_count=10, context_mean=50, context_std=5)
    assert r["anomaly_type"] == "NORMAL"
    assert r["is_anomaly"] is False


def test_high_energy_anomaly():
    r = classify_anomaly(observed_energy=75, context_count=10, context_mean=50, context_std=5)
    assert r["anomaly_type"] == "HIGH_ENERGY_ANOMALY"
    assert r["is_anomaly"] is True
    assert r["z_score"] == pytest.approx(5.0)


def test_low_energy_anomaly():
    r = classify_anomaly(observed_energy=25, context_count=10, context_mean=50, context_std=5)
    assert r["anomaly_type"] == "LOW_ENERGY_ANOMALY"
    assert r["is_anomaly"] is True
    assert r["z_score"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# 4-8: z-score calculation and threshold boundaries
# ---------------------------------------------------------------------------
def test_z_score_calculation_is_correct():
    r = classify_anomaly(observed_energy=60, context_count=10, context_mean=50, context_std=5)
    assert r["z_score"] == pytest.approx(2.0)


def test_threshold_boundary_exactly_plus_2_0_is_anomaly():
    r = classify_anomaly(observed_energy=60, context_count=10, context_mean=50, context_std=5)
    assert r["z_score"] == pytest.approx(2.0)
    assert r["anomaly_type"] == "HIGH_ENERGY_ANOMALY"


def test_threshold_boundary_exactly_minus_2_0_is_anomaly():
    r = classify_anomaly(observed_energy=40, context_count=10, context_mean=50, context_std=5)
    assert r["z_score"] == pytest.approx(-2.0)
    assert r["anomaly_type"] == "LOW_ENERGY_ANOMALY"


def test_value_just_below_threshold_is_normal():
    r = classify_anomaly(observed_energy=59.99, context_count=10, context_mean=50, context_std=5)
    assert r["z_score"] < config.ANOMALY_Z_THRESHOLD
    assert r["anomaly_type"] == "NORMAL"


def test_value_just_above_threshold_is_anomaly():
    r = classify_anomaly(observed_energy=60.01, context_count=10, context_mean=50, context_std=5)
    assert r["z_score"] > config.ANOMALY_Z_THRESHOLD
    assert r["anomaly_type"] == "HIGH_ENERGY_ANOMALY"


# ---------------------------------------------------------------------------
# 9: Zero standard deviation -- must never divide by zero
# ---------------------------------------------------------------------------
def test_zero_std_is_insufficient_baseline_not_a_crash():
    r = classify_anomaly(observed_energy=55, context_count=10, context_mean=50, context_std=0)
    assert r["anomaly_type"] == "INSUFFICIENT_BASELINE"
    assert r["z_score"] is None
    assert r["is_anomaly"] is False


def test_near_zero_std_is_also_insufficient_baseline():
    r = classify_anomaly(observed_energy=55, context_count=10, context_mean=50,
                          context_std=config.ANOMALY_STD_EPSILON / 2)
    assert r["anomaly_type"] == "INSUFFICIENT_BASELINE"


def test_none_std_is_insufficient_baseline():
    r = classify_anomaly(observed_energy=55, context_count=10, context_mean=50, context_std=None)
    assert r["anomaly_type"] == "INSUFFICIENT_BASELINE"


# ---------------------------------------------------------------------------
# 10: Insufficient group size
# ---------------------------------------------------------------------------
def test_group_smaller_than_minimum_is_insufficient_baseline():
    r = classify_anomaly(observed_energy=1000, context_count=config.MIN_CONTEXT_GROUP_SIZE - 1,
                          context_mean=50, context_std=5)
    assert r["anomaly_type"] == "INSUFFICIENT_BASELINE"
    assert r["is_anomaly"] is False
    assert "Insufficient observations" in r["reason"]


def test_group_exactly_at_minimum_is_classified_normally():
    r = classify_anomaly(observed_energy=60, context_count=config.MIN_CONTEXT_GROUP_SIZE,
                          context_mean=50, context_std=5)
    assert r["anomaly_type"] != "INSUFFICIENT_BASELINE"


# ---------------------------------------------------------------------------
# 11-15: Context grouping -- occupancy / HVAC / lighting / hour separation,
# and confirming source_day_of_week is never used.
# ---------------------------------------------------------------------------
def test_get_occupancy_state_from_condition_code():
    assert get_occupancy_state("U_HVAC") == "Unoccupied"
    assert get_occupancy_state("U_NONE") == "Unoccupied"
    assert get_occupancy_state("O_HVAC") == "Occupied"
    assert get_occupancy_state("O_NONE") == "Occupied"


def test_context_key_differs_by_occupancy_state():
    k1 = build_context_key("Occupied", "On", "Off", 10)
    k2 = build_context_key("Unoccupied", "On", "Off", 10)
    assert k1 != k2


def test_context_key_differs_by_hvac_state():
    k1 = build_context_key("Occupied", "On", "Off", 10)
    k2 = build_context_key("Occupied", "Off", "Off", 10)
    assert k1 != k2


def test_context_key_differs_by_lighting_state():
    k1 = build_context_key("Occupied", "On", "On", 10)
    k2 = build_context_key("Occupied", "On", "Off", 10)
    assert k1 != k2


def test_context_key_differs_by_hour():
    k1 = build_context_key("Occupied", "On", "Off", 10)
    k2 = build_context_key("Occupied", "On", "Off", 11)
    assert k1 != k2


def test_context_label_is_human_readable_and_round_trips_distinctly():
    k1 = build_context_key("Occupied", "On", "Off", 9)
    k2 = build_context_key("Occupied", "On", "Off", 10)
    assert format_context_label(k1) != format_context_label(k2)
    assert "Occupied" in format_context_label(k1)
    assert "HVAC-On" in format_context_label(k1)


def test_source_day_of_week_is_never_referenced():
    """The processed dataset does not even contain source_day_of_week
    (Phase 0 removed it as statistically unreliable). Confirm the module
    never ACCESSES such a column (e.g. row["source_day_of_week"] or
    .source_day_of_week) -- the module's docstring is allowed to mention the
    name in prose explaining why it is excluded, so this checks for actual
    code access patterns rather than the bare substring. Also confirm a
    dataset without that column still analyzes without error."""
    import inspect
    from modules import anomaly_detection
    source = inspect.getsource(anomaly_detection)
    assert '"source_day_of_week"' not in source  # no bracket/key access anywhere in code
    assert "'source_day_of_week'" not in source
    assert ".source_day_of_week" not in source

    df = load_processed_dataset()
    assert "source_day_of_week" not in df.columns
    results = analyze_anomalies(df.head(20))
    assert len(results) == 20  # ran without error despite the column's absence


# ---------------------------------------------------------------------------
# 16-18: Wastage/anomaly independence, same condition + different energy
# ---------------------------------------------------------------------------
@pytest.fixture
def mixed_wastage_anomaly_df():
    """
    Engineered to cover all four wastage x anomaly combinations:
      - Occupied + HVAC + Lighting, hour=10: typical values, one high outlier
        -> that outlier is ANOMALY but NOT wastage (Rule 5: occupied is never wastage).
      - Unoccupied + HVAC, hour=14: typical values (low variance) around ~40
        -> these rows are WASTAGE (Rule 1) but NOT anomalous (close to their own baseline).
      - Same Unoccupied + HVAC, hour=14 group, with one extreme outlier
        -> that specific row is BOTH wastage AND anomalous.
      - Occupied + both off, hour=18: typical values, no outlier
        -> NEITHER wastage NOR anomaly.
    """
    rows = []
    # Group A: Occupied + HVAC + Lighting, hour=10 -- 9 typical + 1 high outlier
    typical_a = [60, 61, 59, 62, 58, 60, 61, 59, 62]
    for v in typical_a:
        rows.append(_base_record(occupancy=4, hvac_usage="On", lighting_usage="On", hour=10, energy_consumption=v))
    rows.append(_base_record(occupancy=4, hvac_usage="On", lighting_usage="On", hour=10, energy_consumption=150))  # outlier

    # Group B: Unoccupied + HVAC, hour=14 -- 9 typical (tight) + 1 extreme outlier
    typical_b = [40, 41, 39, 40, 41, 39, 40, 41, 39]
    for v in typical_b:
        rows.append(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=14, energy_consumption=v))
    rows.append(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=14, energy_consumption=300))  # outlier

    # Group C: Occupied + both off, hour=18 -- typical values only, no outlier
    typical_c = [30, 31, 29, 30, 31, 29, 30]
    for v in typical_c:
        rows.append(_base_record(occupancy=2, hvac_usage="Off", lighting_usage="Off", hour=18, energy_consumption=v))

    return pd.DataFrame(rows)


def test_anomalous_but_not_wastage(mixed_wastage_anomaly_df):
    """Occupied group's outlier: anomalous, but Rule 5 forbids wastage for occupied rooms."""
    results = analyze_anomalies(mixed_wastage_anomaly_df)
    outlier = [r for r in results if r["energy_consumption"] == 150][0]
    assert outlier["is_anomaly"] is True
    assert outlier["anomaly_type"] == "HIGH_ENERGY_ANOMALY"
    assert outlier["potential_wastage"] is False


def test_wastage_but_not_anomalous(mixed_wastage_anomaly_df):
    """A typical (non-outlier) row in the Unoccupied+HVAC group: wastage=True,
    but close enough to its own group's mean to be NORMAL."""
    results = analyze_anomalies(mixed_wastage_anomaly_df)
    typical_b_rows = [r for r in results if r["condition_code"] == "U_HVAC" and r["energy_consumption"] == 40]
    assert len(typical_b_rows) >= 1
    for r in typical_b_rows:
        assert r["potential_wastage"] is True
        assert r["anomaly_type"] == "NORMAL"
        assert r["is_anomaly"] is False


def test_both_wastage_and_anomalous(mixed_wastage_anomaly_df):
    """The Unoccupied+HVAC group's outlier: wastage=True AND anomalous=True simultaneously."""
    results = analyze_anomalies(mixed_wastage_anomaly_df)
    outlier = [r for r in results if r["energy_consumption"] == 300][0]
    assert outlier["potential_wastage"] is True
    assert outlier["is_anomaly"] is True
    assert outlier["anomaly_type"] == "HIGH_ENERGY_ANOMALY"


def test_neither_wastage_nor_anomalous(mixed_wastage_anomaly_df):
    """Occupied + both off group, no outlier: neither flag should be set."""
    results = analyze_anomalies(mixed_wastage_anomaly_df)
    group_c_rows = [r for r in results if r["condition_code"] == "O_NONE"]
    assert len(group_c_rows) == 7
    for r in group_c_rows:
        assert r["potential_wastage"] is False
        assert r["is_anomaly"] is False
        assert r["anomaly_type"] == "NORMAL"


def test_same_condition_different_energy_values_can_differ_in_anomaly_status(mixed_wastage_anomaly_df):
    """Two records sharing the exact same context group can get different
    anomaly classifications based purely on their own energy_consumption."""
    results = analyze_anomalies(mixed_wastage_anomaly_df)
    group_b = [r for r in results if r["context_group"] == "Unoccupied_HVAC-On_Lighting-Off_Hour-14"]
    assert len(group_b) == 10
    typical = [r for r in group_b if r["energy_consumption"] == 40][0]
    outlier = [r for r in group_b if r["energy_consumption"] == 300][0]
    assert typical["context_mean"] == outlier["context_mean"]  # same baseline
    assert typical["context_std"] == outlier["context_std"]
    assert typical["anomaly_type"] != outlier["anomaly_type"]


def test_detector_does_not_classify_merely_from_state_flags(mixed_wastage_anomaly_df):
    """HVAC ON, lighting ON, unoccupied, peak tariff, or high temperature
    alone must never by themselves produce an anomaly -- only deviation from
    the contextual baseline does. The typical (non-outlier) rows in every
    group here have HVAC/lighting ON and/or are unoccupied, yet are NORMAL."""
    results = analyze_anomalies(mixed_wastage_anomaly_df)
    non_outlier_results = [r for r in results if r["energy_consumption"] not in (150, 300)]
    assert all(r["anomaly_type"] in ("NORMAL", "INSUFFICIENT_BASELINE") for r in non_outlier_results)
    assert any(r["hvac_status"] == "On" for r in non_outlier_results)
    assert any(r["occupancy_state"] == "Unoccupied" for r in non_outlier_results)


# ---------------------------------------------------------------------------
# 19: Dataset aggregation
# ---------------------------------------------------------------------------
def test_analyze_anomaly_dataset_counts(mixed_wastage_anomaly_df):
    summary = analyze_anomaly_dataset(mixed_wastage_anomaly_df)
    assert summary["total_records"] == 27
    assert summary["high_energy_anomalies"] >= 2   # the two injected high outliers
    # All three fixture groups have >= MIN_CONTEXT_GROUP_SIZE (10, 10, and 7
    # rows respectively; MIN_CONTEXT_GROUP_SIZE=5), so none is insufficient.
    assert summary["insufficient_baseline_records"] == 0
    assert summary["number_of_context_groups"] == 3
    assert set(ANOMALY_TYPES) >= {"NORMAL", "HIGH_ENERGY_ANOMALY", "LOW_ENERGY_ANOMALY", "INSUFFICIENT_BASELINE"}


def test_analyze_anomaly_dataset_empty_df_returns_zero():
    summary = analyze_anomaly_dataset(pd.DataFrame())
    assert summary["total_records"] == 0


def test_anomaly_rate_denominator_excludes_insufficient_baseline():
    """anomaly_rate must be computed over records WITH a valid baseline,
    never over all records including INSUFFICIENT_BASELINE ones."""
    rows = []
    # One tiny group (count=2, < MIN_CONTEXT_GROUP_SIZE) -- insufficient.
    rows.append(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=3, energy_consumption=40))
    rows.append(_base_record(occupancy=0, hvac_usage="On", lighting_usage="Off", hour=3, energy_consumption=41))
    # One sufficient group with a clear anomaly.
    for v in [50, 51, 49, 52, 48, 150]:
        rows.append(_base_record(occupancy=3, hvac_usage="On", lighting_usage="On", hour=8, energy_consumption=v))
    df = pd.DataFrame(rows)
    summary = analyze_anomaly_dataset(df)
    assert summary["insufficient_baseline_records"] == 2
    assert summary["records_with_valid_baseline"] == 6
    expected_rate = round(summary["high_energy_anomalies"] / 6 * 100, 2)
    assert summary["anomaly_rate_pct_of_records_with_valid_baseline"] == expected_rate


# ---------------------------------------------------------------------------
# 20: Real 1000-row dataset integration
# ---------------------------------------------------------------------------
def test_real_dataset_integration_runs_without_error():
    df = load_processed_dataset()
    results = analyze_anomalies(df)
    assert len(results) == 1000
    for r in results:
        assert r["anomaly_type"] in ANOMALY_TYPES


def test_real_dataset_aggregate_is_internally_consistent():
    df = load_processed_dataset()
    summary = analyze_anomaly_dataset(df)
    assert summary["total_records"] == 1000
    assert (
        summary["normal_records"] + summary["high_energy_anomalies"]
        + summary["low_energy_anomalies"] + summary["insufficient_baseline_records"]
    ) == 1000
    assert summary["records_with_valid_baseline"] == (
        summary["normal_records"] + summary["high_energy_anomalies"] + summary["low_energy_anomalies"]
    )
    assert summary["min_context_group_size"] <= summary["max_context_group_size"]


def test_real_dataset_does_not_modify_energy_consumption():
    """Confirm the output's energy_consumption values are identical to the
    source dataset's -- never normalized, never injected, never altered."""
    df = load_processed_dataset()
    results = analyze_anomalies(df.head(100))
    for (_, row), result in zip(df.head(100).iterrows(), results):
        assert result["energy_consumption"] == row["energy_consumption"]
