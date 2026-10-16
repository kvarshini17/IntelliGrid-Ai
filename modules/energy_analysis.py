"""
modules/energy_analysis.py
----------------------------
Purpose : Energy Condition Analysis Engine (Phase 1).

Takes a single energy record or a collection of records and classifies the
CURRENT OPERATING CONDITION of the building/room from contextual fields
(occupancy, HVAC status, lighting status, temperature, tariff, energy
consumption, room, time).

This module answers exactly one question: "What is happening?"

It explicitly does NOT answer:
    - "Does this represent confirmed energy wastage?"        (Phase 3)
    - "Is this behavior statistically unusual?"                (Phase 4, anomaly detection)
    - "What should the system do about it?"                    (Phase 5, optimization)
    - anything requiring an LLM, RAG, or a trained ML model.

AI technique used : Deterministic contextual rule-based classification
                     (an explicit lookup table keyed on occupancy/HVAC/
                     lighting state). No machine learning, no statistics,
                     no thresholds applied to energy_consumption itself.

Research-integrity notes (see TRANSFORM_LOG.md and the Phase 0 data-quality
report for full detail, reproduced here because they directly constrain this
module's design):
    - The processed dataset has NO ground-truth wastage or anomaly labels.
      Nothing in this module claims otherwise.
    - The "unoccupied + HVAC on" condition does NOT show elevated
      energy_consumption versus the dataset's overall mean. This module
      therefore NEVER uses energy_consumption magnitude to decide
      `potential_wastage_indicator` -- that flag is purely a function of
      (occupancy, hvac_usage, lighting_usage), i.e. a condition indicator,
      not a validated wastage detector. Magnitude-based reasoning belongs to
      later phases (anomaly detection, controlled fault injection).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd

import config

Record = Union[pd.Series, dict]


class EnergyAnalysisError(ValueError):
    """Raised when a record is missing a required classification field or a
    required field has a value outside its expected domain. Phase 1 never
    guesses a condition for an invalid record -- it reports the problem."""


# ---------------------------------------------------------------------------
# Condition rule table
# ---------------------------------------------------------------------------
# Keyed by (occupied: bool, hvac_on: bool, lighting_on: bool).
# Each entry: (condition_code, condition_label, assessment, potential_wastage_indicator)
#
# Wording deliberately uses "Potential" / "indicator" throughout -- per the
# project's research-integrity requirement, none of these labels assert
# confirmed wastage, since the dataset provides no ground-truth labels and
# (per the Phase 0 audit) no consumption-magnitude signal corroborates the
# "unoccupied + HVAC" pattern.
# ---------------------------------------------------------------------------
CONDITION_RULES: Dict[tuple, tuple] = {
    (False, False, False): (
        "U_NONE", "Unoccupied - No Active Appliances",
        "No HVAC or lighting activity while unoccupied.", False,
    ),
    (False, True, False): (
        "U_HVAC", "Unoccupied - HVAC Active",
        "Potential unnecessary HVAC operation.", True,
    ),
    (False, False, True): (
        "U_LIGHT", "Unoccupied - Lighting Active",
        "Potential unnecessary lighting operation.", True,
    ),
    (False, True, True): (
        "U_HVAC_LIGHT", "Unoccupied - HVAC and Lighting Active",
        "Higher-priority potential unnecessary appliance activity.", True,
    ),
    (True, True, False): (
        "O_HVAC", "Occupied - HVAC Active",
        "Normal HVAC operation.", False,
    ),
    (True, False, True): (
        "O_LIGHT", "Occupied - Lighting Active",
        "Normal lighting operation.", False,
    ),
    (True, True, True): (
        "O_HVAC_LIGHT", "Occupied - HVAC and Lighting Active",
        "Normal active usage.", False,
    ),
    (True, False, False): (
        "O_NONE", "Occupied - No Active Appliances",
        "No HVAC or lighting activity detected.", False,
    ),
}


# ---------------------------------------------------------------------------
# Field access helpers (work uniformly for pandas.Series or plain dict)
# ---------------------------------------------------------------------------
def _has_field(record: Record, field: str) -> bool:
    if isinstance(record, pd.Series):
        return field in record.index and pd.notna(record[field])
    return field in record and record[field] is not None


def _get(record: Record, field: str, default=None):
    if not _has_field(record, field):
        return default
    return record[field]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_required_fields(record: Record) -> None:
    """
    Raise EnergyAnalysisError if any required classification field is
    missing or has a value outside its expected domain. Required fields are
    exactly those needed to classify the operating condition: occupancy,
    hvac_usage, lighting_usage. Everything else is contextual (see module
    docstring) and is reported as unavailable rather than raised on.
    """
    for field in config.REQUIRED_CLASSIFICATION_FIELDS:
        if not _has_field(record, field):
            raise EnergyAnalysisError(
                f"Required field '{field}' is missing or null; cannot classify energy condition."
            )

    occupancy = _get(record, "occupancy")
    try:
        occupancy_numeric = float(occupancy)
    except (TypeError, ValueError):
        raise EnergyAnalysisError(f"Field 'occupancy' must be numeric, got: {occupancy!r}")
    if occupancy_numeric < 0:
        raise EnergyAnalysisError(f"Field 'occupancy' cannot be negative, got: {occupancy_numeric}")

    for field in ("hvac_usage", "lighting_usage"):
        value = _get(record, field)
        if value not in config.VALID_STATUS_VALUES:
            raise EnergyAnalysisError(
                f"Field '{field}' must be one of {sorted(config.VALID_STATUS_VALUES)}, got: {value!r}"
            )


# ---------------------------------------------------------------------------
# Tariff context
# ---------------------------------------------------------------------------
def get_tariff_category(hour: int) -> str:
    """
    Classify an hour-of-day (0-23) into its documented tariff category
    (Off-Peak / Standard / Peak), using the exact bands verified against the
    Phase 0 source file (see config.py and TRANSFORM_LOG.md section 4).

    Raises EnergyAnalysisError if hour is outside 0-23.
    """
    try:
        hour_int = int(hour)
    except (TypeError, ValueError):
        raise EnergyAnalysisError(f"'hour' must be an integer 0-23, got: {hour!r}")
    if hour_int not in config.TARIFF_CATEGORY_BY_HOUR:
        raise EnergyAnalysisError(f"'hour' must be in range 0-23, got: {hour_int}")
    return config.TARIFF_CATEGORY_BY_HOUR[hour_int]


def _cross_check_tariff_rate(electricity_tariff: Optional[float], tariff_category: Optional[str]) -> Optional[str]:
    """
    Independently cross-check the row's electricity_tariff rate against the
    documented rate->category table. Returns a warning string if the rate is
    unrecognized or disagrees with the hour-derived category; returns None
    if consistent or if there isn't enough information to check.

    This never overrides tariff_category -- the hour-based classification
    remains the documented source of truth. This is a transparency check,
    not a correction mechanism (per the project's "report, don't silently
    fix" rule).
    """
    if electricity_tariff is None or tariff_category is None:
        return None
    expected_category = config.TARIFF_RATE_TO_CATEGORY.get(float(electricity_tariff))
    if expected_category is None:
        return f"electricity_tariff value {electricity_tariff} is not in the documented rate table {config.TARIFF_RATE_TO_CATEGORY}."
    if expected_category != tariff_category:
        return (
            f"electricity_tariff value {electricity_tariff} (expected category '{expected_category}') "
            f"does not match the hour-derived tariff_category '{tariff_category}'."
        )
    return None


# ---------------------------------------------------------------------------
# Temperature context (descriptive only -- see config.py for the documented
# assumption behind these bands; never used in condition classification)
# ---------------------------------------------------------------------------
def get_temperature_context(temperature: float) -> str:
    """Descriptive label only. See config.py for the documented thresholds
    (an explicit project assumption, not an engineering/comfort standard)."""
    temp = float(temperature)
    if temp < config.TEMPERATURE_COOL_MAX:
        return "Cool"
    if temp <= config.TEMPERATURE_MODERATE_MAX:
        return "Moderate"
    return "Warm"


# ---------------------------------------------------------------------------
# Core: single-record analysis
# ---------------------------------------------------------------------------
def analyze_record(record: Record) -> Dict:
    """
    Classify the energy condition of a single record.

    Input:
        record: a pandas.Series (one row of the processed dataset) or an
                equivalent dict. Must contain at least `occupancy`,
                `hvac_usage`, `lighting_usage` (see
                config.REQUIRED_CLASSIFICATION_FIELDS).
    Output:
        A structured, machine-readable dict (see module docstring / project
        spec section 12 for the schema). Optional contextual fields that are
        unavailable on the input record are set to None and noted in
        `data_quality_warnings` -- they never block classification.
    Raises:
        EnergyAnalysisError if a required classification field is missing
        or invalid.
    """
    _validate_required_fields(record)

    warnings: List[str] = []

    occupancy = float(_get(record, "occupancy"))
    hvac_usage = _get(record, "hvac_usage")
    lighting_usage = _get(record, "lighting_usage")

    occupied = occupancy > 0
    hvac_on = hvac_usage == "On"
    lighting_on = lighting_usage == "On"

    condition_code, condition_label, assessment, potential_wastage_indicator = CONDITION_RULES[
        (occupied, hvac_on, lighting_on)
    ]

    # --- Time / tariff context (contextual, not required) ---
    hour = _get(record, "hour")
    if hour is None and _has_field(record, "timestamp"):
        ts = record["timestamp"]
        if isinstance(ts, str):
            ts = pd.to_datetime(ts, errors="coerce")
        hour = ts.hour if pd.notna(ts) else None
    if hour is None:
        warnings.append("'hour' and a parseable 'timestamp' are both unavailable; tariff_category not computed.")

    tariff_category = None
    if hour is not None:
        try:
            tariff_category = get_tariff_category(hour)
        except EnergyAnalysisError as exc:
            warnings.append(str(exc))

    electricity_tariff = _get(record, "electricity_tariff")
    tariff_warning = _cross_check_tariff_rate(electricity_tariff, tariff_category)
    if tariff_warning:
        warnings.append(tariff_warning)

    # --- Temperature context (descriptive only) ---
    temperature = _get(record, "temperature")
    temperature_context = None
    if temperature is not None:
        temperature_context = get_temperature_context(temperature)
    else:
        warnings.append("'temperature' is unavailable; temperature_context not computed.")

    # --- Room context ---
    room_id = _get(record, "room_id")
    square_footage = _get(record, "square_footage")
    if room_id is None:
        warnings.append("'room_id' is unavailable.")
    if square_footage is None:
        warnings.append("'square_footage' is unavailable.")

    # --- Appliance context (passthrough only, not used in classification) ---
    appliance = _get(record, "appliance")
    appliance_status = _get(record, "appliance_status")
    if appliance is None:
        warnings.append("'appliance' is unavailable.")
    if appliance_status is None:
        warnings.append("'appliance_status' is unavailable.")

    # --- Energy consumption: contextual passthrough, no magnitude judgement ---
    energy_consumption = _get(record, "energy_consumption")
    if energy_consumption is None:
        warnings.append("'energy_consumption' is unavailable.")

    timestamp = _get(record, "timestamp")

    return {
        "timestamp": timestamp,
        "room_id": room_id,
        "square_footage": square_footage,
        "occupancy": occupancy,
        "temperature": temperature,
        "temperature_context": temperature_context,
        "hvac_status": hvac_usage,
        "lighting_status": lighting_usage,
        "appliance": appliance,
        "appliance_status": appliance_status,
        "electricity_tariff": electricity_tariff,
        "tariff_category": tariff_category,
        "energy_consumption": energy_consumption,
        "condition_code": condition_code,
        "condition_label": condition_label,
        "assessment": assessment,
        "potential_wastage_indicator": potential_wastage_indicator,
        "data_quality_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Core: dataset-level analysis
# ---------------------------------------------------------------------------
def analyze_many(df: pd.DataFrame) -> List[Dict]:
    """
    Run analyze_record() over every row of a DataFrame.

    Output: list of per-record result dicts, same order as the input rows.
    Reuses analyze_record() for every row -- no duplicated classification logic.
    """
    return [analyze_record(row) for _, row in df.iterrows()]


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """
    Produce aggregate, dataset-level energy condition statistics.

    Input:  the processed DataFrame (or any filtered subset of it).
    Output: a dict of counts -- see project spec section 13. All counts are
            described as "conditions" / "indicators", never as confirmed
            wastage, per the research-integrity requirement.
    """
    if df is None or df.empty:
        return {"total_records": 0}

    records = analyze_many(df)
    results_df = pd.DataFrame(records)

    total = len(results_df)
    occupied_mask = results_df["occupancy"] > 0

    condition_counts = results_df["condition_label"].value_counts().to_dict()
    # Ensure every defined condition label appears in the output, even with count 0.
    for _, label, _, _ in CONDITION_RULES.values():
        condition_counts.setdefault(label, 0)

    tariff_counts = results_df["tariff_category"].value_counts(dropna=False).to_dict()

    summary = {
        "total_records": total,
        "occupied_records": int(occupied_mask.sum()),
        "unoccupied_records": int((~occupied_mask).sum()),
        "hvac_on_records": int((results_df["hvac_status"] == "On").sum()),
        "lighting_on_records": int((results_df["lighting_status"] == "On").sum()),
        "hvac_and_lighting_on_records": int(
            ((results_df["hvac_status"] == "On") & (results_df["lighting_status"] == "On")).sum()
        ),
        "no_active_appliance_records": int(
            ((results_df["hvac_status"] == "Off") & (results_df["lighting_status"] == "Off")).sum()
        ),
        "potential_hvac_wastage_condition_records": int(
            results_df["condition_code"].isin(["U_HVAC", "U_HVAC_LIGHT"]).sum()
        ),
        "potential_lighting_wastage_condition_records": int(
            results_df["condition_code"].isin(["U_LIGHT", "U_HVAC_LIGHT"]).sum()
        ),
        "total_potential_wastage_condition_records": int(results_df["potential_wastage_indicator"].sum()),
        "peak_tariff_records": int(tariff_counts.get("Peak", 0)),
        "standard_tariff_records": int(tariff_counts.get("Standard", 0)),
        "off_peak_tariff_records": int(tariff_counts.get("Off-Peak", 0)),
        "condition_counts": condition_counts,
        "tariff_category_counts": tariff_counts,
        "records_with_data_quality_warnings": int(
            results_df["data_quality_warnings"].apply(lambda w: len(w) > 0).sum()
        ),
    }
    return summary
