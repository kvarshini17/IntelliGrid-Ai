"""
modules/wastage_detection.py
-------------------------------
Purpose : Wastage Detection Engine (Phase 2).

Converts the contextual energy conditions produced by Phase 1
(`modules.energy_analysis`) into explicit, explainable POTENTIAL WASTAGE
detections. Answers exactly one question: "Given the current occupancy and
appliance operating condition, is there an indication that energy use may
be unnecessary?"

This module explicitly does NOT:
    - prove actual energy wastage (no ground-truth wastage labels exist),
    - perform statistical anomaly detection (Phase 4),
    - perform optimization or recommend an action (Phase 5+),
    - use energy_consumption magnitude as a detector or a threshold,
    - use machine learning of any kind.

Design note -- reuse, not duplication:
    This module does NOT re-derive occupancy/HVAC/lighting booleans or the
    8 condition codes. It calls `energy_analysis.analyze_record()` (Phase 1)
    for classification and field validation, then maps the resulting
    `condition_code` onto a wastage_type/severity/reason via the
    WASTAGE_RULES table below. Phase 1's error-handling behavior
    (`EnergyAnalysisError` for missing/invalid required fields) is reused
    unchanged -- Phase 2 introduces no parallel validation mechanism.

AI technique used : Deterministic rule-based classification (a lookup table
                     keyed on Phase 1's condition_code). No ML, no
                     statistics, no energy_consumption thresholds.

Research-integrity notes (see TRANSFORM_LOG.md and the Phase 0 data-quality
report for full detail):
    - The processed dataset has NO ground-truth wastage labels. Nothing in
      this module claims to have detected confirmed wastage -- only
      "potential wastage conditions."
    - `energy_consumption` is carried through the output as context ONLY.
      It is never compared against a threshold and never used to decide
      `potential_wastage`, `wastage_type`, or `severity`.
    - `severity` represents the PRIORITY of the detected condition (how many
      appliance categories are implicated), NOT a measurement or estimate
      of energy lost. See the WASTAGE_RULES table and module-level
      SEVERITY_LEVELS documentation below.
"""

from typing import Dict, List, Optional, Union

import pandas as pd

from modules.energy_analysis import EnergyAnalysisError, analyze_record

Record = Union[pd.Series, dict]


# ---------------------------------------------------------------------------
# Wastage type and severity vocabularies
# ---------------------------------------------------------------------------
WASTAGE_TYPES = {"NONE", "HVAC", "LIGHTING", "HVAC_AND_LIGHTING"}

# NONE/LOW/MEDIUM/HIGH is the full declared severity scale. No current rule
# produces LOW -- it is reserved for future rule extensions (e.g. a
# duration-aware or room-size-aware rule introduced in a later phase) so
# that adding such a rule later does not require widening this vocabulary.
# Severity is a PRIORITY ranking of the detected condition type, never a
# measurement or estimate of energy lost.
SEVERITY_LEVELS = {"NONE", "LOW", "MEDIUM", "HIGH"}


# ---------------------------------------------------------------------------
# Wastage rules -- keyed on Phase 1's condition_code (single source of truth
# for occupancy/HVAC/lighting state; not re-derived here).
#
# Each entry: (wastage_type, severity, potential_wastage, reason)
# ---------------------------------------------------------------------------
WASTAGE_RULES: Dict[str, tuple] = {
    # RULE 4 -- unoccupied, nothing active -> no potential wastage
    "U_NONE": (
        "NONE", "NONE", False,
        "No active HVAC or lighting was detected while the room was unoccupied.",
    ),
    # RULE 1 -- unoccupied + HVAC on
    "U_HVAC": (
        "HVAC", "MEDIUM", True,
        "Potential HVAC wastage: the room is unoccupied while HVAC is active.",
    ),
    # RULE 2 -- unoccupied + lighting on
    "U_LIGHT": (
        "LIGHTING", "MEDIUM", True,
        "Potential lighting wastage: the room is unoccupied while lighting is active.",
    ),
    # RULE 3 -- unoccupied + HVAC and lighting on (higher priority: 2 appliance categories)
    "U_HVAC_LIGHT": (
        "HVAC_AND_LIGHTING", "HIGH", True,
        "Potential HVAC and lighting wastage: the room is unoccupied while both HVAC and lighting are active.",
    ),
    # RULE 5 -- occupied rooms are never classified as wastage in Phase 2,
    # regardless of which appliances are active.
    "O_HVAC": (
        "NONE", "NONE", False,
        "Room is occupied with HVAC active; this is normal active usage, not potential wastage.",
    ),
    "O_LIGHT": (
        "NONE", "NONE", False,
        "Room is occupied with lighting active; this is normal active usage, not potential wastage.",
    ),
    "O_HVAC_LIGHT": (
        "NONE", "NONE", False,
        "Room is occupied with HVAC and lighting active; this is normal active usage, not potential wastage.",
    ),
    "O_NONE": (
        "NONE", "NONE", False,
        "Room is occupied with no HVAC or lighting active; no potential wastage condition applies.",
    ),
}


# ---------------------------------------------------------------------------
# Core: single-record wastage detection
# ---------------------------------------------------------------------------
def analyze_wastage_record(record: Record, record_id: Optional[int] = None) -> Dict:
    """
    Detect the potential-wastage condition of a single record.

    Input:
        record: a pandas.Series or dict, same requirements as
                energy_analysis.analyze_record() (occupancy, hvac_usage,
                lighting_usage are required; everything else is contextual).
        record_id: optional identifier (e.g. the source DataFrame index) to
                   include in the output for traceability.
    Output:
        A structured dict: every field produced by Phase 1's
        analyze_record(), plus `record_id`, `wastage_type`, `severity`,
        `potential_wastage`, and `reason`.
    Raises:
        EnergyAnalysisError (re-raised unchanged from Phase 1) if a required
        classification field is missing or invalid. Phase 2 introduces no
        separate validation mechanism.
    """
    condition_result = analyze_record(record)  # Phase 1 -- validates & classifies

    wastage_type, severity, potential_wastage, reason = WASTAGE_RULES[condition_result["condition_code"]]

    result = dict(condition_result)
    result["record_id"] = record_id
    result["wastage_type"] = wastage_type
    result["severity"] = severity
    result["potential_wastage"] = potential_wastage
    result["reason"] = reason
    return result


# ---------------------------------------------------------------------------
# Core: dataset-level wastage detection
# ---------------------------------------------------------------------------
def analyze_wastage_many(df: pd.DataFrame) -> List[Dict]:
    """
    Run analyze_wastage_record() over every row of a DataFrame.

    `record_id` for each result is the row's original DataFrame index, so
    identity is preserved even if `df` is a filtered subset of the full
    processed dataset.

    Output: list of per-record result dicts, same order as the input rows.
    """
    return [analyze_wastage_record(row, record_id=idx) for idx, row in df.iterrows()]


def analyze_wastage_dataset(df: pd.DataFrame) -> Dict:
    """
    Produce aggregate, dataset-level potential-wastage statistics.

    Input:  the processed DataFrame (or any filtered subset of it).
    Output: a dict of counts and rates. Denominators are named explicitly in
            every rate field to avoid ambiguity (see project spec section
            "DATASET-LEVEL OUTPUT"). These are rates of RECORDS satisfying a
            condition, never a measurement of energy wasted.
    """
    if df is None or df.empty:
        return {"total_records": 0}

    records = analyze_wastage_many(df)
    results_df = pd.DataFrame(records)

    total = len(results_df)
    occupied_mask = results_df["occupancy"] > 0
    occupied_count = int(occupied_mask.sum())
    unoccupied_count = int((~occupied_mask).sum())

    potential_wastage_count = int(results_df["potential_wastage"].sum())
    no_wastage_count = total - potential_wastage_count

    # NOTE on these three counts -- two different, both-correct groupings:
    #
    # `wastage_type` (per-record field, set above) is a MUTUALLY EXCLUSIVE
    # category: exactly one of NONE / HVAC / LIGHTING / HVAC_AND_LIGHTING.
    #
    # The dataset-level "potential_hvac_wastage_records" and
    # "potential_lighting_wastage_records" counts below are NOT the same as
    # counting wastage_type=="HVAC" / =="LIGHTING". Per Rule 1 and Rule 2
    # (see WASTAGE_RULES docstring and project spec: "This includes both
    # U_HVAC and U_HVAC_LIGHT conditions"), they are RULE-level counts that
    # OVERLAP with the combined count -- any record where HVAC contributed
    # to a potential-wastage condition, whether alone (wastage_type=="HVAC")
    # or combined with lighting (wastage_type=="HVAC_AND_LIGHTING"). This
    # matches Phase 1's own "potential_hvac_wastage_condition_records"
    # definition exactly, so the two phases' figures stay directly
    # comparable (verified in tests/test_wastage_detection.py).
    hvac_wastage_count = int(
        (results_df["potential_wastage"] & (results_df["hvac_status"] == "On")).sum()
    )
    lighting_wastage_count = int(
        (results_df["potential_wastage"] & (results_df["lighting_status"] == "On")).sum()
    )
    combined_wastage_count = int((results_df["wastage_type"] == "HVAC_AND_LIGHTING").sum())

    severity_counts = results_df["severity"].value_counts().to_dict()
    for level in SEVERITY_LEVELS:
        severity_counts.setdefault(level, 0)

    wastage_rate_of_all_records = round(potential_wastage_count / total * 100, 2) if total > 0 else None
    wastage_rate_of_unoccupied_records = (
        round(potential_wastage_count / unoccupied_count * 100, 2) if unoccupied_count > 0 else None
    )

    return {
        "total_records": total,
        "occupied_records": occupied_count,
        "unoccupied_records": unoccupied_count,
        "no_potential_wastage_records": no_wastage_count,
        "potential_wastage_records": potential_wastage_count,
        "potential_hvac_wastage_records": hvac_wastage_count,
        "potential_lighting_wastage_records": lighting_wastage_count,
        "potential_hvac_and_lighting_wastage_records": combined_wastage_count,
        "severity_counts": severity_counts,
        # Explicit denominators, per the research-integrity requirement that
        # these never be described as "percentage of energy wasted":
        "potential_wastage_rate_of_all_records_pct": wastage_rate_of_all_records,
        "potential_wastage_rate_of_unoccupied_records_pct": wastage_rate_of_unoccupied_records,
    }
