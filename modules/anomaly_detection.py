"""
modules/anomaly_detection.py
-------------------------------
Purpose : Context-Aware Anomaly Detection Engine (Phase 3).

Answers: "Is this energy consumption unusually high or low for a similar
operating context?" -- fundamentally different from Phase 2's question
("could this operating condition represent potential wastage?").

This module explicitly does NOT:
    - use a single global mean/std threshold for the whole dataset,
    - use machine learning of any kind (no scikit-learn, no clustering,
      no trained models),
    - treat wastage and anomaly as the same thing (see WASTAGE VS ANOMALY
      below),
    - modify, normalize, or inject values into energy_consumption,
    - claim a physical energy-efficiency meaning for the z-score threshold.

Design note -- reuse, not duplication:
    This module does NOT re-derive condition codes or wastage logic. It
    calls `wastage_detection.analyze_wastage_record()` (which itself calls
    Phase 1's `energy_analysis.analyze_record()`) to obtain condition_code,
    hvac_status, lighting_status, potential_wastage, etc., then adds only
    what is new here: context grouping, baseline statistics, z-scores, and
    anomaly classification.

AI technique used : Deterministic descriptive statistics (context-conditioned
                     mean/standard deviation + z-score thresholding). No ML.

===============================================================================
BASELINE DESIGN -- READ BEFORE USE
===============================================================================
Context key: (occupancy_state, hvac_state, lighting_state, hour)
    occupancy_state : "Occupied" / "Unoccupied", derived from Phase 1's
                       condition_code prefix ("O_" / "U_") -- not re-derived
                       from occupancy independently.
    hvac_state       : "On" / "Off" (Phase 1's hvac_status, unchanged)
    lighting_state   : "On" / "Off" (Phase 1's lighting_status, unchanged)
    hour             : 0-23, read directly from the record's `hour` field
                       (falls back to deriving it from `timestamp` only if
                       `hour` is absent, e.g. for a hand-built test record).
                       `source_day_of_week` is never used -- Phase 0 already
                       established it is statistically unreliable.

Baseline calculation -- Option A (in-sample / offline), explicitly chosen
over leave-one-out or historical-only:
    For every context group, mean and standard deviation (sample std,
    ddof=1) are computed over ALL records in that group within the
    DataFrame being analyzed -- INCLUDING the record whose z-score is being
    computed. This is the simplest defensible approach for a first research
    prototype and is documented here as an explicit limitation: this is an
    OFFLINE, IN-SAMPLE analysis method, not a real-time detector. A record
    contributes to its own baseline, which slightly dampens how extreme its
    own z-score can appear. A future phase could implement leave-one-out or
    historical-only (causal) baselines for a stricter evaluation; that is
    out of scope here.

Room is deliberately NOT part of the context key (per Phase 0's correction
of the original room/hour aliasing problem and the project's explicit
instruction not to treat room identity as an independent learned profile).
Room is preserved in the output only for traceability.

Minimum group size and zero-variance handling:
    Groups with fewer than config.MIN_CONTEXT_GROUP_SIZE observations, or
    with a standard deviation below config.ANOMALY_STD_EPSILON, are marked
    INSUFFICIENT_BASELINE rather than silently falling back to a global
    baseline or dividing by zero.

Threshold:
    |z_score| >= config.ANOMALY_Z_THRESHOLD (2.0) -> HIGH/LOW_ENERGY_ANOMALY.
    This is a STATISTICAL detection threshold, not a physical
    energy-efficiency claim. A record flagged HIGH_ENERGY_ANOMALY is
    "statistically unusual relative to the observed distribution for its
    contextual group" -- never "proof the building is wasting energy."
===============================================================================

WASTAGE VS ANOMALY -- kept strictly independent:
    `potential_wastage` (Phase 2) is a pure function of
    (occupancy, hvac_usage, lighting_usage) -- it never looks at
    energy_consumption. `anomaly_type` (this module) is a pure function of
    energy_consumption relative to its context baseline -- it never looks at
    whether the condition is a "wastage" condition. A record can be:
    wastage-only, anomaly-only, both, or neither. Nothing in this module
    equates the two.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config
from modules.wastage_detection import analyze_wastage_many, analyze_wastage_record

Record = Union[pd.Series, dict]

ANOMALY_TYPES = {"NORMAL", "HIGH_ENERGY_ANOMALY", "LOW_ENERGY_ANOMALY", "INSUFFICIENT_BASELINE"}


# ---------------------------------------------------------------------------
# Context key construction
# ---------------------------------------------------------------------------
def get_occupancy_state(condition_code: str) -> str:
    """Derive 'Occupied'/'Unoccupied' from Phase 1's condition_code prefix
    (O_* / U_*) rather than re-deriving it from occupancy independently."""
    return "Occupied" if condition_code.startswith("O_") else "Unoccupied"


def _get_hour(record: Record, phase1_hour_hint: Optional[int] = None) -> Optional[int]:
    """
    Read `hour` directly from the record (the processed dataset already has
    this column). Falls back to deriving it from `timestamp` only when
    `hour` is absent -- e.g. a hand-built test record that supplies only a
    timestamp. This is a plain field-access convenience, not a duplication
    of any Phase 1 classification rule.
    """
    if isinstance(record, pd.Series):
        has_hour = "hour" in record.index and pd.notna(record["hour"])
    else:
        has_hour = "hour" in record and record["hour"] is not None

    if has_hour:
        return int(record["hour"])

    timestamp = record["timestamp"] if (
        (isinstance(record, pd.Series) and "timestamp" in record.index) or
        (isinstance(record, dict) and "timestamp" in record)
    ) else None
    if timestamp is None:
        return None
    ts = pd.to_datetime(timestamp, errors="coerce")
    return int(ts.hour) if pd.notna(ts) else None


def build_context_key(occupancy_state: str, hvac_state: str, lighting_state: str, hour: Optional[int]) -> Tuple:
    """The grouping key used for the contextual baseline."""
    return (occupancy_state, hvac_state, lighting_state, hour)


def format_context_label(context_key: Tuple) -> str:
    """Human-readable label for a context key, for display/reporting."""
    occupancy_state, hvac_state, lighting_state, hour = context_key
    hour_str = f"{hour:02d}" if hour is not None else "NA"
    return f"{occupancy_state}_HVAC-{hvac_state}_Lighting-{lighting_state}_Hour-{hour_str}"


# ---------------------------------------------------------------------------
# Pure classification function -- decoupled from DataFrame grouping so it can
# be unit-tested in isolation with exact, hand-chosen mean/std/count values.
# ---------------------------------------------------------------------------
def classify_anomaly(observed_energy: float, context_count: int, context_mean: float,
                      context_std: Optional[float]) -> Dict:
    """
    Classify a single observation against an already-computed context
    baseline. Pure function: does no grouping, no DataFrame access.

    Input:
        observed_energy : the record's energy_consumption.
        context_count    : number of observations in the context group.
        context_mean      : the context group's mean energy_consumption.
        context_std        : the context group's sample standard deviation
                              (ddof=1), or None/NaN if undefined (e.g. count < 2).
    Output:
        {"z_score": float | None, "anomaly_type": str, "is_anomaly": bool, "reason": str}
    """
    if context_count < config.MIN_CONTEXT_GROUP_SIZE:
        return {
            "z_score": None,
            "anomaly_type": "INSUFFICIENT_BASELINE",
            "is_anomaly": False,
            "reason": (
                "Insufficient observations are available to establish a reliable contextual "
                f"baseline (n={context_count}, minimum required={config.MIN_CONTEXT_GROUP_SIZE})."
            ),
        }

    if context_std is None or pd.isna(context_std) or context_std < config.ANOMALY_STD_EPSILON:
        return {
            "z_score": None,
            "anomaly_type": "INSUFFICIENT_BASELINE",
            "is_anomaly": False,
            "reason": (
                "The contextual baseline has zero or near-zero variance "
                f"(std={context_std}), so a z-score cannot be reliably computed."
            ),
        }

    z_score = (observed_energy - context_mean) / context_std

    if z_score >= config.ANOMALY_Z_THRESHOLD:
        return {
            "z_score": z_score,
            "anomaly_type": "HIGH_ENERGY_ANOMALY",
            "is_anomaly": True,
            "reason": "Energy consumption is unusually high for the observed occupancy, HVAC, lighting, and hour context.",
        }
    if z_score <= -config.ANOMALY_Z_THRESHOLD:
        return {
            "z_score": z_score,
            "anomaly_type": "LOW_ENERGY_ANOMALY",
            "is_anomaly": True,
            "reason": "Energy consumption is unusually low for the observed occupancy, HVAC, lighting, and hour context.",
        }
    return {
        "z_score": z_score,
        "anomaly_type": "NORMAL",
        "is_anomaly": False,
        "reason": "Energy consumption is within the expected range for the observed contextual group.",
    }


# ---------------------------------------------------------------------------
# Core: dataset-level context-aware anomaly detection
# ---------------------------------------------------------------------------
def analyze_anomalies(df: pd.DataFrame) -> List[Dict]:
    """
    Compute context-aware anomaly results for every row of `df`.

    Two passes, as the method requires:
      1. Classify every row (reusing Phase 1 + Phase 2 via
         wastage_detection.analyze_wastage_many) and compute its context key.
      2. Group by context key to get count/mean/std per group (in-sample,
         see module docstring), then classify each record against its own
         group's baseline via classify_anomaly().

    Output: list of per-record dicts (Phase 1 + Phase 2 fields, plus
    occupancy_state, hour, context_group, context_count, context_mean,
    context_std, z_score, anomaly_type, is_anomaly, anomaly_reason). Same
    row order as the input. NOTE: `reason` (Phase 2's wastage explanation)
    and `anomaly_reason` (this phase's explanation) are kept as separate
    fields -- never overwrite one with the other.
    Raises:
        EnergyAnalysisError (propagated unchanged from Phase 1/2) if a
        required classification field is missing or invalid in any row.
    """
    if df is None or df.empty:
        return []

    wastage_results = analyze_wastage_many(df)

    enriched = []
    for (_, row), wr in zip(df.iterrows(), wastage_results):
        occupancy_state = get_occupancy_state(wr["condition_code"])
        hour = _get_hour(row)
        context_key = build_context_key(occupancy_state, wr["hvac_status"], wr["lighting_status"], hour)
        record = dict(wr)
        record["occupancy_state"] = occupancy_state
        record["hour"] = hour
        # context_group is a string label (format_context_label is a bijective
        # mapping from the tuple context key) -- used as the groupby/lookup
        # key instead of the raw tuple, since pandas .loc[] misinterprets a
        # tuple key as a multi-level indexer rather than a single group key.
        record["context_group"] = format_context_label(context_key)
        enriched.append(record)

    enriched_df = pd.DataFrame(enriched)

    # In-sample baseline per context group (sample std, ddof=1 -- pandas default).
    baseline = enriched_df.groupby("context_group")["energy_consumption"].agg(
        context_count="count", context_mean="mean", context_std="std"
    )

    results = []
    for record in enriched:
        stats = baseline.loc[record["context_group"]]
        count = int(stats["context_count"])
        mean = float(stats["context_mean"])
        std = float(stats["context_std"]) if pd.notna(stats["context_std"]) else None

        classification = classify_anomaly(record["energy_consumption"], count, mean, std)

        final_record = dict(record)
        final_record["context_count"] = count
        final_record["context_mean"] = round(mean, 4)
        final_record["context_std"] = round(std, 4) if std is not None else None
        # NOTE: classification's own "reason" key is renamed to "anomaly_reason"
        # here specifically to avoid silently overwriting Phase 2's "reason"
        # field (the wastage explanation) already present on `record`. Both
        # explanations are distinct pieces of information and must survive
        # independently for later phases (e.g. the recommendation engine)
        # to use either one correctly.
        final_record["z_score"] = classification["z_score"]
        final_record["anomaly_type"] = classification["anomaly_type"]
        final_record["is_anomaly"] = classification["is_anomaly"]
        final_record["anomaly_reason"] = classification["reason"]
        results.append(final_record)

    return results


def analyze_anomaly_dataset(df: pd.DataFrame) -> Dict:
    """
    Produce aggregate, dataset-level anomaly statistics.

    Output: counts and rates, with denominators named explicitly. See
    project spec "DATASET-LEVEL RESULTS" -- `anomaly_rate_pct` is defined
    over records with a VALID baseline only (total - insufficient_baseline),
    never over all records, and is never described as "percentage of energy
    wasted."
    """
    if df is None or df.empty:
        return {"total_records": 0}

    results = analyze_anomalies(df)
    results_df = pd.DataFrame(results)

    total = len(results_df)
    type_counts = results_df["anomaly_type"].value_counts().to_dict()
    for t in ANOMALY_TYPES:
        type_counts.setdefault(t, 0)

    normal = type_counts["NORMAL"]
    high = type_counts["HIGH_ENERGY_ANOMALY"]
    low = type_counts["LOW_ENERGY_ANOMALY"]
    insufficient = type_counts["INSUFFICIENT_BASELINE"]

    records_with_valid_baseline = total - insufficient
    anomaly_rate_pct = round((high + low) / records_with_valid_baseline * 100, 2) if records_with_valid_baseline > 0 else None

    group_sizes = results_df.groupby("context_group")["context_count"].first()
    num_context_groups = int(group_sizes.shape[0])
    groups_with_insufficient_observations = int((group_sizes < config.MIN_CONTEXT_GROUP_SIZE).sum())

    return {
        "total_records": total,
        "normal_records": int(normal),
        "high_energy_anomalies": int(high),
        "low_energy_anomalies": int(low),
        "insufficient_baseline_records": int(insufficient),
        "records_with_valid_baseline": int(records_with_valid_baseline),
        "anomaly_rate_pct_of_records_with_valid_baseline": anomaly_rate_pct,
        "number_of_context_groups": num_context_groups,
        "context_groups_with_insufficient_observations": groups_with_insufficient_observations,
        "min_context_group_size": int(group_sizes.min()) if num_context_groups else None,
        "max_context_group_size": int(group_sizes.max()) if num_context_groups else None,
    }
