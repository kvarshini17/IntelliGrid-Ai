"""
modules/optimization.py
-------------------------
Purpose : Energy Optimization Engine (Phase 4).

For each controllable appliance (HVAC, Lighting) on a record, generates
candidate actions, applies hard operating constraints, scores the feasible
candidates with a documented, deterministic objective function, and selects
the best feasible action -- with the reasoning fully visible.

This module explicitly does NOT:
    - automatically turn off equipment in an occupied room (hard constraint),
    - fabricate kWh/cost savings estimates (no appliance-level sub-metering
      exists in this dataset -- see SCORING DESIGN below),
    - use machine learning or a trained model,
    - modify the dataset.

Design note -- reuse, not duplication:
    Built on `anomaly_detection.analyze_anomalies()` (which itself reuses
    Phase 1 + Phase 2), so optimization sees the full condition/wastage/
    anomaly picture without re-deriving any of it.

AI technique used : Deterministic candidate-action scoring (not a trained
                     model, not A*/search -- see SEARCH DESIGN below).

===============================================================================
SEARCH DESIGN -- why no A*/Best-First Search was used
===============================================================================
A search algorithm earns its place when there is a multi-step state space
with transitions to explore (e.g. planning an HVAC setpoint trajectory over
several future hours). Here, the decision is a one-shot choice between a
small, fixed set of candidate actions per appliance per record -- there is no
sequence, no transition model, and no horizon. A deterministic
candidate-scoring approach is strictly clearer and equally correct for this
problem, so that is what is implemented. (Documented per the project's own
instruction not to add search merely to look more advanced.)
===============================================================================

===============================================================================
SCORING DESIGN -- documented assumptions
===============================================================================
Score = energy_cost_term + constraint_penalty + comfort_penalty + operational_penalty

- energy_cost_term: an ORDINAL priority weight (TURN_OFF = -1, KEEP = 0),
  NOT a quantitative kWh or currency estimate. This dataset has no
  appliance-level sub-metering -- energy_consumption is a single whole-record
  value covering all active appliances combined, so there is no valid basis
  to compute "how much the HVAC alone would save" without fabricating a
  split. Lower score = more favorable action. This is a ranking signal only.
- constraint_penalty: a HARD constraint, not a soft score term in the current
  rule set. TURN_OFF is never offered as a candidate when occupancy > 0 --
  it is excluded from the candidate set entirely, never merely penalized.
  (Hence this term is always 0 among candidates actually generated.)
- comfort_penalty: always 0 in this phase. TURN_OFF is only ever offered when
  the room is unoccupied, so there is no occupant to experience discomfort.
  Retained as an explicit field (not silently omitted) so a future phase
  that models pre-conditioning or occupancy-return timing can populate it
  without changing the output schema.
- operational_penalty: always 0 in this phase. Modeling equipment-cycling
  stress would require cross-record/time-series state (e.g. how recently an
  appliance was toggled), which Phase 1-3 do not track. Retained as an
  explicit field for the same reason as comfort_penalty.

Given the above, the CURRENT effective rule is: prefer TURN_OFF over KEEP
whenever TURN_OFF is feasible (unoccupied + appliance on); otherwise KEEP is
the only feasible candidate. This matches the project's worked examples
exactly, and is intentionally this simple -- adding unverified precision to
comfort/operational terms would violate the project's research-integrity
requirement against inventing scientifically unsupported thresholds.
===============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union

import pandas as pd

from modules.anomaly_detection import analyze_anomalies

Record = Union[pd.Series, dict]

ACTION_TYPES = {"KEEP", "REDUCE", "TURN_OFF"}
# REDUCE is part of the declared action vocabulary but is never generated as
# a candidate in the current implementation: HVAC/lighting state in this
# dataset is strictly binary (On/Off), so there is no data-grounded
# intermediate "reduced" state to offer without fabricating one. Reserved
# for a future phase with multi-level equipment telemetry.

CONTROLLABLE_APPLIANCES = ["HVAC", "Lighting"]


@dataclass
class ActionCandidate:
    appliance: str
    action: str
    feasible: bool
    score: float
    reason: str


def _appliance_state(record_result: Dict, appliance: str) -> str:
    return record_result["hvac_status"] if appliance == "HVAC" else record_result["lighting_status"]


def generate_candidate_actions(record_result: Dict, appliance: str) -> List[ActionCandidate]:
    """
    Generate the feasible candidate actions for one appliance on one record.

    Input:
        record_result: a dict as produced by anomaly_detection.analyze_anomalies()
                        (carries condition_code, occupancy, hvac_status, lighting_status, ...).
        appliance: "HVAC" or "Lighting".
    Output:
        List of ActionCandidate. TURN_OFF is included only when occupancy==0
        AND the appliance is currently On (hard constraint -- never offered
        for an occupied room, per the project's explicit instruction).
    """
    state = _appliance_state(record_result, appliance)
    occupancy = record_result["occupancy"]

    candidates = [ActionCandidate(
        appliance=appliance, action="KEEP", feasible=True, score=0.0,
        reason=f"{appliance} remains in its current state ({state}).",
    )]

    if state == "On" and occupancy == 0:
        candidates.append(ActionCandidate(
            appliance=appliance, action="TURN_OFF", feasible=True, score=-1.0,
            reason=f"Room is unoccupied while {appliance} is active; turning it off is feasible with no occupant impact.",
        ))

    return candidates


def select_best_action(candidates: List[ActionCandidate]) -> ActionCandidate:
    """Select the lowest-scoring feasible candidate (lower score = more
    favorable, per the documented ordinal scoring convention)."""
    feasible = [c for c in candidates if c.feasible]
    return min(feasible, key=lambda c: c.score)


def optimize_record(record_result: Dict) -> Dict:
    """
    Run optimization for both controllable appliances on a single record
    (as produced by anomaly_detection.analyze_anomalies()).

    Output:
        {
            "record_id": ..., "condition_code": ...,
            "actions": [ {appliance, candidates, selected_action, score, reason}, ... ],
        }
    """
    actions = []
    for appliance in CONTROLLABLE_APPLIANCES:
        candidates = generate_candidate_actions(record_result, appliance)
        best = select_best_action(candidates)
        actions.append({
            "appliance": appliance,
            "candidates": [
                {"action": c.action, "feasible": c.feasible, "score": c.score, "reason": c.reason}
                for c in candidates
            ],
            "selected_action": best.action,
            "score": best.score,
            "reason": best.reason,
            "constraint_penalty": 0.0,   # see module docstring SCORING DESIGN
            "comfort_penalty": 0.0,
            "operational_penalty": 0.0,
        })

    result = dict(record_result)
    result["optimization_actions"] = actions
    result["has_turn_off_recommendation"] = any(a["selected_action"] == "TURN_OFF" for a in actions)
    return result


def optimize_many(df: pd.DataFrame) -> List[Dict]:
    """Run optimize_record() over every row of df, reusing analyze_anomalies()."""
    anomaly_results = analyze_anomalies(df)
    return [optimize_record(r) for r in anomaly_results]


def optimize_dataset(df: pd.DataFrame) -> Dict:
    """
    Aggregate optimization statistics for a dataset (or a filtered subset).

    Output: total records, per-appliance TURN_OFF recommendation counts, and
    records with at least one TURN_OFF recommendation. No kWh/cost totals are
    reported -- see module docstring SCORING DESIGN for why none can be
    validly computed from this dataset.
    """
    if df is None or df.empty:
        return {"total_records": 0}

    results = optimize_many(df)
    total = len(results)

    turn_off_counts = {appliance: 0 for appliance in CONTROLLABLE_APPLIANCES}
    records_with_any_turn_off = 0
    for r in results:
        any_turn_off = False
        for a in r["optimization_actions"]:
            if a["selected_action"] == "TURN_OFF":
                turn_off_counts[a["appliance"]] += 1
                any_turn_off = True
        if any_turn_off:
            records_with_any_turn_off += 1

    return {
        "total_records": total,
        "hvac_turn_off_recommendations": turn_off_counts["HVAC"],
        "lighting_turn_off_recommendations": turn_off_counts["Lighting"],
        "records_with_any_turn_off_recommendation": records_with_any_turn_off,
    }
