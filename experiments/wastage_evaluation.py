"""
experiments/wastage_evaluation.py
-------------------------------------
EXPERIMENT 1 -- Potential Wastage Condition Analysis, with controlled
synthetic fault injection as an evaluation protocol (RQ1: "Can contextual
rules identify potential energy-wastage conditions?").

IMPORTANT -- what this experiment can and cannot tell you:
    modules/wastage_detection.py is a DETERMINISTIC rule-based detector:
    `potential_wastage = True` if and only if `occupancy == 0 AND
    (hvac_usage == "On" OR lighting_usage == "On")`. If we inject exactly
    that condition into a record and ask the SAME rule whether it fired,
    precision and recall on the injected subset will be 1.0 by logical
    necessity -- this is a CORRECTNESS CHECK confirming the implementation
    matches its own specification, not an empirical discovery about
    detection performance (which is what fault injection is actually useful
    for with a probabilistic/learned detector). This is stated honestly
    here rather than presented as a meaningful accuracy result.

    What IS a genuine (if modest) empirical check: that injecting the
    condition does NOT cause false positives on the untouched remainder of
    the dataset, and that the real dataset's un-injected potential-wastage
    rate is correctly reproduced as a baseline for comparison.

Never modifies the real processed dataset -- operates only on an in-memory
COPY, clearly labeled SYNTHETIC EVALUATION DATA.
"""

from typing import Dict

import numpy as np
import pandas as pd

from modules.data_manager import load_processed_dataset
from modules.wastage_detection import analyze_wastage_dataset, analyze_wastage_many

RANDOM_SEED = 123


def inject_wastage_faults(df: pd.DataFrame, n_hvac: int = 20, n_lighting: int = 20,
                           n_combined: int = 10, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Return a COPY of df with synthetic wastage conditions injected into
    randomly selected rows (occupancy forced to 0, HVAC/lighting forced On).
    `energy_consumption` and all other fields are left untouched -- only
    occupancy/hvac_usage/lighting_usage are modified, and only on this copy.

    Adds an `is_injected_fault` column marking which rows were modified, to
    serve as ground truth for this evaluation.
    """
    df = df.copy()
    df["is_injected_fault"] = False
    df["injected_fault_type"] = None

    rng = np.random.RandomState(seed)
    available_idx = list(df.index)
    rng.shuffle(available_idx)

    hvac_idx = available_idx[:n_hvac]
    lighting_idx = available_idx[n_hvac:n_hvac + n_lighting]
    combined_idx = available_idx[n_hvac + n_lighting:n_hvac + n_lighting + n_combined]

    df.loc[hvac_idx, ["occupancy", "hvac_usage", "lighting_usage"]] = [0, "On", "Off"]
    df.loc[hvac_idx, "is_injected_fault"] = True
    df.loc[hvac_idx, "injected_fault_type"] = "HVAC"

    df.loc[lighting_idx, ["occupancy", "hvac_usage", "lighting_usage"]] = [0, "Off", "On"]
    df.loc[lighting_idx, "is_injected_fault"] = True
    df.loc[lighting_idx, "injected_fault_type"] = "LIGHTING"

    df.loc[combined_idx, ["occupancy", "hvac_usage", "lighting_usage"]] = [0, "On", "On"]
    df.loc[combined_idx, "is_injected_fault"] = True
    df.loc[combined_idx, "injected_fault_type"] = "HVAC_AND_LIGHTING"

    return df


def run_experiment() -> Dict:
    """
    Run the baseline (real, un-injected) wastage analysis, then the
    fault-injection correctness check, and return both results.
    """
    real_df = load_processed_dataset()
    baseline_summary = analyze_wastage_dataset(real_df)

    injected_df = inject_wastage_faults(real_df)
    injected_results = analyze_wastage_many(injected_df)
    results_df = pd.DataFrame(injected_results)
    results_df["is_injected_fault"] = injected_df["is_injected_fault"].values
    results_df["injected_fault_type"] = injected_df["injected_fault_type"].values

    injected_rows = results_df[results_df["is_injected_fault"]]
    non_injected_rows = results_df[~results_df["is_injected_fault"]]

    true_positives = int(injected_rows["potential_wastage"].sum())
    false_negatives = int((~injected_rows["potential_wastage"]).sum())
    precision = true_positives / len(injected_rows) if len(injected_rows) else None
    recall = true_positives / len(injected_rows) if len(injected_rows) else None  # identical by construction here

    # Genuine check: injection must not spuriously flag untouched rows differently
    # than the real, un-injected baseline would have for those same rows.
    real_results = pd.DataFrame(analyze_wastage_many(real_df))
    non_injected_match = (
        results_df.loc[non_injected_rows.index, "potential_wastage"].values
        == real_results.loc[non_injected_rows.index, "potential_wastage"].values
    ).all()

    return {
        "label": "SYNTHETIC EVALUATION DATA -- fault injection, not real observations",
        "baseline_real_dataset_summary": baseline_summary,
        "fault_injection": {
            "n_injected": len(injected_rows),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "note": (
                "Precision/recall = 1.0 here is EXPECTED and not a meaningful empirical result: "
                "the detector is a deterministic rule checking exactly the condition that was "
                "injected. This confirms the implementation is correct, not that it would detect "
                "a wastage condition it cannot structurally observe."
            ),
            "non_injected_rows_unaffected_by_injection": bool(non_injected_match),
        },
    }


if __name__ == "__main__":
    import json
    result = run_experiment()
    print(json.dumps(result, indent=2, default=str))
