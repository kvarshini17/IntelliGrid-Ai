"""
experiments/anomaly_evaluation.py
-------------------------------------
EXPERIMENT 2 -- Context-Aware Anomaly Detection evaluation via controlled
synthetic fault injection.

EXPERIMENT 5 -- Global baseline vs. context-aware baseline comparison
(RQ2: "Does context-aware anomaly detection identify unusual energy
consumption better than a global baseline?").

Unlike the wastage-detection fault injection (see wastage_evaluation.py),
this IS a genuine empirical comparison: both methods are magnitude-based
statistical detectors, so they can meaningfully disagree on the same
injected ground truth, and the comparison has real research content.

Never modifies the real processed dataset -- operates only on an in-memory
COPY, clearly labeled SYNTHETIC EVALUATION DATA.
"""

from typing import Dict

import numpy as np
import pandas as pd

import config
from modules.data_manager import load_processed_dataset
from modules.anomaly_detection import analyze_anomalies

RANDOM_SEED = 456
INJECTION_MULTIPLIER = 2.0   # documented assumption: double energy_consumption at injected rows
N_INJECTED = 30


def inject_consumption_spikes(df: pd.DataFrame, n_injected: int = N_INJECTED,
                               multiplier: float = INJECTION_MULTIPLIER,
                               seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Return a COPY of df with energy_consumption multiplied at n_injected
    randomly selected rows, simulating an unusual consumption spike. Adds
    `is_injected_anomaly` as ground truth. All other fields are untouched.
    """
    df = df.copy()
    df["is_injected_anomaly"] = False

    rng = np.random.RandomState(seed)
    injected_idx = rng.choice(df.index, size=n_injected, replace=False)

    df.loc[injected_idx, "energy_consumption"] = df.loc[injected_idx, "energy_consumption"] * multiplier
    df.loc[injected_idx, "is_injected_anomaly"] = True

    return df


def _global_baseline_classify(df: pd.DataFrame) -> pd.Series:
    """
    Strawman comparator: a SINGLE global mean/std z-score threshold across
    the entire dataset, with no contextual conditioning. Implemented locally
    here (not as a reusable project module) -- it exists only to measure the
    value of contextual conditioning, and is explicitly NOT the project's
    anomaly detector.
    """
    mean = df["energy_consumption"].mean()
    std = df["energy_consumption"].std(ddof=1)
    z = (df["energy_consumption"] - mean) / std
    return z.abs() >= config.ANOMALY_Z_THRESHOLD


def run_experiment() -> Dict:
    real_df = load_processed_dataset()
    injected_df = inject_consumption_spikes(real_df)

    # --- Context-aware method (the project's actual detector) ---
    context_results = pd.DataFrame(analyze_anomalies(injected_df))
    context_results["is_injected_anomaly"] = injected_df["is_injected_anomaly"].values
    context_flagged = context_results["is_anomaly"]

    # --- Global baseline comparator (this experiment only) ---
    global_flagged = _global_baseline_classify(injected_df)

    def _metrics(flagged: pd.Series, ground_truth: pd.Series) -> Dict:
        flagged = flagged.reset_index(drop=True)
        ground_truth = ground_truth.reset_index(drop=True)
        tp = int((flagged & ground_truth).sum())
        fp = int((flagged & ~ground_truth).sum())
        fn = int((~flagged & ground_truth).sum())
        tn = int((~flagged & ~ground_truth).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else None
        return {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
                "recall": recall, "precision": precision, "false_positive_rate": false_positive_rate}

    ground_truth = pd.Series(injected_df["is_injected_anomaly"].values)
    context_metrics = _metrics(context_flagged, ground_truth)
    global_metrics = _metrics(global_flagged, ground_truth)

    # Coverage note: context-aware method cannot classify INSUFFICIENT_BASELINE rows at all.
    insufficient_count = int((context_results["anomaly_type"] == "INSUFFICIENT_BASELINE").sum())

    # Break down WHY the context-aware method missed injected anomalies (false negatives),
    # since "false negative" alone hides two structurally different causes.
    injected_rows = context_results[context_results["is_injected_anomaly"]]
    fn_rows = injected_rows[~injected_rows["is_anomaly"]]
    fn_insufficient_baseline = int((fn_rows["anomaly_type"] == "INSUFFICIENT_BASELINE").sum())
    fn_normal_despite_injection = int((fn_rows["anomaly_type"] == "NORMAL").sum())

    return {
        "label": "SYNTHETIC EVALUATION DATA -- fault injection, not real observations",
        "injection_design": {
            "n_injected": N_INJECTED,
            "multiplier": INJECTION_MULTIPLIER,
            "seed": RANDOM_SEED,
            "note": "energy_consumption multiplied by a fixed factor at randomly selected rows; a documented experimental assumption, not a physically modeled fault.",
        },
        "context_aware_method": context_metrics,
        "global_baseline_method": global_metrics,
        "context_aware_coverage_note": (
            f"{insufficient_count} of {len(injected_df)} records fall in context groups below "
            f"the minimum size and are never classified (INSUFFICIENT_BASELINE) by the context-aware "
            f"method; the global method has no such coverage gap. This is a genuine trade-off, not a flaw "
            f"in either method -- reported here rather than hidden."
        ),
        "context_aware_false_negative_breakdown": {
            "total_false_negatives": len(fn_rows),
            "due_to_insufficient_baseline_group": fn_insufficient_baseline,
            "due_to_high_within_group_variance_diluting_z_score": fn_normal_despite_injection,
            "interpretation": (
                "Two structurally different causes: (1) the injected row fell in a context group too "
                "small to classify at all; (2) the row's context group had high residual natural "
                "variance in energy_consumption, lowering the injected spike's z-score below threshold "
                "even though the spike is extreme relative to the WHOLE dataset. Cause (2) is plausibly "
                "linked to temperature, which correlates with energy_consumption (see Phase 0 data-quality "
                "findings) but is NOT part of the Phase 3 context key -- a concrete, evidence-based "
                "direction for a future context-key extension, not a design flaw being hidden."
            ),
        },
    }


if __name__ == "__main__":
    import json
    result = run_experiment()
    print(json.dumps(result, indent=2, default=str))
