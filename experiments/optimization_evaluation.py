"""
experiments/optimization_evaluation.py
-------------------------------------------
EXPERIMENT 3 -- Optimization Action Comparison (RQ3: "Can deterministic
optimization produce explainable energy-saving actions?").

Reports candidate-action and selected-action statistics on the real
processed dataset. Per the project's research-integrity requirement, this
experiment reports ONLY counts and the (ordinal, documented) score
statistics already produced by modules/optimization.py -- it does NOT
compute or claim a kWh/cost savings total, because no appliance-level
sub-metering exists in this dataset to support one (see
modules/optimization.py's SCORING DESIGN section for the full justification).
"""

from typing import Dict

import pandas as pd

from modules.data_manager import load_processed_dataset
from modules.optimization import optimize_dataset, optimize_many


def run_experiment() -> Dict:
    df = load_processed_dataset()
    summary = optimize_dataset(df)

    results = optimize_many(df)
    action_rows = []
    for r in results:
        for a in r["optimization_actions"]:
            action_rows.append({
                "appliance": a["appliance"],
                "selected_action": a["selected_action"],
                "score": a["score"],
                "occupancy_state": r["occupancy_state"],
            })
    actions_df = pd.DataFrame(action_rows)

    by_action = actions_df.groupby(["appliance", "selected_action"]).size().unstack(fill_value=0).to_dict()
    mean_score_by_action = actions_df.groupby("selected_action")["score"].mean().to_dict()

    # Constraint check: TURN_OFF must NEVER appear for an occupied record (hard constraint).
    turn_off_rows = actions_df[actions_df["selected_action"] == "TURN_OFF"]
    constraint_violations = int((turn_off_rows["occupancy_state"] == "Occupied").sum())

    return {
        "dataset_summary": summary,
        "action_counts_by_appliance_and_type": by_action,
        "mean_ordinal_score_by_action_type": mean_score_by_action,
        "hard_constraint_check": {
            "turn_off_recommendations_for_occupied_rooms": constraint_violations,
            "expected": 0,
            "constraint_holds": constraint_violations == 0,
        },
        "savings_estimate": "Not scientifically established from the available data -- no appliance-level "
                             "sub-metering exists to compute a valid kWh/cost figure. Scores are ordinal "
                             "priority rankings only (see modules/optimization.py).",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_experiment(), indent=2, default=str))
