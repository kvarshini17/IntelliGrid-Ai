"""
tests/test_experiments.py
-----------------------------
Phase 10 -- confirms each research experiment runs end-to-end without error,
produces the expected top-level structure, and never modifies the real
dataset files on disk (hash check before/after each experiment run).
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from experiments import anomaly_evaluation, optimization_evaluation, rag_evaluation, wastage_evaluation


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_hashes():
    return {
        "original": _file_sha256(config.ORIGINAL_DATASET_PATH),
        "processed": _file_sha256(config.PROCESSED_DATASET_PATH),
        "room_reference": _file_sha256(config.ROOM_REFERENCE_PATH),
    }


def test_wastage_evaluation_runs_and_preserves_dataset():
    before = _dataset_hashes()
    result = wastage_evaluation.run_experiment()
    after = _dataset_hashes()
    assert before == after

    assert "SYNTHETIC EVALUATION DATA" in result["label"]
    assert result["baseline_real_dataset_summary"]["total_records"] == 1000
    assert result["fault_injection"]["non_injected_rows_unaffected_by_injection"] is True


def test_wastage_evaluation_injection_only_touches_a_copy():
    """Confirm the real dataset loaded fresh afterwards is unaffected."""
    from modules.data_manager import load_processed_dataset
    df1 = load_processed_dataset()
    wastage_evaluation.run_experiment()
    df2 = load_processed_dataset()
    assert df1["occupancy"].equals(df2["occupancy"])
    assert df1["hvac_usage"].equals(df2["hvac_usage"])


def test_anomaly_evaluation_runs_and_preserves_dataset():
    before = _dataset_hashes()
    result = anomaly_evaluation.run_experiment()
    after = _dataset_hashes()
    assert before == after

    assert "SYNTHETIC EVALUATION DATA" in result["label"]
    for method in ("context_aware_method", "global_baseline_method"):
        for key in ("true_positives", "false_positives", "false_negatives", "true_negatives"):
            assert key in result[method]
    assert result["context_aware_false_negative_breakdown"]["total_false_negatives"] == (
        result["context_aware_false_negative_breakdown"]["due_to_insufficient_baseline_group"]
        + result["context_aware_false_negative_breakdown"]["due_to_high_within_group_variance_diluting_z_score"]
    )


def test_optimization_evaluation_runs_and_preserves_dataset():
    before = _dataset_hashes()
    result = optimization_evaluation.run_experiment()
    after = _dataset_hashes()
    assert before == after

    assert result["dataset_summary"]["total_records"] == 1000
    assert result["hard_constraint_check"]["constraint_holds"] is True
    assert "Not scientifically established" in result["savings_estimate"]


def test_rag_evaluation_runs_and_preserves_dataset():
    before = _dataset_hashes()
    result = rag_evaluation.run_experiment()
    after = _dataset_hashes()
    assert before == after

    assert result["n_on_topic_questions"] > 0
    assert result["n_off_topic_questions"] > 0
    assert 0.0 <= result["precision_at_1_on_topic"] <= 1.0
    assert result["off_topic_correct_no_evidence_rate"] == 1.0  # both off-topic questions must be correctly refused
