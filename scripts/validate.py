"""
scripts/validate.py
---------------------
Independent validation of the Phase 0 transformation. Re-derives everything
from the original file and the processed outputs and checks it -- does not
trust the transform script's own print statements.

Runs the 10 validation checks approved in the design review and prints a
pass/fail report with actual numbers.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_PATH = BASE_DIR / "data" / "original" / "intelligrid_energy_dataset_with_4_fields.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "intelligrid_energy_dataset.csv"
ROOM_REF_PATH = BASE_DIR / "data" / "processed" / "room_reference.csv"

results = []  # (check_number, description, pass_bool, detail_string)


def record(n, desc, passed, detail):
    results.append((n, desc, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Check {n}: {desc}")
    print(f"        {detail}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    original = pd.read_csv(ORIGINAL_PATH)
    # IMPORTANT: pandas' default na_values list includes the literal string
    # "None", which would silently re-convert our intentional "None" appliance
    # label back into NaN on reload. keep_default_na=False (with an explicit,
    # narrow na_values set) is required any time this processed file is read.
    processed = pd.read_csv(PROCESSED_PATH, keep_default_na=False, na_values=[""])
    room_ref = pd.read_csv(ROOM_REF_PATH)

    # ---- Check 1: row count preserved ----
    record(1, "Row count preserved (1000 in, 1000 out, no drops)",
           len(original) == len(processed) == 1000,
           f"original={len(original)} rows, processed={len(processed)} rows")

    # ---- Check 2: untouched-field integrity ----
    rename_map = {
        "Electricity Tariff": "electricity_tariff",  # not checked here (retained but renamed, not "untouched name")
    }
    untouched_fields = [
        ("timestamp", "timestamp"),
        ("temperature", "temperature"),
        ("humidity", "humidity"),
        ("occupancy", "occupancy"),
        ("hvac_usage", "hvac_usage"),
        ("lighting_usage", "lighting_usage"),
        ("renewable_energy", "renewable_energy"),
        ("energy_consumption", "energy_consumption"),
        ("day_of_week", "day_of_week"),
        ("holiday", "holiday"),
    ]
    mismatches = {}
    for orig_col, proc_col in untouched_fields:
        if orig_col not in original.columns or proc_col not in processed.columns:
            mismatches[orig_col] = "column missing"
            continue
        o = original[orig_col].reset_index(drop=True)
        p = processed[proc_col].reset_index(drop=True)
        if o.dtype == float or p.dtype == float:
            equal = np.allclose(pd.to_numeric(o, errors="coerce"),
                                 pd.to_numeric(p, errors="coerce"), equal_nan=True)
        else:
            equal = (o.astype(str) == p.astype(str)).all()
        if not equal:
            mismatches[orig_col] = "VALUES DIFFER"
    record(2, "Untouched fields are byte/value-identical to original",
           len(mismatches) == 0,
           f"fields checked={len(untouched_fields)}, mismatches={mismatches if mismatches else 'none'}")

    # ---- Check 3: room balance ----
    counts = processed["room_id"].value_counts()
    within_tolerance = ((counts - 100).abs() <= 2).all()
    record(3, "Room totals within +/-2 of 100",
           within_tolerance,
           f"min={counts.min()}, max={counts.max()}, per-room counts={counts.sort_index().to_dict()}")

    # ---- Check 4: room x hour independence ----
    ct = pd.crosstab(processed["room_id"], processed["hour"])
    chi2, p_value, dof, expected = chi2_contingency(ct)
    record(4, "Room x hour independence (chi-square test, expect high p-value)",
           p_value > 0.05,
           f"chi2={chi2:.3f}, dof={dof}, p-value={p_value:.4f}, cell min={ct.values.min()}, cell max={ct.values.max()}")

    # ---- Check 5: hour coverage ----
    coverage = (ct > 0).sum(axis=1)
    record(5, "Every room observed at all 24 hours",
           (coverage == 24).all(),
           f"coverage per room={coverage.to_dict()}")

    # ---- Check 6: fixed square footage per room ----
    sqft_nunique = processed.groupby("room_id")["square_footage"].nunique()
    record(6, "square_footage is constant (single value) per room_id",
           (sqft_nunique == 1).all(),
           f"unique sqft values per room={sqft_nunique.to_dict()}")

    # ---- Check 7: Appliance / Appliance Status consistency preserved ----
    processed["hvac_on_check"] = (processed["hvac_usage"] == "On").astype(int)
    processed["lighting_on_check"] = (processed["lighting_usage"] == "On").astype(int)

    def expected_appliance(row):
        h, l = row["hvac_on_check"], row["lighting_on_check"]
        if h and l:
            return "HVAC + Lighting"
        if h:
            return "HVAC"
        if l:
            return "Lighting"
        return "None"

    expected_app = processed.apply(expected_appliance, axis=1)
    appliance_ok = (expected_app == processed["appliance"]).all()

    expected_status = np.where(
        (processed["hvac_on_check"] == 1) | (processed["lighting_on_check"] == 1), "On", "Off"
    )
    status_ok = (expected_status == processed["appliance_status"]).all()

    record(7, "Appliance / Appliance Status still consistent with hvac/lighting usage after NaN->None relabel",
           bool(appliance_ok and status_ok),
           f"appliance_label_consistent={bool(appliance_ok)}, appliance_status_consistent={bool(status_ok)}, "
           f"appliance_mismatches={(expected_app != processed['appliance']).sum()}, "
           f"status_mismatches={(expected_status != processed['appliance_status']).sum()}")

    # ---- Check 8: Electricity Tariff banding unaffected ----
    orig_tariff_by_hour = original.assign(hour=pd.to_datetime(original["timestamp"]).dt.hour) \
                                   .groupby("hour")["Electricity Tariff"].unique()
    proc_tariff_by_hour = processed.groupby("hour")["electricity_tariff"].unique()
    tariff_match = all(
        set(orig_tariff_by_hour[h]) == set(proc_tariff_by_hour[h]) for h in range(24)
    )
    record(8, "Electricity Tariff hour-banding unchanged from original",
           tariff_match,
           f"hour->tariff mapping identical for all 24 hours: {tariff_match}")

    # ---- Check 9: capacity sanity report (descriptive only, not corrective) ----
    merged = processed.merge(room_ref, on="room_id", how="left")
    exceed = merged[merged["occupancy"] > merged["capacity"]]
    record(9, "Capacity sanity report (descriptive -- violations are NOT corrected)",
           True,  # always "pass" as a check -- this is a report, not a pass/fail gate
           f"{len(exceed)} of {len(merged)} rows have occupancy > assigned room capacity "
           f"({len(exceed)/len(merged)*100:.1f}%) -- documented as a known limitation, not altered")

    # ---- Check 10: reproducibility ----
    import subprocess
    script_path = BASE_DIR / "scripts" / "preprocess.py"
    tmp_dataset_before = PROCESSED_PATH.read_bytes()
    tmp_room_before = ROOM_REF_PATH.read_bytes()
    subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=True, capture_output=True)
    tmp_dataset_after = PROCESSED_PATH.read_bytes()
    tmp_room_after = ROOM_REF_PATH.read_bytes()
    reproducible = (tmp_dataset_before == tmp_dataset_after) and (tmp_room_before == tmp_room_after)
    record(10, "Re-running the pipeline produces byte-identical output",
           reproducible,
           f"dataset identical={tmp_dataset_before == tmp_dataset_after}, "
           f"room_reference identical={tmp_room_before == tmp_room_after}")

    # ---- Original file untouched check (bonus, not numbered in the original 10) ----
    print()
    print(f"Original file SHA-256 (post-run): {file_sha256(ORIGINAL_PATH)}")
    print("(compare against the pre-run hash recorded in TRANSFORM_LOG.md)")

    print()
    passed_count = sum(1 for r in results if r[2])
    print(f"=== {passed_count}/{len(results)} checks passed ===")


if __name__ == "__main__":
    main()
