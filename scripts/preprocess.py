"""
scripts/preprocess.py
------------------------
Phase 0 — IntelliGrid AI dataset transformation.

Purpose:
    Transform the uploaded dataset (which already contains ad-hoc,
    inconsistent synthetic fields from a prior process) into the approved
    IntelliGrid processed dataset + a stable room reference table, following
    the exact design approved by the project owner (see TRANSFORM_LOG.md).

This script is the single source of truth for the transformation. Running it
twice with the same input produces byte-identical output (reproducibility
check #10 in the validation suite).

INPUT  (read-only, never modified):
    data/original/intelligrid_energy_dataset_with_4_fields.csv

OUTPUT:
    data/processed/intelligrid_energy_dataset.csv
    data/processed/room_reference.csv

Usage:
    python scripts/preprocess.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_PATH = BASE_DIR / "data" / "original" / "intelligrid_energy_dataset_with_4_fields.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "intelligrid_energy_dataset.csv"
ROOM_REFERENCE_PATH = PROCESSED_DIR / "room_reference.csv"

# ---------------------------------------------------------------------------
# Approved configuration (see TRANSFORM_LOG.md for the full justification)
# ---------------------------------------------------------------------------
ROOM_IDS = [
    "Room 101", "Room 102", "Room 103", "Room 104", "Room 105",
    "Room 201", "Room 202", "Room 203", "Room 204", "Room 205",
]
ROOM_ASSIGNMENT_SEED = 42

# Deterministic, evenly-spaced square footage per room (approved design).
# Floor 1 rooms: 1000-1400. Floor 2 rooms: 1600-2000 (gap preserves the
# visual floor separation implied by the 101-105 / 201-205 naming).
ROOM_SQUARE_FOOTAGE = {
    "Room 101": 1000, "Room 102": 1100, "Room 103": 1200,
    "Room 104": 1300, "Room 105": 1400,
    "Room 201": 1600, "Room 202": 1700, "Room 203": 1800,
    "Room 204": 1900, "Room 205": 2000,
}

# Documented assumption: 100 sq ft per occupant (coarse commercial-space
# planning figure). Descriptive metadata only -- never used to alter data.
SQFT_PER_OCCUPANT = 100


# ---------------------------------------------------------------------------
# Step 1: Load the original file (read-only)
# ---------------------------------------------------------------------------
def load_original(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 2: Build the room reference table
# ---------------------------------------------------------------------------
def build_room_reference_table() -> pd.DataFrame:
    rows = []
    for room_id in ROOM_IDS:
        floor = int(room_id.split(" ")[1][0])
        sqft = ROOM_SQUARE_FOOTAGE[room_id]
        capacity = round(sqft / SQFT_PER_OCCUPANT)
        rows.append({
            "room_id": room_id,
            "floor": floor,
            "square_footage_fixed": sqft,
            "capacity": capacity,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: Balanced, per-hour, reproducible room assignment
# ---------------------------------------------------------------------------
def assign_rooms_balanced_per_hour(
    df: pd.DataFrame,
    hour_col: str = "hour",
    seed: int = ROOM_ASSIGNMENT_SEED,
    room_ids=None,
) -> pd.Series:
    """
    Assign a room_id to every row such that:
      - every room is observed at (approximately) every hour of the day
        (balanced within each hour-of-day group, counts differ by <= 1),
      - the assignment does NOT use row_index % n (which aliases with the
        24-hour clock when n does not share a factor of 24),
      - the "remainder" room (when an hour-group's size is not an exact
        multiple of 10) is rotated per hour, so no single room is
        systematically overrepresented,
      - the result is deterministic and reproducible given the same seed.

    Returns a pandas Series aligned to df's index.
    """
    room_ids = room_ids or ROOM_IDS
    n_rooms = len(room_ids)

    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    assigned = pd.Series(index=df.index, dtype=object)

    for h in sorted(df[hour_col].unique()):
        idx = df.loc[df.index[df[hour_col] == h]].sort_values("_orig_order").index
        n = len(idx)

        rotation = h % n_rooms
        rotated_room_ids = room_ids[rotation:] + room_ids[:rotation]

        reps = int(np.ceil(n / n_rooms))
        pool = (rotated_room_ids * reps)[:n]

        rng = np.random.RandomState(seed + h)
        pool = np.array(pool)
        rng.shuffle(pool)

        assigned.loc[idx] = pool

    return assigned


# ---------------------------------------------------------------------------
# Step 4: Full transformation pipeline
# ---------------------------------------------------------------------------
def transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- Derived timestamp fields (re-derived and verified, not trusted blindly) ---
    ts = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = ts.dt.day_name()
    df["hour"] = ts.dt.hour
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    # --- Room reassignment (replaces old Room + old per-row square_footage) ---
    df["room_id"] = assign_rooms_balanced_per_hour(df, hour_col="hour")
    df["square_footage"] = df["room_id"].map(ROOM_SQUARE_FOOTAGE)

    # --- Appliance NaN -> explicit "None" label (logic itself untouched) ---
    df["Appliance"] = df["Appliance"].fillna("None")

    # --- Rename retained columns to final snake_case schema ---
    df = df.rename(columns={
        "Appliance": "appliance",
        "Appliance Status": "appliance_status",
        "Electricity Tariff": "electricity_tariff",
    })

    # --- Drop fields removed by the approved design ---
    drop_cols = [
        "source_day_of_week",  # statistically indistinguishable from random
        "Room",                # replaced by regenerated room_id
        "hvac_on",              # redundant duplicate of hvac_usage
        "lighting_on",          # redundant duplicate of lighting_usage
        "occupied_status",      # redundant duplicate of occupancy == 0
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- Final column order ---
    final_cols = [
        "timestamp", "day_of_week", "hour", "is_weekend", "holiday",
        "room_id", "square_footage",
        "temperature", "humidity", "occupancy",
        "hvac_usage", "lighting_usage",
        "appliance", "appliance_status",
        "renewable_energy", "electricity_tariff",
        "energy_consumption",
    ]
    df = df[final_cols]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading original dataset from: {ORIGINAL_PATH}")
    df_raw = load_original(ORIGINAL_PATH)
    print(f"Original dataset SHA-256: {file_sha256(ORIGINAL_PATH)}")
    print(f"Original shape: {df_raw.shape}")

    room_reference = build_room_reference_table()
    room_reference.to_csv(ROOM_REFERENCE_PATH, index=False)
    print(f"Room reference table written to: {ROOM_REFERENCE_PATH}")

    df_processed = transform(df_raw)
    df_processed.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"Processed dataset written to: {PROCESSED_DATASET_PATH}")
    print(f"Processed shape: {df_processed.shape}")


if __name__ == "__main__":
    main()
