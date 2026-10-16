"""
modules/data_manager.py
-------------------------
Purpose : Single, reusable loader for the Phase 0 processed dataset and room
          reference table. Every later module/phase should load data through
          this file rather than calling pandas.read_csv() independently, so
          the "do not silently modify the dataset" and "avoid the None/NaN
          CSV hazard" guarantees from Phase 0 are enforced in exactly one
          place.
Input   : Paths from config.py (data/processed/*.csv).
Output  : Pandas DataFrames, loaded read-only -- this module performs no
          cleaning, imputation, or row modification of any kind.
AI technique used : None (data access layer only).

Known hazard this module exists to prevent (see TRANSFORM_LOG.md section 5):
    pandas.read_csv()'s default na_values list includes the literal string
    "None". The processed dataset intentionally uses "None" as a real
    Appliance value (not missing data). Loading it with default arguments
    silently turns that back into NaN. Every read in this module uses
    keep_default_na=False, na_values=[""] to prevent this.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

import config


class DataLoadError(Exception):
    """Raised when the processed dataset or room reference table cannot be
    loaded or does not match the expected schema."""


# ---------------------------------------------------------------------------
# Expected schema (as produced by Phase 0's scripts/preprocess.py)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "timestamp", "day_of_week", "hour", "is_weekend", "holiday",
    "room_id", "square_footage", "temperature", "humidity", "occupancy",
    "hvac_usage", "lighting_usage", "appliance", "appliance_status",
    "renewable_energy", "electricity_tariff", "energy_consumption",
]

EXPECTED_ROOM_REFERENCE_COLUMNS = ["room_id", "floor", "square_footage_fixed", "capacity"]


def load_processed_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Phase 0 processed IntelliGrid dataset.

    This function performs NO cleaning or modification. It only:
      - reads the CSV with arguments that preserve the literal "None"
        Appliance value instead of silently converting it to NaN,
      - parses `timestamp` as a proper datetime column,
      - verifies the expected columns are present (raises DataLoadError if not).

    Input:
        path: optional override; defaults to config.PROCESSED_DATASET_PATH.
    Output:
        pandas.DataFrame, one row per original record, unmodified otherwise.
    Raises:
        DataLoadError if the file is missing, empty, or missing expected columns.
    """
    csv_path = path or config.PROCESSED_DATASET_PATH

    if not csv_path.exists():
        raise DataLoadError(f"Processed dataset not found at '{csv_path}'.")

    try:
        df = pd.read_csv(
            csv_path,
            keep_default_na=False,   # prevents "None" (Appliance) -> NaN
            na_values=[""],          # only a genuinely empty cell counts as missing
        )
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Processed dataset at '{csv_path}' is empty.") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to read processed dataset: {exc}") from exc

    if df.empty:
        raise DataLoadError("Processed dataset contains no rows.")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"Processed dataset is missing expected columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    return df


def load_room_reference(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Phase 0 room reference table (room_id, floor, square_footage_fixed, capacity).

    Input:
        path: optional override; defaults to config.ROOM_REFERENCE_PATH.
    Output:
        pandas.DataFrame with one row per simulated room.
    Raises:
        DataLoadError if the file is missing or missing expected columns.
    """
    csv_path = path or config.ROOM_REFERENCE_PATH

    if not csv_path.exists():
        raise DataLoadError(f"Room reference table not found at '{csv_path}'.")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to read room reference table: {exc}") from exc

    missing = [c for c in EXPECTED_ROOM_REFERENCE_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"Room reference table is missing expected columns: {missing}")

    return df


def validate_schema(df: pd.DataFrame, expected: List[str] = None) -> List[str]:
    """
    Report (without fixing) any expected columns missing from `df`.

    Output: list of missing column names (empty list if the schema is complete).
    """
    expected = expected or EXPECTED_COLUMNS
    return [c for c in expected if c not in df.columns]
