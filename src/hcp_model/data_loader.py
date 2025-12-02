from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "booking_id",
    "user_id",
    "hotel_id",
    "booking_datetime",
    "checkin_date",
    "checkout_date",
    "booking_channel",
    "device_type",
    "rate_plan",
    "payment_status",
    "booking_amount",
    "currency",
    "num_guests",
    "num_rooms",
    "user_country",
    "status",
    "is_cancelled",
    "cancellation_datetime",
    "no_show_flag",
]

DATETIME_COLUMNS = [
    "booking_datetime",
    "cancellation_datetime",
]

DATE_COLUMNS = [
    "checkin_date",
    "checkout_date",
]

INT_COLUMNS = [
    "num_guests",
    "num_rooms",
    "is_cancelled",
    "no_show_flag",
]

FLOAT_COLUMNS = [
    "booking_amount",
]


def _ensure_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")


def load_bookings_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a bookings CSV and apply basic schema normalization.

    - Validates presence of REQUIRED_COLUMNS
    - Parses datetime and date columns
    - Casts numeric columns
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bookings CSV not found at: {path}")

    df = pd.read_csv(path)

    # Validate columns
    _ensure_columns(df, REQUIRED_COLUMNS)

    # Parse datetime columns
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Parse date columns
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Cast numeric columns
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
