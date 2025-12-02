from __future__ import annotations

from typing import List

import pandas as pd


LABEL_COL = "is_cancelled"


def _compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Force these columns to proper datetime64[ns]
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"], errors="coerce")
    df["checkin_date"] = pd.to_datetime(df["checkin_date"], errors="coerce")
    df["checkout_date"] = pd.to_datetime(df["checkout_date"], errors="coerce")

    # Core time features (no .dt.date here)
    df["lead_time_days"] = (df["checkin_date"] - df["booking_datetime"]).dt.days
    df["length_of_stay_nights"] = (df["checkout_date"] - df["checkin_date"]).dt.days

    # Booking and check-in temporal patterns
    df["booking_dow"] = df["booking_datetime"].dt.dayofweek  # 0=Mon
    df["booking_hour"] = df["booking_datetime"].dt.hour
    df["checkin_dow"] = df["checkin_date"].dt.dayofweek
    df["is_weekend_checkin"] = (df["checkin_dow"] >= 5).astype("Int64")

    return df


def _filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows missing critical dates
    mask_dates = (
        df["booking_datetime"].notna()
        & df["checkin_date"].notna()
        & df["checkout_date"].notna()
    )
    df = df.loc[mask_dates].copy()

    # Drop obviously invalid values (negative lead time, non-positive length of stay)
    df = df[df["lead_time_days"] >= 0]
    df = df[df["length_of_stay_nights"] > 0]

    # Drop rows with missing label
    df = df[df[LABEL_COL].notna()].copy()

    return df


def make_basic_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw bookings dataframe (as loaded by data_loader.load_bookings_csv)
    and return a processed dataframe with engineered features, ready for modeling.

    This keeps:
    - ID columns for reference (booking_id, user_id, hotel_id)
    - Core categorical features
    - Time-based engineered features
    - Label column: is_cancelled
    """
    df = df_raw.copy()

    # Compute time-based features
    df = _compute_time_features(df)

    # Filter invalid rows
    df = _filter_invalid_rows(df)

    # Define columns to keep (for now we keep it fairly wide)
    feature_cols: List[str] = [
        # IDs (useful for debugging, but can be dropped at training time)
        "booking_id",
        "user_id",
        "hotel_id",
        # Original categorical/numeric features
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
        "no_show_flag",
        # Engineered time features
        "lead_time_days",
        "length_of_stay_nights",
        "booking_dow",
        "booking_hour",
        "checkin_dow",
        "is_weekend_checkin",
        # Label
        LABEL_COL,
    ]

    # Intersect with available columns to be robust
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_out = df[feature_cols].reset_index(drop=True)

    return df_out
