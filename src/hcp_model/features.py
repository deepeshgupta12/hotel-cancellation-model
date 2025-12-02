from __future__ import annotations

from typing import List

import pandas as pd


LABEL_COL = "is_cancelled"


def _compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Force these columns to proper datetime64[ns]
    df["booking_datetime"] = pd.to_datetime(df.get("booking_datetime"), errors="coerce")
    df["checkin_date"] = pd.to_datetime(df.get("checkin_date"), errors="coerce")
    df["checkout_date"] = pd.to_datetime(df.get("checkout_date"), errors="coerce")

    # Core time features
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

    return df


def make_basic_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Training-time feature builder.

    Takes the raw bookings dataframe (as loaded by data_loader.load_bookings_csv)
    and returns a processed dataframe with engineered features **including** the label.

    This keeps:
    - ID columns (booking_id, user_id, hotel_id)
    - Core categorical/numeric features
    - Time-based engineered features
    - Label column: is_cancelled
    """
    df = df_raw.copy()

    # Compute time-based features
    df = _compute_time_features(df)

    # Filter invalid rows (dates / durations)
    df = _filter_invalid_rows(df)

    # Drop rows with missing label
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL].notna()].copy()

    # Define columns to keep
    feature_cols: List[str] = [
        # IDs
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

    feature_cols = [c for c in feature_cols if c in df.columns]
    df_out = df[feature_cols].reset_index(drop=True)
    return df_out


def make_features_for_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Inference-time feature builder.

    Takes a raw bookings dataframe (without label) and returns feature columns
    matching what the trained model expects, **excluding** the label.
    """
    df = df_raw.copy()

    # Compute time-based features and filter invalid rows
    df = _compute_time_features(df)
    df = _filter_invalid_rows(df)

    # Columns expected by the model at inference time
    feature_cols: List[str] = [
        "booking_id",
        "user_id",
        "hotel_id",
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
        "lead_time_days",
        "length_of_stay_nights",
        "booking_dow",
        "booking_hour",
        "checkin_dow",
        "is_weekend_checkin",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    df_out = df[feature_cols].reset_index(drop=True)

    return df_out
