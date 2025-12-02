from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hcp_model.config import load_config  # noqa: E402
from hcp_model.logging_utils import get_logger  # noqa: E402


logger = get_logger("build_master_dataset")


def add_kaggle_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for the Kaggle-style hotel_booking data."""
    df = df.copy()

    # total nights
    df["total_nights"] = (
        df.get("stays_in_weekend_nights", 0).fillna(0)
        + df.get("stays_in_week_nights", 0).fillna(0)
    )

    # total guests
    df["total_guests"] = (
        df.get("adults", 0).fillna(0)
        + df.get("children", 0).fillna(0)
        + df.get("babies", 0).fillna(0)
    )

    # ADR per guest
    df["adr_per_guest"] = df.get("adr", 0) / df["total_guests"].replace(0, 1)

    # short lead time (<= 2 days)
    df["is_short_lead"] = (df.get("lead_time", 0) <= 2).astype("Int64")

    # long stay (>= 7 nights)
    df["is_long_stay"] = (df["total_nights"] >= 7).astype("Int64")

    # family flag: at least 1 child or 3+ guests
    df["is_family"] = (
        (df.get("children", 0).fillna(0) > 0)
        | (df["total_guests"] >= 3)
    ).astype("Int64")

    return df


def add_booking_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for the custom booking.csv data."""
    df = df.copy()

    # total nights
    df["total_nights"] = (
        df.get("number of weekend nights", 0).fillna(0)
        + df.get("number of week nights", 0).fillna(0)
    )

    # total guests
    df["total_guests"] = (
        df.get("number of adults", 0).fillna(0)
        + df.get("number of children", 0).fillna(0)
    )

    # ADR per guest
    df["adr_per_guest"] = df.get("average price", 0) / df["total_guests"].replace(0, 1)

    # short lead time (<= 2 days)
    df["is_short_lead"] = (df.get("lead time", 0) <= 2).astype("Int64")

    # long stay (>= 7 nights)
    df["is_long_stay"] = (df["total_nights"] >= 7).astype("Int64")

    # family flag: at least 1 child or 3+ guests
    df["is_family"] = (
        (df.get("number of children", 0).fillna(0) > 0)
        | (df["total_guests"] >= 3)
    ).astype("Int64")

    return df


def main() -> None:
    cfg = load_config(ROOT_DIR / "config" / "config.yaml")

    raw_dir = ROOT_DIR / "data" / "raw"
    booking_path = raw_dir / "booking.csv"
    hotel_booking_path = raw_dir / "hotel_booking.csv"
    updated_hotel_path = raw_dir / "updated_hotel_data.csv"

    logger.info(f"Loading booking CSV from: {booking_path}")
    booking = pd.read_csv(booking_path)
    logger.info(f"booking.csv rows: {len(booking)}")

    logger.info(f"Loading hotel_booking CSV from: {hotel_booking_path}")
    hotel_booking = pd.read_csv(hotel_booking_path)
    logger.info(f"hotel_booking.csv rows: {len(hotel_booking)}")

    logger.info(f"Loading updated_hotel_data CSV from: {updated_hotel_path}")
    updated = pd.read_csv(updated_hotel_path)
    logger.info(f"updated_hotel_data.csv rows: {len(updated)}")

    # 1) Collapse updated_hotel_data to one row per original booking index
    updated_uni = (
        updated
        .sort_values("index")
        .drop_duplicates("index", keep="first")
    )
    logger.info(
        "updated_hotel_data unique by index: "
        f"{len(updated_uni)} rows (should match hotel_booking)"
    )

    # 2) Add an index column to hotel_booking and join enrichment
    hb = hotel_booking.reset_index().rename(columns={"index": "index"})
    hb_full = hb.merge(
        updated_uni,
        on="index",
        how="left",
        suffixes=("", "_upd")
    )
    logger.info(f"hotel_booking + enrichment shape: {hb_full.shape}")

    # 3) Create unified label "is_cancelled" (double-l spelling) everywhere
    if "is_canceled" in hb_full.columns:
        hb_full["is_cancelled"] = hb_full["is_canceled"]

    if "booking status" in booking.columns:
        booking["is_cancelled"] = booking["booking status"].map(
            {
                "Canceled": 1,
                "Not_Canceled": 0,
            }
        )

    # 4) Add source marker for debugging later
    hb_full["source_dataset"] = "hotel_booking_with_enrichment"
    booking["source_dataset"] = "booking_csv"

    # 5) Add derived features per source
    hb_full = add_kaggle_derived_features(hb_full)
    booking = add_booking_derived_features(booking)

    # 6) Concatenate both datasets row-wise, preserving all columns
    master = pd.concat(
        [hb_full, booking],
        axis=0,
        join="outer",
        ignore_index=True,
    )

    # 7) Keep only rows where we actually have a label
    if "is_cancelled" not in master.columns:
        raise ValueError("Master dataset does not contain 'is_cancelled' column.")

    master = master[master["is_cancelled"].notna()].copy()

    logger.info(f"Master dataset shape: {master.shape}")
    logger.info("Label distribution (is_cancelled):")
    logger.info(f"\n{master['is_cancelled'].value_counts(dropna=False)}")

    # 8) Save to the configured processed path
    output_path = ROOT_DIR / cfg.paths.processed_data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)

    logger.info(f"Saved master dataset to: {output_path}")


if __name__ == "__main__":
    main()
