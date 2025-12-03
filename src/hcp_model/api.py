from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hcp_model.config import load_config
from hcp_model.risk import RiskConfig, bucket_from_proba
from hcp_model.predict_logger import PredictionLogger

# -----------------------------------------------------------------------------
# Config, logging, model paths
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

CONFIG = load_config()
RISK_CFG = RiskConfig.from_dict(CONFIG.risk_thresholds)
PRED_LOGGER = PredictionLogger()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"


class PredictionResponse(BaseModel):
    """
    Standard response for a single-booking prediction.
    """
    booking_id: str | None = None
    cancellation_probability: float
    predicted_label: int
    risk_bucket: Literal["low", "medium", "high"]


def load_model():
    """
    Load the trained sklearn Pipeline from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    version="0.2.0",
    description=(
        "Predicts cancellation probability for hotel bookings using the "
        "trained Random Forest model over the master_bookings dataset."
    ),
)

# Load model once at startup
MODEL = load_model()


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Feature alignment helper
# -----------------------------------------------------------------------------

def _prepare_features_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Take an arbitrary booking JSON (keys should ideally match training columns)
    and align it to what the trained sklearn Pipeline expects.

    - Ensures all expected columns exist.
    - Fills missing numeric columns with 0.0.
    - Fills missing categorical columns with 'Unknown'.
    - Drops any extra columns not used by the model.

    This is aligned with how the model was trained on `master_bookings.csv`.
    """
    # 1) One-row DataFrame from the incoming JSON
    df = pd.DataFrame([payload])

    # 2) Try to introspect the preprocessing step from the sklearn Pipeline
    preprocess = None
    if hasattr(MODEL, "named_steps"):
        preprocess = MODEL.named_steps.get("preprocess")

    if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
        expected_cols = list(preprocess.feature_names_in_)
    else:
        # Fallback: if we can't introspect, just use whatever comes in.
        # This should still work if the pipeline itself handles column selection.
        logger.warning(
            "Could not introspect preprocess.feature_names_in_; "
            "passing raw payload columns directly to the model."
        )
        return df

    # 3) Figure out which columns the preprocessor treated as numeric vs categorical
    numeric_cols = set()
    categorical_cols = set()

    if hasattr(preprocess, "transformers_"):
        for name, _, cols in preprocess.transformers_:
            # This relies on the naming used when building the ColumnTransformer
            if name.startswith("num"):
                numeric_cols.update(cols)
            elif name.startswith("cat"):
                categorical_cols.update(cols)

    # 4) Add missing columns with sensible defaults
    for col in expected_cols:
        if col not in df.columns:
            if col in numeric_cols:
                df[col] = 0.0
            elif col in categorical_cols:
                df[col] = "Unknown"
            else:
                # Default to numeric-ish 0.0 if type is unknown;
                # this is safer than None for most sklearn transformers.
                df[col] = 0.0

    # 5) Keep only the columns the model was actually trained on, in the right order
    df = df[expected_cols]

    return df


# -----------------------------------------------------------------------------
# Inference endpoint
# -----------------------------------------------------------------------------

@app.post("/predict_raw", response_model=PredictionResponse)
def predict_raw(payload: Dict[str, Any]) -> PredictionResponse:
    """
    Score a single booking using the trained model.

    Expects a JSON object where keys match (as closely as possible)
    the columns used in training, i.e. the headers you see in
    `data/processed/master_bookings.csv`.

    Example shape (illustrative):

    {
      "booking_id": "BKG-123",
      "hotel": "Resort Hotel",
      "lead_time": 342,
      "arrival_date_year": 2015,
      "arrival_date_month": "July",
      "arrival_date_week_number": 27,
      "arrival_date_day_of_month": 1,
      "stays_in_weekend_nights": 0,
      "stays_in_week_nights": 0,
      "adults": 2,
      "children": 0,
      "babies": 0,
      "meal": "BB",
      "country": "PRT",
      "market_segment": "Direct",
      "distribution_channel": "Direct",
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "reserved_room_type": "C",
      "assigned_room_type": "C",
      "booking_changes": 3,
      "deposit_type": "No Deposit",
      "agent": null,
      "company": null,
      "days_in_waiting_list": 0,
      "customer_type": "Transient",
      "adr": 0.0,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 0,
      "total_nights": 0,
      "total_guests": 2,
      "adr_per_guest": 0.0,
      "is_short_lead": 0,
      "is_long_stay": 0,
      "is_family": 0
    }
    """
    # 1) Align incoming JSON to the expected feature space
    df_features = _prepare_features_from_payload(payload)

    if df_features.empty:
        raise HTTPException(
            status_code=400,
            detail="Input payload produced an empty feature frame. "
                   "Check that keys roughly match training columns.",
        )

    # 2) Predict probability of cancellation
    proba = float(MODEL.predict_proba(df_features)[0, 1])

    # Optional: configurable decision threshold; default 0.5
    threshold = 0.5
    model_cfg = getattr(CONFIG, "model", None)
    if model_cfg is not None and getattr(model_cfg, "decision_threshold", None) is not None:
        threshold = float(model_cfg.decision_threshold)

    pred_label = int(proba >= threshold)

    # 3) Map to risk bucket using shared risk thresholds
    risk_bucket = bucket_from_proba(proba, cfg=RISK_CFG)

    # 4) Best-effort prediction logging for monitoring / retraining
    try:
        model_version = getattr(model_cfg, "version", None) if model_cfg is not None else None

        PRED_LOGGER.log_prediction(
            booking_id=payload.get("booking_id"),
            cancellation_probability=proba,
            predicted_label=pred_label,
            risk_bucket=risk_bucket,
            model_version=model_version,
            source="api",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to log prediction: {e}")

    # 5) Return structured response
    return PredictionResponse(
        booking_id=payload.get("booking_id"),
        cancellation_probability=round(proba, 3),
        predicted_label=pred_label,
        risk_bucket=risk_bucket,
    )