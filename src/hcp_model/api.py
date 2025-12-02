from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Path to the saved model (baseline_model.joblib)
# src/hcp_model/api.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"


class BookingFeatures(BaseModel):
    """
    Processed feature schema expected by the model.
    Matches columns in data/processed/bookings_processed.csv, except the label.
    """
    booking_id: str
    user_id: str
    hotel_id: str

    booking_channel: str
    device_type: str
    rate_plan: str
    payment_status: str

    booking_amount: float
    currency: str

    num_guests: int
    num_rooms: int
    user_country: str
    status: str

    no_show_flag: int = Field(..., description="0 or 1")

    lead_time_days: int
    length_of_stay_nights: int
    booking_dow: int
    booking_hour: int
    checkin_dow: int
    is_weekend_checkin: int


class PredictionResponse(BaseModel):
    booking_id: str
    cancellation_probability: float
    predicted_label: int
    risk_bucket: Literal["low", "medium", "high"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    version="0.1.0",
    description="Predicts cancellation probability for hotel bookings using a trained ML model.",
)

# Load model at startup
MODEL = load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _risk_bucket(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.7:
        return "medium"
    return "high"


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BookingFeatures) -> PredictionResponse:
    # Convert input to DataFrame with a single row
    df = pd.DataFrame([features.dict()])

    # Predict probability of cancellation (class 1)
    proba = float(MODEL.predict_proba(df)[0, 1])
    label = int(proba >= 0.5)
    bucket = _risk_bucket(proba)

    return PredictionResponse(
        booking_id=features.booking_id,
        cancellation_probability=proba,
        predicted_label=label,
        risk_bucket=bucket,
    )
