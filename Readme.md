
# 🏨 Hotel Booking Cancellation Prediction Model

An end-to-end, production-minded ML system that predicts **hotel booking cancellation probability** and assigns a **risk bucket** to support smarter B2C and revenue decisions.  
Built to be practical, modular, and extensible — not just a notebook experiment.

**GitHub:** https://github.com/deepeshgupta12/hotel-cancellation-model

---

## ✨ What this product does

This project delivers a full ML + API + diagnostics pipeline:

✅ **Multi-source dataset unification** (3 CSVs → master dataset)  
✅ **Feature engineering** (including derived behavioural and pricing features)  
✅ **Baseline + improved model training**  
✅ **RandomForest hyperparameter tuning**  
✅ **Batch scoring with shared risk thresholds**  
✅ **FastAPI service** with a **raw booking → features → predict** wrapper  
✅ **Prediction logging** for observability  
✅ **Model diagnostics** with plots for interpretability + slice-level sanity checks

---

## 🧩 The problem (in plain terms)

Cancellations are a silent revenue killer in hotel B2C:

- Booking intent is high at checkout, but commitment is uncertain.
- Late cancellations damage occupancy planning.
- Overbooking decisions become riskier without reliable signals.
- Marketing, inventory, and support teams all work with partial visibility.

**The gap:** Hotels often react *after* cancellations happen.  
**The opportunity:** Predict risk *at booking time* and act early.

---

## 🚀 The solution

We predict cancellation risk and translate it into operationally usable buckets:

- **Low risk** ✅  
- **Medium risk** ⚠️  
- **High risk** 🔥  

This enables:

- smarter confirmation strategies  
- dynamic pricing guardrails  
- targeted re-engagement  
- channel-specific policy design  
- better overbooking confidence

---

## 🏗️ Project structure

```text
hotel-cancellation-model/
├── data/
│   ├── raw/
│   │   ├── booking.csv
│   │   ├── hotel_booking.csv
│   │   └── updated_hotel_data.csv
│   └── processed/
│       ├── master_bookings.csv
│       └── master_bookings_scored.csv
├── models/
│   └── baseline_model.joblib
├── reports/
│   └── diagnostics/
│       ├── feature_importances.png
│       ├── risk_bucket_distribution.png
│       ├── slice_country_cancel_vs_pred.png
│       ├── slice_customer_type_cancel_vs_pred.png
│       ├── slice_deposit_type_cancel_vs_pred.png
│       ├── slice_distribution_channel_cancel_vs_pred.png
│       ├── slice_hotel_cancel_vs_pred.png
│       ├── slice_market_segment_cancel_vs_pred.png
│       └── slice_source_dataset_cancel_vs_pred.png
├── scripts/
│   ├── build_master_dataset.py
│   ├── train_baseline.py
│   ├── tune_random_forest.py
│   ├── batch_score.py
│   └── diagnostics_feature_and_slices.py
├── src/
│   └── hcp_model/
│       ├── api.py
│       ├── config.py
│       ├── features.py
│       ├── risk.py
│       └── predict_logger.py
├── configs/
│   └── config.yaml
├── Readme.md
├── model_diagnostics.md
└── requirements.txt
```

---

## 🧠 Data sources & master dataset

We merged three datasets into a unified master file while preserving columns:

1. **booking.csv**  
2. **hotel_booking.csv**  
3. **updated_hotel_data.csv** (used as enrichment where it matched the base index)

Your Step 8 output confirms:

- **Master dataset shape:** **(155,675 rows, 105 columns)**  
- **Label distribution (`is_cancelled`):**  
  - 0 → **99,562**  
  - 1 → **56,113**

📌 Output file:  
`data/processed/master_bookings.csv`

---

## 🛠️ Tech stack

### Core ML
- Python 3.10+
- Pandas, NumPy
- scikit-learn
- joblib

### Backend/API
- FastAPI
- Uvicorn
- Pydantic

### Observability & config
- YAML-based configs
- Python `logging`
- Lightweight prediction logger

### Diagnostics
- Matplotlib

---

## 🧪 Features (current phase)

We moved beyond a toy baseline and added richer signals such as:

- `total_nights`  
- `total_guests`  
- `adr_per_guest`  
- `is_short_lead`  
- `is_long_stay`  
- `is_family`

These features were designed for interpretability and business relevance.

---

## 📈 Model performance (your latest verified runs)

### Train/validation metrics (baseline + improved features)

You observed stable, realistic performance after tightening the training logic:

- **RandomForest (selected best)**  
  - Accuracy around **0.88** on holdout  
  - ROC AUC around **0.95** range on validation runs  
  - Strong class-level balance vs early overfit-looking numbers

### Full-dataset evaluation (batch score on master)

After scoring:

- **ROC AUC:** **0.9833**  
- Strong classification report with high precision/recall across labels  
📌 Output file:  
`data/processed/master_bookings_scored.csv`

> Note: We intentionally improved the training pipeline to reduce leakage and avoid “too-perfect” early results. ✅

---

## 🔧 Hyperparameter tuning

Your tuning run produced:

**Best ROC AUC:** **0.9380**  
**Best params:**
```json
{
  "model__n_estimators": 200,
  "model__min_samples_split": 10,
  "model__min_samples_leaf": 1,
  "model__max_features": 0.5,
  "model__max_depth": 20,
  "model__bootstrap": false
}
```

We also tested `max_depth = null` to increase generalization and improve real-world stability.

---

## 🧯 Risk buckets

Risk bucketing is centralized and shared across:

- batch scoring  
- API responses  
- prediction logs  

Defined in config (example pattern):

```yaml
risk_thresholds:
  low: 0.30
  medium: 0.70
```

---

## 🌐 FastAPI service

### Start the API

```bash
# From project root
source .venv/bin/activate
PYTHONPATH=src uvicorn hcp_model.api:app --reload
```

### Endpoints

- `GET /health`
- `POST /predict_raw`

### ✅ `/predict_raw` supports raw booking JSON

**Example input:**

```json
{
  "booking_id": "BKG-NEW-001",
  "user_id": "U-NEW",
  "hotel_id": "H-01",
  "booking_datetime": "2025-01-01T09:15:00",
  "checkin_date": "2025-02-10",
  "checkout_date": "2025-02-12",
  "booking_channel": "web",
  "device_type": "desktop",
  "rate_plan": "refundable",
  "payment_status": "prepaid",
  "booking_amount": 12000.0,
  "currency": "INR",
  "num_guests": 2,
  "num_rooms": 1,
  "user_country": "IN",
  "status": "confirmed",
  "no_show_flag": 0
}
```

**Example output you validated:**

```json
{
  "booking_id": "BKG-NEW-001",
  "cancellation_probability": 0.281,
  "predicted_label": 0,
  "risk_bucket": "medium"
}
```

---

## 🧾 Prediction logging

The service logs predictions in a structured, lightweight manner for:

- debugging
- monitoring
- future retraining corpuses

This supports real-world ML operations without heavy infra.

---

## 🧪 Batch scoring

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/batch_score.py
```

This will:

- load `models/baseline_model.joblib`
- read `data/processed/master_bookings.csv`
- add:
  - `pred_cancellation_proba`
  - `pred_label`
  - `risk_bucket`
- write:
  - `data/processed/master_bookings_scored.csv`

---

## 🔍 Model diagnostics & interpretability

Generate all diagnostic outputs:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/diagnostics_feature_and_slices.py
```

This produces graphs like:

### 1. Feature importance 🧠  
`reports/diagnostics/feature_importances.png`

### 2. Risk bucket distribution 🚦  
`reports/diagnostics/risk_bucket_distribution.png`

### 3. Calibration sanity checks by slice 🎯  
- country  
- customer type  
- deposit type  
- distribution channel  
- hotel type  
- market segment  
- source dataset

These plots validate that **predicted probabilities track real cancellation rates** within critical business segments.

📌 Detailed narrative:  
See `model_diagnostics.md`

---

## 🧪 Local setup

### 1) Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏭 Reproduce the full pipeline

```bash
# Step A: Build master dataset
PYTHONPATH=src python scripts/build_master_dataset.py

# Step B: Train baseline + improved features
PYTHONPATH=src python scripts/train_baseline.py

# Step C: Tune RF (optional but recommended)
PYTHONPATH=src python scripts/tune_random_forest.py

# Step D: Batch score full master file
PYTHONPATH=src python scripts/batch_score.py

# Step E: Generate diagnostics
PYTHONPATH=src python scripts/diagnostics_feature_and_slices.py
```

---

## 🎯 Product use cases

### 1) Revenue management 📊
- Adjust overbooking strategy based on risk distribution.
- Set confidence thresholds for high-demand periods.

### 2) CRM & retention 💬
- Trigger proactive nudges for high-risk bookings.
- Offer limited-time upgrade or price-lock incentives.

### 3) Channel strategy 🧭
- Apply tighter policies on risk-heavy channels.
- Work with OTA partners using data-backed cancellation patterns.

### 4) Ops allocation 🧑‍💼
- Focus human confirmation effort on high risk.
- Reduce manual overhead for low-risk bookings.

---

## ⚠️ Things we consciously handled

✅ Reduced **“too perfect” metrics** by tightening training logic  
✅ Ensured **shared config** across scripts + API  
✅ Added **feature alignment safeguards** for inference  
✅ Strengthened slice-level trust with diagnostics

---

## 🧭 Next upgrades (Step 12+ roadmap)

### Model improvements
- Gradient boosting baselines (e.g., XGBoost/LightGBM)
- Probability calibration (Platt/Isotonic)
- Cost-sensitive learning aligned to revenue loss

### Data & splitting
- Strict time-aware train/val split based on reservation date
- Drift checks by month/season

### Productization
- Dockerize the service
- Add `/predict_batch` endpoint
- Structured model registry & versioning

### Business metrics
- Estimate expected revenue saved per bucket
- Compute lift from targeted interventions

---

## ✅ Status

This project is now a solid **product-grade ML foundation** with:

- real dataset scale  
- robust pipelines  
- explainability  
- operational risk outputs  
- and a live service layer

Ready for scaling into a more advanced MLOps setup. 🚀
