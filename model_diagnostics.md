
# Model Diagnostics & Monitoring

This document summarises the current behaviour of the hotel cancellation model
trained on the merged master dataset (~155k bookings across three sources).

It complements the main `Readme.md` (which focuses on problem statement and
architecture) with **evidence that the model is reliable and business-ready**.

---

## 1. Top feature importances

![Top feature importances](../reports/diagnostics/feature_importances.png)

The current best model is a `RandomForestClassifier` trained on engineered
features (including enrichment like historic cancellation rates and ADR per
guest).

Key observations:

- **Lead time** (days between booking and check-in) is the single most
  important driver.
- **ADR per guest** and **average price** follow next – higher per-guest spend
  behaves differently from low-spend, high-churn bookings.
- **Deposit / refundability** (`deposit_type_upd_*`, `cancellation_rate_by_deposit`)
  is a strong signal of cancellation behaviour.
- Enrichment fields like **total bookings / cancellations**, **previous
  cancellations**, and **special requests** are all used by the model, which
  validates the enrichment pipeline.

This matches domain intuition: *when* the user books, *how much* they pay, and
*how flexible* their booking is are all key to predicting cancellations.

---

## 2. Risk bucket distribution

![Risk bucket distribution](../reports/diagnostics/risk_bucket_distribution.png)

We currently use three risk buckets:

- `low` – low likelihood of cancellation
- `medium` – moderate risk
- `high` – high likelihood of cancellation

The distribution shows:

- Most bookings fall into the **low** risk bucket.
- There is still a sizeable **high-risk** segment and a smaller **medium-risk**
  segment.

From an ops perspective this means:

- **Front office / call-centre can focus on high-risk bookings first**, for
  confirmation calls, payment follow-ups, or re-marketing.
- Low-risk bookings can be handled in a lighter-touch way, saving human effort.

---

## 3. Slice diagnostics (actual vs predicted)

We also validate the model on key business segments to ensure that predicted
probabilities match real-world cancellation behaviour.

### 3.1 Country slices

![Cancellation vs predicted prob by country](../reports/diagnostics/slice_country_cancel_vs_pred.png)

For the top countries by volume:

- The **orange bar (avg predicted probability)** tracks closely with the
  **blue bar (actual cancellation rate)**.
- This suggests **good calibration** across geographies rather than the model
  being biased to one region.

### 3.2 Customer type

![Cancellation vs predicted prob by customer_type](../reports/diagnostics/slice_customer_type_cancel_vs_pred.png)

- Segment-wise behaviour (Transient, Contract, Transient-Party, Group) is
  captured cleanly.
- The model’s probabilities closely match realised cancellation rates, which is
  important for B2B discussions with revenue and sales.

### 3.3 Deposit type

![Cancellation vs predicted prob by deposit_type](../reports/diagnostics/slice_deposit_type_cancel_vs_pred.png)

- **Non-refundable** bookings are correctly identified as much higher risk if
  they do cancel, and the predicted probability is very close to the actual
  rate.
- **No-deposit** bookings show a lower but well-calibrated rate.

This validates that the model respects core commercial rules.

### 3.4 Distribution channel

![Cancellation vs predicted prob by distribution_channel](../reports/diagnostics/slice_distribution_channel_cancel_vs_pred.png)

- Channels like **TA/TO**, **Corporate**, and **Direct** have different base
  cancellation rates.
- The model’s predicted probabilities follow those patterns, which is
  important if you want channel-specific policies (e.g. overbooking strategy).

### 3.5 Hotel type (City vs Resort)

![Cancellation vs predicted prob by hotel](../reports/diagnostics/slice_hotel_cancel_vs_pred.png)

- City hotels generally see higher cancellation rates than resort properties.
- The model captures that difference, which can later drive **segment-wise
  risk thresholds**.

### 3.6 Market segment

![Cancellation vs predicted prob by market_segment](../reports/diagnostics/slice_market_segment_cancel_vs_pred.png)

- **Groups** and **Online TA** segments are riskier than **Direct** or
  **Complementary**.
- Again, predicted vs actual lines up tightly.

### 3.7 Source dataset

![Cancellation vs predicted prob by source_dataset](../reports/diagnostics/slice_source_dataset_cancel_vs_pred.png)

- Both main datasets (`hotel_booking_with_enrichment`, `booking_csv`) show
  comparable calibration – the model behaves consistently in spite of them
  being merged and enriched.

---

## 4. How to regenerate these diagnostics

From the project root:

```bash
# 1. Score the full master dataset (writes master_bookings_scored.csv)
python scripts/batch_score.py

# 2. Generate feature importances, bucket distribution and slice plots
#    (writes PNGs under reports/diagnostics/)
python scripts/diagnostics_feature_and_slices.py
```

You can run this periodically (e.g. weekly / monthly) when new data is added to:

- Check for **drift** (feature importances & slice plots changing drastically).
- Spot **calibration issues** (orange vs blue bars diverging).
- Decide whether a **retrain** is needed.

---

## 5. Next monitoring ideas (backlog)

- Simple **time-series CSV** logging of AUC / F1 by month to visualise model
  performance over time.
- Adding **per-bucket cancellation rates** to ensure `high` really stays high.
- Automatic **alerting** if any segment’s predicted vs actual gap exceeds a
  threshold.
