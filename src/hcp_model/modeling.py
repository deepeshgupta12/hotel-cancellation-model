from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


LABEL_COL = "is_cancelled"


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features."""
    label_col = LABEL_COL

    # Columns we NEVER want to use as features (IDs / PII etc.)
    id_like_cols = [
        "booking_id",
        "user_id",
        "hotel_id",
        "Booking_ID",
        "index",
        "name",
        "email",
        "phone-number",
        "credit_card",
    ]

    # Direct leakage or post-outcome columns
    leakage_cols = [
        # explicit label variants
        "is_canceled",
        "is_canceled_upd",
        # status / final outcome fields
        "booking status",
        "reservation_status",
        "reservation_status_date",
        "reservation_status_upd",
        "reservation_status_date_upd",
    ]

    # Raw date strings that would be one-hot encoded with huge cardinality
    raw_date_cols = [
        "date of reservation",
    ]

    exclude_cols = id_like_cols + leakage_cols + raw_date_cols

    # Candidate feature columns = everything except label + excluded cols
    feature_cols = [
        c
        for c in feature_df.columns
        if c not in exclude_cols + [label_col]
    ]

    # Simple heuristic: numeric vs categorical
    numeric_features = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(feature_df[c])
    ]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_models(
    df: pd.DataFrame,
    config: TrainConfig | None = None,
    rf_params: Dict[str, object] | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline]]:
    """
    Train a couple of baseline models and return metrics + fitted pipelines.

    Args:
        df: processed dataframe (includes LABEL_COL).
        config: train/test split parameters.
        rf_params: hyperparameters for RandomForestClassifier (dict).

    Returns:
        metrics: dict[model_name -> metric_name -> value]
        models: dict[model_name -> fitted sklearn Pipeline]
    """
    if config is None:
        config = TrainConfig()

    if rf_params is None:
        rf_params = {
            "n_estimators": 200,
            "random_state": config.random_state,
        }
    else:
        # Ensure random_state is always set for reproducibility
        rf_params = dict(rf_params)
        rf_params.setdefault("random_state", config.random_state)

    # Drop any rows without a label
    df = df[df[LABEL_COL].notna()].copy()

    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])

    preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    models: Dict[str, Pipeline] = {}

    # Logistic Regression baseline
    log_reg_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    models["log_reg"] = log_reg_clf

    # Random Forest baseline
    rf_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(**rf_params)),
        ]
    )
    models["random_forest"] = rf_clf

    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # For ROC AUC, we need probabilities for the positive class
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            # Fallback in weird edge cases; not expected for these models
            y_proba = np.zeros_like(y_pred, dtype=float)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = float("nan")

        metrics[name] = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        print(f"\n=== {name} ===")
        print("Accuracy:", acc)
        print("F1 score:", f1)
        print("ROC AUC:", roc_auc)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    return metrics, models


def select_best_model(
    metrics: Dict[str, Dict[str, float]],
    models: Dict[str, Pipeline],
    metric_name: str = "roc_auc",
) -> Tuple[str, Pipeline]:
    """Select the best model based on the chosen metric."""
    # Fallback: if all ROC AUC are NaN, use accuracy
    values = [metrics[m].get(metric_name) for m in metrics]
    if all(np.isnan(v) for v in values if v is not None):
        metric_name = "accuracy"

    best_name = None
    best_value = -np.inf

    for name, m in metrics.items():
        value = m.get(metric_name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if value > best_value:
            best_value = value
            best_name = name

    if best_name is None:
        # As a last resort, just pick one deterministically
        best_name = sorted(models.keys())[0]

    return best_name, models[best_name]


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to: {path}")