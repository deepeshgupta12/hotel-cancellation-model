from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hcp_model.config import load_config  # noqa: E402
from hcp_model.logging_utils import get_logger  # noqa: E402
from hcp_model.modeling import (  # noqa: E402
    LABEL_COL,
    TrainConfig,
    build_preprocessor,
)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


logger = get_logger("tune_rf")


def main() -> None:
    cfg = load_config(ROOT_DIR / "config" / "config.yaml")
    processed_path = ROOT_DIR / cfg.paths.processed_data

    logger.info(f"Loading processed data from: {processed_path}")
    df = pd.read_csv(processed_path)

    # Drop rows without label
    df = df[df[LABEL_COL].notna()].copy()

    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])

    logger.info(f"Total rows with label: {len(df)}")

    # Sample a subset for tuning to keep it manageable
    max_rows = 50000
    if len(df) > max_rows:
        df_sample = df.sample(n=max_rows, random_state=cfg.training.random_state)
        y = df_sample[LABEL_COL].astype(int)
        X = df_sample.drop(columns=[LABEL_COL])
        logger.info(f"Using a sample of {max_rows} rows for tuning")
    else:
        logger.info("Using full dataset for tuning")

    preprocessor = build_preprocessor(df)

    rf = RandomForestClassifier(
        n_estimators=cfg.model.rf.n_estimators,
        max_depth=cfg.model.rf.max_depth,
        min_samples_split=cfg.model.rf.min_samples_split,
        min_samples_leaf=cfg.model.rf.min_samples_leaf,
        random_state=cfg.training.random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )

    # Hyperparameter search space
    param_distributions = {
        "model__n_estimators": [100, 200, 400, 600],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=2,
        random_state=cfg.training.random_state,
    )

    logger.info("Starting RandomizedSearchCV for RandomForest...")
    search.fit(X, y)

    logger.info(f"Best ROC AUC: {search.best_score_}")
    logger.info(f"Best params: {search.best_params_}")

    print("\nBest ROC AUC:", search.best_score_)
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
