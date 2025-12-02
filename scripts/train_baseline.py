from pathlib import Path
import sys

# --- ensure src/ is on sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from hcp_model.config import load_config  # noqa: E402
from hcp_model.logging_utils import get_logger  # noqa: E402
from hcp_model.modeling import (  # noqa: E402
    LABEL_COL,
    TrainConfig,
    save_model,
    select_best_model,
    train_models,
)

logger = get_logger("train_baseline")


def main() -> None:
    cfg = load_config(ROOT_DIR / "config" / "config.yaml")

    processed_path = ROOT_DIR / cfg.paths.processed_data
    model_dir = ROOT_DIR / cfg.paths.model_dir
    model_path = model_dir / cfg.paths.model_file

    logger.info(f"Loading processed data from: {processed_path}")
    df = pd.read_csv(processed_path)

    logger.info(f"Rows in processed dataset: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")

    logger.info("Label distribution:")
    logger.info(f"\n{df[LABEL_COL].value_counts(dropna=False)}")

    train_conf = TrainConfig(
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
    )

    rf_params = {
        "n_estimators": cfg.model.rf.n_estimators,
        "max_depth": cfg.model.rf.max_depth,
        "min_samples_split": cfg.model.rf.min_samples_split,
        "min_samples_leaf": cfg.model.rf.min_samples_leaf,
    }

    logger.info("Training baseline models...")
    metrics, models = train_models(df, config=train_conf, rf_params=rf_params)

    logger.info("All model metrics:")
    for name, m in metrics.items():
        logger.info(f"- {name}: {m}")

    best_name, best_model = select_best_model(metrics, models, metric_name="roc_auc")
    logger.info(f"Selected best model: {best_name}")

    save_model(best_model, model_path)


if __name__ == "__main__":
    main()