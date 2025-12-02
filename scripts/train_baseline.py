from pathlib import Path
import sys

# --- ensure src/ is on sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from hcp_model.modeling import (  # noqa: E402
    LABEL_COL,
    TrainConfig,
    save_model,
    select_best_model,
    train_models,
)


def main() -> None:
    processed_path = ROOT_DIR / "data" / "processed" / "bookings_processed.csv"
    models_dir = ROOT_DIR / "models"
    model_path = models_dir / "baseline_model.joblib"

    print(f"Loading processed data from: {processed_path}")
    df = pd.read_csv(processed_path)

    print("Rows in processed dataset:", len(df))
    print("Columns:", list(df.columns))

    # Basic sanity check on label
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")

    print("\nLabel distribution:")
    print(df[LABEL_COL].value_counts(dropna=False))

    config = TrainConfig(test_size=0.3, random_state=42)

    print("\nTraining baseline models...")
    metrics, models = train_models(df, config=config)

    print("\nAll model metrics:")
    for name, m in metrics.items():
        print(f"- {name}: {m}")

    best_name, best_model = select_best_model(metrics, models, metric_name="roc_auc")
    print(f"\nSelected best model: {best_name}")

    save_model(best_model, model_path)


if __name__ == "__main__":
    main()
