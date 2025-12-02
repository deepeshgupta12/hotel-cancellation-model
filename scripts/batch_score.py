from pathlib import Path
import sys

# --- ensure src/ is on sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, classification_report  # noqa: E402

from hcp_model.modeling import LABEL_COL  # noqa: E402


MODEL_PATH = ROOT_DIR / "models" / "baseline_model.joblib"
INPUT_PATH = ROOT_DIR / "data" / "processed" / "bookings_processed.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "bookings_scored.csv"


def _risk_bucket(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.7:
        return "medium"
    return "high"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Loading input data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    if LABEL_COL in df.columns:
        print(f"Found label column '{LABEL_COL}' in input data.")
    else:
        print(f"Label column '{LABEL_COL}' not found. Proceeding without evaluation.")

    # Features = all columns except label (if present)
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols]

    print(f"Scoring {len(df)} rows...")
    proba = model.predict_proba(X)[:, 1]
    pred_labels = (proba >= 0.5).astype(int)
    risk_buckets = [_risk_bucket(p) for p in proba]

    df_out = df.copy()
    df_out["pred_cancellation_proba"] = proba
    df_out["pred_label"] = pred_labels
    df_out["risk_bucket"] = risk_buckets

    # If we have true labels, print evaluation metrics as a sanity check
    if LABEL_COL in df.columns:
        y_true = df[LABEL_COL].astype(int)
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = float("nan")

        print("\nEvaluation on input data (using existing labels):")
        print(f"ROC AUC: {auc}")
        print("\nClassification report:")
        print(classification_report(y_true, pred_labels, zero_division=0))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved scored data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
