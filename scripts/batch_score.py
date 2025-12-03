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

from hcp_model.risk import bucket_from_proba, RiskConfig  # noqa: E402
from hcp_model.config import load_config  # noqa: E402
from hcp_model.logging_utils import get_logger  # noqa: E402
from hcp_model.modeling import LABEL_COL  # noqa: E402

logger = get_logger("batch_score")


def _risk_bucket(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.7:
        return "medium"
    return "high"


def main() -> None:
    cfg = load_config(ROOT_DIR / "config" / "config.yaml")
    risk_cfg = RiskConfig.from_dict(cfg.risk_thresholds)

    model_path = ROOT_DIR / cfg.paths.model_dir / cfg.paths.model_file
    input_path = ROOT_DIR / cfg.paths.processed_data
    output_path = ROOT_DIR / cfg.paths.scored_data

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading input data from: {input_path}")
    df = pd.read_csv(input_path)

    if LABEL_COL in df.columns:
        logger.info(f"Found label column '{LABEL_COL}' in input data.")
    else:
        logger.info(f"Label column '{LABEL_COL}' not found. Proceeding without evaluation.")

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols]

    logger.info(f"Scoring {len(df)} rows...")
    proba = model.predict_proba(X)[:, 1]
    pred_labels = (proba >= 0.5).astype(int)
    risk_buckets = [_risk_bucket(p) for p in proba]
    risk_cfg = RiskConfig()


    df_out = df.copy()
    df_out["pred_cancellation_proba"] = proba
    df_out["pred_label"] = pred_labels
    df_out["risk_bucket"] = risk_buckets
    df["risk_bucket"] = [
                            bucket_from_proba(p, cfg=risk_cfg) for p in proba
                        ]

    if LABEL_COL in df.columns:
        y_true = df[LABEL_COL].astype(int)
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = float("nan")

        logger.info("Evaluation on input data (using existing labels):")
        logger.info(f"ROC AUC: {auc}")
        logger.info(
            "Classification report:\n" + classification_report(y_true, pred_labels, zero_division=0)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"Saved scored data to: {output_path}")


if __name__ == "__main__":
    main()