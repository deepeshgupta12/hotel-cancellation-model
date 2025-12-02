from pathlib import Path
import sys

# --- ensure src/ is on sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hcp_model.data_loader import load_bookings_csv  # noqa: E402
from hcp_model.features import make_basic_features, LABEL_COL  # noqa: E402


def main() -> None:
    raw_path = ROOT_DIR / "data" / "raw" / "bookings_sample.csv"
    processed_path = ROOT_DIR / "data" / "processed" / "bookings_processed.csv"

    print(f"Loading raw data from: {raw_path}")
    df_raw = load_bookings_csv(raw_path)

    print(f"Raw rows: {len(df_raw)}")

    df_proc = make_basic_features(df_raw)

    print(f"Processed rows (after filtering): {len(df_proc)}")
    print("\nProcessed columns:")
    print(list(df_proc.columns))

    print("\nLabel distribution in processed data:")
    print(df_proc[LABEL_COL].value_counts(dropna=False))
    print("\nLabel distribution ratio in processed data:")
    print(df_proc[LABEL_COL].value_counts(normalize=True, dropna=False))

    # Ensure output directory exists
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(processed_path, index=False)
    print(f"\nSaved processed dataset to: {processed_path}")


if __name__ == "__main__":
    main()
