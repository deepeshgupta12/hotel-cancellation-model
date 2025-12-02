from pathlib import Path
import sys

# --- ensure src/ is on sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hcp_model.data_loader import load_bookings_csv  # noqa: E402


def main() -> None:
    data_path = ROOT_DIR / "data" / "raw" / "bookings_sample.csv"
    df = load_bookings_csv(data_path)

    print("Loaded rows:", len(df))
    print("\nColumns:", list(df.columns))

    print("\nCancellation label distribution (counts):")
    print(df["is_cancelled"].value_counts(dropna=False))

    print("\nCancellation label distribution (ratio):")
    print(df["is_cancelled"].value_counts(normalize=True, dropna=False))


if __name__ == "__main__":
    main()