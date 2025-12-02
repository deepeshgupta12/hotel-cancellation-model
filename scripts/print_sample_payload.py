from pathlib import Path
import json

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_path = root / "data" / "processed" / "bookings_processed.csv"

    df = pd.read_csv(processed_path)

    # Take first row as example
    row = df.iloc[0].drop(labels=["is_cancelled"])
    payload = row.to_dict()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
