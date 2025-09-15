import math
import pandas as pd
from pathlib import Path

TICKER = "AAPL"
PROCESSED_DIR = Path("data/processed") / TICKER
FEATURES_CSV = PROCESSED_DIR / "features.csv"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV   = PROCESSED_DIR / "val.csv"
TEST_CSV  = PROCESSED_DIR / "test.csv"

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

def load_features(csv_path : Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory = False, encoding_errors = "ignore")
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")

    for col in ["close", "close_daily_return", "close_rolling_return_mean20", "close_rolling_return_std20", "close_zscore20"]:
        df[col] = pd.to_numeric(df[col], errors = "coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop = True)
    return df

def main():
    PROCESSED_DIR.mkdir(parents = True, exist_ok = True)
    features_df = load_features(FEATURES_CSV)

    n = len(features_df)
    if n < 3:
        raise ValueError("Not enough rows to split (need >= 3).")

    train_end = int(math.floor(TRAIN_FRAC * n))
    val_end = int(math.floor((TRAIN_FRAC + VAL_FRAC) * n))

    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_df = features_df.iloc[:train_end].copy()
    val_df = features_df.iloc[train_end:val_end].copy()
    test_df = features_df.iloc[val_end:].copy()

    train_df.to_csv(TRAIN_CSV, index = False)
    val_df.to_csv(VAL_CSV, index = False)
    test_df.to_csv(TEST_CSV, index = False)

    print("Saved:", TRAIN_CSV, len(train_df))
    print("Saved:", VAL_CSV, len(val_df))
    print("Saved:", TEST_CSV, len(test_df))

if __name__ == "__main__":
    main()
