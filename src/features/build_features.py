import numpy as np
import pandas as pd
from pathlib import Path

TICKER = "AAPL"
RAW_DIR = Path("data/raw")
FEATURES_DIR = Path("data/processed") / TICKER
RAW_CSV_PATH = RAW_DIR / f"{TICKER}_1d.csv"
FEATURES_CSV_PATH = FEATURES_DIR / "features.csv"


ROLLING_WINDOW_DAYS = 20

def load_raw_and_clean_ohlcv_csv(csv_path : Path) -> pd.DataFrame:
     if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
     
     use_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
     raw_ohlcv_df = pd.read_csv(csv_path, usecols = use_columns, low_memory = False, encoding_errors = "ignore")

     raw_ohlcv_df['date'] = pd.to_datetime(raw_ohlcv_df['date'], errors = "coerce")
     for col in ['open', 'high', 'low', 'close', 'volume']:
         raw_ohlcv_df[col] = pd.to_numeric(raw_ohlcv_df[col], errors = "coerce")

     clean_ohlcv_df = (
         raw_ohlcv_df
         .dropna(subset = ['close', 'date'])
         .drop_duplicates(subset = ['date'], keep = 'last')
         .sort_values('date')
         .reset_index(drop = True)
     )
     return clean_ohlcv_df

def compute_feature_engineering_on_close(ohlcv_df : pd.DataFrame, rolling_window_days : int) -> pd.DataFrame:
    close = ohlcv_df['close']

    close_daily_return = close.pct_change()

    close_rolling_return_mean = close_daily_return.rolling(window = rolling_window_days).mean().shift(1)

    close_rolling_return_std = close_daily_return.rolling(window = rolling_window_days).std(ddof = 1).shift(1)

    close_zscore = (close_daily_return - close_rolling_return_mean) / close_rolling_return_std

    features_df = pd.DataFrame({
        "date" : ohlcv_df["date"],
        "close" : close,
        "close_daily_return" : close_daily_return,
        f"close_rolling_return_mean{rolling_window_days}" : close_rolling_return_mean,
        f"close_rolling_return_std{rolling_window_days}" : close_rolling_return_std,
        f"close_zscore{rolling_window_days}" : close_zscore
    }).dropna().reset_index(drop = True)
    
    std_col = f"close_rolling_return_std{rolling_window_days}"
    if features_df.empty:
        raise ValueError("Insufficient data")
    if (features_df[std_col] <= 0).any():
        raise ValueError("Non positive std found for close")

    return features_df

def save_features_to_csv(features_df : pd.DataFrame, output_csv_path : Path) :
    output_csv_path.parent.mkdir(parents = True, exist_ok = True)
    features_df.to_csv(output_csv_path, index = False)

    print(f"Saved: {output_csv_path} ({len(features_df)} rows)")
    print("Last three rows:\n" + features_df.tail(3).to_string(index = False))

def main():
    ohlcv_df = load_raw_and_clean_ohlcv_csv(RAW_CSV_PATH)
    features_df = compute_feature_engineering_on_close(ohlcv_df, ROLLING_WINDOW_DAYS)
    save_features_to_csv(features_df, FEATURES_CSV_PATH)

if __name__ == "__main__":
    main()

