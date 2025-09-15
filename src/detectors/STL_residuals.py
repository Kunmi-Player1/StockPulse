import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.seasonal import STL

TICKER = "AAPL"
PROCESSED_DIR = Path("data/processed") / TICKER

TRAIN_CSV_PATH = PROCESSED_DIR / "train.csv"
VAL_CSV_PATH   = PROCESSED_DIR / "val.csv"
TEST_CSV_PATH  = PROCESSED_DIR / "test.csv"

TRAIN_OUT_CSV = PROCESSED_DIR / "train_stl_residuals.csv"
VAL_OUT_CSV   = PROCESSED_DIR / "val_stl_residuals.csv"
TEST_OUT_CSV  = PROCESSED_DIR / "test_stl_residuals.csv"

FEATURE = "close"
STL_PERIOD = 5
STL_ROBUST = True 

def load_split_csv(csv_path : Path, feature : str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError("Cant find csv splits")
    
    df = pd.read_csv(csv_path, low_memory = False, encoding_errors = "ignore")

    if feature not in df.columns:
        raise ValueError(f"Column '{feature}' not found in {csv_path}. Columns: {list(df.columns)}")

    df['date'] = pd.to_datetime(df['date'], errors = "coerce")
    df[feature] = pd.to_numeric(df[feature], errors = "coerce")
    df = df.dropna(subset = ['date', feature]).sort_values('date').reset_index(drop = True)
    return df

def compute_stl_residual(df : pd.DataFrame, feature : str, period : int, robust : bool) -> np.ndarray:
    feature = df[feature]
    values = pd.to_numeric(feature, errors="coerce").astype("float64").to_numpy()
    num_values = len(values)

    if num_values < max(8, 2 * period):
        mean_val = np.nanmean(values)
        return values - mean_val
    
    stl = STL(values, period = period, robust = robust)
    fit = stl.fit()
    return fit.resid

def make_split_residual_df(split_df : pd.DataFrame, feature : str, residual : np.ndarray, out_path : Path) -> pd.DataFrame:
    out_df = pd.DataFrame({
        "date" : pd.to_datetime(split_df["date"], errors = "coerce").to_numpy(),
        "close" : pd.to_numeric(split_df[feature], errors = "coerce").to_numpy(),
        "stl_residual" : residual
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return out_df

def main():
    train_df = load_split_csv(TRAIN_CSV_PATH, FEATURE)
    train_residual = compute_stl_residual(train_df, FEATURE, STL_PERIOD, STL_ROBUST)
    train_out = make_split_residual_df(train_df, FEATURE, train_residual, TRAIN_OUT_CSV)
    print(f"Saved: {TRAIN_CSV_PATH} ({len(train_out)} rows)")
    print("Last three rows:\n" + train_out.tail(3).to_string(index = False))

    val_df = load_split_csv(VAL_CSV_PATH, FEATURE)
    val_residual = compute_stl_residual(val_df, FEATURE, STL_PERIOD, STL_ROBUST)
    val_out = make_split_residual_df(val_df, FEATURE, val_residual, VAL_OUT_CSV)
    print(f"Saved: {VAL_CSV_PATH} ({len(val_out)} rows)")
    print("Last three rows:\n" + val_out.tail(3).to_string(index = False))

    test_df = load_split_csv(TEST_CSV_PATH, FEATURE)
    test_residual = compute_stl_residual(test_df, FEATURE, STL_PERIOD, STL_ROBUST)
    test_out = make_split_residual_df(test_df, FEATURE, test_residual, TEST_OUT_CSV)
    print(f"Saved: {TEST_CSV_PATH} ({len(test_out)} rows)")
    print("Last three rows:\n" + test_out.tail(3).to_string(index = False))

if __name__ == "__main__":
    main()
