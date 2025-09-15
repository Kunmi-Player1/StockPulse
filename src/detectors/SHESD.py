import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from scikit_posthocs import outliers_gesd 

TICKER = "AAPL"
PROCESSED_DIR = Path("data/processed") / TICKER

TRAIN_RESID_CSV = PROCESSED_DIR / "train_stl_residuals.csv"
VAL_RESID_CSV   = PROCESSED_DIR / "val_stl_residuals.csv"
TEST_RESID_CSV  = PROCESSED_DIR / "test_stl_residuals.csv"

TRAIN_OUT_CSV = PROCESSED_DIR / "train_shesd.csv"
VAL_OUT_CSV   = PROCESSED_DIR / "val_shesd.csv"
TEST_OUT_CSV  = PROCESSED_DIR / "test_shesd.csv"

ALPHA = 0.05 
MAX_OUTLIER_RATIO = 0.05   
TWO_SIDED = True
WINDOW_SIZE = 260
FEATURES = ["date", "close", "stl_residual"]

def load_residuals(csv_path : Path, features : list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError("Cant find csv")
    
    df = pd.read_csv(csv_path, low_memory = False, encoding_errors = "ignore")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Features: {missing} not in feature dataframe:")
    
    df['date'] = pd.to_datetime(df['date'], errors = "coerce")
    df['close'] = pd.to_numeric(df['close'], errors = "coerce")
    df['stl_residual'] = pd.to_numeric(df['stl_residual'], errors = "coerce")

    df = df.dropna(subset = features).sort_values('date').reset_index(drop = True)
    return df

def esd_flag(residual_values : np.ndarray, alpha : float, max_outliers : int, two_sided : bool) -> np.ndarray:
    x_residuals = np.array(residual_values, dtype = float)

    if x_residuals.size == 0 or max_outliers <= 0:
        return np.zeros_like(x_residuals, dtype = bool)
    
    flag = outliers_gesd(x_residuals, outliers = max_outliers, hypo = True, alpha = alpha)

    if not two_sided:
        flag = np.logical_and(flag, x_residuals >= np.nanmean(x_residuals))
    return flag.astype(bool)

def store_flags_in_csv(residuals_df : pd.DataFrame, flags : np.ndarray, out_path : Path) -> pd.DataFrame:
    out_df = pd.DataFrame({
        "date" : residuals_df['date'].to_numpy(),
        "close" : residuals_df['close'].to_numpy(),
        "stl_residual" : residuals_df['stl_residual'].to_numpy(),
        "shesd_flag" : flags.astype(bool),
    })

    out_path.parent.mkdir(parents = True, exist_ok = True)
    out_df.to_csv(out_path, index = False)
    return out_df

def main():
    train_df = load_residuals(TRAIN_RESID_CSV, FEATURES)
    n_train = len(train_df)
    max_num_to_flag = max(1, int(np.floor(MAX_OUTLIER_RATIO * n_train)))

    if WINDOW_SIZE and 0 < WINDOW_SIZE < n_train:
        train_flags = np.zeros(n_train, dtype = bool)
        for start in range(0, n_train, WINDOW_SIZE):
            end = min(start + WINDOW_SIZE, n_train)
            window_vals = train_df["stl_residual"].to_numpy()[start:end]
            max_num_to_flag_window = max(1, int(np.floor(MAX_OUTLIER_RATIO * (end - start))))
            window_mask = esd_flag(window_vals, ALPHA, max_num_to_flag_window, TWO_SIDED)
            train_flags[start:end] = np.logical_or(train_flags[start:end], window_mask)
    else:
        train_flags = esd_flag(train_df["stl_residual"].to_numpy(), ALPHA, max_num_to_flag, TWO_SIDED)

    train_out = store_flags_in_csv(train_df, train_flags, TRAIN_OUT_CSV)    
    print(f"Saved: {TRAIN_OUT_CSV} ({len(train_out)} rows)")
    print("Last three rows:\n" + train_out.tail(3).to_string(index = False))

    val_df = load_residuals(VAL_RESID_CSV, FEATURES)
    n_val = len(val_df)
    max_num_to_flag = max(1, int(np.floor(MAX_OUTLIER_RATIO * n_val)))

    if WINDOW_SIZE and 0 < WINDOW_SIZE < n_val:
        val_flags = np.zeros(n_val, dtype = bool)
        for start in range(0, n_val, WINDOW_SIZE):
            end = min(start + WINDOW_SIZE, n_val)
            window_vals = val_df["stl_residual"].to_numpy()[start:end]
            max_num_to_flag_window = max(1, int(np.floor(MAX_OUTLIER_RATIO * (end - start))))
            window_mask = esd_flag(window_vals, ALPHA, max_num_to_flag_window, TWO_SIDED)
            val_flags[start:end] = np.logical_or(val_flags[start:end], window_mask)
    else:
        val_flags = esd_flag(val_df["stl_residual"].to_numpy(), ALPHA, max_num_to_flag, TWO_SIDED)

    val_out = store_flags_in_csv(val_df, val_flags, VAL_OUT_CSV)    
    print(f"Saved: {VAL_OUT_CSV} ({len(val_out)} rows)")
    print("Last three rows:\n" + val_out.tail(3).to_string(index = False))

    test_df = load_residuals(TEST_RESID_CSV, FEATURES)
    n_test = len(test_df)
    max_num_to_flag = max(1, int(np.floor(MAX_OUTLIER_RATIO * n_test)))

    if WINDOW_SIZE and 0 < WINDOW_SIZE < n_test:
        test_flags = np.zeros(n_test, dtype = bool)
        for start in range(0, n_test, WINDOW_SIZE):
            end = min(start + WINDOW_SIZE, n_test)
            window_vals = test_df["stl_residual"].to_numpy()[start:end]
            max_num_to_flag_window= max(1, int(np.floor(MAX_OUTLIER_RATIO * (end - start))))
            window_mask = esd_flag(window_vals, ALPHA, max_num_to_flag_window, TWO_SIDED)
            test_flags[start:end] = np.logical_or(test_flags[start:end], window_mask)
    else:
        test_flags = esd_flag(test_df["stl_residual"].to_numpy(), ALPHA, max_num_to_flag, TWO_SIDED)

    test_out = store_flags_in_csv(test_df, test_flags, TEST_OUT_CSV)    
    print(f"Saved: {TEST_OUT_CSV} ({len(test_out)} rows)")
    print("Last three rows:\n" + test_out.tail(3).to_string(index = False))

if __name__ == "__main__":
    main()

