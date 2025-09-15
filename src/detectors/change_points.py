import numpy as np
import pandas as pd
import json
import ruptures as rpt  
from typing import Dict, List
from pathlib import Path

TICKER = "AAPL"
PROCESSED_DIR = Path("data/processed") / TICKER
LOGS_DIR = Path("logs")

FEATURE_FOR_CP = "stl_residual"

PELT_MODEL = "l2"
PENALTY = 20.0  
MIN_SIZE = 5 
JUMP = 1  

SPLITS = ["train", "val", "test"]

def load_split_dataframe(split_name : str, feature_for_cp : str) -> pd.DataFrame:
    if feature_for_cp == "stl_residual":
        csv_path = PROCESSED_DIR / f"{split_name}_stl_residuals.csv"  # NOTE: 'residuals' plural
        chosen_col = "stl_residual"
    else:
        csv_path = PROCESSED_DIR / f"{split_name}.csv"
        chosen_col = "close"
    
    df = pd.read_csv(csv_path, low_memory = False)

    df["date"] = pd.to_datetime(df.get("date"), errors = "coerce")

    for col in ["close", "stl_residual"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = "coerce")

    df = df.dropna(subset = ["date", chosen_col])

    df = df.sort_values("date").drop_duplicates(subset = ["date"], keep = "first").reset_index(drop = True)
    
    df = df.dropna(subset = ["date", "close"]).reset_index(drop = True)
    return df

def detect_change_points(values_array : np.ndarray, penalty_value : float, min_spacing : int, jump_step : int, cost_model : str) -> List[int]:
    n = len(values_array)
    if n < max(2 * min_spacing, 10):
        return [n]
    
    pelt_segmenter = rpt.Pelt(model = cost_model, min_size = min_spacing, jump = jump_step).fit(values_array)
    breakpoint_indices = pelt_segmenter.predict(pen = penalty_value)
    return breakpoint_indices

def breakpoints_to_cp_flags(n_rows : int, breakpoints : List[int]) -> np.ndarray:
    cp_flags = np.zeros(n_rows, dtype = bool)
    for end_idx in breakpoints[:-1]:
        mark_idx = max(0, min(n_rows - 1, end_idx - 1))
        cp_flags[mark_idx] = True
    return cp_flags

def store_change_points_csv(split_name: str, split_df: pd.DataFrame, cp_flags: np.ndarray):
    out_df = pd.DataFrame({
        "date" : split_df["date"].dt.strftime("%Y-%m-%d"),
        "close" : split_df["close"].values,
        "cp_flag" : cp_flags.astype(bool)
    })
    out_path = PROCESSED_DIR / f"{split_name}_cp.csv"
    out_path.parent.mkdir(parents = True, exist_ok = True)
    out_df.to_csv(out_path, index = False)

def save_params(feature_for_cp : str, cost_model : str, penalty_value : float, min_spacing : int, jump_step : int):
    LOGS_DIR.mkdir(parents = True, exist_ok = True)
    Params = {
        "ticker" : TICKER,
        "feature" : feature_for_cp,
        "model" : cost_model,
        "pen" : penalty_value,
        "min_size" : min_spacing,
        "jump" : jump_step
    }
    with open(LOGS_DIR / "cp_params.json", "w", encoding = "utf-8") as f:
        json.dump(Params, f, indent = 2)

def main():
    for split_name in SPLITS:
        split_df = load_split_dataframe(split_name, FEATURE_FOR_CP)

        selected_col = "stl_residual" if FEATURE_FOR_CP == "stl_residual" else "close"
        values_array = split_df[selected_col].to_numpy()

        breakpoint_indices = detect_change_points(values_array, PENALTY, MIN_SIZE, JUMP, PELT_MODEL)

        cp_flags = breakpoints_to_cp_flags(len(values_array), breakpoint_indices)

        store_change_points_csv(split_name, split_df, cp_flags)

        print(f"[{split_name.upper()}] rows={len(split_df)} | CPs={cp_flags.sum()}")

        save_params(FEATURE_FOR_CP, PELT_MODEL, PENALTY, MIN_SIZE, JUMP)

if __name__ == "__main__":
    main()
