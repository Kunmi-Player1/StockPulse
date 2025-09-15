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
MIN_SIZE = 5 
JUMP = 1

PENALTY_START = 5.0
PENALTY_STOP  = 130.0
PENALTY_STEP  = 5.0

SPLIT_FOR_SWEEP = "val"

def make_penalty_grid(start: float, stop: float, step: float) -> list[float]:
    n_steps = int(round((stop - start) / step))
    return list(np.linspace(start, stop, num = n_steps + 1, endpoint = True))

def load_split_dataframe(split_name : str, feature_for_cp : str) -> pd.DataFrame:
    if feature_for_cp == "stl_residual":
        csv_path = PROCESSED_DIR / f"{split_name}_stl_residuals.csv"  # NOTE: 'residuals' plural
        chosen_col = "stl_residual"
    else:
        csv_path = PROCESSED_DIR / f"{split_name}.csv"
        chosen_col = "close"
    
    df = pd.read_csv(csv_path, low_memory=False)

    df["date"] = pd.to_datetime(df.get("date"), errors = "coerce")

    for col in ["close", "stl_residual"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = "coerce")

    df = df.dropna(subset = ["date", chosen_col])

    df = df.sort_values("date").drop_duplicates(subset = ["date"], keep = "first").reset_index(drop = True)
    
    df = df.dropna(subset = ["date", "close"]).reset_index(drop = True)
    return df

def count_change_points(values_array : np.ndarray, penalty_value : float, min_spacing : int, jump_step : int, cost_model : str) -> int:
    n = len(values_array)
    if n < max(2 * min_spacing, 10):
        return 0
    
    pelt_segmenter = rpt.Pelt(model = cost_model, min_size = min_spacing, jump = jump_step).fit(values_array)
    breakpoint_indices = pelt_segmenter.predict(pen = penalty_value)
    return max(0, len(breakpoint_indices) - 1)

def choose_best_penalty_at_diminishing_returns(sweep_df : pd.DataFrame) -> tuple[float, dict]:
    if sweep_df.empty or len(sweep_df) < 3:
        mid = float(sweep_df["penalty"].median()) if not sweep_df.empty else 0.0
        return mid, {"reason": "fallback_median_or_zero"}

    x = sweep_df["penalty"].to_numpy(dtype = float)
    y = sweep_df["num_cps"].to_numpy(dtype = float)

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    denom = np.hypot((y1 - y0), (x1 - x0))
    if denom == 0:
        return float(np.median(x)), {"reason": "flat_curve"}

    distances = np.abs((y1 - y0) * x - (x1 - x0) * y + (x1 * y0 - y1 * x0)) / denom
    best_idx = int(np.argmax(distances))
    best_penalty = float(x[best_idx])

    pen_results = {
    "best_index" : best_idx,
    "best_distance" : float(distances[best_idx]),
    "first_point" : [float(x0), float(y0)],
    "last_point" : [float(x1), float(y1)]
    }
    return best_penalty, pen_results

def save_sweep_values(split_name : str, feature_for_cp : str, sweep_df : pd.DataFrame, best_penalty : float, pen_results : dict):
    LOGS_DIR.mkdir(parents = True, exist_ok = True)
    feature_tag = "stl_residual" if feature_for_cp == "stl_residual" else "close"

    csv_path = LOGS_DIR / f"pen_cp_sweep_summary__{split_name}__{feature_tag}"
    sweep_df.to_csv(csv_path, index = False)

    params_uesd_and_values_gotten= {
        "ticker" : TICKER,
        "split" : split_name,
        "feature" : feature_tag,
        "model" : PELT_MODEL,
        "min_size" : MIN_SIZE,
        "jump" : JUMP,
        "penalty_grid" : sweep_df["penalty"].tolist(),
        "num_cps_curve" : sweep_df["num_cps"].tolist(),
        "chosen_penalty_pdr" : best_penalty,
        "pen_esults" : pen_results
    }

    with open(LOGS_DIR / "cp_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(params_uesd_and_values_gotten, f, indent = 2)

    print(f"Best penalty → {best_penalty:.1f}")

def main():
    penalty_grid = make_penalty_grid(PENALTY_START, PENALTY_STOP, PENALTY_STEP)

    val_df = load_split_dataframe(SPLIT_FOR_SWEEP, FEATURE_FOR_CP)

    selected_col: str = "stl_residual" if FEATURE_FOR_CP == "stl_residual" else "close"
    values_array: np.ndarray = val_df[selected_col].to_numpy()

    rows = []
    for pen in penalty_grid:
        num_cps = count_change_points(values_array, pen, MIN_SIZE, JUMP, PELT_MODEL)
        rows.append({"penalty" : float(pen), "num_cps" : int(num_cps)})
        print(f"penalty = {pen:6.1f}  →  CPs = {num_cps}")

    sweep_df = pd.DataFrame(rows).sort_values("penalty").reset_index(drop = True)

    chosen_penalty, pen_results = choose_best_penalty_at_diminishing_returns(sweep_df)

    save_sweep_values(SPLIT_FOR_SWEEP, FEATURE_FOR_CP, sweep_df, chosen_penalty, pen_results)

if __name__ == "__main__":
    main()



