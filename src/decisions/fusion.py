import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

TICKER = "AAPL"
PROCESSED_DIR = Path("data/processed") / TICKER
TABLEAU_DIR = Path("tableau")

NEAR_CP_WINDOW_K = 1
VERY_STRONG_PERCENTILE = 99.5 
SPLITS = ["train", "val", "test"]

def load_split_iof_csv(split_name : str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split_name}_iof_scores.csv"
    df = pd.read_csv(path, low_memory = False)
    needed = {"date", "close", "weirdness", "iof_flag"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")
    df = df.dropna(subset=["date", "close", "weirdness", "iof_flag"])
    df = df.sort_values("date").drop_duplicates(subset = ["date"], keep = "first").reset_index(drop = True)
    df["iof_flag"] = df["iof_flag"].astype(bool)
    return df

def load_split_shesd_csv(split_name : str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split_name}_shesd.csv"
    df = pd.read_csv(path, low_memory = False)
    needed = {"date", "close", "shesd_flag"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")
    df = df.dropna(subset = ["date", "close", "shesd_flag"])
    df = df.sort_values("date").drop_duplicates(subset = ["date"], keep = "first").reset_index(drop = True)
    df["shesd_flag"] = df["shesd_flag"].astype(bool)
    return df

def load_split_cp_csv(split_name : str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split_name}_cp.csv"
    df = pd.read_csv(path, low_memory = False)
    needed = {"date", "close", "cp_flag"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")
    df = df.dropna(subset = ["date", "close", "cp_flag"])
    df = df.sort_values("date").drop_duplicates(subset = ["date"], keep = "first").reset_index(drop = True)
    df["cp_flag"] = df["cp_flag"].astype(bool)
    return df

def compute_very_strong_cutoff(weirdness_values : np.ndarray, percentile : float) -> float:
    return float(np.percentile(weirdness_values, percentile))

def make_near_cp_flag(cp_flag_series : pd.Series, k : int) -> pd.Series:
    as_int = cp_flag_series.astype(int)
    near = as_int.rolling(window = 2 * k + 1, center = True, min_periods = 1).max()
    return near.astype(bool)

def assign_tier_level_row(iof_flag : bool, shesd_flag : bool, near_cp : bool, weirdness : float, very_strong_cutoff : float) -> Tuple[str, str]:
    if (shesd_flag and near_cp) or (weirdness >= very_strong_cutoff):
        return "A", ("A1 spike_near_cp" if (shesd_flag and near_cp) else "A2 very_strong_iof")
    if iof_flag and (shesd_flag or near_cp):
        return "B", "B iof_and_(spike_or_near_cp)"
    if (shesd_flag and not near_cp) or (iof_flag and weirdness < very_strong_cutoff):
        return "C", ("C isolated_spike_far_cp" if shesd_flag else "C weak_iof_only")
    return "", ""

def store_outputs_in_csv(split_name : str, fusion_df : pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents = True, exist_ok = True)
    TABLEAU_DIR.mkdir(parents = True, exist_ok = True)

    fusion_path = PROCESSED_DIR / f"{split_name}_fusion.csv"
    keep_cols = ["date", "close", "weirdness", "iof_flag", "shesd_flag", "cp_flag", "near_cp", "tier", "reason"]
    fusion_df.to_csv(fusion_path, columns = keep_cols, index=False)

    tableau_path = TABLEAU_DIR / f"signals_{split_name}.csv"
    fusion_df.to_csv(tableau_path, columns = ["date", "close", "weirdness", "iof_flag", "shesd_flag", "cp_flag", "near_cp","tier","reason"], index = False)

def main():
    val_iof = load_split_iof_csv("val")
    val_weirdness = val_iof["weirdness"].to_numpy()
    very_strong_cutoff = compute_very_strong_cutoff(val_weirdness, VERY_STRONG_PERCENTILE)
    print(f"[CUT] very_strong_cutoff (p{VERY_STRONG_PERCENTILE}) from VAL = {very_strong_cutoff:.3f}")

    for split_name in SPLITS:
        iof_df = load_split_iof_csv(split_name)
        sh_df = load_split_shesd_csv(split_name)
        cp_df = load_split_cp_csv(split_name)

        iof_cols_keep = iof_df[["date", "close", "weirdness", "iof_flag"]].copy()
        sh_cols_keep  = sh_df[["date", "shesd_flag"]].copy()
        cp_cols_keep  = cp_df[["date", "cp_flag"]].copy()

        merge1 = pd.merge(iof_cols_keep, sh_cols_keep, on = "date", how = "inner")
        merge2 = pd.merge(merge1, cp_cols_keep, on = "date", how = "inner")
        merged = merge2.sort_values("date").reset_index(drop = True)

        merged_step1 = pd.merge(iof_cols_keep, sh_cols_keep, on = "date", how = "inner")
        merged = pd.merge(merged_step1, cp_cols_keep, on = "date", how = "inner")
        merged = merged.sort_values("date").reset_index(drop = True)

        merged["near_cp"] = make_near_cp_flag(merged["cp_flag"], NEAR_CP_WINDOW_K)

        tiers, reasons = [], []
        for iof_flag, shesd_flag, near_cp, weirdness in merged[["iof_flag", "shesd_flag", "near_cp", "weirdness"]].itertuples(index=False, name=None):
            t, r = assign_tier_level_row(
                iof_flag = bool(iof_flag),
                shesd_flag = bool(shesd_flag),
                near_cp = bool(near_cp),
                weirdness = float(weirdness),
                very_strong_cutoff=very_strong_cutoff
            )
            tiers.append(t); reasons.append(r)
        merged["tier"] = tiers
        merged["reason"] = reasons

        store_outputs_in_csv(split_name, merged)
        a_count = int((merged["tier"] == "A").sum())
        b_count = int((merged["tier"] == "B").sum())
        c_count = int((merged["tier"] == "C").sum())
        print(f"[{split_name.upper()}] rows = {len(merged)} | A = {a_count} B = {b_count} C = {c_count}")

if __name__ == "__main__":
    main()

