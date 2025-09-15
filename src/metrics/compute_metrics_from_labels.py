import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.metrics import (
    precision_score, recall_score,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)

TICKER = "AAPL"
SIGNALS_CSV = Path("tableau/signals.csv") 
LABELS_CSV  = Path(f"data/processed/{TICKER}/labels_event_windows.csv") 
IOF_JSON    = Path("saved_models/iof/iof_best_metrics.json")

OUT_METRICS = Path("tableau/metrics_summary.csv")
OUT_PR = Path("tableau/pr_points.csv")
OUT_ROC = Path("tableau/roc_points.csv")

def load_signals() -> pd.DataFrame:
    df = pd.read_csv(SIGNALS_CSV, low_memory = False, encoding_errors = "ignore")
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")
    df["weirdness"] = pd.to_numeric(df["weirdness"], errors = "coerce")
    df = df.dropna(subset=["date", "weirdness"]).sort_values("date")
    return df

def load_labels() -> pd.DataFrame:
    lab = pd.read_csv(LABELS_CSV, low_memory = False, encoding_errors = "ignore")
    lab["date"] = pd.to_datetime(lab["date"], errors = "coerce")
    lab = lab.dropna(subset = ["date"]).sort_values("date")
    lab["is_earnings_window"] = lab["is_earnings_window"].astype(int)
    return lab

def load_cutoff() -> float:
    with open(IOF_JSON, "r", encoding = "utf-8") as f:
        meta = json.load(f)
    return float(meta["best_numeric_cutoff"])

def join_signals_with_labels(signals : pd.DataFrame, labels : pd.DataFrame) -> pd.DataFrame:
    df = signals.merge(labels, on = "date", how = "left")
    df["is_earnings_window"] = df["is_earnings_window"].fillna(0).astype(int)
    return df

def get_metrics(df : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_true = df["is_earnings_window"].to_numpy(dtype = int)
    y_score = df["weirdness"].to_numpy(dtype = float)
    return y_true, y_score

def compute_precision_and_recall(y_true : np.ndarray, y_score : np.ndarray, cutoff : float) -> tuple[float, float]:
    y_pred = (y_score >= cutoff).astype(int)
    precision = precision_score(y_true, y_pred, zero_division = 0)
    recall = recall_score(y_true, y_pred, zero_division = 0)
    return float(precision), float(recall)

def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[pd.DataFrame, float]:
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    pr_df = pd.DataFrame({
        "threshold": np.append(pr_thresholds, np.nan),
        "precision": pr_precision,
        "recall": pr_recall,
    })
    return pr_df, float(pr_auc)

def compute_roc_curve(y_true : np.ndarray, y_score : np.ndarray) -> tuple[pd.DataFrame, float]:
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    roc_df = pd.DataFrame({
        "threshold" : roc_thr,
        "fpr" : fpr,
        "tpr" : tpr
    })
    return roc_df, float(roc_auc)

def build_kpis_row(ticker : str, df_joined : pd.DataFrame, cutoff : float, precision : float, recall : float, pr_auc : float, roc_auc : float) -> pd.DataFrame:
    date_min = df_joined["date"].min().date().isoformat()
    date_max = df_joined["date"].max().date().isoformat()
    kpis = pd.DataFrame([{
        "ticker" : ticker,
        "date_range" : f"{date_min} to {date_max}",
        "cutoff_numeric" : cutoff,
        "precision" : precision,
        "recall" : recall,
        "pr_auc" : pr_auc,
        "roc_auc" : roc_auc
    }])
    OUT_METRICS.parent.mkdir(parents = True, exist_ok = True)
    kpis.to_csv(OUT_METRICS, index = False)

def save_curves_for_tableau(pr_df : pd.DataFrame, roc_df : pd.DataFrame) -> None:
    OUT_PR.parent.mkdir(parents = True, exist_ok = True)
    pr_df.to_csv(OUT_PR, index = False)
    roc_df.to_csv(OUT_ROC, index = False)

def main():
    signals = load_signals()
    labels = load_labels()
    cutoff = load_cutoff()

    df = join_signals_with_labels(signals, labels)
    y_true, y_score = get_metrics(df)

    precision, recall = compute_precision_and_recall(y_true, y_score, cutoff)
    pr_df, pr_auc = compute_pr_curve(y_true, y_score)
    roc_df, roc_auc = compute_roc_curve(y_true, y_score)

    build_kpis_row(TICKER, df, cutoff, precision, recall, pr_auc, roc_auc)
    save_curves_for_tableau(pr_df, roc_df)

if __name__ == "__main__":
    main()

