import numpy as np
import pandas as pd
import json
from pathlib import Path
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score 

TICKER = "AAPL"
PROCESSED_DIR = Path("data/processed") / TICKER
TRAIN_CSV_PATH = PROCESSED_DIR / "train.csv"
VAL_CSV_PATH = PROCESSED_DIR / "val.csv"
TEST_CSV_PATH = PROCESSED_DIR / "test.csv"

MODEL_DIR = Path("saved_models/iof")
MODEL_PATH = MODEL_DIR / "iof_model.joblib"
BEST_METRICS_JSON_PATH = MODEL_DIR / "iof_best_metrics.json"

TRAIN_SCORES_AND_FLAGS_CSV = PROCESSED_DIR / "train_iof_scores.csv"
VAL_SCORES_AND_FLAGS_CSV = PROCESSED_DIR / "val_iof_scores.csv"
TEST_SCORES_AND_FLAGS_CSV = PROCESSED_DIR / "test_iof_scores.csv"
LABELS_CSV_PATH = PROCESSED_DIR / "labels_event_windows.csv"

FEATURE_COLUMNS = [
    "close_daily_return",
    "close_rolling_return_mean20",
    "close_rolling_return_std20",
    "close_zscore20",
]

IOF_N_ESTIMATORS = 600
IOF_MAX_SAMPLES = "auto"

SCORE_METHOD = "score_samples"
WEIRDNESS_DEFINITION = "weirdness =  -score_samples cus like normally small or negative values mean wierd values but when we reverse the sign those values become the big values so like bigger values now rep wierd rows i think"
TARGET_RECALL_ON_VALIDATION = 0.60
TARGET_PRECISION_ON_VALIDATION = 0.50
CUTOFF_SELECTION_METHOD = "target_recall"
CANDIDATE_PERCENTILES = np.linspace(30.0, 99.9, 400)

def load_split_data(csv_path : Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError("Cant find csv splits")
    
    df = pd.read_csv(csv_path, low_memory = False, encoding_errors = "ignore")

    df['date'] = pd.to_datetime(df['date'], errors = "coerce")
    df = df.dropna(subset = ['date']).sort_values('date').reset_index(drop = True)
    return df

def transform_features_to_numpy_for_iof_model(features_df : pd.DataFrame, feature_cols : list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Features: {missing} not in feature dataframe:")

    x = features_df[feature_cols].to_numpy(dtype = float, copy = False)

    return x

def fit_isolation_forest_on_train(x_train : np.ndarray) -> IsolationForest:
    model = IsolationForest(
        n_estimators = IOF_N_ESTIMATORS,
        max_samples = IOF_MAX_SAMPLES,
        random_state = 42,
        n_jobs = -1
    )

    model.fit(x_train)
    return model

def wierdness_score(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    return -model.score_samples(x)

def choose_numeric_cutoff_by_target_recall(validation_df : pd.DataFrame, labels_csv_path : Path, cutoff_percentiles: np.ndarray, target_recall_on_validation : float, target_precision_on_validation: float) -> dict:
    labels_df = pd.read_csv(labels_csv_path, low_memory = False, encoding_errors = "ignore")
    labels_df["date"] = pd.to_datetime(labels_df["date"], errors = "coerce")
    labels_df = labels_df.dropna(subset = ["date"]).sort_values("date")
    labels_df = labels_df.rename(columns = {"is_earnings_window": "is_anomaly"})

    validation_df["date"] = pd.to_datetime(validation_df["date"], errors = "coerce")
    validation_df["weirdness"] = pd.to_numeric(validation_df["weirdness"], errors = "coerce")
    validation_df = validation_df.dropna(subset = ["date", "weirdness"]).sort_values("date")

    merged_df = validation_df.merge(labels_df[["date", "is_anomaly"]], on = "date", how = "inner")
    y_true_labels = merged_df["is_anomaly"].astype(int).to_numpy()
    y_weirdness_score = merged_df["weirdness"].to_numpy()

    cut_off_values = np.unique(np.percentile(y_weirdness_score, cutoff_percentiles))

    records = []
    for cutoff_value in cut_off_values:
        y_predicted_flags = (y_weirdness_score >= cutoff_value).astype(int)
        precision_val = precision_score(y_true_labels, y_predicted_flags, zero_division = 0)
        recall_val = recall_score(y_true_labels, y_predicted_flags, zero_division = 0)
        records.append((float(cutoff_value), precision_val, recall_val))

    record_table = pd.DataFrame(
        records,
        columns = ["numeric_cutoff_value", "precision_on_val", "recall_on_val"]
    )

    viable_record_rows = record_table[record_table["recall_on_val"] >= target_recall_on_validation]
    if not viable_record_rows.empty:
        viable_record_rows = viable_record_rows[viable_record_rows["precision_on_val"] >= target_precision_on_validation]
        if not viable_record_rows.empty:
            best_row = viable_record_rows.sort_values(["precision_on_val", "numeric_cutoff_value"], ascending = [False, True]).iloc[0]
        else:
            best_row = record_table[record_table["recall_on_val"] >= target_recall_on_validation].sort_values(["precision_on_val", "numeric_cutoff_value"], ascending = [False, True]).iloc[0]
    else:
        best_row = record_table.sort_values(["recall_on_val", "precision_on_val", "numeric_cutoff_value"], ascending = [False, False, True]).iloc[0]

    approx_percentile_on_val = float((y_weirdness_score <= float(best_row["numeric_cutoff_value"])).mean() * 100.0)

    return {
        "numeric_cutoff_value" : float(best_row["numeric_cutoff_value"]),
        "achieved_recall_on_val" : float(best_row["recall_on_val"]),
        "achieved_precision_on_val" : float(best_row["precision_on_val"]),
        "approx_percentile_on_val" : approx_percentile_on_val,
        "record_table" : record_table
    }

def csv_file_wierdness_score_and_flags_for_df(df : pd.DataFrame, wierdness : np.ndarray, cutoff : float, out_path : Path):
    scored_df = pd.DataFrame({
        "date" : df["date"].to_numpy(),
        "close" : df["close"].to_numpy(),
        "weirdness" : wierdness,
        "iof_flag" : (wierdness >= cutoff).astype(int),
    })

    out_path.parent.mkdir(parents = True, exist_ok = True)
    scored_df.to_csv(out_path, index = False)
    print(f"Saved: {out_path} ({len(scored_df)} rows)\n Last three rows : " + scored_df.tail(3).to_string(index = False))


def save_model_and_best_metrics(model : IsolationForest, model_path : Path, best_metrics_json_path : Path, best_percentile : float, best_numeric_cutoff : float, alert_rate_val: float, alert_rate_test: float):
    model_path.parent.mkdir(parents = True, exist_ok = True)
    dump(model, model_path)

    meta = {
        "score_method" : SCORE_METHOD,
        "weirdness_definition" : WEIRDNESS_DEFINITION,
        "best_percentile" : best_percentile,
        "best_numeric_cutoff" : best_numeric_cutoff,
        "selected_row_method" : CUTOFF_SELECTION_METHOD,
        "target_recall_val" : TARGET_RECALL_ON_VALIDATION,
        "target_prec_val" : TARGET_PRECISION_ON_VALIDATION,
        "n_estimators" : IOF_N_ESTIMATORS,
        "max_samples" : IOF_MAX_SAMPLES,
        "kpis" : {
            "alert_rate_val": float(alert_rate_val),
            "alert_rate_test": float(alert_rate_test)
        }
    }

    best_metrics_json_path.parent.mkdir(parents = True, exist_ok = True)
    with open(best_metrics_json_path, "w", encoding = "utf-8") as f:
        json.dump(meta, f, indent = 2)

def evaluate_test_precision_recall(test_df : pd.DataFrame, test_wierdness : np.ndarray, labels_csv_path : Path, cutoff_value : float) -> dict:
    labels_df = pd.read_csv(labels_csv_path, low_memory = False, encoding_errors = "ignore")
    labels_df["date"] = pd.to_datetime(labels_df["date"], errors="coerce")
    labels_df = labels_df.dropna(subset = ["date"]).sort_values("date")

    labels_df = labels_df.rename(columns = {"is_earnings_window": "is_anomaly"})

    test_eval_df = pd.DataFrame({"date" : test_df["date"], "weirdness" : test_wierdness})
    test_eval_df["date"] = pd.to_datetime(test_eval_df["date"], errors = "coerce")
    test_eval_df["weirdness"] = pd.to_numeric(test_eval_df["weirdness"], errors = "coerce")
    test_eval_df = test_eval_df.dropna(subset = ["date", "weirdness"]).sort_values("date")

    merged_test = test_eval_df.merge(labels_df[["date", "is_anomaly"]], on = "date", how = "inner")
    y_true_test = merged_test["is_anomaly"].astype(int).to_numpy()
    y_pred_test = (merged_test["weirdness"] >= cutoff_value).astype(int)

    test_precision = precision_score(y_true_test, y_pred_test, zero_division = 0)
    test_recall    = recall_score(y_true_test,  y_pred_test, zero_division = 0)

    return {
        "precision_on_test": float(test_precision),
        "recall_on_test": float(test_recall),
        "n_rows_scored": int(len(merged_test)),
        "n_positives": int(merged_test["is_anomaly"].sum()),
        "n_flags": int(y_pred_test.sum())
    }

def main():
    train_df = load_split_data(TRAIN_CSV_PATH)
    val_df = load_split_data(VAL_CSV_PATH)
    test_df = load_split_data(TEST_CSV_PATH)

    x_train = transform_features_to_numpy_for_iof_model(train_df, FEATURE_COLUMNS)
    x_val = transform_features_to_numpy_for_iof_model(val_df, FEATURE_COLUMNS)
    x_test = transform_features_to_numpy_for_iof_model(test_df, FEATURE_COLUMNS)

    iof_model = fit_isolation_forest_on_train(x_train)

    train_wierdness = wierdness_score(iof_model, x_train)
    val_wierdness = wierdness_score(iof_model, x_val)
    test_wierdness = wierdness_score(iof_model, x_test)

    train_mean = float(train_wierdness.mean())
    train_std  = float(train_wierdness.std(ddof = 0))

    train_wierdness = (train_wierdness - train_mean) / train_std
    val_wierdness = (val_wierdness - train_mean) / train_std
    test_wierdness = (test_wierdness - train_mean) / train_std


    val_selected_features_df = pd.DataFrame({"date" : val_df["date"], "weirdness" : val_wierdness})

    selected_row = choose_numeric_cutoff_by_target_recall(val_selected_features_df, LABELS_CSV_PATH, CANDIDATE_PERCENTILES, TARGET_RECALL_ON_VALIDATION, TARGET_PRECISION_ON_VALIDATION)
    best_cutoff_value = selected_row["numeric_cutoff_value"]
    approx_percentile_on_val = selected_row["approx_percentile_on_val"]

    print("Validation PR sweep (first 10 rows):")
    print(selected_row["record_table"].head(10).to_string(index=False))
    print(
        f"\nChosen cutoff by target recall ({TARGET_RECALL_ON_VALIDATION:.2f}): {best_cutoff_value:.6f} "
        f"achieved recall = {selected_row['achieved_recall_on_val']:.3f}, "
        f"precision = {selected_row['achieved_precision_on_val']:.3f}, "
        f"percentile = {approx_percentile_on_val:.1f}]"
    )

    csv_file_wierdness_score_and_flags_for_df(train_df, train_wierdness, best_cutoff_value, TRAIN_SCORES_AND_FLAGS_CSV)
    csv_file_wierdness_score_and_flags_for_df(val_df, val_wierdness, best_cutoff_value, VAL_SCORES_AND_FLAGS_CSV)
    csv_file_wierdness_score_and_flags_for_df(test_df, test_wierdness, best_cutoff_value, TEST_SCORES_AND_FLAGS_CSV)

    val_alert_rate = float((val_wierdness  >= best_cutoff_value).mean())
    test_alert_rate = float((test_wierdness >= best_cutoff_value).mean())

    save_model_and_best_metrics(
        iof_model, MODEL_PATH, BEST_METRICS_JSON_PATH,
        approx_percentile_on_val, best_cutoff_value,
        val_alert_rate, test_alert_rate
    )

    train_scored_flagged = pd.read_csv(TRAIN_SCORES_AND_FLAGS_CSV, low_memory=False, encoding_errors="ignore")
    val_scored_flagged = pd.read_csv(VAL_SCORES_AND_FLAGS_CSV,   low_memory=False, encoding_errors="ignore")
    test_scored_flagged = pd.read_csv(TEST_SCORES_AND_FLAGS_CSV,  low_memory=False, encoding_errors="ignore")

    signals_df = pd.concat([train_scored_flagged, val_scored_flagged, test_scored_flagged], ignore_index=True)
    Path("tableau").mkdir(parents = True, exist_ok = True)
    signals_csv_path = Path("tableau") / "iof_signals.csv"
    signals_df.to_csv(signals_csv_path, index = False)

    print(f"Alert rates = train : {train_scored_flagged['iof_flag'].astype(bool).mean():.3%} | val: {val_scored_flagged['iof_flag'].astype(bool).mean():.3%} | test: {test_scored_flagged['iof_flag'].astype(bool).mean():.3%}")

    test_metrics = evaluate_test_precision_recall(test_df, test_wierdness, LABELS_CSV_PATH, best_cutoff_value)
    print(f"TEST metrics — precision = {test_metrics['precision_on_test']:.3f} | recall = {test_metrics['recall_on_test']:.3f}")


if __name__ == "__main__":
    main()


