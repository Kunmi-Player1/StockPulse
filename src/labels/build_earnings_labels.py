import pandas as pd
from pathlib import Path

TICKER = "AAPL"
EARNINGS_CSV = Path("data/external/calendars/aapl_earnings.csv")
PROCESSED_DIR = Path("data/processed") / TICKER
OUT_LABELS = PROCESSED_DIR / "labels_event_windows.csv"

def load_trading_dates_from_splits() -> pd.Series:
    frames = []
    for split in ["train", "val", "test"]:
        p = PROCESSED_DIR / f"{split}.csv"
        if p.exists():
            df = pd.read_csv(p, low_memory = False, encoding_errors = "ignore")
            df["date"] = pd.to_datetime(df["date"], errors = "coerce")
            frames.append(df)
    if not frames:
        raise SystemExit(
            f"No split CSVs found under {PROCESSED_DIR}. "
            "Expected train.csv, val.csv, test.csv with a 'date' column."
        )
    all_dates = pd.concat(frames, ignore_index = True)
    all_dates = all_dates.dropna(subset = ["date"]).drop_duplicates().sort_values("date")
    return all_dates["date"].reset_index(drop = True)

def next_trading_day(dates : pd.Series, day : pd.Timestamp):
    nxt = dates[dates > day].min()
    return None if pd.isna(nxt) else pd.Timestamp(nxt)

def build_labels(trading_dates : pd.Series, earnings_df : pd.DataFrame) -> pd.DataFrame:
    trading_dates = pd.to_datetime(trading_dates, errors = "coerce")
    trading_dates = trading_dates.dropna().drop_duplicates().sort_values()

    edf = earnings_df.copy()
    edf["date"] = pd.to_datetime(edf["date"], errors = "coerce")
    edf = edf.dropna(subset = ["date"]).sort_values("date")
    edf["time_of_day"] = edf["time_of_day"].astype(str).str.upper().str.strip()

    label_set = []

    for row in edf.itertuples(index = False):
        d = pd.Timestamp(row.date)
        tod = row.time_of_day

        if tod in ("AMC", "PM", "AFTER CLOSE"):
            label_set.append(d)
            nd = next_trading_day(trading_dates, d)
            if nd is not None:
                label_set.append(nd)

        elif tod in ("BMO", "AM", "BEFORE OPEN"):
            label_set.append(d)

        else:
            label_set.append(d)
            nd = next_trading_day(trading_dates, d)
            if nd is not None:
                label_set.append(nd)

    out = pd.DataFrame({"date" : trading_dates})
    out["is_earnings_window"] = out["date"].isin(label_set).astype("int8")
    out["event_type"] = "earnings"
    return out

def main():
    if not EARNINGS_CSV.exists():
        EARNINGS_CSV.parent.mkdir(parents = True, exist_ok = True)
        raise SystemExit("Missing data earnings.csv")

    trading_dates = load_trading_dates_from_splits()
    earnings_df = pd.read_csv(EARNINGS_CSV, low_memory = False)

    labels_df = build_labels(trading_dates, earnings_df)

    OUT_LABELS.parent.mkdir(parents = True, exist_ok = True)
    labels_df.to_csv(OUT_LABELS, index = False)
    print(f"Saved labels to: {OUT_LABELS}")
    print(labels_df.tail(5))

if __name__ == "__main__":
    main()