import pandas as pd
import yfinance as yf
from pathlib import Path

TICKER = "AAPL"
START = "2018-01-01"
END = "2025-01-01"
INTERVAL = "1d"

def main():
    data = yf.download(TICKER, start = START, end = END, interval = INTERVAL, auto_adjust = True)
    if data is None or data.empty:
        raise SystemExit(f"Nov data available for {TICKER} from {START} to {END}.")
    
    data = data.reset_index().rename(columns = {
        "Date" : "date", "Open" : "open", "High" : "high", "Low" : "low", "Close" : "close", "Volume" : "volume"
    })

    data['date'] = pd.to_datetime(data["date"], errors = "coerce").dt.strftime("%Y-%m-%d")
    out_cols = ["date", "open", "high", "low", "close", "volume"]

    output_file_dir = Path("data/raw") / f"{TICKER}_{INTERVAL}.csv"
    output_file_dir.parent.mkdir(parents = True, exist_ok = True)
    data[out_cols].to_csv(output_file_dir, index = False)
    print(f"Saved: {output_file_dir.name} ({len(data)} rows)")
    print("First three rows:\n " + data[out_cols].head(3).to_string(index = False) + "\n")
    print("Last three rows:\n" + data[out_cols].tail(3).to_string(index = False))

if __name__ == "__main__":
    main()