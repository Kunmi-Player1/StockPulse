import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_CSV = Path("data/raw/AAPL_1d.csv")
OUT_DIR = Path("tableau")
OUT_IMG = OUT_DIR / "StockPulse_Smoke.png"

def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"CSV not found: {RAW_CSV} .")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, low_memory = False, encoding_errors = "ignore")
    df['date'] = pd.to_datetime(df['date'], errors = "coerce")
    df = df.dropna(subset = ['close', 'date'])
    df["close"] = pd.to_numeric(df["close"], errors = "coerce") 
    df = df.dropna(subset = ["date", "close"]).sort_values("date")

    plt.figure(figsize = (10, 4))
    plt.plot(df['date'], df['close'])
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title("AAPL daily close")
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi = 150)
    print(f"Saved: {OUT_IMG}  (rows plotted: {len(df)})")

if __name__ == "__main__":
    main()

