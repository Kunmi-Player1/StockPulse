import pandas as pd
from pathlib import Path

TICKER = "AAPL"
TABLEAU_DIR = Path("tableau")
SPLITS = ["train", "val", "test"]

REQUIRED = ["date", "close", "weirdness", "iof_flag", "shesd_flag", "cp_flag", "tier", "reason"]

def load_signals_csv(split_name : str) -> pd.DataFrame:
    path = TABLEAU_DIR / f"signals_{split_name}.csv"
    df = pd.read_csv(path, low_memory = False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[REQUIRED].copy()

def main():
    frames = []
    for split in SPLITS:
        df = load_signals_csv(split)
        df["split"] = split
        frames.append(df)

    out = pd.concat(frames, ignore_index = True)
    TABLEAU_DIR.mkdir(parents = True, exist_ok = True)
    out.to_csv(TABLEAU_DIR / "signals.csv", index = False)
    print(f"Built: {TABLEAU_DIR / 'signals.csv'}  rows = {len(out)}  cols = {list(out.columns)}")

if __name__ == "__main__":
    main()
