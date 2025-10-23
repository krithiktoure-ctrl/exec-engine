import pandas as pd
import numpy as np
from pathlib import Path

def load_quotes(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = "".join(p.suffixes).lower()
    if suf.endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
        
    if "symbol" in df.columns:
        df = df[df["symbol"] == "AAPL"].reset_index(drop=True)
    return df

def ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_ms" in df.columns:
        df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    elif "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        start = pd.Timestamp("2024-03-29", tz="UTC")
        df["time"] = start + pd.to_timedelta(np.arange(len(df)), unit="ms")
    return df

def load_quotes_bucketed(csv_path: str = "data/quotes_bucketed.csv",
                         pkl_path: str = "data/quotes_bucketed.pkl") -> pd.DataFrame:
    
    pkl = Path(pkl_path)
    csv = Path(csv_path)
    if pkl.exists():
        df = pd.read_pickle(pkl)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Neither {pkl} nor {csv} found. Run nb20 to export first.")
    if "time" not in df.columns:
        raise KeyError(f"'time' not found in {df.columns.tolist()}")
    if "price_mid" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["price_mid"] = (df["bid"] + df["ask"]) / 2.0
        elif "mid$" in df.columns:
            df["price_mid"] = df["mid$"]
        else:
            raise KeyError("Could not find 'price_mid' (or bid/ask or mid$)")
    if np.issubdtype(df["time"].dtype, np.number):
        base = pd.Timestamp("1970-01-01", tz="UTC")
        df["time"] = base + pd.to_timedelta(df["time"].astype(float), unit="s")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if "t" not in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
        df.insert(0, "t", np.arange(len(df), dtype=int))
    return df