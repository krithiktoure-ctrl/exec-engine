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