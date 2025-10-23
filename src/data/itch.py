from pathlib import Path
import pandas as pd
import numpy as np
import glob

def _find_lobster_files(root: Path, symbol: str):
    root = Path(root)
    ob = glob.glob(str(root / f"{symbol}_*_orderbook_10.csv*"))
    if not ob:
        raise FileNotFoundError(f"Could not find {symbol}_*_orderbook_10.csv in {root}")
    ob_path = Path(sorted(ob)[0])

    msg = []
    msg += glob.glob(str(root / f"{symbol}_*_message*.csv*"))
    msg += glob.glob(str(root / f"{symbol}_*_messages*.csv*"))
    if not msg:
        raise FileNotFoundError(f"Could not find {symbol}_*_message*.csv in {root}")
    msg_path = Path(sorted(msg)[0])
    return ob_path, msg_path

def load_lobster_top1_day(root: Path, symbol: str) -> pd.DataFrame:
    """Load LOBSTER level-1 (top of book) as a tidy DataFrame with time, bid, ask, sizes."""
    ob_path, msg_path = _find_lobster_files(root, symbol)

    msg = pd.read_csv(msg_path, header=None)
    time_sec = msg.iloc[:, 0].astype(float)

    ob = pd.read_csv(ob_path, header=None)
    n = min(len(ob), len(time_sec))
    if n == 0:
        return pd.DataFrame(columns=["time", "bid", "bid_size", "ask", "ask_size", "mid", "spread"])
    ob = ob.iloc[:n, :]
    time_sec = time_sec.iloc[:n].reset_index(drop=True)

    c0, c1, c2, c3 = ob.columns[:4]  

    bidA  = ob[c0].astype(float)
    bszA  = ob[c1].astype(float)
    askA  = ob[c2].astype(float)
    aszA  = ob[c3].astype(float)
    okA   = (bidA <= askA).mean()

    bidB  = ob[c2].astype(float)
    bszB  = ob[c3].astype(float)
    askB  = ob[c0].astype(float)
    aszB  = ob[c1].astype(float)
    okB   = (bidB <= askB).mean()

    if okB > okA:
        bid, bid_sz, ask, ask_sz = bidB, bszB, askB, aszB
    else:
        bid, bid_sz, ask, ask_sz = bidA, bszA, askA, aszA

    df = pd.DataFrame({
        "time": time_sec.values,        
        "bid":  bid.values,
        "bid_size": bid_sz.values,
        "ask":  ask.values,
        "ask_size": ask_sz.values,
    })

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]

    df = df[df["spread"] >= 0].reset_index(drop=True)
    return df


from __future__ import annotations
from pathlib import Path

def load_messages(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["time_s","type","order_id","size","price","dir"])
    return df

def volume_1s_from_messages(messages: pd.DataFrame) -> pd.DataFrame:
    execs = messages.copy()
    execs["type_num"] = pd.to_numeric(execs["type"], errors="coerce")
    execs = execs[execs["type_num"].isin([4,5])]
    base = pd.Timestamp("1970-01-01", tz="UTC")
    execs["ts"] = base + pd.to_timedelta(execs["time_s"].astype(float), unit="s")
    vol_1s = (
        execs.set_index("ts")["size"].astype(float).resample("1s").sum()
             .rename("vol").reset_index().rename(columns={"ts":"time"})
    )
    return vol_1s

def quotes_to_mid_1s(quotes: pd.DataFrame) -> pd.DataFrame:
    q = quotes.copy()
    if not pd.api.types.is_datetime64_any_dtype(q["time"]):
        base = pd.Timestamp("1970-01-01", tz="UTC")
        q["time"] = base + pd.to_timedelta(pd.to_numeric(q["time"], errors="coerce"), unit="s")
    else:
        if q["time"].dt.tz is None:
            q["time"] = q["time"].dt.tz_localize("UTC")
    mid_col = "mid$" if "mid$" in q.columns else ("mid" if "mid" in q.columns else "price_mid")
    q["_t1s"] = q["time"].dt.floor("1s")
    out = (q.groupby("_t1s", as_index=False)[mid_col].last()
             .rename(columns={"_t1s":"time", mid_col:"price_mid"})
             .sort_values("time").reset_index(drop=True))
    return out

def make_1s_frame(quotes: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    mid_1s = quotes_to_mid_1s(quotes)
    vol_1s = volume_1s_from_messages(messages)
    df = mid_1s.merge(vol_1s, on="time", how="left")
    df["vol"] = df["vol"].fillna(0.0)
    df.insert(0, "t", np.arange(len(df), dtype=int))
    return df

