from __future__ import annotations
import pandas as pd
import numpy as np


def score_metrics(
    fills: pd.DataFrame,
    book: pd.DataFrame,
    targets: pd.Series | None = None,
    horizon_secs: int = 5,
    side: str = "buy",
):
    
    if fills is None or len(fills) == 0:
        return {
            "fills": 0,
            "fill_rate": 0.0,
            "CPS": np.nan,
            "SpreadCapture%": np.nan,
            f"PTM@{horizon_secs}s": 0.0,
        }

    f = fills.copy()

    if "qty" not in f.columns:
        cand = [c for c in f.columns if c.lower() in ("qty", "quantity", "shares", "size")]
        if cand:
            f = f.rename(columns={cand[0]: "qty"})
        else:
            raise ValueError("fills must contain a 'qty' (or quantity/shares/size) column")

    if "price" not in f.columns:
        cand = [c for c in f.columns if c.lower() in ("price", "px", "fill_price", "exec_px")]
        if cand:
            f = f.rename(columns={cand[0]: "price"})
        else:
            raise ValueError("fills must contain a 'price' (or px/fill_price/exec_px) column")
        
    if "time" not in f.columns:
        f = f.assign(time=f.index)

    b = book[["bid", "ask", "mid"]].copy()
    b["spread"] = (b["ask"] - b["bid"]).astype(float)

    joined = f.join(b, on="time", how="left")

    joined = joined.dropna(subset=["mid", "bid", "ask", "spread"])


    total_filled = float(joined["qty"].sum())
    total_target = float(targets.sum()) if targets is not None else total_filled
    fill_rate = (total_filled / total_target) * 100.0 if total_target > 0 else 0.0

 
    sign = 1.0 if side.lower() == "buy" else -1.0

  
    cps_num = ((joined["price"] - joined["mid"]) * sign * 100.0 * joined["qty"]).sum()
    cps = cps_num / max(total_filled, 1.0)

    if side.lower() == "buy":
        capture = (joined["ask"] - joined["price"]) / joined["spread"]
    else:
        capture = (joined["price"] - joined["bid"]) / joined["spread"]

    capture = capture.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sc = (capture * joined["qty"]).sum() / max(total_filled, 1.0) * 100.0


    future_mid = book["mid"].shift(-int(horizon_secs))
    ptm_vals = (future_mid.loc[joined["time"]].values - joined["mid"].values) * sign * 100.0
    ptm = (pd.Series(ptm_vals) * joined["qty"].values).sum() / max(total_filled, 1.0)

    return {
        "fills": int((joined["qty"] > 0).sum()),
        "fill_rate": float(fill_rate),
        "CPS": float(cps),
        "SpreadCapture%": float(sc),
        f"PTM@{horizon_secs}s": float(ptm),
    }

def realized_sigma_from_mid(mid, dt):
    mid = np.asarray(mid, dtype=float)
    if len(mid) < 2: return 0.0
    r = np.diff(np.log(mid))
    var = np.var(r, ddof=1) if len(r) > 1 else np.var(r)
    return float(np.sqrt(max(var, 0.0) / dt))

def calibrate_sigma_by_group(df, price_col="price_mid", group_cols=("date_id","stock_id"), dt=1.0):

    if isinstance(group_cols, str): group_cols = (group_cols,)
    rows = []
    for gvals, gdf in df.groupby(list(group_cols)):
        mid = gdf[price_col].values
        sigma = realized_sigma_from_mid(mid, dt)
        key = gvals if isinstance(gvals, tuple) else (gvals,)
        rows.append({**{c:v for c,v in zip(group_cols, key)}, "sigma": sigma})
    return pd.DataFrame(rows)

def summarize_is(samples):
    x = np.asarray(samples, dtype=float)
    mu = float(np.mean(x)) if len(x) else 0.0
    var = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
    std = float(np.sqrt(max(var,0.0)))
    ci = (mu - 1.96*std/np.sqrt(len(x)), mu + 1.96*std/np.sqrt(len(x))) if len(x)>1 else (mu, mu)
    return {"mean": mu, "std": std, "var": var, "ci95": ci}

def certainty_equivalent(mean_cost, var_cost, lam):
    return float(mean_cost + lam * var_cost)


def realized_sigma_from_mid(prices, dt: float) -> float:
    p = np.asarray(prices, dtype=float)
    r = np.diff(np.log(p + 1e-12))
    return float(np.std(r, ddof=1) / np.sqrt(dt))

def summarize_is(samples) -> dict:
    a = np.asarray(samples, dtype=float)
    return {"mean": float(a.mean()), "var": float(a.var(ddof=1)), "std": float(a.std(ddof=1))}

def certainty_equivalent(mean: float, var: float, lam: float) -> float:
    return float(mean + lam * var)

