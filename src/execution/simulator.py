import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def per_second_targets_float(Q_total, n_secs):
    per = float(Q_total) / float(n_secs)
    t = np.full(n_secs, per, dtype=float)
    t[0] += (Q_total - t.sum())
    return pd.Series(t, index=df.index[:n_secs])

def vwapy_targets(df, Q_total):
    vol = (df["bid_size"] + df["ask_size"]).clip(lower=1e-9)
    w = vol / vol.sum()
    t = (w * Q_total).values
    t[0] += (Q_total - t.sum())  
    return pd.Series(t, index=df.index[:len(t)])

def simulate_passive_then_flip(df, targets, timeout_secs=5, cap_frac=0.20, queue_share=0.25):
    n = len(df)
    fills = []
    remaining = 0.0
    for i in range(n):
        row = df.iloc[i]
        planned = targets.iloc[i] if i < len(targets) else 0.0
        remaining += planned

 
        passive_cap = queue_share * max(row["bid_size"], 1e-9)
        passive_take = float(min(remaining, passive_cap))
        if passive_take > 0:
            fills.append((row["time"], "buy_passive", passive_take, float(row["bid"])))
            remaining -= passive_take

    
        if timeout_secs > 0 and i >= timeout_secs:
            flip_cap = cap_frac * max(row["ask_size"], 1e-9)
            flip_take = float(min(remaining, flip_cap))
            if flip_take > 0:
                fills.append((row["time"], "buy_flip", flip_take, float(row["ask"])))
                remaining -= flip_take

    return pd.DataFrame(fills, columns=["time","side","qty","price"])

def simulate_aggressive(df, targets, cap_frac=0.20):
    records = []
    for i, (idx, row) in enumerate(df.iterrows()):
        want = targets.iloc[i] if i < len(targets) else 0.0
        if want <= 0:
            records.append((row["time"], "buy", 0.0, np.nan))
            continue
        max_take = cap_frac * max(row["ask_size"], 1e-9)
        got = float(min(want, max_take))
        px  = float(row["ask"])
        records.append((row["time"], "buy", got, px))
    return pd.DataFrame(records, columns=["time","side","qty","price"])

def simulate_passive_then_smart_flip(df, targets, base_timeout=5, cap_frac=0.20, queue_share=0.25,
                                     tight_spread=1.0, mom_look=3):
    fills, remaining = [], 0.0
    mid = df["mid"].values
    spr = (df["ask"] - df["bid"]).values
    for i, row in enumerate(df.itertuples(index=False)):
        planned = targets.iloc[i] if i < len(targets) else 0.0
        remaining += planned
        passive_cap = queue_share * max(row.bid_size, 1e-9)
        take_p = float(min(remaining, passive_cap))
        if take_p > 0:
            fills.append((row.time, "buy_passive", take_p, float(row.bid)))
            remaining -= take_p
        mom = 0.0
        if i >= mom_look:
            mom = mid[i] - mid[i-mom_look]   

        flip_now = (spr[i] <= tight_spread) or (mom > 0) or (i >= base_timeout)
        if flip_now and remaining > 0:
            flip_cap = cap_frac * max(row.ask_size, 1e-9)
            take_a = float(min(remaining, flip_cap))
            if take_a > 0:
                fills.append((row.time, "buy_flip", take_a, float(row.ask)))
                remaining -= take_a

    return pd.DataFrame(fills, columns=["time","side","qty","price"])

def _cps(paid_prices, mid_prices, qtys):
    if len(paid_prices) == 0:
        return 0.0
    paid = np.average(paid_prices, weights=qtys)
    mid  = np.average(mid_prices,  weights=qtys)
    return (paid - mid) * 100.0

def _spread_capture_percent(paid_prices, bid_prices, ask_prices, qtys):
    if len(paid_prices) == 0:
        return 0.0
    paid = np.average(paid_prices, weights=qtys)
    best_bid = np.average(bid_prices, weights=qtys)
    best_ask = np.average(ask_prices, weights=qtys)
    if best_ask == best_bid:
        return 0.0
    return (best_ask - paid) / (best_ask - best_bid) * 100.0

def simulate_model_driven(q, targets, clf, X, k=4.0, mode="mixed"):
    paid_prices = []
    mid_prices  = []
    bid_prices  = []
    ask_prices  = []
    qtys        = []

    filled = 0
    for i in range(len(q)):
        want = float(targets.iloc[i])
        if want <= 0:
            continue

        X_curr = X.iloc[[i]]
        p_up = float(clf.predict_proba(X_curr)[:, 1][0])
        qshare = float(np.clip((p_up - 0.5) * k, 0.0, 1.0))

        if mode == "all_agg":
            agg = want
            pas = 0.0
        elif mode == "all_pass":
            agg = 0.0
            pas = want
        else:
            agg = want * qshare
            pas = want - agg

        if agg > 0:
            paid_prices.append(q["ask"].iloc[i])
            mid_prices.append(q["mid"].iloc[i])
            bid_prices.append(q["bid"].iloc[i])
            ask_prices.append(q["ask"].iloc[i])
            qtys.append(agg)
            filled += agg

        if pas > 0:
            if p_up < 0.5:
                fill_frac = 0.6
            else:
                fill_frac = 0.4
            got = pas * fill_frac
            if got > 0:
                paid_prices.append(q["bid"].iloc[i])
                mid_prices.append(q["mid"].iloc[i])
                bid_prices.append(q["bid"].iloc[i])
                ask_prices.append(q["ask"].iloc[i])
                qtys.append(got)
                filled += got

    if len(qtys) == 0:
        return pd.DataFrame([{
            "fills": 0, "fill_rate": 0.0, "CPS": 0.0, "SpreadCapture%": 0.0
        }])

    cps  = _cps(paid_prices, mid_prices, qtys)
    scp  = _spread_capture_percent(paid_prices, bid_prices, ask_prices, qtys)
    total = float(targets.sum())
    res = pd.DataFrame([{
        "fills": int(round(filled)),
        "fill_rate": float(filled / total * 100.0) if total > 0 else 0.0,
        "CPS": cps,
        "SpreadCapture%": scp,
    }])
    return res

def simulate_path(p0, n, dt, sigma, rs):
    z = rs.normal(0.0, 1.0, size=n)
    dW = np.sqrt(dt) * z
    base = np.empty(n+1, dtype=float)
    base[0] = p0
    base[1:] = p0 + np.cumsum(sigma * dW)
    return base  

def simulate_exec(p0, u, dt, sigma, eta, gamma, side=+1, seed=42):
    rs = np.random.RandomState(seed)
    u = np.asarray(u, dtype=float)
    n = len(u)
    path = simulate_path(p0, n, dt, sigma, gamma, rs).copy()
    mid = path.copy()
    exec_px = np.empty(n, dtype=float)
    for t in range(n):
        exec_px[t] = mid[t] + (gamma * u[t])/(2.0*dt) + (eta * u[t])/dt
        mid[t+1:] += (gamma * u[t]) / dt
    Q = u.sum()
    is_cost = float(np.dot(u, exec_px) - Q * p0)
    out = pd.DataFrame({
        "t": np.arange(n, dtype=int),
        "u": u,
        "mid_t": mid[:-1],
        "mid_t1": mid[1:],
        "exec_px": exec_px,
    })
    totals = {"Q": Q, "p0": p0, "IS": is_cost}
    return out, totals

def simulate_exec_fast(p0, u, dt, sigma, eta, gamma, seed=42):
    rs = np.random.RandomState(seed)
    u = np.asarray(u, dtype=float)
    n = len(u)

    z = rs.normal(0.0, 1.0, size=n)
    base = np.empty(n + 1, dtype=float)
    base[0] = p0
    base[1:] = p0 + np.cumsum(sigma * np.sqrt(dt) * z)

    
    csum_u = np.cumsum(u)                         
    perm_shift = np.r_[0.0, csum_u[:-1]] * (gamma / dt)
    mid = base.copy()
    mid[1:] += perm_shift

    exec_px = mid[:-1] + (gamma * u) / (2.0 * dt) + (eta * u) / dt

    Q = float(u.sum())
    is_cost = float(np.dot(u, exec_px) - Q * p0)

    out = pd.DataFrame({
        "t": np.arange(n, dtype=int),
        "u": u,
        "mid_t": mid[:-1],
        "mid_t1": mid[1:],
        "exec_px": exec_px,
    })
    totals = {"Q": Q, "p0": p0, "IS": is_cost}
    return out, totals


