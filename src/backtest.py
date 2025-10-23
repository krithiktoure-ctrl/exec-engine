from __future__ import annotations
import numpy as np
import pandas as pd
from math import ceil

from src.eval.metrics import realized_sigma_from_mid
from src.execution.schedules import twap_schedule, ac_discrete_schedule, apply_pov_cap
from src.execution.simulator import simulate_exec_fast


def calibrate_eta_gamma_from_quotes(df_like: pd.DataFrame, dt: float, target_pov: float = 0.10) -> tuple[float, float]:
    if "spread$" in df_like.columns:
        half_spread = float(np.median(df_like["spread$"])) / 2.0
    elif {"ask$", "bid$"}.issubset(df_like.columns):
        half_spread = float(np.median(df_like["ask$"] - df_like["bid$"])) / 2.0
    else:
        half_spread = 0.005
    med_vol = max(1.0, float(np.median(df_like.get("vol", pd.Series(np.zeros(len(df_like)))))))
    ETA = (half_spread * dt) / (target_pov * med_vol)
    GAMMA = 0.1 * ETA
    return ETA, GAMMA


def run_backtest(
    df_full: pd.DataFrame,
    *,
    q_for_spread: pd.DataFrame | None = None,
    k: int = 3,
    Q: float = 100_000.0,
    DT: float = 1.0,
    LAMBDAS: list[float] = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6),
    MAX_POV: float = 0.20,
    mc: int = 64,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    n = len(df_full)
    fold_len = ceil(n / k)
    results, winners = [], []

    for i in range(k):
        test_lo = i * fold_len
        test_hi = min((i + 1) * fold_len, n)
        if test_hi - test_lo < 10:
            continue

        df_train = df_full.iloc[:test_lo]
        df_test  = df_full.iloc[test_lo:test_hi].reset_index(drop=True)

    
        SIG = realized_sigma_from_mid(df_train["price_mid"].values, dt=DT)

        
        cal_df = df_train if q_for_spread is None else q_for_spread
        ETA, GAMMA = calibrate_eta_gamma_from_quotes(cal_df, dt=DT)

        n_test = len(df_test)
        vols   = df_test["vol"].to_numpy(float)

        
        cap = MAX_POV * vols
        Q_eff = float(min(Q, cap.sum() * 0.98)) if cap.sum() + 1e-9 < Q else Q

        
        scheds = {}
        twap = twap_schedule(Q_eff, n_test)
        twap = apply_pov_cap(twap, vols, MAX_POV)
        scheds["TWAP"] = twap

        for lam in LAMBDAS:
            u = ac_discrete_schedule(Q_eff, n_test, DT, sigma=SIG, eta=ETA, lam=lam)
            u = apply_pov_cap(u, vols, MAX_POV)
            scheds[f"AC(lam={lam:g})"] = u

        def _eval(u):
            rs = np.random.RandomState(seed)
            p0 = float(df_test["price_mid"].iloc[0])
            smp = []
            for _ in range(mc):
                s = int(rs.randint(0, 2**31 - 1))
                _, totals = simulate_exec_fast(p0=p0, u=u, dt=DT, sigma=SIG, eta=ETA, gamma=GAMMA, seed=s)
                smp.append(totals["IS"])
            arr = np.asarray(smp)
            return {"mean": float(arr.mean()), "var": float(arr.var(ddof=1)), "std": float(arr.std(ddof=1))}

        rows = []
        for name, u in scheds.items():
            stats = _eval(u)
            for lam in (0.0, *LAMBDAS):
                CE = stats["mean"] + lam * stats["var"]
                rows.append({
                    "fold": i + 1,
                    "t_start": int(test_lo),
                    "t_end": int(test_hi - 1),
                    "schedule": name,
                    "lam": lam,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "var": stats["var"],
                    "CE": CE,
                    "Q_eff": Q_eff,
                    "SIG": SIG, "ETA": ETA, "GAMMA": GAMMA,
                })

        fold_df = pd.DataFrame(rows).sort_values(["lam", "CE"]).reset_index(drop=True)
        win = fold_df.loc[fold_df.groupby("lam")["CE"].idxmin()].copy()
        results.append(fold_df); winners.append(win)

    all_rows = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    winners  = pd.concat(winners,  ignore_index=True) if winners  else pd.DataFrame()
    return all_rows, winners
