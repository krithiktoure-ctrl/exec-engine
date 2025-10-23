import numpy as np
import pandas as pd

def per_second_targets(total_shares, n_secs):
    base = total_shares // n_secs
    rem = total_shares % n_secs
    t = np.full(n_secs, base, dtype=float)
    if rem > 0:
        t[:rem] += 1
    return pd.Series(t)

def twap_schedule(Q, n):
    n = int(n)
    if n <= 0: return np.array([])
    return np.full(n, Q / n, dtype=float)

def ac_discrete_schedule(Q, n, dt, sigma, eta, lam):
    Q = float(Q); n = int(n); dt = float(dt)
    if n <= 0: return np.array([])
    T = n * dt

    sigma = max(1e-12, float(sigma))
    eta   = max(1e-12, float(eta))
    lam   = max(1e-12, float(lam))
    kappa = np.sqrt(lam * (sigma**2) / eta)
    t = np.arange(0, n+1, dtype=float) * dt 

    sh = np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    x = Q * sh
    
    u = x[:-1] - x[1:]
 
    u[-1] += (Q - u.sum())
    return u

def apply_pov_cap(u, vols, max_pov):
    u = np.asarray(u, dtype=float).copy()
    vols = np.asarray(vols, dtype=float)
    cap = np.maximum(0.0, np.minimum(1.0, float(max_pov))) * vols
    carry = 0.0
    for i in range(len(u)):
        want = u[i] + carry
        take = min(want, cap[i])
        carry = want - take
        u[i] = take
    
    if carry > 0 and len(u) > 0:
        u[-1] += carry
    return u
