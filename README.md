
# Execution Engine (NASDAQ ITCH) — Almgren–Chriss vs TWAP

**What**: A minimal, fast execution engine that builds an Almgren–Chriss (AC) trading schedule on NASDAQ ITCH (LOBSTER) data, compares it to TWAP across risk aversion λ, enforces per-second PoV, and evaluates certainty-equivalent cost via simulation.

**Data**:
- *Quotes* → bucket to 1-second, take last mid per bucket.
- *Messages* (LOBSTER `messages.csv`) → keep executions (types 4/5) and sum traded **volume** per second.
- Join to get `df[t, time, price_mid, vol]`.

**Model**:
- σ estimated from mid returns.
- η calibrated so ~10% PoV costs about **half-spread** per bucket; γ ≈ 0.1·η (simple, interview-friendly).
- PoV cap: per-second trade `u_t ≤ MAX_POV · vol_t`.

**Evaluation**:
- Build TWAP and AC(λ) schedules, cap with PoV, simulate with an O(n) AC path (vectorized permanent impact).
- Report mean, variance, and **certainty-equivalent** (CE = mean + λ·var). Frontier + schedule plots.

**Run**:
1. Open `notebooks/40_ac_frontier.ipynb`, set `MSG_DIR` (messages) and load quotes CSV.
2. Run all cells → saves `data/ac_frontier_results.csv` and plots in `README_assets/`.

**Backtest**:
- A chronological 3-fold intraday backtest (`src/backtest.py`) trains σ, η, γ on the past slice and tests on the next slice.
