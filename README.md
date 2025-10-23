
# Execution Engine (NASDAQ ITCH) — Almgren–Chriss vs TWAP

A compact, readable project that turns NASDAQ ITCH (LOBSTER) data into a 1-second “tape”, builds **Almgren–Chriss (AC)** and **TWAP** schedules with a **per-second PoV cap**, and compares their execution costs with a simple, fast simulator. It also includes a small ML notebook (LogReg, Random Forest, XGBoost) to benchmark a next-second direction signal.

---

## What this repo does

- **Data prep:**  
  - Quotes → bucket to **1 second**, take the **last mid** each second.  
  - Messages (`messages.csv`) → keep execution types **4/5**, sum **volume** per second.  
  - Join into one frame: `df[t, time, price_mid, vol]`.

- **Schedules:**  
  - **TWAP** over the horizon.  
  - **AC(λ)** (risk-aware shape) across a small λ grid.  
  - **PoV cap:** enforce `u_t ≤ MAX_POV × vol_t` every second.

- **Simulation & eval:**  
  - Vectorized **O(n)** simulator with temporary (η) and permanent (γ) impact.  
  - Report **implementation shortfall** statistics and **certainty-equivalent** scores.  
  - Small **rolling backtest** to compare schedules out-of-sample.

- **Optional ML add-on:**  
  - Build simple features from the 1s tape.  
  - Train **Logistic Regression**, **Random Forest**, and **XGBoost** on next-second direction.  
  - Compare accuracy/F1 and plot a tiny “toy PnL” for intuition.

---
