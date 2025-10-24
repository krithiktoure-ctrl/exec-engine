
# Execution Engine (NASDAQ ITCH) — Almgren–Chriss vs TWAP

This project turns raw NASDAQ ITCH (LOBSTER) data into a simple 1-second tape and asks the question: **When should you trade faster vs spread out?**  
It compares two schedules—**TWAP** (even pacing) and **Almgren–Chriss** (risk-aware pacing) under a per-second **participation cap (PoV)**, then uses a small simulator to estimate cost and risk. It also includes an ML notebook that benchmarks a next-second direction signal using LogReg, Random Forest, and XGBoost.

---

## What does it do?

- Builds a 1-second feed with **mid price** and **traded volume** from ITCH messages and quotes.  
- Generates **TWAP** and **AC** schedules for “buy Q shares over N seconds,” with a per-second **PoV limit**.  
- Simulates execution with a lightweight impact model to get **implementation shortfall** (mean, variance) and a **certainty-equivalent** score.  
- Runs a small, chronological **backtest** so you can compare strategies across slices of the day.  
- Contains an **ML** notebook that trains three models on next-second direction for context.

---

## What I learned

- **Trade-offs are concrete:** front-loading cuts timing risk but raises impact, but spreading out does the opposite. The AC knob (λ) makes that trade-off visible.
- **Participation limits bite:** a poV restriction may soften a strongly competitive distribution, since feasibility, as much as theoretical form, carries weight.
- **ALIGNED DATA = EVERYTHING:** Quotes and executions are on separate clocks, but getting to a clean set of `time, price_mid, vol` per second time series is half the battle.
- **Simple calibration:** no need for a complicated fit. Basic defaults for η/γ, along with a volatility parameter σ, will suffice for a comparison between schedules.
- **Split in time, not at random:** Chronologically split data avoids leakage and provides a fairer view than a random split for training/testing.
- **ML models:** labels at a 1-second level are noisy, but an ML model may aid with filtering, with a final decision provided by this execution model,
- **Clarity pays off:** Short modules, sanity checks, and plots/graphs make the story easier to tell

---

## Run this quickly

```bash
python -m venv .venv
source .venv/bin/activate            
pip install -r requirements.txt
