
# Execution Engine (NASDAQ ITCH) — Almgren–Chriss vs TWAP

This project turns raw NASDAQ ITCH (LOBSTER) data into a simple 1-second tape and asks the question: **When should you trade faster vs spread out?**  
It compares two schedules—**TWAP** (even pacing) and **Almgren–Chriss** (risk-aware pacing) under a per-second **participation cap (PoV)**, then uses a small simulator to estimate cost and risk. It also includes an ML notebook that benchmarks a next-second direction signal using LogReg, Random Forest, and XGBoost.

---

## What does this project do?

- Builds a 1-second feed with **mid price** and **traded volume** from ITCH messages and quotes.  
- Generates **TWAP** and **AC** schedules for “buy Q shares over N seconds,” with a per-second **PoV limit**.  
- Simulates execution with a lightweight impact model to get **implementation shortfall** (mean, variance) and a **certainty-equivalent** score.  
- Runs a small, chronological **backtest** so you can compare strategies across slices of the day.  
- Contains a **ML** notebook that trains three models on next-second direction for context.

---

## What I learned

- **Trade-offs are concrete:** front-loading cuts timing risk but raises impact, but spreading out does the opposite. The AC knob (λ) makes that trade-off visible.  
- **Participation caps bite:** a PoV limit can flatten an otherwise aggressive schedule; feasibility matters as much as the theoretical shape.  
- **Data alignment is everything:** quotes and executions live on different clocks; getting to a clean `time, price_mid, vol` per second is half the work.  
- **Keep calibration simple:** you don’t need a complex fit. Reasonable defaults for η/γ + a volatility estimate (σ) are enough to compare schedules.  
- **Validate in time, not at random:** chronological splits avoid leakage and give a truer picture than random train/test.  
- **ML is a garnish here:** at 1-second resolution, labels are noisy; a small model can help prioritize, but the core decision still comes from the execution model.  
- **Clarity pays off:** short modules, visible sanity checks, and small tables/plots make the story easier to tell.

---

## Run this quickly

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
