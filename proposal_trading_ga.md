---
name: GA Trading Algorithm Proposal
description: Proposal to extend the DEAP demo into a financial trading strategy optimiser
type: project
---

Extend the existing DEAP demo to evolve trading strategy parameters using genetic algorithms, with a full backtest engine and walk-forward validation.

**Why:** Natural extension of the GA demo into a real-world use case — financial trading.

**How to apply:** When the user asks to build this, refer to the design below.

## Core Idea

Each individual in the GA = a trading strategy's parameter set (e.g. RSI period, MA windows, stop-loss %, position size). Crossover mixes parameters from two profitable strategies; mutation nudges them. Fitness = backtest result (Sharpe ratio, Calmar ratio, etc.).

## Genome Design

```
individual = [
  rsi_period,      # int, e.g. 14
  rsi_overbought,  # float, e.g. 70
  rsi_oversold,    # float, e.g. 30
  ma_fast,         # int, e.g. 10
  ma_slow,         # int, e.g. 50
  stop_loss_pct,   # float, e.g. 0.02
  take_profit_pct, # float, e.g. 0.05
  position_size,   # float, e.g. 0.1
]
```

## Fitness Function

Run a full backtest on historical OHLC data and return risk-adjusted metrics:
- Sharpe ratio (primary)
- Calmar ratio (return / max drawdown) for trend strategies
- Multi-objective: maximise return AND minimise drawdown simultaneously
  → DEAP `weights=(1.0, -1.0)` → produces a Pareto front of non-dominated strategies

Must include transaction costs to avoid overfitting to high-frequency noise.

## Key Engineering Pieces to Build

1. `backtest()` function — pandas + vectorbt or backtrader
2. OHLC data loader — synthetic or real (yfinance)
3. Multi-objective DEAP fitness
4. Walk-forward validation loop wrapping the GA (train on window A, test on B, advance)
5. UI extensions:
   - Pareto front scatter plot (return vs drawdown)
   - Equity curve chart for best individual
   - Walk-forward performance breakdown

## Risks / Mitigations

- **Overfitting (curve-fitting):** Walk-forward validation + held-out test set the GA never sees
- **Fitness gaming:** Multi-objective fitness penalising drawdown and requiring minimum trade count
- **Transaction costs:** Must be baked into backtest, not added as afterthought

## Stack additions needed

- `yfinance` or synthetic OHLC generator
- `pandas`, `numpy`
- `vectorbt` or custom backtest engine
- No frontend stack changes needed — same Tailwind/Chart.js/jQuery UI
