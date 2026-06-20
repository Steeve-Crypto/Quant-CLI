# Prediction Market Quant Arbitrage Pipeline

**End-to-end Python framework** for statistical arbitrage between prediction markets (Polymarket, Kalshi, etc.) using cointegration, Ornstein-Uhlenbeck calibration, order-book-inspired signals, and portfolio construction.

Built following the concepts from @ridark_eth's thread on quantitative hedge fund techniques for prediction market spreads.

## Features
- Realistic synthetic data generator (regimes, shocks, latency, time-of-day effects)
- Engle-Granger + **Johansen cointegration** testing
- OU process calibration (MLE via OLS on discretized form)
- Improved mean-reversion signals with momentum confirmation
- Position sizing based on OU parameters (z-score scaled or vol-target)
- Single-asset + **multi-asset portfolio backtesting** (correlation-aware min-variance, equal-risk)
- Rich diagnostics: rolling beta, residual analysis, equity curves, etc.
- Git-tracked development with regular commits

## Quick Start
```bash
cd /home/workdir/artifacts
python prediction_market_arb_pipeline.py
```

This runs the full pipeline:
- Generates synthetic data
- Performs cointegration + beta estimation
- Calibrates OU
- Runs single + multi-asset backtests
- Saves all plots, processed data, and results JSON

## Files Overview
- `prediction_market_arb_pipeline.py` — Main pipeline
- `ou_calibration.py` — Core OU calibration module
- `*.pkl` — Synthetic/processed data
- `*.png` — All diagnostic and equity plots
- `pipeline_results.json` — Full run summary
- `README.md`, `Specs.md` — Documentation

## Real-World Use Cases
1. **Election / Political Markets**: Calibrate spreads between Polymarket and Kalshi on the same race. Use portfolio across multiple states/events for diversified mean-reversion trading.
2. **Crypto / Macro Events**: Trade Fed rate decisions, ETF approvals, or token launches across platforms. The multi-asset version allocates across correlated outcomes.
3. **Sports / Awards**: Oscar predictions, Super Bowl props — fast mean-reversion on liquidity imbalances.
4. **Backtesting Strategy Ideas**: Load your own L2 Parquet data (mid prices) and run the pipeline to evaluate cointegration strength, reversion speed (half-life vs latency), and portfolio Sharpe.
5. **Institutional Research**: Use as a sandbox for testing order book imbalance signals + OU thresholds from the original thread.

## Future Ideas (see Specs.md)
- Full L2 order book integration (OBI + micro-price signals)
- Live API connectors (Polymarket + Kalshi)
- Dynamic risk scaling + regime detection
- ML-enhanced signals and optimal stopping
- Visualization dashboard / Streamlit UI

## Dependencies
numpy, pandas, statsmodels, matplotlib, scipy (available in the sandbox environment)

Contributions / extensions welcome via pull requests or direct edits to the scripts.

**Last updated**: June 2026
