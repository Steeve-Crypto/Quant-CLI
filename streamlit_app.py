#!/usr/bin/env python3
"""
Streamlit Dashboard for Prediction Market Quant Pipeline
=======================================================
Full UI with existing charts, live feed integration, backtest results, and controls.

Run locally (after pip install streamlit):
    streamlit run streamlit_app.py

Features:
- Live portfolio monitoring (from WS runner)
- Interactive charts from pipeline outputs
- Parameter tuning sliders
- Diagnostics & statistics tables
- Multi-tab layout

Assumes artifacts/ files are present from pipeline runs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path

# Import from project modules (adjust paths if needed)
import sys
sys.path.append('.')

try:
    from prediction_market_arb_pipeline import run_prediction_market_arb_pipeline
    from live_l2_obi_integration import compute_obi
    # from live_portfolio_runner import LivePortfolioRunner  # Run separately for full live
except ImportError:
    st.warning("Some modules not found - running in limited mode.")

st.set_page_config(page_title="Prediction Market Quant Dashboard", layout="wide")
st.title("Prediction Market Quant Arbitrage Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
run_pipeline = st.sidebar.button("Run Full Pipeline (Synthetic)")
refresh_live = st.sidebar.button("Refresh Live Data")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Portfolio", "Backtest & Diagnostics", "Settings"])

# Load latest results
artifacts = Path("/home/workdir/artifacts")
results_file = artifacts / "pipeline_results.json"
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
else:
    results = {}

with tab1:
    st.header("Overview")
    st.markdown("**Real-time quant arbitrage pipeline for Polymarket/Kalshi**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Markets Monitored", "3 (demo)")
    col2.metric("Latest Equity", f"{results.get('portfolio_demo', {}).get('portfolio_metrics', {}).get('total_return', 0)*100:.1f}%")
    col3.metric("Active Signals", "2")  # Placeholder

    if artifacts.glob("*.png"):
        st.subheader("Recent Charts")
        for img in list(artifacts.glob("*.png"))[:6]:
            st.image(str(img), caption=img.name, use_column_width=True)

with tab2:
    st.header("Live Portfolio")
    st.info("Run live_portfolio_runner.py in another terminal for full WebSocket feed.")
    st.subheader("Current Status (from latest run)")
    if 'portfolio_demo' in results:
        port = results['portfolio_demo'].get('portfolio_metrics', {})
        st.metric("Portfolio Return", f"{port.get('total_return', 0)*100:.2f}%")
        st.metric("Sharpe", f"{port.get('sharpe', 0):.2f}")
        st.metric("Max DD", f"{port.get('max_drawdown', 0)*100:.2f}%")
        
        # Placeholder live table (update with real state file in production)
        st.dataframe(pd.DataFrame({
            'Market': ['market_1', 'market_2', 'market_3'],
            'Position': [1.2, -0.8, 0.5],
            'Current OBI': [0.28, -0.15, 0.42]
        }))
    else:
        st.write("No live data yet. Run the pipeline or live runner.")

    if refresh_live:
        st.rerun()

with tab3:
    st.header("Backtest & Diagnostics")
    st.subheader("Key Charts")
    for img_name in ["portfolio_equity_curve.png", "spread_ou_mean.png", "rolling_beta.png", "residual_diagnostics.png"]:
        img_path = artifacts / img_name
        if img_path.exists():
            st.image(str(img_path), caption=img_name.replace('.png', '').replace('_', ' ').title())

    st.subheader("Statistics")
    if results:
        st.json(results.get('portfolio_demo', {}))
        st.json(results.get('ou_calibration', {}))

with tab4:
    st.header("Settings & Tuning")
    entry_z = st.slider("Entry Z-Score Threshold", 0.5, 3.0, 1.1)
    dd_threshold = st.slider("Max DD Threshold for Scaling", 0.01, 0.20, 0.05)
    st.button("Re-run Pipeline with New Params")  # Can hook to run_pipeline function

    st.markdown("**Real-World Tips**")
    st.markdown("- Use live token_ids/tickers in `live_portfolio_runner.py`")
    - Combine with `live_l2_obi_integration.py` for OBI signals")

if run_pipeline:
    with st.spinner("Running full pipeline..."):
        run_prediction_market_arb_pipeline()
    st.success("Pipeline complete! Refresh charts.")
    st.rerun()

st.caption("Built on the prediction market quant pipeline. All artifacts in /home/workdir/artifacts/")
