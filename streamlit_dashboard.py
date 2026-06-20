#!/usr/bin/env python3
"""
Full UI Dashboard - Streamlit
=============================
Interactive dashboard for the Prediction Market Quant Arbitrage Pipeline.

Features:
- Displays all existing charts and diagnostics (PNG files)
- Live feed simulation / integration with WebSocket runner
- Interactive controls (sliders for thresholds, OBI, DD scaling)
- Stats from JSON results
- Tabs for Overview, Live Monitor, Backtest Results, Diagnostics, Portfolio
- Easy to run: streamlit run streamlit_dashboard.py

Install: pip install streamlit  (free, lightweight)
Run: cd /home/workdir/artifacts && streamlit run streamlit_dashboard.py

Integrates:
- Existing plots from pipeline runs
- live_l2_obi_integration.py (OBI + DD scaling)
- live_portfolio_runner.py concepts (live state)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import numpy as np

# Optional imports for live simulation
try:
    from live_l2_obi_integration import (
        fetch_polymarket_orderbook, compute_obi, 
        dynamic_drawdown_scale, enhanced_backtest_with_obi_and_dd
    )
    LIVE_AVAILABLE = True
except ImportError:
    LIVE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Prediction Market Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json_results():
    """Load the latest pipeline results JSON."""
    json_path = "pipeline_results.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def display_image_safe(image_path, caption="", width=None):
    """Safely display a PNG if it exists."""
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width=width)
    else:
        st.warning(f"Image not found: {image_path}")

def get_available_plots():
    """List all PNG plots in the directory."""
    plots = [f for f in os.listdir('.') if f.endswith('.png')]
    return sorted(plots)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("⚙️ Controls & Settings")

st.sidebar.header("Signal Parameters")
entry_z = st.sidebar.slider("Entry Z-Score Threshold", 0.5, 3.0, 1.1, 0.1)
exit_z = st.sidebar.slider("Exit Z-Score Threshold", 0.1, 1.0, 0.2, 0.05)
obi_threshold = st.sidebar.slider("OBI Confirmation Threshold", 0.0, 0.5, 0.15, 0.05)
max_dd_threshold = st.sidebar.slider("Max Drawdown Threshold (%)", 1, 20, 8, 1) / 100
dd_scale_factor = st.sidebar.slider("DD Scale Factor", 0.3, 0.9, 0.6, 0.05)

st.sidebar.header("Portfolio Settings")
max_total_exposure = st.sidebar.slider("Max Total Exposure", 2.0, 12.0, 6.0, 0.5)
allocation_method = st.sidebar.selectbox(
    "Allocation Method", 
    ["min_variance", "equal_risk", "equal_weight"]
)

st.sidebar.header("Live / Simulation")
use_live = st.sidebar.checkbox("Enable Live Data Mode (requires internet)", value=False)
selected_market = st.sidebar.text_input("Market Token/Ticker (for live)", "example-token-or-ticker")

if st.sidebar.button("🔄 Refresh All Data"):
    st.rerun()

# ============================================================
# MAIN HEADER
# ============================================================

st.title("📊 Prediction Market Quant Arbitrage Dashboard")
st.markdown("""
**Real-time monitoring + Backtesting + Live Signals**  
Built on OU calibration, L2 OBI, cointegration, and dynamic risk scaling.
""")

# Load results
results = load_json_results()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview & Live Monitor", 
    "🔄 Backtest Results", 
    "🔍 Diagnostics & Charts", 
    "💼 Portfolio & Multi-Asset", 
    "⚡ Live Controls & Execution"
])

# ============================================================
# TAB 1: OVERVIEW + LIVE MONITOR
# ============================================================

with tab1:
    st.header("Overview & Live Market Monitor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Metrics (Latest Run)")
        if results:
            st.metric("Johansen Cointegrated", "Yes" if results.get('cointegration', {}).get('johansen', {}).get('is_cointegrated_johansen') else "No")
            st.metric("Beta Used", f"{results.get('beta_used', 0):.4f}")
            ou = results.get('ou_calibration', {})
            st.metric("OU Theta (Reversion)", f"{ou.get('theta', 0):.2f}")
            st.metric("OU R²", f"{ou.get('r_squared', 0):.4f}")
        else:
            st.info("Run the pipeline first to see metrics.")
    
    with col2:
        st.subheader("Live Market Snapshot")
        if use_live and LIVE_AVAILABLE:
            if st.button("Fetch Live Order Book"):
                with st.spinner("Fetching live data..."):
                    bids, asks = fetch_polymarket_orderbook(selected_market)
                    if not bids.empty:
                        obi = compute_obi(bids, asks, levels=5)
                        st.success(f"Live OBI: {obi:.4f}")
                        st.dataframe(bids.head(5), use_container_width=True)
                        st.dataframe(asks.head(5), use_container_width=True)
                    else:
                        st.warning("Live fetch failed or invalid token. Using simulation.")
        else:
            st.info("Enable 'Live Data Mode' in sidebar and provide a real token/ticker for live fetch.")
            # Simulated live display
            st.metric("Simulated Mid Price", "0.512")
            st.metric("Simulated OBI (Live)", f"{np.random.uniform(-0.4, 0.4):.3f}")
    
    st.divider()
    
    # Quick equity preview
    st.subheader("Latest Portfolio Equity Curve")
    display_image_safe("portfolio_equity_curve.png", "Multi-Asset Portfolio Equity (Min-Variance)", width=700)

# ============================================================
# TAB 2: BACKTEST RESULTS
# ============================================================

with tab2:
    st.header("Backtest Results")
    
    if results:
        st.json(results.get('portfolio_demo', {}))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Single-Asset Backtest Equity")
        display_image_safe("backtest_equity_curve.png", width=600)
    
    with col2:
        st.subheader("Multi-Asset Portfolio Equity")
        display_image_safe("portfolio_equity_curve.png", width=600)
    
    st.subheader("Key Backtest Stats")
    # Placeholder stats (in real use, parse from results)
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Total Return", "Simulated: +X%")
    stats_col2.metric("Sharpe Ratio", "Y.YY")
    stats_col3.metric("Max Drawdown", "Z.Z%")

# ============================================================
# TAB 3: DIAGNOSTICS & CHARTS
# ============================================================

with tab3:
    st.header("Full Diagnostics & Charts")
    
    st.subheader("Price Overlay & Spread Analysis")
    col1, col2 = st.columns(2)
    with col1:
        display_image_safe("prices_overlay.png", width=550)
    with col2:
        display_image_safe("spread_ou_mean.png", width=550)
    
    st.subheader("Residual & Statistical Diagnostics")
    display_image_safe("residual_diagnostics.png", width=800)
    
    st.subheader("Rolling Beta & Stability")
    display_image_safe("rolling_beta.png", width=700)
    
    st.subheader("Spread Distribution")
    display_image_safe("spread_histogram.png", width=550)

# ============================================================
# TAB 4: PORTFOLIO & MULTI-ASSET
# ============================================================

with tab4:
    st.header("Multi-Asset Portfolio View")
    
    st.subheader("Allocation & Weights")
    if results and 'portfolio_demo' in results:
        weights = results['portfolio_demo'].get('weights', {})
        if weights:
            st.bar_chart(pd.Series(weights))
    
    st.subheader("Portfolio Equity Curve")
    display_image_safe("portfolio_equity_curve.png", width=700)
    
    st.subheader("Individual Market Equities (from latest run)")
    # Show any other equity plots if available
    for plot in get_available_plots():
        if "equity" in plot.lower() and "portfolio" not in plot.lower():
            display_image_safe(plot, width=500)

# ============================================================
# TAB 5: LIVE CONTROLS & EXECUTION
# ============================================================

with tab5:
    st.header("Live Controls & Execution (Demo)")
    
    st.warning("This tab demonstrates live integration. Real execution requires authenticated private APIs.")
    
    st.subheader("Live Parameters (from Sidebar)")
    st.write(f"Entry Z: {entry_z} | Exit Z: {exit_z} | OBI Threshold: {obi_threshold}")
    st.write(f"Max DD Threshold: {max_dd_threshold*100:.1f}% | DD Scale: {dd_scale_factor}")
    
    if st.button("▶️ Run Enhanced Backtest with Current Params (Simulated)"):
        with st.spinner("Running enhanced backtest..."):
            # Simulate using dummy data (in real: load real spread + OU)
            np.random.seed(42)
            dummy_spread = pd.Series(np.cumsum(np.random.randn(3000)*0.001))
            dummy_ou = {'mu': 0.0, 'sigma': 0.012}
            dummy_obi = pd.Series(np.random.uniform(-0.6, 0.6, 3000))
            
            result = enhanced_backtest_with_obi_and_dd(
                dummy_spread, dummy_ou, obi_series=dummy_obi,
                entry_z=entry_z, exit_z=exit_z, 
                obi_threshold=obi_threshold,
                max_dd_threshold=max_dd_threshold,
                dd_scale_factor=dd_scale_factor
            )
            st.success(f"Simulated Return: {result['total_return']*100:.2f}%")
            st.metric("Final Equity", f"{result['final_equity']:.4f}")
            st.metric("Max DD", f"{result['max_dd']*100:.2f}%")
    
    st.subheader("Live WebSocket Status (Demo)")
    if st.button("🔌 Connect to Live WS (Demo Mode)"):
        st.info("In a real deployment, this would start the async WebSocket connections from live_portfolio_runner.py and stream updates here.")
        st.write("Simulated live updates would appear below with OBI, positions, and DD scaling in action.")
    
    st.caption("For full live execution: Extend with authenticated order placement from Polymarket/Kalshi private APIs.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Dashboard powered by the Prediction Market Quant Pipeline • All charts auto-generated from pipeline runs • Live mode uses free public APIs + WebSocket")

# Optional: Auto-refresh note
st.caption("Tip: Use `streamlit run streamlit_dashboard.py --server.runOnSave true` for auto-reload during development.")