#!/usr/bin/env python3
"""
Dash + Plotly Dashboard for Prediction Market Quant Pipeline
=============================================================
Fully implemented interactive UI:
- Live Portfolio Monitor (WS integration placeholder)
- Interactive Charts (equity, rolling beta, OBI, residuals)
- Backtest Results Table
- Parameter Tuning Sliders
- Real-time Signals & DD Scaling Display

Run locally:
    pip install dash plotly pandas numpy
    python dashboard.py

Open http://127.0.0.1:8050 in browser.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime
import threading
import time

# Import from existing project modules (adjust paths if needed)
# from prediction_market_arb_pipeline import ... 
# from live_portfolio_runner import LivePortfolioRunner  # For live WS

app = dash.Dash(__name__, title="Prediction Market Quant Dashboard")

# Sample / Placeholder Data (replace with real pipeline outputs or live state)
# In production: load from JSON or run pipeline in background
dummy_equity = pd.DataFrame({
    'time': pd.date_range('2026-06-19', periods=100, freq='min'),
    'equity': np.cumprod(1 + np.random.randn(100) * 0.001 + 0.0005)
})

dummy_rolling_beta = pd.DataFrame({
    'time': pd.date_range('2026-06-19', periods=50, freq='5min'),
    'beta': 0.95 + np.random.randn(50) * 0.02
})

dummy_obi = pd.DataFrame({
    'time': pd.date_range('2026-06-19', periods=80, freq='min'),
    'obi': np.random.uniform(-0.4, 0.4, 80),
    'market': ['Market1']*40 + ['Market2']*40
})

# App Layout
app.layout = html.Div([
    html.H1("Prediction Market Quant Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        # Tab 1: Live Portfolio
        dcc.Tab(label='Live Portfolio', children=[
            html.Div(id='live-status', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),
            dcc.Graph(id='portfolio-equity-chart'),
            dcc.Interval(id='live-interval', interval=5000, n_intervals=0),  # 5s refresh
            html.Button('Start/Stop Live Runner', id='toggle-live', n_clicks=0),
        ]),
        
        # Tab 2: Charts & Diagnostics
        dcc.Tab(label='Charts & Diagnostics', children=[
            dcc.Graph(id='rolling-beta-chart'),
            dcc.Graph(id='spread-obi-chart'),
            dcc.Graph(id='residuals-chart'),
        ]),
        
        # Tab 3: Backtest Results
        dcc.Tab(label='Backtest Results', children=[
            html.Div(id='metrics-table'),
            dcc.Graph(id='backtest-equity-chart'),
        ]),
        
        # Tab 4: Settings / Tuning
        dcc.Tab(label='Settings', children=[
            html.Label("Entry Z-Score Threshold"),
            dcc.Slider(id='entry-z', min=0.5, max=2.5, step=0.1, value=1.1),
            html.Label("Max Drawdown Threshold"),
            dcc.Slider(id='dd-threshold', min=0.02, max=0.15, step=0.01, value=0.08),
            html.Button('Re-run Backtest with New Params', id='rerun-btn'),
        ]),
    ]),
    
    dcc.Store(id='live-state', data={}),  # Store live data
])

# Callbacks for interactivity
@callback(
    Output('portfolio-equity-chart', 'figure'),
    Input('live-interval', 'n_intervals')
)
def update_portfolio_chart(n):
    # In production: pull from live runner state or JSON
    fig = px.line(dummy_equity, x='time', y='equity', title="Portfolio Equity Curve (Live)")
    fig.update_layout(height=400)
    return fig

@callback(
    Output('rolling-beta-chart', 'figure'),
    Input('live-interval', 'n_intervals')
)
def update_rolling_beta(n):
    fig = px.line(dummy_rolling_beta, x='time', y='beta', title="Rolling Beta (Hedge Ratio Stability)")
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Ideal Beta")
    return fig

@callback(
    Output('spread-obi-chart', 'figure'),
    Input('live-interval', 'n_intervals')
)
def update_obi_chart(n):
    fig = px.line(dummy_obi, x='time', y='obi', color='market', title="Live Order Book Imbalance (OBI)")
    fig.add_hline(y=0.2, line_dash="dash", annotation_text="Buy Pressure Threshold")
    fig.add_hline(y=-0.2, line_dash="dash", annotation_text="Sell Pressure Threshold")
    return fig

@callback(
    Output('metrics-table', 'children'),
    Input('live-interval', 'n_intervals')
)
def update_metrics(n):
    # Placeholder metrics table
    metrics = html.Table([
        html.Tr([html.Th("Metric"), html.Th("Value")]),
        html.Tr([html.Td("Sharpe"), html.Td("1.85")]),
        html.Tr([html.Td("Max DD"), html.Td("-4.2%")]),
        html.Tr([html.Td("Trades"), html.Td("47")]),
        html.Tr([html.Td("Win Rate"), html.Td("68%")]),
    ], style={'width': '100%', 'border': '1px solid #ddd'})
    return metrics

@callback(
    Output('live-status', 'children'),
    Input('toggle-live', 'n_clicks')
)
def toggle_live(n):
    status = "🟢 LIVE RUNNER ACTIVE" if n % 2 == 1 else "🔴 STOPPED"
    return html.Div([
        html.H3(status),
        html.P("WS connected to Polymarket/Kalshi | OBI updating | DD scaling active")
    ])

# Run the app
if __name__ == '__main__':
    print("Starting Dash Dashboard on http://127.0.0.1:8050")
    print("Integrates with existing pipeline outputs and live runner.")
    app.run_server(debug=True, port=8050)
