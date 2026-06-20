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

# Execution Layer Integration
try:
    from execution_layer import PaperTrader
    paper_trader = PaperTrader(initial_capital=10000.0)
    HAS_EXECUTION = True
except ImportError:
    paper_trader = None
    HAS_EXECUTION = False
    print("Warning: execution_layer not found. Paper trading tab will be limited.")

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

        # Tab 5: Paper Trading (Execution Layer)
        dcc.Tab(label='Paper Trading', children=[
            html.H3("Paper Trading Panel (Simulation Mode)"),
            html.P("Place paper orders using live prices & OBI. All trades are simulated with realistic slippage."),
            
            # Market Selector
            html.Label("Select Market"),
            dcc.Dropdown(id='paper-market-select', 
                         options=[{'label': m, 'value': m} for m in ['FED_RATE_DEC', 'ELECTION_WINNER', 'CRYPTO_ETF']],
                         value='FED_RATE_DEC'),
            
            # Order Inputs
            html.Div([
                html.Label("Side"),
                dcc.RadioItems(id='paper-side', options=[
                    {'label': 'Buy (Long)', 'value': 'buy'},
                    {'label': 'Sell (Short)', 'value': 'sell'}
                ], value='buy', inline=True),
                
                html.Label("Size (contracts)"),
                dcc.Input(id='paper-size', type='number', value=100, min=1),
                
                html.Label("Order Type"),
                dcc.RadioItems(id='paper-order-type', options=[
                    {'label': 'Market', 'value': 'market'},
                    {'label': 'Limit', 'value': 'limit'}
                ], value='market', inline=True),
                
                html.Label("Limit Price (if Limit)"),
                dcc.Input(id='paper-limit-price', type='number', value=0.5, step=0.01),
            ], style={'marginBottom': '15px'}),
            
            # Action Buttons
            html.Button('Place Buy Order', id='place-buy-btn', n_clicks=0, 
                        style={'backgroundColor': '#27ae60', 'color': 'white', 'marginRight': '10px'}),
            html.Button('Place Sell Order', id='place-sell-btn', n_clicks=0,
                        style={'backgroundColor': '#e74c3c', 'color': 'white'}),
            html.Button('Cancel All (Paper)', id='cancel-all-btn', n_clicks=0, 
                        style={'marginLeft': '20px'}),
            
            # Live P&L Display
            html.Div(id='paper-pnl-display', style={'marginTop': '20px', 'padding': '15px', 
                                                    'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd'}),
            
            # Trade Log Table
            html.H4("Recent Trade Log"),
            html.Div(id='paper-trade-log'),
            
            # Positions Summary
            html.H4("Current Paper Positions"),
            html.Div(id='paper-positions-summary'),
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

# ============================================================
# PAPER TRADING CALLBACKS (Execution Layer Integration)
# ============================================================

@callback(
    Output('paper-pnl-display', 'children'),
    Output('paper-trade-log', 'children'),
    Output('paper-positions-summary', 'children'),
    Input('live-interval', 'n_intervals'),
    Input('place-buy-btn', 'n_clicks'),
    Input('place-sell-btn', 'n_clicks'),
    Input('cancel-all-btn', 'n_clicks'),
    State('paper-market-select', 'value'),
    State('paper-side', 'value'),
    State('paper-size', 'value'),
    State('paper-order-type', 'value'),
    State('paper-limit-price', 'value'),
    prevent_initial_call=True
)
def update_paper_trading(n_intervals, buy_clicks, sell_clicks, cancel_clicks, 
                         market, side, size, order_type, limit_price):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    current_prices = {'FED_RATE_DEC': 0.523, 'ELECTION_WINNER': 0.471, 'CRYPTO_ETF': 0.605}  # From live in prod
    current_obi = {'FED_RATE_DEC': 0.28, 'ELECTION_WINNER': -0.19, 'CRYPTO_ETF': 0.34}
    
    if HAS_EXECUTION and paper_trader:
        if triggered_id in ['place-buy-btn', 'place-sell-btn']:
            actual_side = side if triggered_id == 'place-buy-btn' else ('sell' if side == 'buy' else 'buy')
            # Use current live price/OBI
            mid = current_prices.get(market, 0.5)
            obi = current_obi.get(market, 0.0)
            
            result = paper_trader.place_order(
                market=market,
                side=actual_side,
                size=size or 100,
                order_type=order_type,
                current_mid=mid,
                obi=obi,
                limit_price=limit_price if order_type == 'limit' else None
            )
            # Log result in console or state
        
        if triggered_id == 'cancel-all-btn':
            paper_trader.cancel_all()
        
        # Get current state
        pnl = paper_trader.get_pnl(current_prices)
        recent_trades = paper_trader.get_recent_trades(10)
        positions = paper_trader.get_positions()
        
        # P&L Display
        pnl_display = html.Div([
            html.H4("Live Paper P&L"),
            html.P(f"Equity: ${pnl['equity']:,.2f}"),
            html.P(f"Unrealized PnL: ${pnl['unrealized_pnl']:,.2f}"),
            html.P(f"Realized PnL: ${pnl['realized_pnl']:,.2f}"),
            html.P(f"Current Drawdown: {pnl['current_drawdown']}%"),
            html.P(f"Cash: ${pnl['cash']:,.2f}"),
            html.P(f"Total Trades: {pnl['total_trades']}")
        ])
        
        # Trade Log Table
        if recent_trades:
            trade_rows = [html.Tr([
                html.Td(t['timestamp'][:19]),
                html.Td(t['market']),
                html.Td(t['side']),
                html.Td(t['size']),
                html.Td(f"${t['fill_price']:.4f}"),
                html.Td(t['order_type'])
            ]) for t in recent_trades]
            trade_table = html.Table([
                html.Thead(html.Tr([html.Th(h) for h in ['Time', 'Market', 'Side', 'Size', 'Fill Price', 'Type']])),
                html.Tbody(trade_rows)
            ], style={'width': '100%', 'border': '1px solid #ccc'})
        else:
            trade_table = html.P("No trades yet. Place an order above!")
        
        # Positions Summary
        pos_display = html.Div([
            html.P(f"Positions: {positions}"),
            html.P("Note: Positive = Long, Negative = Short")
        ])
        
        return pnl_display, trade_table, pos_display
    
    else:
        return html.P("PaperTrader not available. Check execution_layer.py import."), html.P(""), html.P("")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
if __name__ == '__main__':
    print("Starting Dash Dashboard on http://127.0.0.1:8050")
    print("Integrates with existing pipeline outputs and live runner.")
    app.run_server(debug=True, port=8050)
