#!/usr/bin/env python3
"""
Live API Integration + Full L2 OBI + Dynamic Drawdown Scaling
=============================================================
Free public endpoints only (no API keys/cost for market data & order books).

- Polymarket CLOB: https://clob.polymarket.com (public reads)
- Kalshi: external-api.kalshi.com (public market data)

Features:
- Fetch live order books (L2)
- Compute Order Book Imbalance (OBI) at multiple depth levels
- Integrate OBI into signals (filter or combine with OU z-score)
- Dynamic drawdown-based position scaling in backtests
- Example live monitor + backtest enhancement

Run: python live_l2_obi_integration.py
(Requires internet for live fetches; works with synthetic for offline)

Dependencies: requests, pandas, numpy (available)
"""

import requests
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import time

# ============================================================
# 1. LIVE FREE API FETCHERS (Polymarket primary, Kalshi similar)
# ============================================================

def fetch_polymarket_orderbook(token_id: str, timeout: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch current L2 order book from Polymarket CLOB (free, no auth).
    Returns (bids_df, asks_df) with columns ['price', 'size']
    """
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        bids = pd.DataFrame(data.get("bids", []), columns=["price", "size"])
        asks = pd.DataFrame(data.get("asks", []), columns=["price", "size"])
        bids = bids.astype(float)
        asks = asks.astype(float)
        return bids.sort_values("price", ascending=False), asks.sort_values("price")
    except Exception as e:
        print(f"Error fetching Polymarket book for {token_id}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def fetch_kalshi_orderbook(ticker: str, timeout: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch current order book from Kalshi (public market data endpoint).
    Returns (yes_bids, yes_asks) or similar – adjust parsing as needed.
    Note: Some endpoints may benefit from headers; this is read-only market data.
    """
    url = f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Typical structure: {'orderbook': {'yes': [...], 'no': [...]}}
        ob = data.get("orderbook", {})
        # Example parsing – adapt to actual response structure
        yes_bids = pd.DataFrame(ob.get("yes", []))  # or bids/asks depending on API
        # For simplicity, assume similar price/size; real parsing may differ
        return yes_bids, pd.DataFrame()  # Placeholder – extend parsing
    except Exception as e:
        print(f"Error fetching Kalshi book for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ============================================================
# 2. FULL L2 ORDER BOOK IMBALANCE (OBI)
# ============================================================

def compute_obi(bids: pd.DataFrame, asks: pd.DataFrame, 
                levels: int = 5, 
                price_col: str = "price", 
                size_col: str = "size") -> float:
    """
    Compute Order Book Imbalance from L2 data.
    OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    at top N levels. Positive = buying pressure.
    """
    if bids.empty or asks.empty:
        return 0.0
    
    bid_vol = bids.head(levels)[size_col].astype(float).sum()
    ask_vol = asks.head(levels)[size_col].astype(float).sum()
    total = bid_vol + ask_vol
    
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def compute_weighted_obi(bids: pd.DataFrame, asks: pd.DataFrame, 
                         levels: int = 10) -> float:
    """
    Weighted OBI giving more weight to closer-to-mid levels.
    """
    if bids.empty or asks.empty:
        return 0.0
    
    mid = (bids.iloc[0]['price'] + asks.iloc[0]['price']) / 2 if not bids.empty and not asks.empty else 0
    
    def weighted_vol(df, is_bid=True):
        df = df.head(levels).copy()
        df['dist'] = np.abs(df['price'] - mid)
        weights = 1 / (1 + df['dist'])  # closer = higher weight
        return (df['size'].astype(float) * weights).sum()
    
    bid_w = weighted_vol(bids, True)
    ask_w = weighted_vol(asks, False)
    total = bid_w + ask_w
    return (bid_w - ask_w) / total if total > 0 else 0.0


# ============================================================
# 3. DYNAMIC DRAWDOWN SCALING
# ============================================================

def dynamic_drawdown_scale(current_equity: float, 
                           peak_equity: float, 
                           max_dd_threshold: float = 0.05,
                           scale_factor: float = 0.5) -> float:
    """
    Compute position scaling factor based on current drawdown.
    If current DD > threshold, reduce exposure.
    """
    if peak_equity <= 0:
        return 1.0
    current_dd = (peak_equity - current_equity) / peak_equity
    if current_dd > max_dd_threshold:
        # Linear scaling down
        excess_dd = current_dd - max_dd_threshold
        scale = max(0.1, 1 - (excess_dd / (1 - max_dd_threshold)) * (1 - scale_factor))
        return scale
    return 1.0


# ============================================================
# 4. ENHANCED BACKTEST WITH OBI + DYNAMIC DD SCALING
# ============================================================

def enhanced_backtest_with_obi_and_dd(
    spread: pd.Series,
    ou_params: dict,
    obi_series: Optional[pd.Series] = None,  # Optional precomputed OBI aligned to spread
    entry_z: float = 1.1,
    exit_z: float = 0.2,
    obi_threshold: float = 0.2,  # Only trade if |OBI| confirms direction
    max_dd_threshold: float = 0.08,
    dd_scale_factor: float = 0.6,
    **kwargs
) -> dict:
    """
    Enhanced single-asset backtest combining:
    - OU z-score + momentum (existing)
    - L2 OBI confirmation
    - Dynamic drawdown scaling
    """
    s = spread.dropna().copy()
    mu = ou_params.get('mu', s.mean())
    sigma = ou_params.get('sigma', s.std() or 1.0)
    z = (s - mu) / sigma
    
    if obi_series is not None:
        obi_aligned = obi_series.reindex(s.index).fillna(0)
    else:
        obi_aligned = pd.Series(0.0, index=s.index)  # No OBI -> neutral
    
    position = np.zeros(len(s), dtype=float)
    current_pos = 0.0
    peak_equity = 1.0
    equity = [1.0]
    
    for i in range(1, len(s)):
        z_val = z.iloc[i]
        obi_val = obi_aligned.iloc[i]
        prev_equity = equity[-1]
        peak_equity = max(peak_equity, prev_equity)
        
        # Dynamic DD scale
        dd_scale = dynamic_drawdown_scale(prev_equity, peak_equity, 
                                          max_dd_threshold, dd_scale_factor)
        
        # Signal logic (OU + momentum + OBI confirmation)
        # Example: Long only if z low AND OBI positive (buying pressure)
        long_signal = (z_val < -entry_z) and (obi_val > obi_threshold)
        short_signal = (z_val > entry_z) and (obi_val < -obi_threshold)
        
        if current_pos == 0:
            if long_signal:
                current_pos = min(4.0, abs(z_val) / entry_z * 2.0) * dd_scale
            elif short_signal:
                current_pos = -min(4.0, abs(z_val) / entry_z * 2.0) * dd_scale
        else:
            # Exit or scale down on DD
            exit_cond = abs(z_val) < exit_z or abs(obi_val) < 0.05
            if exit_cond:
                current_pos = 0.0
            else:
                # Dynamically scale existing position on DD
                current_pos *= dd_scale
        
        position[i] = current_pos
        
        # Update equity (simplified)
        ret = current_pos * (s.iloc[i] - s.iloc[i-1])
        new_equity = equity[-1] * (1 + ret)
        equity.append(new_equity)
    
    equity = pd.Series(equity, index=s.index)
    total_return = equity.iloc[-1] - 1.0
    # ... (add more metrics as before)
    
    return {
        'total_return': float(total_return),
        'final_equity': float(equity.iloc[-1]),
        'max_dd': float((equity / equity.cummax() - 1).min()),
        'positions': position,
        'equity_curve': equity
    }


# ============================================================
# 5. EXAMPLE USAGE / DEMO
# ============================================================

if __name__ == "__main__":
    print("=== Live L2 OBI + Dynamic DD Demo ===")
    print("Note: Live fetches require internet. Falling back to synthetic if needed.\n")
    
    # Example live fetch (replace token_id with real one, e.g. from a current market)
    example_token = "0x..."  # Get real token_id from Polymarket market page or API
    print("Attempting live Polymarket order book fetch (example token)...")
    bids, asks = fetch_polymarket_orderbook(example_token)
    if not bids.empty:
        obi = compute_obi(bids, asks, levels=5)
        print(f"Live OBI (top 5 levels): {obi:.4f}")
        print(f"Top bids sample:\n{bids.head(3)}")
    else:
        print("Live fetch skipped or failed (no internet / invalid token). Using synthetic example.")
    
    # Synthetic OBI example for backtest
    print("\nRunning enhanced backtest with simulated OBI + DD scaling...")
    # Assume 'spread' and 'ou_params' from main pipeline
    # For demo, create dummy data
    np.random.seed(42)
    dummy_spread = pd.Series(np.cumsum(np.random.randn(5000) * 0.001) + 0.01)
    dummy_ou = {'mu': 0.0, 'sigma': 0.015}
    dummy_obi = pd.Series(np.random.uniform(-0.5, 0.5, 5000))  # Simulated OBI
    
    result = enhanced_backtest_with_obi_and_dd(
        dummy_spread, dummy_ou, obi_series=dummy_obi,
        obi_threshold=0.15, max_dd_threshold=0.06
    )
    print(f"Enhanced backtest result: Return={result['total_return']*100:.2f}%, MaxDD={result['max_dd']*100:.2f}%")
    
    print("\nIntegration complete. Import functions into your pipeline for live OBI signals and DD scaling.")
