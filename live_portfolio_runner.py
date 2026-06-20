#!/usr/bin/env python3
"""
Live Portfolio Runner with Full WebSocket Streaming + L2 OBI + Dynamic DD Scaling
=================================================================================
Real-time streaming for prediction markets (Polymarket & Kalshi).

- WebSocket connections for low-latency updates (order books, prices, trades).
- Maintains live order book state (snapshot + deltas where supported).
- Computes real-time OBI from L2 data.
- Fuses with OU-style signals (or running statistics).
- Dynamic drawdown scaling on portfolio equity.
- Simple multi-market portfolio runner (print decisions; extend to execution).

Free / low-cost: Public market data channels preferred. Auth only for private/user channels if needed.

Run:
    python live_portfolio_runner.py

Requirements: websockets, asyncio, pandas, numpy (available in env)
Internet required for live WS.

Example markets (replace with active ones):
- Kalshi: FED rate or election tickers
- Polymarket: Use slugs or token_ids for activity/price channels
"""

import asyncio
import json
import websockets
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Callable
import time

# ============================================================
# CONFIG - Replace with real active market identifiers
# ============================================================
KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/"  # or real-time-data-streaming variant
# Example subscriptions (user to customize with real slugs/tickers/token_ids)
EXAMPLE_KALSHI_MARKETS = ["FED-25DEC-T3.00", "ELECTION-2028-PRES"]  # placeholders
EXAMPLE_POLYMARKET_MARKETS = ["presidential-election-winner-2028", "some-crypto-event"]  # slugs or tokens

# ============================================================
# 1. LIVE MARKET STATE (maintains L2 book + metrics)
# ============================================================

class LiveMarketState:
    def __init__(self, market_id: str):
        self.market_id = market_id
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}
        self.last_price: Optional[float] = None
        self.last_update = datetime.now()
        self.ws_latency_history: List[float] = []  # ms between updates
        self.last_ws_timestamp = datetime.now()
        self.obi_history: List[float] = []

    def apply_snapshot(self, bids: List[dict], asks: List[dict]):
        """Apply full orderbook snapshot."""
        self.bids = {float(b['price']): float(b['size']) for b in bids}
        self.asks = {float(a['price']): float(a['size']) for a in asks}
        self._update_latency()
        self._update_obi()

    def apply_delta(self, changes: List[dict], side: str = "bids"):
        """Apply incremental delta (common in WS)."""
        book = self.bids if side == "bids" else self.asks
        for change in changes:
            price = float(change.get('price', change.get('p', 0)))
            size = float(change.get('size', change.get('s', 0)))
            if size <= 0:
                book.pop(price, None)
            else:
                book[price] = size
        self._update_latency()
        self._update_obi()

    def _update_latency(self):
        """Track WS update latency in ms."""
        now = datetime.now()
        latency_ms = (now - self.last_ws_timestamp).total_seconds() * 1000
        self.ws_latency_history.append(latency_ms)
        if len(self.ws_latency_history) > 100:
            self.ws_latency_history.pop(0)
        self.last_ws_timestamp = now

    def _update_obi(self, levels: int = 5):
        if not self.bids or not self.asks:
            obi = 0.0
        else:
            sorted_bids = sorted(self.bids.items(), reverse=True)[:levels]
            sorted_asks = sorted(self.asks.items())[:levels]
            bid_vol = sum(size for _, size in sorted_bids)
            ask_vol = sum(size for _, size in sorted_asks)
            total = bid_vol + ask_vol
            obi = (bid_vol - ask_vol) / total if total > 0 else 0.0
        self.obi_history.append(obi)
        if len(self.obi_history) > 100:
            self.obi_history.pop(0)

    @property
    def current_obi(self) -> float:
        return self.obi_history[-1] if self.obi_history else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return np.mean(self.ws_latency_history) if self.ws_latency_history else 0.0

    @property
    def mid_price(self) -> Optional[float]:
        if not self.bids or not self.asks:
            return self.last_price
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return best_bid or best_ask or self.last_price

    def get_summary(self) -> dict:
        return {
            "market": self.market_id,
            "mid": self.mid_price,
            "obi": round(self.current_obi, 4),
            "best_bid": max(self.bids.keys()) if self.bids else None,
            "best_ask": min(self.asks.keys()) if self.asks else None,
            "last_update": self.last_update.isoformat()
        }


# ============================================================
# 2. KALSHI WEBSOCKET HANDLER (Snapshot + Delta)
# ============================================================

async def kalshi_ws_handler(market_ticker: str, state: LiveMarketState, 
                            on_update: Optional[Callable] = None):
    """Connect to Kalshi WS, handle snapshot + delta for orderbook."""
    async with websockets.connect(KALSHI_WS_URL) as ws:
        # Subscribe to orderbook (auth may be needed for some channels; market data often public)
        sub_msg = {
            "id": 1,
            "type": "orderbook_snapshot",  # or subscribe command per docs
            "params": {"market_ticker": market_ticker}
        }
        await ws.send(json.dumps(sub_msg))
        print(f"[Kalshi] Subscribed to {market_ticker}")

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "orderbook_snapshot":
                    ob = data.get("msg", {})
                    state.apply_snapshot(ob.get("bids", []), ob.get("asks", []))
                elif msg_type == "orderbook_delta":
                    # Handle bids/asks deltas
                    delta = data.get("msg", {})
                    if "bids" in delta:
                        state.apply_delta(delta["bids"], "bids")
                    if "asks" in delta:
                        state.apply_delta(delta["asks"], "asks")

                state.last_update = datetime.now()
                if on_update:
                    on_update(state)

            except Exception as e:
                print(f"[Kalshi {market_ticker}] Error: {e}")
                await asyncio.sleep(5)  # Reconnect delay


# ============================================================
# 3. POLYMARKET WEBSOCKET (Activity / Price / Market Data)
# ============================================================

async def polymarket_ws_handler(market_slug_or_token: str, state: LiveMarketState,
                                on_update: Optional[Callable] = None):
    """Polymarket real-time data WS (activity, prices, trades). 
    For full orderbook, combine with periodic REST /book or CLOB WS if available."""
    # Use the real-time data streaming pattern (or CLOB WS)
    ws_url = POLYMARKET_WS_URL
    async with websockets.connect(ws_url) as ws:
        # Example subscription (adapt to actual protocol from docs/GitHub client)
        sub = {
            "subscriptions": [
                {
                    "topic": "activity",
                    "type": "trades",
                    "filters": json.dumps({"market_slug": market_slug_or_token})
                },
                {
                    "topic": "crypto_prices",  # or market prices if available
                    "type": "update",
                    "filters": json.dumps({"symbol": market_slug_or_token})
                }
            ]
        }
        await ws.send(json.dumps(sub))
        print(f"[Polymarket] Subscribed to {market_slug_or_token}")

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                # Example: update last_price from trade or price update
                if "payload" in data and "price" in str(data.get("payload", {})):
                    # Parse appropriately
                    payload = data.get("payload", {})
                    if isinstance(payload, dict) and "value" in payload:
                        state.last_price = float(payload["value"])
                # For full L2, you would also poll REST /book periodically or use dedicated CLOB WS

                state.last_update = datetime.now()
                if on_update:
                    on_update(state)

            except Exception as e:
                print(f"[Polymarket {market_slug_or_token}] Error: {e}")
                await asyncio.sleep(5)


# ============================================================
# 4. LIVE PORTFOLIO RUNNER (Multi-market + Signals + DD Scaling)
# ============================================================

class LivePortfolioRunner:
    def __init__(self, markets: List[str], platform: str = "kalshi"):
        self.states: Dict[str, LiveMarketState] = {m: LiveMarketState(m) for m in markets}
        self.positions: Dict[str, float] = {m: 0.0 for m in markets}
        self.equity = 1.0
        self.peak_equity = 1.0
        self.platform = platform
        self.running = True

    async def start(self):
        tasks = []
        for market in self.states.keys():
            state = self.states[market]
            if self.platform == "kalshi":
                tasks.append(kalshi_ws_handler(market, state, self.on_market_update))
            else:
                tasks.append(polymarket_ws_handler(market, state, self.on_market_update))
        await asyncio.gather(*tasks, return_exceptions=True)

    def on_market_update(self, state: LiveMarketState):
        """Called on every WS update. Evaluate signals + portfolio logic."""
        summary = state.get_summary()
        avg_latency = state.avg_latency_ms
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {summary} | Latency: {avg_latency:.1f}ms")

        # Simple signal example: OBI + price momentum (extend with pre-calibrated OU)
        obi = state.current_obi
        mid = state.mid_price or 0

        # Placeholder for OU z-score (in production: maintain running mean/std or load calibrated params)
        # For demo: use recent price change as proxy
        recent_change = 0  # Would come from history or last_price tracking

        # Decision with OBI confirmation
        signal = 0
        if obi > 0.2 and recent_change <= 0:  # Buying pressure + not already running up
            signal = 1
        elif obi < -0.2 and recent_change >= 0:
            signal = -1

        # Dynamic DD scaling
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        dd_scale = max(0.2, 1 - current_dd * 5) if current_dd > 0.03 else 1.0  # Example scaling

        # Update position (simplified; in real: send orders via private API)
        target_pos = signal * 2.0 * dd_scale  # Scale with DD protection
        old_pos = self.positions.get(state.market_id, 0)
        self.positions[state.market_id] = target_pos

        # Simulate P&L (replace with real fills)
        if old_pos != target_pos and mid:
            # Rough P&L on change
            self.equity += (target_pos - old_pos) * (mid * 0.001)  # Placeholder

        self.peak_equity = max(self.peak_equity, self.equity)

        if abs(target_pos) > 0.1:
            print(f"  >>> SIGNAL: {state.market_id} | Pos: {target_pos:.2f} | DD Scale: {dd_scale:.2f} | Equity: {self.equity:.4f} | Latency: {avg_latency:.1f}ms")

    async def run_forever(self):
        print("Starting Live Portfolio Runner...")
        try:
            await self.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False


# ============================================================
# MAIN DEMO
# ============================================================

async def main():
    print("=" * 70)
    print("LIVE PORTFOLIO RUNNER - WebSocket + L2 OBI + Dynamic DD Scaling")
    print("=" * 70)
    print("WARNING: This is a demonstration. Real trading requires private API auth, risk management, and capital.")
    print("Replace example markets with live ones. Monitor output for signals.\n")

    # Choose platform and markets
    platform = "kalshi"  # or "polymarket"
    markets = EXAMPLE_KALSHI_MARKETS if platform == "kalshi" else EXAMPLE_POLYMARKET_MARKETS

    runner = LivePortfolioRunner(markets, platform=platform)
    await runner.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
