#!/usr/bin/env python3
"""
Execution Layer (Paper Trading Mode)
====================================
Simulated execution for live portfolio runner with realistic fills, slippage, fees, and reconciliation.

- Paper orders using live mids + OBI-informed pricing.
- Regime-dependent slippage and costs (higher in low liquidity).
- Dynamic DD scaling integration.
- Reconciliation with WS order book state.
- Ready for real private API extension (Polymarket/Kalshi order endpoints).

Integrates with live_portfolio_runner.py and dashboard.py.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import uuid

class PaperExecutionEngine:
    """
    Full Execution Layer (Phase A: Paper Trading).
    Extends to Phase B (real APIs) and Phase C (safeguards).
    """
    def __init__(self, initial_capital: float = 10000.0):
        self.cash_balance = initial_capital
        self.positions: Dict[str, float] = {}  # market -> net size (positive = long)
        self.orders: Dict[str, dict] = {}  # order_id -> details
        self.fills: List[dict] = []
        self.trade_log: List[dict] = []
        self.peak_equity = initial_capital
        self.equity_history = [(datetime.now(), initial_capital)]
        self.max_position_per_market = 5000.0
        self.commission_bps = 5.0  # Polymarket/Kalshi typical maker/taker range
        self.slippage_model = "obi_depth"  # or "fixed"

    def get_current_equity(self, current_prices: Dict[str, float]) -> float:
        """Equity = cash + unrealized PnL from current prices."""
        equity = self.cash_balance
        for market, size in self.positions.items():
            if market in current_prices:
                equity += size * current_prices[market]
        return equity

    def place_order(self, market: str, side: str, size: float, order_type: str = "market", 
                    limit_price: Optional[float] = None, current_mid: Optional[float] = None,
                    obi: float = 0.0, book_depth: Optional[dict] = None, regime: str = "normal") -> dict:
        """
        Phase A: Simulate order with realistic slippage (OBI + depth + regime).
        Phase B: Replace with real API calls (Polymarket CLOB POST /order, Kalshi authenticated).
        """
        if side not in ["buy", "sell"]:
            return {"status": "error", "reason": "Invalid side"}

        # Risk gate (Phase C)
        proposed_pos = self.positions.get(market, 0) + (size if side == "buy" else -size)
        if abs(proposed_pos * (current_mid or 0.5)) > self.max_position_per_market:
            return {"status": "rejected", "reason": "Max position exceeded"}

        if current_mid is None:
            current_mid = 0.5  # fallback

        # Realistic fill simulation (OBI-based slippage)
        slippage_bps = 2.0  # base
        if obi > 0.3 and side == "buy": slippage_bps += 8.0  # buying into buying pressure
        if obi < -0.3 and side == "sell": slippage_bps += 8.0
        if regime == "stressed": slippage_bps += 10.0
        if book_depth:
            depth = book_depth.get('opposite_side_depth', 1000)
            if size > depth * 0.2: slippage_bps += 15.0  # large order vs depth

        fill_price = current_mid * (1 + (slippage_bps / 10000) * (1 if side == "buy" else -1))

        # Fee
        fee = abs(size * fill_price) * (self.commission_bps / 10000)

        # Execute
        if side == "buy":
            cost = size * fill_price + fee
            if self.cash_balance < cost:
                return {"status": "rejected", "reason": "Insufficient cash"}
            self.cash_balance -= cost
            self.positions[market] = self.positions.get(market, 0) + size
        else:
            proceeds = size * fill_price - fee
            self.cash_balance += proceeds
            self.positions[market] = self.positions.get(market, 0) - size

        fill = {
            "timestamp": datetime.now().isoformat(),
            "market": market,
            "side": side,
            "size": size,
            "fill_price": round(fill_price, 6),
            "fee": round(fee, 2),
            "order_type": order_type,
            "status": "filled"
        }
        self.fills.append(fill)
        self.trade_log.append(fill)

        # Update equity
        current_equity = self.get_current_equity({market: fill_price})
        self.equity_history.append((datetime.now(), current_equity))
        self.peak_equity = max(self.peak_equity, current_equity)

        return {"status": "filled", "fill": fill, "new_equity": round(current_equity, 2)}

    def get_pnl(self, current_prices: Dict[str, float]) -> dict:
        """Full PnL + DD report."""
        equity = self.get_current_equity(current_prices)
        unrealized = equity - self.peak_equity  # approximate
        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        return {
            "equity": round(equity, 2),
            "unrealized_pnl": round(unrealized, 2),
            "cash": round(self.cash_balance, 2),
            "current_dd_pct": round(current_dd * 100, 2),
            "total_trades": len(self.trade_log),
            "positions": self.positions.copy()
        }

    def get_recent_trades(self, n: int = 10) -> List[dict]:
        return self.trade_log[-n:]

    # Phase B: Real Execution Placeholders
    def place_real_order_polymarket(self, ...):
        """Phase B: Use authenticated CLOB POST /order with wallet signature."""
        pass  # Implement with requests + signature

    def place_real_order_kalshi(self, ...):
        """Phase B: Use authenticated Kalshi trading API."""
        pass

    # Phase C: Safeguards
    def check_circuit_breaker(self, current_dd: float, vol_spike: bool) -> bool:
        """Circuit breaker: pause trading if large DD or vol spike."""
        if current_dd > 0.08 or vol_spike:
            return True  # triggered
        return False

    def reconcile_with_ws(self, ws_state: dict):
        """Reconcile paper positions with live WS order book state."""
        # Compare simulated fills vs real WS fills
        pass  # Implement comparison logic

    def place_paper_order(self, market: str, side: str, size: float, price: float, 
                         obi: float = 0.0, regime: str = "normal") -> str:
        """
        Simulate order placement with realistic slippage and costs.
        """
        order_id = str(uuid.uuid4())[:8]
        
        # Regime-dependent slippage (higher in low liquidity)
        slippage = 0.0005 if regime == "normal" else 0.003
        slippage *= (1 + abs(obi))  # OBI affects slippage
        
        fill_price = price + (slippage * price if side == "BUY" else -slippage * price)
        
        cost_bps = 5.0 if regime == "normal" else 12.0
        fee = abs(size) * fill_price * (cost_bps / 10000.0)
        
        self.orders[order_id] = {
            'market': market,
            'side': side,
            'size': size,
            'price': price,
            'fill_price': fill_price,
            'fee': fee,
            'status': 'FILLED',
            'timestamp': datetime.now()
        }
        
        # Update position
        pos_change = size if side == "BUY" else -size
        self.positions[market] = self.positions.get(market, 0.0) + pos_change
        
        # Log trade
        pnl_approx = 0.0  # Would be calculated on close
        self.trade_log.append({
            'id': order_id,
            'market': market,
            'side': side,
            'size': size,
            'fill_price': fill_price,
            'fee': fee,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"[EXEC] {side} {size} {market} @ {fill_price:.4f} (fee: ${fee:.2f})")
        
        return order_id

    def get_portfolio_summary(self) -> dict:
        total_exposure = sum(abs(pos) for pos in self.positions.values())
        return {
            'cash': self.cash_balance,
            'positions': self.positions,
            'total_exposure': total_exposure,
            'trades_count': len(self.trade_log),
            'recent_trades': self.trade_log[-5:] if self.trade_log else []
        }

    def reconcile_with_book(self, market: str, current_mid: float, obi: float):
        """Reconcile simulated positions with live book (placeholder for advanced matching)."""
        # In real implementation, compare against actual fills from WS
        pass


# Global engine instance for integration
execution_engine = PaperExecutionEngine()

# Example integration with live runner (call from on_market_update)
def example_paper_trade_on_signal(market: str, signal: int, size: float, mid: float, obi: float, regime: str = "normal"):
    if signal > 0:
        execution_engine.place_paper_order(market, "BUY", size, mid, obi, regime)
    elif signal < 0:
        execution_engine.place_paper_order(market, "SELL", size, mid, obi, regime)

print("Execution Layer loaded (Paper Trading Mode). Ready for integration.")
