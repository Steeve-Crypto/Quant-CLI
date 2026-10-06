#!/usr/bin/env python3
"""
Prediction Market Statistical Arbitrage Pipeline
=================================================
Full end-to-end pipeline that combines:
1. Realistic synthetic data generation (regime-switching vol, shocks, latency between platforms)
2. Timestamp alignment & preprocessing
3. Cointegration testing (Engle-Granger) + beta estimation
4. Spread construction
5. Ornstein-Uhlenbeck calibration on the spread
6. Simple OU-based mean-reversion backtest stub
7. Rich diagnostics, multiple plots, JSON report, and saved data

This reproduces the complete workflow described in the @ridark_eth thread
on quantitative arbitrage between prediction platforms (cointegration + OU + order book ideas).

Outputs are saved next to this script (override with QUANT_ARTIFACTS_DIR).

Usage:
    python prediction_market_arb_pipeline.py

Requirements (already available in sandbox):
    numpy, pandas, scipy, statsmodels, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import warnings

# Outputs go next to this script unless QUANT_ARTIFACTS_DIR is set.
ARTIFACTS_DIR = os.environ.get("QUANT_ARTIFACTS_DIR", os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

# Import the OU calibration functions from the previous script
try:
    from ou_calibration import simulate_ou_process, calibrate_ou
except ImportError:
    print("Warning: Could not import from ou_calibration.py — using built-in versions.")
    # Fallback definitions (minimal) if import fails
    def simulate_ou_process(theta, mu, sigma, n_steps=10000, dt=1e-5, S0=0.0, seed=42):
        np.random.seed(seed)
        exp_theta_dt = np.exp(-theta * dt)
        drift = mu * (1 - exp_theta_dt)
        vol = sigma * np.sqrt((1 - np.exp(-2*theta*dt)) / (2*theta))
        S = np.zeros(n_steps)
        S[0] = S0
        for t in range(1, n_steps):
            S[t] = S[t-1]*exp_theta_dt + drift + vol*np.random.normal()
        return pd.Series(S, name='spread')

    def calibrate_ou(spread, dt, method='ols', verbose=True):
        s = pd.to_numeric(spread, errors='coerce').dropna()
        y = s.iloc[1:].values
        x = s.iloc[:-1].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        a, b = model.params
        theta = -np.log(b) / dt if 0 < b < 1 else np.nan
        mu = a / (1 - b) if 0 < b < 1 else np.nan
        resid_var = model.mse_resid
        sigma = np.sqrt(resid_var * (2*theta) / (1-np.exp(-2*theta*dt))) if theta > 0 else np.nan
        half_life = np.log(2)/theta if theta > 0 else np.nan
        return {'theta': theta, 'mu': mu, 'sigma': sigma, 'half_life_periods': half_life,
                'r_squared': model.rsquared, 'n_obs': len(y)}


# ============================================================
# 1. SYNTHETIC DATA GENERATOR (mimics real prediction market mids)
# ============================================================

def generate_cointegrated_prediction_market_data(
    n_obs: int = 30_000,
    dt: float = 1.0 / (365.25 * 24 * 60),  # ~1 minute bars
    beta_true: float = 1.0,
    theta_spread: float = 40.0,
    mu_spread: float = 0.001,
    sigma_spread: float = 0.009,
    base_vol_p: float = 0.007,      # Base Polymarket volatility (higher liquidity but on-chain noise)
    base_vol_k: float = 0.005,      # Base Kalshi volatility (more stable, CFTC)
    regime_switch_prob: float = 0.015,  # Probability of entering high-vol / low-liquidity regime
    high_vol_multiplier: float = 3.5,   # How much vol increases in stressed regime
    shock_prob: float = 0.003,          # Probability of a whale/news shock per bar
    shock_size: float = 0.025,          # Size of occasional jumps
    latency_lag_bars: int = 2,          # Kalshi lags Polymarket by this many bars (realism: on-chain vs CEX)
    seed: int = 2026
) -> pd.DataFrame:
    """
    More realistic synthetic generator for Polymarket vs Kalshi prediction market mids.

    Features:
    - Cointegrated prices with OU spread
    - Regime-switching volatility (normal vs stressed/illiquid periods)
    - Occasional large shocks (whale trades, breaking news)
    - Latency / asynchronicity between platforms (Polymarket leads)
    - Time-of-day volatility pattern (higher during US business hours)
    - Different base microstructure noise between platforms
    """
    np.random.seed(seed)

    # 1. Common fundamental trend (shared information)
    common_trend = np.cumsum(np.random.normal(0, 0.0004, n_obs))

    # 2. Time-of-day volatility multiplier (simulate higher activity 13:00-21:00 UTC ~ US hours)
    hours = np.array([(start_time := pd.Timestamp("2026-05-01 12:00:00") + timedelta(minutes=i)).hour for i in range(n_obs)])
    tod_multiplier = np.where((hours >= 13) & (hours <= 21), 1.6, 0.7)

    # 3. Regime indicator (0 = normal liquidity, 1 = stressed/illiquid)
    regime = np.zeros(n_obs)
    for i in range(1, n_obs):
        if np.random.rand() < regime_switch_prob:
            regime[i] = 1 - regime[i-1]
        else:
            regime[i] = regime[i-1]
    vol_multiplier = 1 + regime * (high_vol_multiplier - 1)

    # 4. Generate the OU spread (core mean-reverting component)
    spread = simulate_ou_process(
        theta=theta_spread,
        mu=mu_spread,
        sigma=sigma_spread,
        n_steps=n_obs,
        dt=dt,
        S0=0.003,
        seed=seed + 1
    ).values

    # 5. Add occasional shocks to the spread (whale dumps, news)
    shocks = np.zeros(n_obs)
    shock_idx = np.random.rand(n_obs) < shock_prob
    shocks[shock_idx] = np.random.choice([-1, 1], size=shock_idx.sum()) * shock_size * (1 + regime[shock_idx])
    spread = spread + np.cumsum(shocks) * 0.3   # cumulative effect fades with mean reversion

    # 6. Construct base prices
    P_k_base = 0.48 + common_trend * 0.25
    P_p_base = beta_true * P_k_base + spread

    # 7. Add microstructure noise + regime + time-of-day vol
    noise_p = np.random.normal(0, base_vol_p, n_obs) * vol_multiplier * tod_multiplier
    noise_k = np.random.normal(0, base_vol_k, n_obs) * vol_multiplier * tod_multiplier * 0.8

    P_p = P_p_base + noise_p
    P_k = P_k_base + noise_k

    # 8. Simulate latency: Kalshi lags Polymarket (realistic for on-chain vs centralized)
    if latency_lag_bars > 0:
        P_k = np.roll(P_k, latency_lag_bars)
        P_k[:latency_lag_bars] = P_k[latency_lag_bars]  # pad beginning

    # 9. Clip to valid probability range
    P_p = np.clip(P_p, 0.005, 0.995)
    P_k = np.clip(P_k, 0.005, 0.995)

    # Create datetime index
    start_time = pd.Timestamp("2026-05-01 12:00:00")
    times = pd.date_range(start=start_time, periods=n_obs, freq='min')

    df = pd.DataFrame({
        'mid_polymarket': P_p,
        'mid_kalshi': P_k,
        'true_spread': spread,
        'true_beta': beta_true,
        'regime': regime,
        'tod_vol_multiplier': tod_multiplier
    }, index=times)

    return df


# ============================================================
# 2. COINTEGRATION + BETA ESTIMATION
# ============================================================

def test_cointegration_and_estimate_beta(
    price_p: pd.Series,
    price_k: pd.Series,
    verbose: bool = True
) -> Dict:
    """
    Perform Engle-Granger cointegration test and estimate hedge ratio (beta).
    """
    # Align on common index
    common_idx = price_p.index.intersection(price_k.index)
    p = price_p.loc[common_idx].dropna()
    k = price_k.loc[common_idx].dropna()

    # Engle-Granger cointegration test
    # (tests if residual of regression is stationary)
    coint_stat, p_value, _ = coint(p, k)

    # Estimate beta via OLS (Engle-Granger style)
    X = sm.add_constant(k)
    model = sm.OLS(p, X).fit()
    beta = model.params.iloc[1]
    intercept = model.params.iloc[0]
    resid = model.resid

    # ADF test on residuals (should be stationary if cointegrated)
    adf_stat, adf_pval, _, _, _, _ = adfuller(resid, maxlag=10, regression='c')

    results = {
        'coint_statistic': float(coint_stat),
        'coint_pvalue': float(p_value),
        'is_cointegrated': p_value < 0.05,
        'beta': float(beta),
        'intercept': float(intercept),
        'adf_stat_resid': float(adf_stat),
        'adf_pvalue_resid': float(adf_pval),
        'resid_std': float(resid.std()),
        'n_obs': len(common_idx)
    }

    if verbose:
        print("\n" + "="*65)
        print("COINTEGRATION TEST RESULTS (Engle-Granger)")
        print("="*65)
        print(f"Observations aligned : {results['n_obs']:,}")
        print(f"Cointegration t-stat : {results['coint_statistic']:.4f}")
        print(f"Cointegration p-value: {results['coint_pvalue']:.2e}")
        print(f"  → Cointegrated?    : {'YES ✓' if results['is_cointegrated'] else 'NO'}")
        print(f"Estimated beta (hedge ratio): {results['beta']:.6f}")
        print(f"Regression intercept        : {results['intercept']:.6f}")
        print(f"ADF on residuals (p-value)  : {results['adf_pvalue_resid']:.2e}")
        print(f"  → Residuals stationary?   : {'YES' if results['adf_pvalue_resid'] < 0.05 else 'NO'}")
        print("="*65 + "\n")

    return results


# ============================================================
# IMPROVED: Johansen Cointegration Test
# ============================================================

def johansen_cointegration_test(
    price_p: pd.Series,
    price_k: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Johansen cointegration test (more robust than Engle-Granger).
    Returns trace statistics, critical values, and the cointegrating vector (beta).
    """
    common_idx = price_p.index.intersection(price_k.index)
    data = pd.concat([price_p.loc[common_idx], price_k.loc[common_idx]], axis=1).dropna()
    data.columns = ['P', 'K']

    # Run Johansen test
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # Trace statistic for rank 0 (no cointegration) vs rank >=1
    trace_stat = result.lr1[0]          # Trace stat for r=0
    crit_value_95 = result.cvt[0, 1]    # 95% critical value for r=0

    is_cointegrated = trace_stat > crit_value_95

    # Cointegrating vector (normalized so first coefficient = 1)
    # result.evec[:, 0] is the first cointegrating vector
    coint_vec = result.evec[:, 0]
    beta_johansen = -coint_vec[1] / coint_vec[0]   # beta such that P - beta*K is stationary

    results = {
        'trace_stat_r0': float(trace_stat),
        'crit_value_95_r0': float(crit_value_95),
        'is_cointegrated_johansen': bool(is_cointegrated),
        'beta_johansen': float(beta_johansen),
        'eigenvalues': result.eig.tolist(),
        'n_obs': len(data)
    }

    if verbose:
        print("\n" + "="*65)
        print("JOHANSEN COINTEGRATION TEST")
        print("="*65)
        print(f"Observations          : {results['n_obs']:,}")
        print(f"Trace statistic (r=0) : {results['trace_stat_r0']:.4f}")
        print(f"95% Critical value    : {results['crit_value_95_r0']:.4f}")
        print(f"  → Cointegrated?     : {'YES ✓' if results['is_cointegrated_johansen'] else 'NO'}")
        print(f"Johansen beta         : {results['beta_johansen']:.6f}")
        print("="*65 + "\n")

    return results


# ============================================================
# NEW: Simple OU-based Mean Reversion Backtest Stub
# ============================================================

def simple_ou_mean_reversion_backtest(
    spread: pd.Series,
    ou_params: Dict,
    entry_z: float = 1.2,
    exit_z: float = 0.25,
    max_hold_bars: int = 120,
    base_cost_bps: float = 6.0,
    max_position: float = 5.0,           # Maximum position size (in units of spread)
    sizing_method: str = 'zscore_scaled', # 'zscore_scaled' or 'vol_target'
    stop_loss_z: float = 4.0,            # Stop loss if spread moves against us
    verbose: bool = True
) -> Dict:
    """
    Improved OU-based mean-reversion backtester with position sizing.

    Position sizing options:
    - 'zscore_scaled': Larger positions when signal is stronger (more aggressive)
    - 'vol_target': Size inversely to OU sigma (risk parity style)

    Uses calibrated OU parameters (mu, sigma) for z-score and sizing.
    More aggressive defaults + basic stop loss.
    """
    s = spread.dropna().copy()
    mu = ou_params.get('mu', s.mean())
    sigma = ou_params.get('sigma', s.std())
    if sigma <= 0 or np.isnan(sigma):
        sigma = s.std() or 1.0

    z = (s - mu) / sigma

    # Determine if we have regime info for variable costs
    has_regime = 'regime' in spread.index.names or hasattr(spread, 'regime')  # simplistic check
    # For simplicity we'll use a constant cost in this version, but can extend

    position = np.zeros(len(s), dtype=float)
    current_pos = 0.0
    entry_idx = 0

    for i in range(1, len(s)):
        z_val = z.iloc[i]
        prev_pos = current_pos

        # --- Signal & Position Sizing ---
        if current_pos == 0:
            # Improved signal quality: OU z-score + short-term momentum confirmation (reduces whipsaws)
            mom_window = 8
            recent_mom = s.iloc[i] - s.iloc[max(0, i - mom_window)]
            mom_confirms_long = (z_val < -entry_z) and (recent_mom < 0)
            mom_confirms_short = (z_val > entry_z) and (recent_mom > 0)

            if mom_confirms_long:
                if sizing_method == 'zscore_scaled':
                    raw_size = min(max_position, (abs(z_val) / entry_z) * 2.5)
                else:
                    raw_size = min(max_position, 2.5 / max(sigma, 0.001))
                current_pos = raw_size
                entry_idx = i
            elif mom_confirms_short:
                if sizing_method == 'zscore_scaled':
                    raw_size = min(max_position, (abs(z_val) / entry_z) * 2.5)
                else:
                    raw_size = min(max_position, 2.5 / max(sigma, 0.001))
                current_pos = -raw_size
                entry_idx = i
        else:
            # Exit or stop loss
            exit_signal = abs(z_val) < exit_z
            time_stop = (i - entry_idx) >= max_hold_bars
            adverse_move = (current_pos > 0 and z_val > stop_loss_z) or (current_pos < 0 and z_val < -stop_loss_z)

            if exit_signal or time_stop or adverse_move:
                current_pos = 0.0

        position[i] = current_pos

    # Calculate P&L while in position
    spread_change = s.diff().fillna(0)
    strategy_returns = position * spread_change

    # More realistic costs: higher when entering large positions or in stress
    position_change = np.abs(np.diff(position, prepend=0))
    # Scale cost with position size (larger trades cost more relatively)
    effective_cost_bps = base_cost_bps * (1 + 0.5 * np.abs(position) / max_position)
    costs = position_change * (effective_cost_bps / 10000.0)
    strategy_returns = strategy_returns - costs

    # Equity curve
    equity = (1 + strategy_returns).cumprod()
    equity.iloc[0] = 1.0

    # Metrics
    total_return = equity.iloc[-1] - 1.0
    n_trades = int((position_change > 0).sum())
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252 * 24 * 60)
    max_dd = (equity / equity.cummax() - 1).min()
    win_rate = (strategy_returns[strategy_returns != 0] > 0).mean() if (strategy_returns != 0).any() else 0.0
    avg_hold = np.mean([i - entry_idx for i in range(len(position)) if position[i] != 0]) if np.any(position != 0) else 0

    results = {
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'n_trades': n_trades,
        'win_rate': float(win_rate),
        'avg_hold_bars': float(avg_hold),
        'max_position_used': float(np.max(np.abs(position))),
        'equity_curve': equity,
        'final_position': float(position[-1])
    }

    if verbose:
        print("\n" + "="*65)
        print("IMPROVED OU MEAN-REVERSION BACKTEST (with Position Sizing)")
        print("="*65)
        print(f"Total Return          : {total_return*100:,.2f}%")
        print(f"Sharpe Ratio          : {sharpe:.2f}")
        print(f"Max Drawdown          : {max_dd*100:.2f}%")
        print(f"Number of Trades      : {n_trades}")
        print(f"Win Rate              : {win_rate*100:.1f}%")
        print(f"Avg Hold Time         : {avg_hold:.1f} bars")
        print(f"Max Position Size     : {results['max_position_used']:.2f} units")
        print("="*65 + "\n")

    return results


# ============================================================
# NEW: Simple Multi-Asset / Portfolio Backtest
# ============================================================

def simple_multi_asset_portfolio_backtest(
    spreads: dict,                    # {'market1': spread_series, 'market2': ...}
    ou_params_dict: dict,
    allocation_method: str = 'equal_risk',  # 'equal_risk' or 'equal_weight'
    max_total_exposure: float = 8.0,
    verbose: bool = True
) -> dict:
    """
    Simple multi-asset portfolio version.
    Runs individual OU signals on multiple spreads and combines them
    with basic risk allocation (equal risk or equal weight).
    Demonstrates portfolio construction across prediction market events.
    """
    equity_curves = {}
    all_returns = []

    for name, spread in spreads.items():
        ou_p = ou_params_dict.get(name, {})
        bt = simple_ou_mean_reversion_backtest(
            spread, ou_p,
            entry_z=1.1, exit_z=0.2, max_position=3.0,
            sizing_method='zscore_scaled',
            verbose=False
        )
        equity_curves[name] = bt['equity_curve']
        all_returns.append(bt['equity_curve'].pct_change().fillna(0))

    # Portfolio allocation: support equal_risk (inverse vol), min_variance (correlation-aware), equal_weight
    returns_df = pd.concat(all_returns, axis=1)
    returns_df.columns = list(spreads.keys())

    if allocation_method == 'min_variance':
        # Correlation-aware minimum variance portfolio
        cov = returns_df.cov().values
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(len(cov))
            w = inv_cov @ ones
            w = w / np.sum(w)
            weights = pd.Series(np.clip(w, 0, None), index=returns_df.columns)  # non-negative
            weights = weights / weights.sum()
        except np.linalg.LinAlgError:
            print("Warning: Singular covariance, falling back to equal_risk")
            vols = returns_df.std()
            weights = (1 / vols) / (1 / vols).sum()
    elif allocation_method == 'equal_risk':
        vols = returns_df.std()
        weights = (1 / vols) / (1 / vols).sum()
        weights = weights.clip(upper=0.5)
    else:
        weights = pd.Series(1.0 / len(spreads), index=returns_df.columns)

    # Normalize to max_total_exposure
    weights = weights * (max_total_exposure / weights.sum())

    portfolio_returns = (returns_df * weights).sum(axis=1)
    portfolio_equity = (1 + portfolio_returns).cumprod()

    portfolio_metrics = {
        'total_return': float(portfolio_equity.iloc[-1] - 1),
        'sharpe': float(portfolio_returns.mean() / (portfolio_returns.std() + 1e-9) * np.sqrt(252*24*60)),
        'max_drawdown': float((portfolio_equity / portfolio_equity.cummax() - 1).min()),
        'n_assets': len(spreads),
        'weights': weights.to_dict(),
        'equity_curve': portfolio_equity
    }

    if verbose:
        print("\n" + "="*60)
        print("SIMPLE MULTI-ASSET PORTFOLIO BACKTEST")
        print("="*60)
        print(f"Assets              : {list(spreads.keys())}")
        print(f"Allocation          : {allocation_method}")
        print(f"Total Return        : {portfolio_metrics['total_return']*100:.2f}%")
        print(f"Portfolio Sharpe    : {portfolio_metrics['sharpe']:.2f}")
        print(f"Portfolio Max DD    : {portfolio_metrics['max_drawdown']*100:.2f}%")
        print(f"Weights             : { {k: round(v,2) for k,v in weights.items()} }")
        print("="*60 + "\n")

    return {
        'portfolio_metrics': portfolio_metrics,
        'individual_equities': equity_curves,
        'weights': weights.to_dict()
    }


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def run_prediction_market_arb_pipeline(
    data_source: str = 'synthetic',
    parquet_path: Optional[str] = None,
    output_dir: str = ARTIFACTS_DIR,
    verbose: bool = True
) -> Dict:
    """
    Full pipeline: load/generate data → cointegration test → beta → spread → OU calibration → diagnostics.
    Everything is saved to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#"*70)
    print("PREDICTION MARKET STATISTICAL ARBITRAGE PIPELINE")
    print("#"*70)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Data source: {data_source}")
    print("#"*70 + "\n")

    # --- Step 1: Load or generate data ---
    if data_source == 'synthetic':
        print("Generating synthetic cointegrated Polymarket + Kalshi mid prices...")
        df = generate_cointegrated_prediction_market_data(n_obs=30_000, seed=42)
        data_path = f"{output_dir}/synthetic_prediction_market_data.pkl"
        df.to_pickle(data_path)
        print(f"Synthetic data saved → {data_path}")
    else:
        print(f"Loading real data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        # Expect columns: mid_polymarket, mid_kalshi (or adjust as needed)

    # Basic data info
    data_info = {
        'n_observations': len(df),
        'time_start': str(df.index.min()),
        'time_end': str(df.index.max()),
        'duration_minutes': (df.index.max() - df.index.min()).total_seconds() / 60,
        'mean_mid_p': float(df['mid_polymarket'].mean()),
        'mean_mid_k': float(df['mid_kalshi'].mean()),
        'std_mid_p': float(df['mid_polymarket'].std()),
        'std_mid_k': float(df['mid_kalshi'].std()),
    }

    if verbose:
        print("\nDATA SUMMARY")
        print("-" * 40)
        for k, v in data_info.items():
            print(f"{k:25s}: {v}")

    # --- Step 2: Cointegration tests (Engle-Granger + Johansen) ---
    eg_results = test_cointegration_and_estimate_beta(
        df['mid_polymarket'], df['mid_kalshi'], verbose=verbose
    )

    johansen_results = johansen_cointegration_test(
        df['mid_polymarket'], df['mid_kalshi'], verbose=verbose
    )

    # Prefer Johansen beta, but fall back if extreme (common in trending synthetic data)
    beta_j = johansen_results.get('beta_johansen', eg_results['beta'])
    if abs(beta_j) > 5 or np.isnan(beta_j):
        beta = eg_results['beta']
        print("Note: Using Engle-Granger beta (Johansen vector was extreme)")
    else:
        beta = beta_j

    # Combine results for reporting
    coint_results = {
        'engle_granger': eg_results,
        'johansen': johansen_results,
        'beta_used': float(beta)
    }

    # --- Step 3: Construct the spread ---
    df['spread'] = df['mid_polymarket'] - beta * df['mid_kalshi']

    # Store residuals from the beta relationship for diagnostics
    df['spread_residuals'] = df['mid_polymarket'] - beta * df['mid_kalshi']

    # --- Step 4: OU Calibration on the spread ---
    # Use ~1 minute frequency → dt in yearly units
    dt_minutes = 1.0 / (365.25 * 24 * 60)
    ou_params = calibrate_ou(df['spread'], dt=dt_minutes, method='ols', verbose=verbose)

    # --- NEW: Simple Backtest using calibrated OU parameters ---
    print("\nRunning simple OU mean-reversion backtest stub...")
    backtest_results = simple_ou_mean_reversion_backtest(
        df['spread'],
        ou_params,
        entry_z=1.1,                    # More aggressive entry
        exit_z=0.2,
        max_hold_bars=90,
        base_cost_bps=5.0,
        max_position=4.0,
        sizing_method='zscore_scaled',  # Aggressive: larger size on stronger signals
        stop_loss_z=3.5,
        verbose=verbose
    )

    # Save equity curve plot
    fig, ax = plt.subplots(figsize=(12, 5))
    backtest_results['equity_curve'].plot(ax=ax, color='darkgreen', linewidth=1.5)
    ax.set_title("Simple OU Mean-Reversion Backtest Equity Curve")
    ax.set_ylabel("Cumulative Return (starting at 1.0)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    equity_plot_path = f"{output_dir}/backtest_equity_curve.png"
    plt.savefig(equity_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved backtest equity curve → {equity_plot_path}")

    # --- MULTI-ASSET DEMO: Generate 2 more synthetic spreads and run portfolio backtest ---
    print("\n" + "="*70)
    print("MULTI-ASSET PORTFOLIO DEMO (auto-generated 3 spreads)")
    print("="*70)
    spreads = {'market_1': df['spread'].copy()}
    ou_params_dict = {'market_1': ou_params.copy() if isinstance(ou_params, dict) else ou_params}

    for i in range(2, 4):
        print(f"  Generating additional synthetic spread for market_{i}...")
        df_i = generate_cointegrated_prediction_market_data(n_obs=8000, seed=42 + i * 100)  # smaller for demo speed
        # Approximate independent spread (use mid_p - mid_k as proxy spread)
        spread_i = (df_i['mid_polymarket'] - df_i['mid_kalshi']).dropna()
        ou_i = calibrate_ou(spread_i, dt=dt_minutes, method='ols', verbose=False)
        spreads[f'market_{i}'] = spread_i
        ou_params_dict[f'market_{i}'] = ou_i

    print("\nRunning multi-asset portfolio backtest with correlation-aware options...")
    portfolio_result = simple_multi_asset_portfolio_backtest(
        spreads,
        ou_params_dict,
        allocation_method='min_variance',  # correlation-aware min variance
        max_total_exposure=6.0,
        verbose=verbose
    )

    # Save portfolio equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    portfolio_result['portfolio_metrics']['equity_curve'].plot(ax=ax, color='purple', linewidth=1.5, label='Portfolio')
    ax.set_title("Multi-Asset Portfolio Equity Curve (Min-Variance Allocation)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    portfolio_equity_path = f"{output_dir}/portfolio_equity_curve.png"
    plt.savefig(portfolio_equity_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved portfolio equity curve → {portfolio_equity_path}")

    # Update results with portfolio info
    portfolio_info = {
        'n_markets': len(spreads),
        'allocation_method': 'min_variance',
        'portfolio_metrics': portfolio_result['portfolio_metrics'],
        'weights': portfolio_result['weights']
    }

    # --- Step 5: Generate and save rich diagnostics ---
    print("\nGenerating diagnostic plots...")

    # Plot 1: Price overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    df['mid_polymarket'].plot(ax=ax, label='Polymarket Mid', alpha=0.8)
    df['mid_kalshi'].plot(ax=ax, label='Kalshi Mid', alpha=0.8)
    ax.set_title("Prediction Market Contract Mids (Synthetic Example)")
    ax.set_ylabel("Price (probability)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    price_plot_path = f"{output_dir}/prices_overlay.png"
    plt.savefig(price_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {price_plot_path}")

    # Plot 2: Spread + OU mean
    fig, ax = plt.subplots(figsize=(12, 5))
    df['spread'].plot(ax=ax, label='Observed Spread (P_p - β·P_k)', alpha=0.7, color='steelblue')
    ax.axhline(ou_params['mu'], color='red', linestyle='--', linewidth=2,
               label=f"OU Long-run Mean μ = {ou_params['mu']:.5f}")
    ax.set_title("Constructed Spread + Calibrated OU Mean Reversion Level")
    ax.set_ylabel("Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    spread_plot_path = f"{output_dir}/spread_ou_mean.png"
    plt.savefig(spread_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {spread_plot_path}")

    # Plot 3: Histogram of spread
    fig, ax = plt.subplots(figsize=(8, 5))
    df['spread'].hist(bins=80, ax=ax, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(ou_params['mu'], color='red', linestyle='--', linewidth=2, label=f"μ = {ou_params['mu']:.4f}")
    ax.set_title("Distribution of the Spread")
    ax.set_xlabel("Spread value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hist_plot_path = f"{output_dir}/spread_histogram.png"
    plt.savefig(hist_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {hist_plot_path}")

    # Plot 4: Rolling Beta (stability of hedge ratio)
    print("  Generating rolling beta plot...")
    window = 2000  # ~2k bars rolling window
    rolling_beta = df['mid_polymarket'].rolling(window).apply(
        lambda x: sm.OLS(x, sm.add_constant(df['mid_kalshi'].loc[x.index])).fit().params.iloc[1],
        raw=False
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    rolling_beta.plot(ax=ax, color='purple', linewidth=1.2)
    ax.axhline(beta, color='red', linestyle='--', label=f'Full-sample beta = {beta:.4f}')
    ax.set_title(f"Rolling Beta (window = {window} bars) — Hedge Ratio Stability")
    ax.set_ylabel("Beta")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rolling_beta_path = f"{output_dir}/rolling_beta.png"
    plt.savefig(rolling_beta_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {rolling_beta_path}")

    # Plot 5: Residual Diagnostics (from beta relationship)
    print("  Generating residual diagnostics...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals time series
    df['spread_residuals'].plot(ax=axes[0, 0], color='darkorange', alpha=0.7)
    axes[0, 0].set_title("Residuals from Cointegrating Relationship")
    axes[0, 0].axhline(0, color='black', linestyle='--')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram of residuals
    df['spread_residuals'].hist(bins=60, ax=axes[0, 1], color='darkorange', alpha=0.7, edgecolor='white')
    axes[0, 1].set_title("Residuals Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # ACF of residuals (simple)
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['spread_residuals'].dropna(), ax=axes[1, 0])
    axes[1, 0].set_title("Autocorrelation of Residuals")

    # QQ plot
    stats.probplot(df['spread_residuals'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot of Residuals (Normality Check)")

    plt.tight_layout()
    residual_diag_path = f"{output_dir}/residual_diagnostics.png"
    plt.savefig(residual_diag_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {residual_diag_path}")

    # --- Step 6: Compile full results ---
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'data_info': data_info,
        'cointegration': coint_results,
        'ou_calibration': ou_params,
        'beta_used': float(beta),
        'portfolio_demo': portfolio_info if 'portfolio_info' in locals() else {},
        'files_saved': {
            'data_parquet': data_path if data_source == 'synthetic' else parquet_path,
            'price_plot': price_plot_path,
            'spread_plot': spread_plot_path,
            'spread_histogram': hist_plot_path,
            'rolling_beta': rolling_beta_path,
            'residual_diagnostics': residual_diag_path,
            'backtest_equity': equity_plot_path,
            'portfolio_equity': portfolio_equity_path if 'portfolio_equity_path' in locals() else None
        }
    }

    # Save results as JSON
    results_json_path = f"{output_dir}/pipeline_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nFull results JSON saved → {results_json_path}")

    # Save the processed DataFrame with spread
    processed_path = f"{output_dir}/processed_data_with_spread.pkl"
    df.to_pickle(processed_path)
    print(f"Processed data with spread saved → {processed_path}")

    print("\n" + "#"*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("#"*70 + "\n")

    return full_results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import os

    results = run_prediction_market_arb_pipeline(
        data_source='synthetic',           # Change to 'parquet' and provide path for real data
        # parquet_path='/path/to/your/l2_data.parquet',
        output_dir=ARTIFACTS_DIR,
        verbose=True
    )

    print("Key Takeaways from this run:")
    print(f"  • Johansen Cointegrated: {results['cointegration']['johansen']['is_cointegrated_johansen']}")
    print(f"  • Beta used for spread : {results['cointegration']['beta_used']:.4f}")
    print(f"  • OU theta (reversion) : {results['ou_calibration']['theta']:.2f}")
    print(f"  • OU R-squared         : {results['ou_calibration']['r_squared']:.4f}")
    print(f"  • All artifacts saved in {ARTIFACTS_DIR}/")

    print("\nYou can now load the saved Parquet files or JSON for further analysis.")