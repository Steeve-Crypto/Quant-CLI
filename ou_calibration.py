#!/usr/bin/env python3
"""
Ornstein-Uhlenbeck Process Calibration
======================================
Implements MLE-style calibration (via OLS on discretized form) for the OU process,
as described in quantitative prediction market arbitrage strategies (e.g. Polymarket vs Kalshi spreads).

The discretized OU process:
    S_t = S_{t-1} * exp(-theta * dt) + mu * (1 - exp(-theta * dt)) + epsilon_t

where epsilon_t ~ N(0, sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta) )

Calibration via linear regression (equivalent to conditional MLE under normality):
    S_t = a + b * S_{t-1} + eps
    theta = -log(b) / dt
    mu    = a / (1 - b)
    Then recover sigma from residual variance.

References:
- Thread by @ridark_eth on cross-platform stat arb using OU + order book imbalance
- Standard quant finance pairs trading / mean-reversion calibration

Usage:
    python ou_calibration.py

    Or import:
    from ou_calibration import calibrate_ou, simulate_ou_process
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def simulate_ou_process(
    theta: float,
    mu: float,
    sigma: float,
    n_steps: int = 10_000,
    dt: float = 1.0 / (252 * 24 * 60),  # ~1 minute in trading years (example)
    S0: float = 0.0,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Simulate an Ornstein-Uhlenbeck process using exact discretization.

    Parameters
    ----------
    theta : float
        Mean reversion speed (higher = faster reversion)
    mu : float
        Long-term mean
    sigma : float
        Volatility (diffusion term)
    n_steps : int
        Number of time steps to simulate
    dt : float
        Time increment (in same units as theta, e.g. years)
    S0 : float
        Initial value
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    pd.Series
        Simulated OU process values indexed by time
    """
    if seed is not None:
        np.random.seed(seed)

    # Exact moments for simulation
    exp_theta_dt = np.exp(-theta * dt)
    drift_term = mu * (1 - exp_theta_dt)
    vol_term = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))

    S = np.zeros(n_steps)
    S[0] = S0

    for t in range(1, n_steps):
        S[t] = S[t-1] * exp_theta_dt + drift_term + vol_term * np.random.normal()

    # Create time index (arbitrary units)
    times = np.arange(n_steps) * dt
    return pd.Series(S, index=pd.to_datetime('2024-01-01') + pd.to_timedelta(times, unit='D'), name='spread')


def calibrate_ou(
    spread: pd.Series,
    dt: float,
    method: str = 'ols',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calibrate Ornstein-Uhlenbeck parameters using discretized MLE / OLS.

    This is the standard, fast, and robust method used by quant desks for
    mean-reversion trading (pairs trading, stat arb on prediction market spreads).

    Parameters
    ----------
    spread : pd.Series
        Time series of the spread S_t (e.g. P_Polymarket - beta * P_Kalshi)
        Must be regularly sampled or you must resample first.
    dt : float
        Time step between observations, in years (or consistent unit with theta).
        Example: for 1-minute bars → dt = 1 / (252 * 24 * 60) ≈ 7.87e-6
    method : str
        'ols' (default, fast & stable) or 'mle' (scipy.optimize, more general)
    verbose : bool
        Print summary if True

    Returns
    -------
    dict
        {
            'theta': float,      # Mean reversion speed
            'mu': float,         # Long-run mean
            'sigma': float,      # Volatility
            'half_life': float,  # Days (or dt units) to halve the deviation
            'r_squared': float,  # Goodness of fit from regression
            'n_obs': int
        }
    """
    if len(spread) < 10:
        raise ValueError("Need at least 10 observations for reliable calibration")

    # Drop NaNs and ensure numeric
    s = pd.to_numeric(spread, errors='coerce').dropna()
    if len(s) < 10:
        raise ValueError("Insufficient valid observations after cleaning")

    # Create lagged variables for regression
    y = s.iloc[1:].values          # S_t
    x = s.iloc[:-1].values         # S_{t-1}
    n = len(y)

    if method == 'ols':
        # === OLS on discretized OU (most common in practice) ===
        # S_t = a + b * S_{t-1} + eps
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        a = model.params[0]          # intercept
        b = model.params[1]          # slope on lagged
        r_squared = model.rsquared
        resid_var = model.mse_resid  # variance of residuals

        # Recover OU parameters
        if b <= 0 or b >= 1:
            # Unstable or explosive — common warning in real data near expiry
            theta = np.nan
            mu = np.nan
        else:
            theta = -np.log(b) / dt
            mu = a / (1 - b)

        # Sigma from residual variance
        # Var(eps) = (sigma^2 / (2*theta)) * (1 - exp(-2*theta*dt))
        if theta > 0 and not np.isnan(theta):
            sigma = np.sqrt(resid_var * (2 * theta) / (1 - np.exp(-2 * theta * dt)))
        else:
            sigma = np.nan

    elif method == 'mle':
        # Full numerical MLE (optional, slower, more flexible for constraints)
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            theta_, mu_, sigma_ = params
            if theta_ <= 0 or sigma_ <= 0:
                return np.inf
            exp_m_theta_dt = np.exp(-theta_ * dt)
            drift = mu_ * (1 - exp_m_theta_dt)
            vol2 = (sigma_**2 / (2 * theta_)) * (1 - np.exp(-2 * theta_ * dt))
            if vol2 <= 0:
                return np.inf

            preds = x * exp_m_theta_dt + drift
            residuals = y - preds
            nll = 0.5 * n * np.log(2 * np.pi * vol2) + 0.5 * np.sum(residuals**2 / vol2)
            return nll

        # Initial guess from OLS
        X = sm.add_constant(x)
        init_model = sm.OLS(y, X).fit()
        init_b = init_model.params[1]
        init_theta = max(-np.log(init_b) / dt, 1e-6) if 0 < init_b < 1 else 1.0
        init_mu = init_model.params[0] / (1 - init_b) if 0 < init_b < 1 else np.mean(y)
        init_sigma = np.std(y) * np.sqrt(2 * init_theta)  # rough

        res = minimize(neg_log_likelihood, [init_theta, init_mu, init_sigma],
                       bounds=[(1e-8, None), (None, None), (1e-8, None)],
                       method='L-BFGS-B')
        theta, mu, sigma = res.x
        r_squared = np.nan  # not directly from MLE

    else:
        raise ValueError("method must be 'ols' or 'mle'")

    # Half-life in number of time steps (periods)
    half_life_periods = np.log(2) / theta if theta > 0 else np.nan
    # Half-life in the same time unit as dt (e.g. years if dt is yearly)
    half_life_in_dt_units = half_life_periods * dt if theta > 0 else np.nan

    result = {
        'theta': float(theta),
        'mu': float(mu),
        'sigma': float(sigma),
        'half_life_periods': float(half_life_periods),
        'half_life_in_dt_units': float(half_life_in_dt_units),
        'r_squared': float(r_squared) if 'r_squared' in locals() else np.nan,
        'n_obs': int(n)
    }

    if verbose:
        print("\n" + "="*60)
        print("ORNSTEIN-UHLENBECK CALIBRATION RESULTS")
        print("="*60)
        print(f"Observations used : {result['n_obs']:,}")
        print(f"Time step (dt)    : {dt:.2e} (in your chosen time units)")
        print("-" * 40)
        print(f"theta (reversion speed) : {result['theta']:.6f}  (per dt unit)")
        print(f"mu (long-run mean)      : {result['mu']:.6f}")
        print(f"sigma (volatility)      : {result['sigma']:.6f}")
        print(f"Half-life               : {result['half_life_periods']:.2f} time steps")
        print(f"Half-life (in dt units) : {result['half_life_in_dt_units']:.6f}")
        print(f"R-squared (regression)  : {result['r_squared']:.6f}")
        print("="*60 + "\n")

    return result


def plot_ou_fit(spread: pd.Series, params: Dict[str, float], title: str = "OU Process Fit"):
    """Quick diagnostic plot of the spread vs fitted mean reversion."""
    fig, ax = plt.subplots(figsize=(10, 5))
    spread.plot(ax=ax, label='Observed Spread', alpha=0.7)
    ax.axhline(params['mu'], color='red', linestyle='--', label=f"Long-run μ = {params['mu']:.4f}")
    ax.set_title(title)
    ax.set_ylabel("Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/workdir/artifacts/ou_diagnostic.png', dpi=150, bbox_inches='tight')
    print("Diagnostic plot saved to: /home/workdir/artifacts/ou_diagnostic.png")


# ====================== DEMO / EXAMPLE ======================
if __name__ == "__main__":
    print("=== Ornstein-Uhlenbeck Calibration Demo ===\n")
    print("This reproduces the calibration technique from the prediction market")
    print("statistical arbitrage thread (cointegration + OU on cross-platform spreads).\n")

    # --- 1. Simulate realistic spread data ---
    TRUE_THETA = 50.0          # fairly fast mean reversion (common in liquid markets)
    TRUE_MU = 0.002            # small persistent mispricing (e.g. 0.2 cents)
    TRUE_SIGMA = 0.015         # reasonable vol for prediction market spread
    DT = 1.0 / (252 * 24 * 60) # 1-minute bars in trading-year units

    print("Simulating synthetic spread with known parameters...")
    spread = simulate_ou_process(
        theta=TRUE_THETA,
        mu=TRUE_MU,
        sigma=TRUE_SIGMA,
        n_steps=50_000,        # ~1 month of minute data
        dt=DT,
        S0=0.01
    )

    print(f"Generated {len(spread):,} observations\n")

    # --- 2. Calibrate (should recover true values closely) ---
    print("Calibrating with OLS (default, recommended for speed & stability)...")
    estimated = calibrate_ou(spread, dt=DT, method='ols', verbose=True)

    print("True parameters (for comparison):")
    print(f"  theta={TRUE_THETA:.4f}, mu={TRUE_MU:.4f}, sigma={TRUE_SIGMA:.4f}")
    hl_true_periods = np.log(2) / TRUE_THETA
    print(f"  True half-life ≈ {hl_true_periods:.2f} periods ({hl_true_periods * DT * 60 * 24 * 365.25:.1f} minutes)\n")

    # --- 3. Optional: Plot ---
    try:
        plot_ou_fit(spread, {'mu': estimated['mu']}, "Synthetic Prediction Market Spread + Calibrated OU Mean")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\n=== How to use with real Polymarket/Kalshi data ===")
    print("""
    1. Load your L2 or trade data (Parquet recommended)
    2. Compute mid prices on both platforms for the same event
    3. Estimate cointegration beta (e.g. via statsmodels.tsa.stattools.coint or OLS)
    4. Create spread = P_polymarket - beta * P_kalshi
    5. Resample to regular interval if needed (e.g. 1min)
    6. Call calibrate_ou(spread_series, dt=1/(252*24*60))

    Example skeleton:
        import pandas as pd
        df = pd.read_parquet('your_l2_data.parquet')
        # ... compute mids and spread ...
        params = calibrate_ou(spread, dt=1/(252*24*60))
        if params['half_life'] < execution_latency_in_same_units:
            print("Tradeable!")
    """)

    print("Done. You can now import calibrate_ou and simulate_ou_process in your own scripts.")