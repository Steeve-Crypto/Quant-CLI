#!/usr/bin/env python3
"""
Kalman Filter for Dynamic Beta (Hedge Ratio)
===========================================
Adaptive beta estimation for cointegrated pairs/spreads.
Better than rolling OLS for noisy, regime-changing data.

Integrates with cointegration, OU calibration, and position sizing.
"""

import numpy as np
import pandas as pd

def kalman_filter_beta(y: pd.Series, x: pd.Series, 
                       Q: float = 1e-5, R: float = 1e-3,
                       initial_beta: float = 1.0) -> pd.Series:
    """
    Simple 1D Kalman filter for dynamic beta (hedge ratio).
    
    y = beta * x + noise
    State: beta_t = beta_{t-1} + process_noise (Q)
    Observation: y_t = beta_t * x_t + measurement_noise (R)
    
    Returns rolling beta series.
    """
    n = len(y)
    beta = np.zeros(n)
    P = np.zeros(n)  # Error covariance
    
    beta[0] = initial_beta
    P[0] = 1.0  # Initial uncertainty
    
    for t in range(1, n):
        # Predict
        beta_pred = beta[t-1]
        P_pred = P[t-1] + Q
        
        # Update
        K = P_pred * x.iloc[t] / (x.iloc[t]**2 * P_pred + R)  # Kalman gain
        innovation = y.iloc[t] - beta_pred * x.iloc[t]
        
        beta[t] = beta_pred + K * innovation
        P[t] = (1 - K * x.iloc[t]) * P_pred
    
    return pd.Series(beta, index=y.index, name='kalman_beta')


# Example usage
if __name__ == "__main__":
    # Dummy data
    np.random.seed(42)
    x = pd.Series(np.cumsum(np.random.randn(1000) * 0.01) + 10)
    true_beta = 0.8
    y = true_beta * x + np.random.randn(1000) * 0.05
    
    beta_series = kalman_filter_beta(y, x, Q=1e-5, R=1e-3)
    
    print("Kalman Beta (last 5):", beta_series.tail())
    print("Mean Kalman Beta:", beta_series.mean())
