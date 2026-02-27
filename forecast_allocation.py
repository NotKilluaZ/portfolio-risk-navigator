"""
Connects the LSTM volatility forecast to actual portfolio construction
decisions.

Two modes:

1. Forecast-Adjusted Optimisation
   Replace the diagonal of the historical covariance matrix with
   LSTM-predicted per-asset volatilities (keeping historical correlations).
   Then run standard optimisers (min-variance, risk parity) on this
   forward-looking covariance.  This means the optimiser reacts to
   predicted risk, not just past risk.

2. Volatility Targeting
   Scale portfolio exposure so that the predicted portfolio volatility
   stays near a user-chosen target.  When predicted vol rises the
   engine reduces exposure; when it falls it increases exposure.
   This is widely used by institutional systematic strategies.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize



# Forecast all tickers
def forecast_all_vols(
    returns: pd.DataFrame,
    forecaster,
    horizon: int = 5,
) -> dict:
    """
    Run the LSTM forecaster on every asset in the portfolio.

    Returns a dict of {ticker: predicted_annualised_vol}.
    For any ticker where the forecast fails (insufficient data),
    falls back to the trailing 30-day historical vol.
    """
    results = {}
    for col in returns.columns:
        series = returns[col].dropna()
        try:
            pred = forecaster.predict(series, horizon=horizon)
            if len(pred) > 0:
                # pred is daily vol; take the terminal (horizon-end) value
                daily_vol = float(pred[-1])
            else:
                daily_vol = float(series.rolling(30).std().iloc[-1])
        except Exception:
            daily_vol = float(series.rolling(30).std().iloc[-1])

        results[col] = {
            "daily_vol": daily_vol,
            "annual_vol": daily_vol * np.sqrt(252),
        }

    return results



# Forecast-adjusted covariance matrix
def build_forecast_covariance(
    returns: pd.DataFrame,
    forecast_vols: dict,
) -> np.ndarray:
    """
    Build a forward-looking covariance matrix by replacing the
    historical volatilities with LSTM-predicted volatilities while
    preserving the historical correlation structure.

    This is the pragmatic middle ground: the LSTM is univariate
    (one ticker at a time), so we can't forecast the full covariance
    directly.  Instead we decompose:

        Σ_forecast = D_forecast · R_hist · D_forecast

    where:
        D_forecast = diag(predicted daily vols)
        R_hist     = historical correlation matrix

    This keeps the co-movement structure from history but scales
    each asset's risk by the forward-looking prediction.
    """
    returns = returns.dropna()
    tickers = list(returns.columns)

    # Historical correlation matrix — clip extremes to prevent concentration
    # in near-identical assets (same logic as _regularize_cov in optimizer.py)
    corr = returns.corr().values
    np.clip(corr, -0.98, 0.98, out=corr)
    np.fill_diagonal(corr, 1.0)

    # Diagonal of predicted daily vols
    pred_vols = np.array([forecast_vols[t]["daily_vol"] for t in tickers])
    D = np.diag(pred_vols)

    # Σ_forecast = D · R · D
    cov_forecast = D @ corr @ D

    return cov_forecast


# Forecast-driven optimisers
def _annualised_stats_from_cov(weights, mean_returns, cov_daily):
    """Annualised return and vol from a daily cov matrix."""
    w = np.array(weights)
    port_return = np.dot(w, mean_returns) * 252
    port_vol = np.sqrt(np.dot(w, np.dot(cov_daily * 252, w)))
    return port_return, port_vol


def forecast_min_variance(
    returns: pd.DataFrame,
    forecast_vols: dict,
    max_weight: float = 1.0,
    long_only: bool = True,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Minimum-variance portfolio using the forecast-adjusted covariance.
    """
    returns = returns.dropna()
    n = returns.shape[1]
    cov = build_forecast_covariance(returns, forecast_vols)
    mean_ret = returns.mean().values

    def objective(w):
        return np.dot(w, np.dot(cov, w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    lb = 0.0 if long_only else -max_weight
    bounds = [(lb, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    weights = result.x
    ann_ret, ann_vol = _annualised_stats_from_cov(weights, mean_ret, cov)

    return {
        "weights": dict(zip(returns.columns, weights)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
        "method": "Forecast Min-Variance",
    }


def forecast_risk_parity(
    returns: pd.DataFrame,
    forecast_vols: dict,
    max_weight: float = 1.0,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Risk parity using the forecast-adjusted covariance.

    Each asset contributes equally to *predicted* portfolio risk,
    so if the LSTM expects one asset's vol to spike, its weight
    is automatically reduced.
    """
    returns = returns.dropna()
    n = returns.shape[1]
    cov = build_forecast_covariance(returns, forecast_vols)
    mean_ret = returns.mean().values

    def _risk_contribution(w):
        port_var = np.dot(w, np.dot(cov, w))
        if port_var < 1e-14:
            return np.zeros(n)
        marginal = np.dot(cov, w)
        return w * marginal / np.sqrt(port_var)

    def objective(w):
        rc = _risk_contribution(w)
        target_rc = np.sqrt(np.dot(w, np.dot(cov, w))) / n
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-14})

    weights = result.x
    ann_ret, ann_vol = _annualised_stats_from_cov(weights, mean_ret, cov)

    rc = _risk_contribution(weights)
    total_risk = np.sum(rc)
    rc_pct = rc / total_risk if total_risk > 0 else np.zeros(n)

    return {
        "weights": dict(zip(returns.columns, weights)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
        "risk_contributions": dict(zip(returns.columns, rc_pct)),
        "method": "Forecast Risk-Parity",
    }


# Volatility targeting
def volatility_target_weights(
    base_weights: dict,
    returns: pd.DataFrame,
    forecast_vols: dict,
    target_vol: float = 0.15,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Scale portfolio exposure so that predicted annualised portfolio
    volatility matches a user-chosen target.

    When predicted vol > target:  reduce exposure (leverage < 1)
    When predicted vol < target:  increase exposure (leverage > 1)

    The scaling factor is:  λ = target_vol / predicted_portfolio_vol

    Individual weights are then:  w_scaled = base_weights * λ

    The residual (1 - sum(w_scaled)) is implicitly "cash" (no risk).
    If λ > 1, the portfolio is leveraged beyond 100% — we cap at 1.5x
    to keep things realistic for a retail context.
    """
    returns = returns.dropna()
    tickers = list(returns.columns)
    n = len(tickers)

    w_base = np.array([base_weights.get(t, 0.0) for t in tickers])
    if np.sum(w_base) < 1e-10:
        w_base = np.ones(n) / n

    cov = build_forecast_covariance(returns, forecast_vols)
    pred_port_vol = np.sqrt(np.dot(w_base, np.dot(cov * 252, w_base)))

    if pred_port_vol < 1e-10:
        leverage = 1.0
    else:
        leverage = target_vol / pred_port_vol

    # Cap leverage for retail context
    leverage = float(np.clip(leverage, 0.1, 1.5))

    w_scaled = w_base * leverage
    cash_weight = max(0.0, 1.0 - np.sum(w_scaled))

    mean_ret = returns.mean().values
    ann_ret, ann_vol = _annualised_stats_from_cov(w_scaled, mean_ret, cov)

    result = {
        "weights": {t: float(w) for t, w in zip(tickers, w_scaled)},
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
        "leverage": leverage,
        "cash_weight": cash_weight,
        "predicted_port_vol": pred_port_vol,
        "target_vol": target_vol,
        "method": "Volatility Target",
    }

    if cash_weight > 1e-4:
        result["weights"]["Cash"] = cash_weight

    return result


# Comparison helper
def build_vol_comparison_table(
    returns: pd.DataFrame,
    forecast_vols: dict,
) -> pd.DataFrame:
    """
    Build a per-asset table comparing historical vs predicted vol,
    with a change indicator.
    """
    rows = []
    for col in returns.columns:
        hist_vol = float(returns[col].std() * np.sqrt(252))
        pred_vol = forecast_vols[col]["annual_vol"]
        change = pred_vol - hist_vol
        direction = "↑ Rising" if change > 0.005 else ("↓ Falling" if change < -0.005 else "→ Stable")

        rows.append({
            "Ticker": col,
            "Historical Vol": hist_vol,
            "Predicted Vol": pred_vol,
            "Change": change,
            "Direction": direction,
        })

    return pd.DataFrame(rows)


def format_vol_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Format for Streamlit display."""
    display = df.copy()
    display["Historical Vol"] = display["Historical Vol"].map("{:.2%}".format)
    display["Predicted Vol"] = display["Predicted Vol"].map("{:.2%}".format)
    display["Change"] = df["Change"].map(lambda c: f"{c:+.2%}")
    return display