from math import exp, log, sqrt
from typing import Literal
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from data_pipeline import fetch_ticker_data, calculate_returns

# Calculates annual portfolio return
def portfolio_return(returns, weights):
    # Annualised (geometric) portfolio return.
    # Using compound growth keeps the Sharpe numerator consistent with the
    # annualised volatility (both reflect a full-year horizon).

    weights_arr = np.array(weights, dtype=float)
    if returns.empty or not np.isclose(weights_arr.sum(), 1.0):
        return 0.0

    aligned = returns.dropna()
    if aligned.empty:
        return 0.0

    # Collapse the matrix of asset returns into a single daily portfolio series.
    portfolio_daily = aligned @ weights_arr

    # Compound the daily returns, then scale to a 252-trading-day year.
    cumulative_growth = (1 + portfolio_daily).prod()
    periods = portfolio_daily.shape[0]
    if cumulative_growth <= 0 or periods == 0:
        return 0.0

    return cumulative_growth ** (252 / periods) - 1

# Calculates the annual volatility of the portfolio
def portfolio_volatility(returns, weights):
    # Returns is a DataFrame of daily returns with each column representing a given ticker
    # .cov() computes the covariance matrix of the returns
    # Covariance shows how two assets move together
    # Multiply by 252 days to convert daily into annual value
    cov_matrix = returns.cov() * 252
    # Convert list to numpy array so we can transpose it later
    weights = np.array(weights) 
    # We use the linear algebra formula for portfolio varience: σ^2​=(w^⊤)Σw
    # where (w) is a vector of portfolio weights and (Σ) is the covariance matrix
    # Since the volatility formula is squared, we sqrt it using np.sqrt() to negate it
    # np.dot(cov_matrix, weights) multiplies the weights vector and the covariance matrix using dot product
    # np.dot(weights.T, ...) multiplies the transposed weights vector to sum the weighted
    # variances and covariances into a scalar value
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def sharpe_ratio(portfolio_return, std_deviation_portfolio_return, risk_free_rate):
    # Sharpe Ratio relates a portfolio's returns (usually annual) to the risk associated with it
    # A higher value (ex. > 1) means more return for the related risk
    # In general, a Sharpe Ratio >= 2 is a strong target to asim for
    # A lower number means more risk is taken than being returned
    # Risk free rate is approximated by the U.S. Treasury yield (13 Week Treasury Bill Yield)
    # We do not need to multiply the ratio by sqrt(252) since all our values are already annualized
    return (portfolio_return - risk_free_rate) / std_deviation_portfolio_return if std_deviation_portfolio_return != 0 else np.nan

OptionType = Literal["call", "put"]

def _portfolio_daily_returns(returns, weights):
    """Collapse asset returns into a single weighted portfolio series."""
    w = np.array(weights, dtype=float)
    aligned = returns.dropna()
    if aligned.empty or not np.isclose(w.sum(), 1.0):
        return np.array([])
    return (aligned @ w).values

def sortino_ratio(returns, weights, risk_free_rate: float = 0.0) -> float:
    """
    Like Sharpe, but only penalises downside volatility.

    Sortino = (R_p - R_f) / σ_downside

    σ_downside is computed from negative returns only, then annualised.
    This directly addresses that upside uncertainty
    is not risk — only losses are.
    """
    daily = _portfolio_daily_returns(returns, weights)
    if len(daily) == 0:
        return np.nan

    ann_ret = portfolio_return(returns, weights)
    negative = daily[daily < 0]
    if len(negative) == 0:
        return np.inf  # no downside observed

    downside_dev = np.sqrt(np.mean(negative ** 2)) * np.sqrt(252)
    if downside_dev < 1e-10:
        return np.nan
    return (ann_ret - risk_free_rate) / downside_dev


def value_at_risk(returns, weights, alpha: float = 0.95) -> float:
    """
    Historical VaR at confidence level alpha.

    Returns a positive number representing the loss threshold:
    "With alpha% confidence, daily loss will not exceed this value."
    """
    daily = _portfolio_daily_returns(returns, weights)
    if len(daily) == 0:
        return np.nan
    return -np.percentile(daily, (1 - alpha) * 100)


def conditional_var(returns, weights, alpha: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall at confidence level alpha.

    This is the 'Expected Loss' — the average
    of all losses that fall beyond the VaR threshold.

    Returns a positive number: "On the worst (1-alpha)% of days,
    the average loss is this much."
    """
    daily = _portfolio_daily_returns(returns, weights)
    if len(daily) == 0:
        return np.nan
    var = np.percentile(daily, (1 - alpha) * 100)
    tail = daily[daily <= var]
    if len(tail) == 0:
        return -var
    return -np.mean(tail)


def max_drawdown(returns, weights) -> float:
    """
    Largest peak-to-trough decline in cumulative portfolio value.

    Returns a negative number (e.g. -0.35 means a 35% drawdown).
    """
    daily = _portfolio_daily_returns(returns, weights)
    if len(daily) == 0:
        return 0.0
    cumulative = np.cumprod(1 + daily)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))


def calmar_ratio(returns, weights) -> float:
    """
    Calmar = Annualised Return / |Max Drawdown|

    Measures return earned per unit of drawdown risk.
    """
    ann_ret = portfolio_return(returns, weights)
    mdd = max_drawdown(returns, weights)
    if abs(mdd) < 1e-10:
        return np.nan
    return ann_ret / abs(mdd)


def downside_metrics(returns, weights, risk_free_rate: float = 0.0, alpha: float = 0.95) -> dict:
    """
    Compute all downside risk metrics in one call.  Returns a dict
    that can be directly displayed in Streamlit.
    """
    return {
        "sortino_ratio": sortino_ratio(returns, weights, risk_free_rate),
        "var_daily": value_at_risk(returns, weights, alpha),
        "cvar_daily": conditional_var(returns, weights, alpha),
        "max_drawdown": max_drawdown(returns, weights),
        "calmar_ratio": calmar_ratio(returns, weights),
    }
