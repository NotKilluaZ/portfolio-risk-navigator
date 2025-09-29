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

# BSM for options pricing model
def black_scholes_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = "call",
) -> float:
    if time_to_maturity <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)

    sigma_sqrt_t = volatility * sqrt(time_to_maturity)
    d1 = (log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    if option_type == "call":
        return spot * norm.cdf(d1) - strike * exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    else:
        return strike * exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)

# Call BSM to get the implied volatility at each strike price (moneyness) & time to maturity
def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    option_type: OptionType = "call",
    *,
    lower: float = 1e-4,
    upper: float = 5.0,
) -> float:
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if market_price <= intrinsic or time_to_maturity <= 0:
        return np.nan

    def objective(vol: float) -> float:
        return black_scholes_price(spot, strike, time_to_maturity, risk_free_rate, vol, option_type) - market_price

    try:
        return brentq(objective, lower, upper, maxiter=100, xtol=1e-6)
    except ValueError:
        return np.nan