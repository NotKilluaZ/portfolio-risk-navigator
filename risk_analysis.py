import numpy as np
from data_pipeline import fetch_ticker_data, calculate_returns

# Calculates annual portfolio return
def portfolio_return(returns, weights):
    # Compute each tickers average daily return with returns.mean()
    # Take the weighted average of the daily means with np.sum(... * weights)
    # Multiply by total number of trading days a year (252 days)
    return np.sum(returns.mean() * weights) * 252

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

def sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate):
    # Sharpe Ratio relates a portfolio's returns to the risk associated with it
    # A higher value (ex. > 1) means more return for the related risk
    # A lower or negative number means more risk is taken than being returned
    # Risk free rate is approximated by the U.S. Treasury yield (13 Week Treasury Bill Yield)
    excess_return = portfolio_return - risk_free_rate
    return excess_return / portfolio_volatility if portfolio_volatility != 0 else np.nan