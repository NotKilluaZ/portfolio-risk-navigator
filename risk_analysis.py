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
    
    # We use the linear algebra formula for portfolio varience: σ^2​=(w^⊤)Σw
    # where (w) is a vector of portfolio weights and (Σ) is the covariance matrix
    # Since the volatility formula is squared, we sqrt it using np.sqrt() to negate it
    # np.dot(cov_matrix, weights) multiplies the weights vector and the covariance matrix using dot product
    # np.dot(weights.T, ...) multiplies the transposed weights vector to sum the weighted
    #  variances and covariances into a scalar value
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))