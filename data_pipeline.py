import yfinance as yf
import matplotlib.pyplot as plt

# Fetch ticker data (default 5 years) for one or multiple tickers
def fetch_ticker_data(tickers, period = "5y"):
    # Turns string into list of 1 ticker
    if isinstance(tickers, str):
        tickers = [tickers]

    # Download data
    data = yf.download(tickers, period = period)

    # Return the Adjusted Close prices
    # Adjusted close accounts for things like dividends that may have affected the prices
    return data["Close"]


def calculate_returns(prices):
    # Change in daily price of a ticker / stock
    returns = prices.pct_change().dropna()
    return returns