import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Fetch ticker data (default 5 years) for one or multiple tickers
def fetch_ticker_data(tickers, period = "5y"):
    # Turns string into list of 1 ticker
    if isinstance(tickers, str):
        tickers = [tickers]

    # Download data
    data = yf.download(tickers, period = period, auto_adjust = False)

    # Return the Adjusted Close prices
    # Adjusted close accounts for things like dividends that may have affected the prices
    return data["Close"]


def calculate_returns(prices):
    # Change in daily price of a ticker / stock
    returns = prices.pct_change().dropna()
    return returns

def fetch_risk_free_rate():
    # Fetch last 13 Week Treasury yield
    irx = yf.Ticker("^IRX")
    hist = irx.history(period="5d") # Last 5 days of data
    latest_yield = hist["Close"].iloc[-1] # Latest closing value
    return latest_yield / 100 # Changes % value to decimal


def fetch_exchange_rates(currencies: list, base: str = "USD") -> dict:
    """
    Fetch live exchange rates for a list of currencies relative to the base (USD).

    Uses Yahoo Finance forex tickers: e.g. "CADUSD=X" returns the price of
    1 CAD in USD.  Multiply any amount in a foreign currency by its rate to
    get the USD equivalent.

    Falls back to 1.0 (i.e. no conversion) for any rate that cannot be fetched,
    and records those currencies in the returned dict's "_failed" key.
    """
    rates: dict = {base: 1.0}
    failed: list = []

    for currency in currencies:
        if currency == base:
            continue
        ticker = f"{currency}{base}=X"
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                rates[currency] = float(hist["Close"].dropna().iloc[-1])
            else:
                rates[currency] = 1.0
                failed.append(currency)
        except Exception:
            rates[currency] = 1.0
            failed.append(currency)

    rates["_failed"] = failed  # type: ignore[assignment]
    return rates
