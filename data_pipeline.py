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

def fetch_option_chain(symbol: str, *, limit_expiries: int = 5, option_type: str = "call") -> pd.DataFrame:
    # Return option chain for the requested ticker.
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options or [])[:limit_expiries]

    if not expiries:
        return pd.DataFrame()

    rows = []
    spot = getattr(ticker.fast_info, "last_price", None)
    if spot is None:
        hist = ticker.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan

    for expiry in expiries:
        try:
            chain = ticker.option_chain(expiry)
        except Exception:
            continue
        table = getattr(chain, f"{option_type}s", None)
        if table is None or table.empty:
            continue
        table = table.copy()
        table["expiry"] = pd.to_datetime(expiry)
        table["underlying_price"] = spot
        rows.append(table[["strike", "bid", "ask", "lastPrice", "expiry", "underlying_price"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["mid_price"] = out[["bid", "ask"]].mean(axis=1, skipna=True)
    out["mid_price"] = out["mid_price"].fillna(out["lastPrice"])
    out = out.dropna(subset=["mid_price", "strike", "expiry", "underlying_price"])
    out["moneyness"] = out["strike"] / out["underlying_price"]
    now = pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert(None)
    else:
        now = now.tz_localize(None)
    expiry = out["expiry"]
    if expiry.dt.tz is not None:
        expiry = expiry.dt.tz_convert(None)
    else:
        expiry = expiry.dt.tz_localize(None)
    out["expiry"] = expiry
    out["time_to_maturity"] = (out["expiry"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    return out.reset_index(drop=True)
