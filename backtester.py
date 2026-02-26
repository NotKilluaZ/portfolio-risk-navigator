"""
Simulates historical portfolio performance under two weight schemes:
  1. "Your Weights" — the allocation the user entered
  2. "Optimised Weights" — the allocation the app's optimiser produced

This answers the question: "If I had followed the app's suggestion
from the start of the historical window, would I have more money?"

Both portfolios start at the same dollar value and are compared
side-by-side with cumulative returns, drawdowns, and summary stats.
"""

import numpy as np
import pandas as pd


def simulate_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
    starting_value: float = 10_000.0,
) -> pd.DataFrame:
    """
    Simulate a buy-and-hold portfolio over the full returns history.

    Parameters
    ----------
    returns : DataFrame
        Daily returns for each asset (columns = tickers).
    weights : dict
        {ticker: weight} allocation.
    starting_value : float
        Initial portfolio dollar value.

    Returns
    -------
    DataFrame with columns:
        daily_return, cumulative_return, portfolio_value, drawdown
    """
    returns = returns.dropna()
    tickers = list(returns.columns)
    w = np.array([weights.get(t, 0.0) for t in tickers])

    # Normalise in case weights don't sum to 1 (e.g., vol targeting with cash)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum

    daily_returns = returns.values @ w

    df = pd.DataFrame(index=returns.index)
    df["daily_return"] = daily_returns
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    df["portfolio_value"] = starting_value * (1 + df["cumulative_return"])

    # Drawdown series
    running_max = df["portfolio_value"].cummax()
    df["drawdown"] = (df["portfolio_value"] - running_max) / running_max

    return df


def compute_backtest_stats(sim: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Compute summary statistics from a simulation DataFrame.

    Returns a dict with: total_return, annual_return, annual_volatility,
    sharpe, sortino, max_drawdown, calmar, best_day, worst_day, win_rate.
    """
    daily = sim["daily_return"]
    n_days = len(daily)
    n_years = n_days / 252

    total_return = float(sim["cumulative_return"].iloc[-1])
    annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    annual_vol = float(daily.std() * np.sqrt(252))

    # Sharpe
    daily_rf = risk_free_rate / 252
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # Sortino
    downside = daily[daily < 0]
    downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252)) if len(downside) > 0 else 0.0
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else float("inf")

    max_dd = float(sim["drawdown"].min())
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else float("inf")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "best_day": float(daily.max()),
        "worst_day": float(daily.min()),
        "win_rate": float((daily > 0).sum() / n_days) if n_days > 0 else 0.0,
    }


def build_comparison_stats(
    user_stats: dict,
    opt_stats: dict,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of backtest statistics.
    """
    metrics = [
        ("Total Return", "total_return", "{:.2%}"),
        ("Annual Return (CAGR)", "annual_return", "{:.2%}"),
        ("Annual Volatility", "annual_volatility", "{:.2%}"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("Sortino Ratio", "sortino", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar Ratio", "calmar", "{:.2f}"),
        ("Best Single Day", "best_day", "{:.2%}"),
        ("Worst Single Day", "worst_day", "{:.2%}"),
        ("Win Rate (% of up days)", "win_rate", "{:.1%}"),
    ]

    rows = []
    for label, key, fmt in metrics:
        u_val = user_stats[key]
        o_val = opt_stats[key]

        # Format, handling inf
        u_str = fmt.format(u_val) if np.isfinite(u_val) else "∞"
        o_str = fmt.format(o_val) if np.isfinite(o_val) else "∞"

        rows.append({
            "Metric": label,
            "Your Weights": u_str,
            "Optimised Weights": o_str,
        })

    return pd.DataFrame(rows)