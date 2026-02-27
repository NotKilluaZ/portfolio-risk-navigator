"""
Rebalancing Engine:

Takes current portfolio positions and optimizer-derived target weights,
then produces:
  - Per-asset drift (how far each position has moved from target)
  - Total portfolio drift
  - A concrete trade list with dollar amounts (buy/sell)
  - Rebalancing urgency flag based on a user-defined drift threshold

"""

import numpy as np
import pandas as pd


def compute_drift(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> pd.DataFrame:
    """
    Compute the per-asset drift between current and target allocations.

    Drift = |current_weight - target_weight| for each asset.
    Total drift = sum of all absolute drifts / 2 (since overweights
    and underweights are symmetric, dividing by 2 gives the fraction
    of the portfolio that needs to move).

    Returns a DataFrame with columns:
        Ticker, Current Weight, Target Weight, Drift, Direction
    """
    tickers = sorted(set(list(current_weights.keys()) + list(target_weights.keys())))
    rows = []

    for t in tickers:
        curr = current_weights.get(t, 0.0)
        tgt = target_weights.get(t, 0.0)
        drift = curr - tgt

        if abs(drift) < 1e-6:
            direction = "On target"
        elif drift > 0:
            direction = "Overweight"
        else:
            direction = "Underweight"

        rows.append({
            "Ticker": t,
            "Current Weight": curr,
            "Target Weight": tgt,
            "Drift": drift,
            "Abs Drift": abs(drift),
            "Direction": direction,
        })

    return pd.DataFrame(rows)


def total_drift(drift_df: pd.DataFrame) -> float:
    """
    Total portfolio drift: the fraction of the portfolio that would
    need to move to reach the target allocation.

    Sum of absolute drifts / 2 (since every sell has a corresponding buy).
    """
    return drift_df["Abs Drift"].sum() / 2.0


def needs_rebalancing(drift_df: pd.DataFrame, threshold: float = 0.05) -> bool:
    """Check if total drift exceeds the rebalancing threshold."""
    return total_drift(drift_df) > threshold


def generate_trade_list(
    drift_df: pd.DataFrame,
    total_portfolio_value: float,
) -> pd.DataFrame:
    """
    Convert weight drifts into concrete dollar-amount trades.

    A negative drift means the asset is underweight → BUY.
    A positive drift means the asset is overweight → SELL.

    Returns a DataFrame with columns:
        Ticker, Action, Amount ($), New Position ($), Direction
    Only includes assets that actually need to trade.
    """
    rows = []

    for _, row in drift_df.iterrows():
        drift = row["Drift"]
        if abs(drift) < 1e-6:
            continue  # no trade needed

        trade_dollars = abs(drift) * total_portfolio_value
        current_dollars = row["Current Weight"] * total_portfolio_value
        target_dollars = row["Target Weight"] * total_portfolio_value

        if drift > 0:
            action = "SELL"
        else:
            action = "BUY"

        rows.append({
            "Ticker": row["Ticker"],
            "Action": action,
            "Amount ($)": trade_dollars,
            "Current Position ($)": current_dollars,
            "Target Position ($)": target_dollars,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort: sells first (to free up cash), then buys
        action_order = {"SELL": 0, "BUY": 1}
        df["_sort"] = df["Action"].map(action_order)
        df = df.sort_values(["_sort", "Amount ($)"], ascending=[True, False])
        df = df.drop(columns=["_sort"]).reset_index(drop=True)

    return df


def format_drift_table(drift_df: pd.DataFrame) -> pd.DataFrame:
    """Format drift table for Streamlit display."""
    display = drift_df[["Ticker", "Current Weight", "Target Weight", "Drift", "Direction"]].copy()
    display["Current Weight"] = display["Current Weight"].map("{:.2%}".format)
    display["Target Weight"] = display["Target Weight"].map("{:.2%}".format)
    display["Drift"] = drift_df["Drift"].map(lambda d: f"{d:+.2%}")
    return display


def format_trade_list(trade_df: pd.DataFrame) -> pd.DataFrame:
    """Format trade list for Streamlit display."""
    if trade_df.empty:
        return trade_df
    display = trade_df.copy()
    display["Amount ($)"] = display["Amount ($)"].map("${:,.2f}".format)
    display["Current Position ($)"] = display["Current Position ($)"].map("${:,.2f}".format)
    display["Target Position ($)"] = display["Target Position ($)"].map("${:,.2f}".format)
    return display