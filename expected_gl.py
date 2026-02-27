"""
Expected Gain / Expected Loss Analysis:

Implements the risk framework from MIT OCW Lecture 13 (Prof. Jake Xia):

    "Replace volatility with a direct assessment of Expected Gain (G)
     and Expected Loss (L).  This reframes the investment objective
     around managing the downside and directly informs the optimal
     investment size."

For each asset and the portfolio as a whole, we compute:
  - Expected Gain (G):  average return on up days
  - Expected Loss (L):  average |return| on down days
  - G/L Ratio:          reward per unit of downside
  - Win Rate:           fraction of positive-return days
  - Skill Ratio:        ranges -1 to +1, maps directly to sizing
  - Kelly Fraction:     approximate optimal position size

The Skill Ratio connects to the Kelly Criterion: a high skill ratio
with a high G/L ratio suggests the investor should size up; a negative
skill ratio means the position is destroying capital.
"""

import numpy as np
import pandas as pd


def _asset_gl_stats(daily_returns: np.ndarray) -> dict:
    """Compute G/L stats for a single return series."""
    if len(daily_returns) == 0:
        return {
            "expected_gain": np.nan,
            "expected_loss": np.nan,
            "gl_ratio": np.nan,
            "win_rate": np.nan,
            "skill_ratio": np.nan,
            "kelly_fraction": np.nan,
        }

    up = daily_returns[daily_returns > 0]
    down = daily_returns[daily_returns < 0]

    expected_gain = float(np.mean(up)) if len(up) > 0 else 0.0
    expected_loss = float(np.abs(np.mean(down))) if len(down) > 0 else 0.0
    win_rate = len(up) / len(daily_returns)
    loss_rate = 1.0 - win_rate

    # G/L Ratio: how much you make on good days per unit of bad-day loss
    gl_ratio = expected_gain / expected_loss if expected_loss > 1e-10 else np.inf

    # Skill Ratio: ranges from -1 (pure loss) to +1 (pure gain)
    # Positive means the asset/portfolio has positive expected value
    # Directly derived from the lecture's framework for sizing decisions
    if expected_loss > 1e-10:
        skill_ratio = (win_rate * expected_gain - loss_rate * expected_loss) / expected_loss
    else:
        skill_ratio = 1.0 if expected_gain > 0 else 0.0

    # Kelly-style optimal fraction: how much of capital to allocate
    # Based on the relationship: f* ≈ skill_ratio / gl_ratio
    # Capped at 1.0 since we're long-only in the app context
    if gl_ratio > 1e-10 and np.isfinite(gl_ratio):
        kelly = skill_ratio / gl_ratio
        kelly = float(np.clip(kelly, 0.0, 1.0))
    else:
        kelly = 0.0

    return {
        "expected_gain": expected_gain,
        "expected_loss": expected_loss,
        "gl_ratio": gl_ratio,
        "win_rate": win_rate,
        "skill_ratio": skill_ratio,
        "kelly_fraction": kelly,
    }


def compute_gl_table(returns: pd.DataFrame, weights: list | np.ndarray) -> pd.DataFrame:
    """
    Compute Expected Gain / Loss metrics for each asset and
    the overall portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, one column per asset.
    weights : array-like
        Portfolio weights (must sum to 1).

    Returns
    -------
    pd.DataFrame with columns:
        Asset, Expected Gain, Expected Loss, G/L Ratio,
        Win Rate, Skill Ratio, Kelly Fraction
    """
    returns = returns.dropna()
    w = np.array(weights, dtype=float)
    rows = []

    # Per-asset stats
    for col in returns.columns:
        stats = _asset_gl_stats(returns[col].values)
        stats["asset"] = col
        rows.append(stats)

    # Portfolio-level stats
    if len(w) == returns.shape[1] and np.isclose(w.sum(), 1.0):
        port_daily = (returns @ w).values
        port_stats = _asset_gl_stats(port_daily)
        port_stats["asset"] = "Portfolio"
        rows.append(port_stats)

    df = pd.DataFrame(rows)

    # Reorder and rename for display
    df = df[["asset", "expected_gain", "expected_loss", "gl_ratio",
             "win_rate", "skill_ratio", "kelly_fraction"]]
    df.columns = [
        "Asset",
        "Exp. Gain (G)",
        "Exp. Loss (L)",
        "G/L Ratio",
        "Win Rate",
        "Skill Ratio",
        "Kelly Fraction",
    ]
    return df


def format_gl_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the G/L table with human-friendly formatting."""
    display = df.copy()
    display["Exp. Gain (G)"] = display["Exp. Gain (G)"].map(
        lambda x: f"{x:.4%}" if np.isfinite(x) else "—"
    )
    display["Exp. Loss (L)"] = display["Exp. Loss (L)"].map(
        lambda x: f"{x:.4%}" if np.isfinite(x) else "—"
    )
    display["G/L Ratio"] = display["G/L Ratio"].map(
        lambda x: f"{x:.2f}" if np.isfinite(x) else "∞"
    )
    display["Win Rate"] = display["Win Rate"].map(
        lambda x: f"{x:.1%}" if np.isfinite(x) else "—"
    )
    display["Skill Ratio"] = display["Skill Ratio"].map(
        lambda x: f"{x:+.3f}" if np.isfinite(x) else "—"
    )
    display["Kelly Fraction"] = display["Kelly Fraction"].map(
        lambda x: f"{x:.1%}" if np.isfinite(x) else "—"
    )
    return display