"""
Asset Suggestion Engine:

Phase 1 — Fast screening
  Each candidate is blended at a fixed 10% trial weight and scored by
  the user's objective (Sharpe improvement, vol reduction, return lift,
  or correlation reduction).  The top 2 per asset-class category are
  kept for diversity, then the overall top 8 are shortlisted.

Phase 2 — Joint optimisation
  The shortlisted candidates are combined with the existing portfolio
  assets and a single optimiser pass is run.  The resulting weight for
  each candidate IS the confidence metric — it's the mathematically
  optimal allocation, not a synthetic score.

Candidate universe is a curated set of ~40 liquid ETFs spanning every
major asset class.  ETFs are used rather than individual stocks because:
  - They are already diversified (no single-event risk)
  - They have 5+ years of clean history
  - They cover every category a retail investor might want
  - Results are stable and actionable
"""

import numpy as np
import pandas as pd

from data_pipeline import fetch_ticker_data, calculate_returns
from optimizer import max_sharpe_weights, min_variance_weights


# Candidate universe
CANDIDATE_UNIVERSE: dict[str, dict] = {
    # US Equity – Broad
    "SPY":  {"name": "S&P 500 ETF (SPDR)",             "category": "US Equity – Broad"},
    "VTI":  {"name": "Total Stock Market ETF (Vanguard)","category": "US Equity – Broad"},
    "QQQ":  {"name": "Nasdaq-100 ETF (Invesco)",         "category": "US Equity – Broad"},
    "IWM":  {"name": "Russell 2000 ETF (iShares)",       "category": "US Equity – Broad"},
    "VTV":  {"name": "Value ETF (Vanguard)",              "category": "US Equity – Broad"},
    "VUG":  {"name": "Growth ETF (Vanguard)",             "category": "US Equity – Broad"},
    # US Equity – Dividend
    "SCHD": {"name": "US Dividend Equity ETF (Schwab)", "category": "US Equity – Dividend"},
    "VIG":  {"name": "Dividend Appreciation ETF (Vanguard)", "category": "US Equity – Dividend"},
    "HDV":  {"name": "High Dividend Yield ETF (iShares)", "category": "US Equity – Dividend"},
    # International Equity
    "VEU":  {"name": "FTSE All-World ex-US ETF (Vanguard)", "category": "International Equity"},
    "EFA":  {"name": "MSCI EAFE ETF (iShares)",           "category": "International Equity"},
    "VWO":  {"name": "FTSE Emerging Markets ETF (Vanguard)", "category": "International Equity"},
    "EEM":  {"name": "MSCI Emerging Markets ETF (iShares)", "category": "International Equity"},
    "VXUS": {"name": "Total International Stock ETF (Vanguard)", "category": "International Equity"},
    # Bonds
    "AGG":  {"name": "Core US Aggregate Bond ETF (iShares)", "category": "Bonds"},
    "BND":  {"name": "Total Bond Market ETF (Vanguard)", "category": "Bonds"},
    "TLT":  {"name": "20+ Year Treasury Bond ETF (iShares)", "category": "Bonds"},
    "IEF":  {"name": "7-10 Year Treasury Bond ETF (iShares)", "category": "Bonds"},
    "LQD":  {"name": "iBoxx $ Investment Grade Corp Bond ETF (iShares)", "category": "Bonds"},
    "HYG":  {"name": "iBoxx $ High Yield Corp Bond ETF (iShares)", "category": "Bonds"},
    "SCHP": {"name": "US TIPS ETF (Schwab)",              "category": "Bonds"},
    "SHY":  {"name": "1-3 Year Treasury Bond ETF (iShares)", "category": "Bonds"},
    # Real Estate
    "VNQ":  {"name": "Real Estate ETF (Vanguard)",        "category": "Real Estate"},
    "VNQI": {"name": "Global ex-US Real Estate ETF (Vanguard)", "category": "Real Estate"},
    # Commodities
    "GLD":  {"name": "Gold ETF (SPDR)",                   "category": "Commodities"},
    "SLV":  {"name": "Silver ETF (iShares)",              "category": "Commodities"},
    "DJP":  {"name": "Diversified Commodity ETF (iPath)", "category": "Commodities"},
    "USO":  {"name": "WTI Crude Oil ETF (United States)", "category": "Commodities"},
    # Sector ETFs
    "XLK":  {"name": "Technology Select Sector SPDR",     "category": "Sector ETFs"},
    "XLV":  {"name": "Health Care Select Sector SPDR",    "category": "Sector ETFs"},
    "XLF":  {"name": "Financial Select Sector SPDR",      "category": "Sector ETFs"},
    "XLE":  {"name": "Energy Select Sector SPDR",         "category": "Sector ETFs"},
    "XLU":  {"name": "Utilities Select Sector SPDR",      "category": "Sector ETFs"},
    "XLP":  {"name": "Consumer Staples Select Sector SPDR", "category": "Sector ETFs"},
    "XLI":  {"name": "Industrial Select Sector SPDR",     "category": "Sector ETFs"},
    # Alternatives / Multi-Asset
    "BNDX": {"name": "Total International Bond ETF (Vanguard)", "category": "Bonds"},
    "PDBC": {"name": "Diversified Commodity Strategy ETF (Invesco)", "category": "Commodities"},
}


# Phase 0 — Fetch candidate returns
def build_candidate_returns(
    existing_tickers: list[str],
    period: str = "5y",
) -> pd.DataFrame:
    """
    Batch-fetch daily returns for all universe tickers not already in the
    portfolio.  Tickers with fewer than 252 rows are silently dropped.

    Parameters
    ----------
    existing_tickers : list of ticker symbols already in the portfolio
    period           : yfinance period string (default '5y')

    Returns
    -------
    pd.DataFrame  — daily returns, columns = ticker symbols
    """
    candidates = [t for t in CANDIDATE_UNIVERSE if t not in existing_tickers]
    if not candidates:
        return pd.DataFrame()

    prices = fetch_ticker_data(candidates, period=period)
    # fetch_ticker_data may return a Series (single ticker) or DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    returns = calculate_returns(prices)

    # Drop columns with insufficient history
    min_rows = 252
    returns = returns.loc[:, returns.count() >= min_rows]

    return returns


# Phase 1 — Candidate screening
def _portfolio_sharpe(port_returns: np.ndarray, risk_free_rate: float) -> float:
    """Annualised Sharpe from a daily return series."""
    if len(port_returns) == 0:
        return -np.inf
    ann_ret = float(np.mean(port_returns)) * 252
    ann_vol = float(np.std(port_returns, ddof=1)) * np.sqrt(252)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def _portfolio_vol(port_returns: np.ndarray) -> float:
    """Annualised volatility from a daily return series."""
    if len(port_returns) == 0:
        return 0.0
    return float(np.std(port_returns, ddof=1)) * np.sqrt(252)


def _portfolio_ann_return(port_returns: np.ndarray) -> float:
    """Annualised return from a daily return series."""
    if len(port_returns) == 0:
        return 0.0
    return float(np.mean(port_returns)) * 252


def screen_candidates(
    candidate_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    weights: list | np.ndarray,
    risk_free_rate: float,
    objective: str,
    trial_weight: float = 0.10,
    top_per_category: int = 2,
    total_top_n: int = 8,
) -> tuple[list[str], dict]:
    """
    Phase 1: score candidates by objective metric and return a diverse shortlist.

    Parameters
    ----------
    candidate_returns  : daily returns for candidate assets (from build_candidate_returns)
    portfolio_returns  : daily returns for existing portfolio assets
    weights            : current portfolio weights (must align with portfolio_returns.columns)
    risk_free_rate     : annualised risk-free rate
    objective          : one of 'sharpe' | 'diversify' | 'returns' | 'stability'
    trial_weight       : fixed weight assigned to each candidate for trial blend (default 0.10)
    top_per_category   : how many to keep per asset category for diversity (default 2)
    total_top_n        : final shortlist size (default 8)

    Returns
    -------
    (shortlisted_tickers, trial_metrics_dict)
      shortlisted_tickers : list of ticker symbols in rank order
      trial_metrics_dict  : {ticker: {"delta_sharpe": float, "delta_vol": float, "delta_ret": float}}
    """
    if candidate_returns.empty:
        return [], {}

    w = np.array(weights, dtype=float)
    if w.sum() < 1e-10:
        w = np.ones(len(w)) / len(w)
    else:
        w = w / w.sum()

    # Align portfolio and candidate returns on common dates
    aligned = portfolio_returns.join(candidate_returns, how="inner").dropna()
    port_cols = list(portfolio_returns.columns)
    n_port = len(port_cols)

    if aligned.empty or n_port == 0:
        return [], {}

    w_trimmed = w[:n_port]  # guard if weights longer than aligned cols
    if len(w_trimmed) < n_port:
        w_trimmed = np.ones(n_port) / n_port

    # Baseline portfolio returns (using existing weights)
    port_daily = aligned[port_cols].values @ w_trimmed

    baseline_sharpe = _portfolio_sharpe(port_daily, risk_free_rate)
    baseline_vol    = _portfolio_vol(port_daily)
    baseline_ret    = _portfolio_ann_return(port_daily)

    candidate_cols = [c for c in candidate_returns.columns if c in aligned.columns]

    scores: dict[str, float] = {}
    trial_metrics: dict[str, dict] = {}

    for ticker in candidate_cols:
        cand_daily = aligned[ticker].values

        # Blend: existing portfolio at (1 - trial_weight), candidate at trial_weight
        blended = (1 - trial_weight) * port_daily + trial_weight * cand_daily

        trial_sharpe = _portfolio_sharpe(blended, risk_free_rate)
        trial_vol    = _portfolio_vol(blended)
        trial_ret    = _portfolio_ann_return(blended)

        delta_sharpe = trial_sharpe - baseline_sharpe
        delta_vol    = trial_vol    - baseline_vol      # negative = improvement
        delta_ret    = trial_ret    - baseline_ret

        trial_metrics[ticker] = {
            "delta_sharpe": delta_sharpe,
            "delta_vol":    delta_vol,
            "delta_ret":    delta_ret,
        }

        # Score by objective
        if objective == "sharpe":
            scores[ticker] = delta_sharpe
        elif objective == "diversify":
            # Lower correlation with existing portfolio = better
            corr = float(np.corrcoef(port_daily, cand_daily)[0, 1])
            scores[ticker] = -corr  # negate so higher = better
        elif objective == "returns":
            scores[ticker] = delta_ret
        elif objective == "stability":
            scores[ticker] = -delta_vol  # negate so higher = better (more vol reduction)
        else:
            scores[ticker] = delta_sharpe

    # Sort by score descending
    ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

    # Enforce category diversity: top_per_category per category
    category_counts: dict[str, int] = {}
    diverse_list: list[str] = []
    for ticker in ranked:
        cat = CANDIDATE_UNIVERSE.get(ticker, {}).get("category", "Other")
        count = category_counts.get(cat, 0)
        if count < top_per_category:
            diverse_list.append(ticker)
            category_counts[cat] = count + 1

    # From the diverse list, take the top total_top_n by original score
    shortlist = sorted(diverse_list, key=lambda t: scores[t], reverse=True)[:total_top_n]

    return shortlist, trial_metrics


# Phase 2 — Joint optimisation
def allocate_suggestions(
    shortlisted_tickers: list[str],
    trial_metrics: dict,
    candidate_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    current_weights_dict: dict,
    risk_free_rate: float,
    objective: str,
    max_weight: float = 0.50,
) -> pd.DataFrame:
    """
    Phase 2: run a single joint optimiser on existing + shortlisted candidates.

    The returned weight for each candidate is the confidence metric —
    it is the mathematically optimal allocation that the optimiser would
    give each new asset if you added it to your portfolio.

    Parameters
    ----------
    shortlisted_tickers  : output of screen_candidates
    trial_metrics        : output of screen_candidates (Δ Sharpe / Δ Vol per ticker)
    candidate_returns    : daily returns for candidates
    portfolio_returns    : daily returns for existing portfolio
    current_weights_dict : {ticker: weight} for existing assets
    risk_free_rate       : annualised risk-free rate
    objective            : 'sharpe' | 'diversify' | 'returns' | 'stability'
    max_weight           : per-asset cap for new candidate assets (default 0.50)

    Returns
    -------
    pd.DataFrame with columns:
      Ticker | Name | Category | Suggested Weight | Δ Sharpe (trial) | Δ Vol (trial)
    Sorted by Suggested Weight descending.  Rows < 0.5% are dropped.
    """
    if not shortlisted_tickers:
        return pd.DataFrame()

    # Build combined returns: existing portfolio + shortlisted candidates
    cand_in_data = [t for t in shortlisted_tickers if t in candidate_returns.columns]
    if not cand_in_data:
        return pd.DataFrame()

    combined = portfolio_returns.join(candidate_returns[cand_in_data], how="inner").dropna()
    if combined.empty or combined.shape[1] < 2:
        return pd.DataFrame()

    # Choose optimiser based on objective
    use_max_sharpe = objective in ("sharpe", "returns")

    # Build per-asset max_weight bounds:
    # existing assets: uncapped (they can go up to 1.0 or whatever)
    # new candidates:  capped at max_weight
    port_tickers = list(portfolio_returns.columns)

    try:
        if use_max_sharpe:
            opt = max_sharpe_weights(
                combined,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                long_only=True,
            )
        else:
            opt = min_variance_weights(
                combined,
                max_weight=max_weight,
                long_only=True,
                risk_free_rate=risk_free_rate,
            )
    except Exception:
        return pd.DataFrame()

    opt_weights = opt.get("weights", {})

    rows = []
    for ticker in cand_in_data:
        weight = opt_weights.get(ticker, 0.0)
        if weight < 0.005:  # drop if < 0.5%
            continue

        info = CANDIDATE_UNIVERSE.get(ticker, {})
        tm = trial_metrics.get(ticker, {})

        rows.append({
            "Ticker":            ticker,
            "Name":              info.get("name", ticker),
            "Category":          info.get("category", "Unknown"),
            "Suggested Weight":  weight,
            "Δ Sharpe (trial)":  tm.get("delta_sharpe", float("nan")),
            "Δ Vol (trial)":     tm.get("delta_vol",    float("nan")),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Suggested Weight", ascending=False).reset_index(drop=True)
    return df
