import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _annualised_stats(weights, mean_returns, cov_matrix):
    """Return (annualised_return, annualised_vol) for a weight vector."""
    w = np.array(weights)
    port_return = np.dot(w, mean_returns) * 252
    port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix * 252, w)))
    return port_return, port_vol


def _check_inputs(returns: pd.DataFrame):
    """Validate and clean the returns DataFrame."""
    returns = returns.dropna()
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets with overlapping return history.")
    return returns


def min_variance_weights(
    returns: pd.DataFrame,
    max_weight: float = 1.0,
    long_only: bool = True,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Find weights that minimise portfolio variance.

    Does NOT require expected return estimates — only the covariance
    matrix — making it far more robust than mean-variance optimisation.
    """
    returns = _check_inputs(returns)
    n = returns.shape[1]
    cov = returns.cov().values  # daily covariance

    def objective(w):
        return np.dot(w, np.dot(cov, w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    lb = 0.0 if long_only else -max_weight
    bounds = [(lb, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    weights = result.x
    mean_ret = returns.mean().values
    ann_ret, ann_vol = _annualised_stats(weights, mean_ret, cov)

    return {
        "weights": dict(zip(returns.columns, weights)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
    }


def max_sharpe_weights(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    long_only: bool = True,
) -> dict:
    """
    Find weights that maximise the Sharpe ratio.

    This is sensitive to expected return estimates. Consider
    using min_variance or risk_parity if return forecasts are uncertain.
    """
    returns = _check_inputs(returns)
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    def neg_sharpe(w):
        ann_ret, ann_vol = _annualised_stats(w, mean_ret, cov)
        if ann_vol < 1e-10:
            return 1e6
        return -(ann_ret - risk_free_rate) / ann_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    lb = 0.0 if long_only else -max_weight
    bounds = [(lb, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    weights = result.x
    ann_ret, ann_vol = _annualised_stats(weights, mean_ret, cov)

    return {
        "weights": dict(zip(returns.columns, weights)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
    }


def risk_parity_weights(
    returns: pd.DataFrame,
    max_weight: float = 1.0,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Find weights so that each asset contributes equally to total
    portfolio variance.

    Does NOT require expected return estimates. Highly robust and
    widely used by institutional allocators.
    """
    returns = _check_inputs(returns)
    n = returns.shape[1]
    cov = returns.cov().values

    def _risk_contribution(w):
        port_var = np.dot(w, np.dot(cov, w))
        if port_var < 1e-14:
            return np.zeros(n)
        marginal = np.dot(cov, w)
        rc = w * marginal / np.sqrt(port_var)
        return rc

    def objective(w):
        rc = _risk_contribution(w)
        target_rc = np.sqrt(np.dot(w, np.dot(cov, w))) / n
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, max_weight)] * n  # strictly positive
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-14})

    weights = result.x
    mean_ret = returns.mean().values
    ann_ret, ann_vol = _annualised_stats(weights, mean_ret, cov)

    # Compute actual risk contributions for display
    rc = _risk_contribution(weights)
    total_risk = np.sum(rc)
    rc_pct = rc / total_risk if total_risk > 0 else np.zeros(n)

    return {
        "weights": dict(zip(returns.columns, weights)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
        "risk_contributions": dict(zip(returns.columns, rc_pct)),
    }


def min_cvar_weights(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    target_return: float | None = None,
    max_weight: float = 1.0,
    long_only: bool = True,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Find weights that minimise CVaR (Expected Shortfall) at confidence
    level alpha.

    This directly minimises the average loss in the worst (1-alpha)%
    of scenarios.

    Uses the Rockafellar-Uryasev (2001) reformulation which converts
    the CVaR minimisation into a smooth optimisation problem.
    """
    returns = _check_inputs(returns)
    n_assets = returns.shape[1]
    T = returns.shape[0]
    ret_matrix = returns.values  # (T, n_assets)

    # Decision variables: [w_1, ..., w_n, zeta]
    # where zeta is the VaR threshold in the R-U formulation.

    def objective(x):
        w = x[:n_assets]
        zeta = x[n_assets]
        portfolio_returns = ret_matrix @ w  # (T,)
        losses = -portfolio_returns
        excess = np.maximum(losses - zeta, 0.0)
        cvar = zeta + (1.0 / (T * (1.0 - alpha))) * np.sum(excess)
        return cvar

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n_assets]) - 1.0}]

    if target_return is not None:
        mean_ret = returns.mean().values
        constraints.append({
            "type": "ineq",
            "fun": lambda x: np.dot(x[:n_assets], mean_ret) * 252 - target_return,
        })

    lb = 0.0 if long_only else -max_weight
    bounds = [(lb, max_weight)] * n_assets + [(-1.0, 1.0)]  # zeta is unbounded-ish
    x0 = np.append(np.ones(n_assets) / n_assets, 0.0)

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-12})

    weights = result.x[:n_assets]
    mean_ret = returns.mean().values
    cov = returns.cov().values
    ann_ret, ann_vol = _annualised_stats(weights, mean_ret, cov)

    # Compute the actual CVaR at the solution
    port_ret = ret_matrix @ weights
    var_level = np.percentile(port_ret, (1 - alpha) * 100)
    cvar_value = -np.mean(port_ret[port_ret <= var_level])

    return {
        "weights": dict(zip(returns.columns, weights)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0,
        "cvar_daily": cvar_value,
        "cvar_annualised": cvar_value * np.sqrt(252),
    }


def compute_efficient_frontier(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_points: int = 50,
    max_weight: float = 1.0,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by sweeping target returns from
    the minimum-variance return to the maximum single-asset return.

    Returns a DataFrame with columns: target_return, volatility, sharpe.
    """
    returns = _check_inputs(returns)
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    # Find the range of feasible returns
    min_var = min_variance_weights(returns, max_weight=max_weight, long_only=long_only)
    min_ret = min_var["annual_return"]
    max_ret = float(np.max(mean_ret) * 252)

    if min_ret >= max_ret:
        max_ret = min_ret + 0.01

    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier = []

    for target in target_returns:
        def obj(w):
            return np.dot(w, np.dot(cov, w))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, t=target: np.dot(w, mean_ret) * 252 - t},
        ]
        lb = 0.0 if long_only else -max_weight
        bounds = [(lb, max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(obj, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})

        if result.success:
            w = result.x
            ann_ret, ann_vol = _annualised_stats(w, mean_ret, cov)
            sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
            frontier.append({
                "target_return": ann_ret,
                "volatility": ann_vol,
                "sharpe": sharpe,
            })

    return pd.DataFrame(frontier)


def compare_portfolios(
    current_weights: dict,
    optimised: dict,
    tickers: list,
) -> pd.DataFrame:
    """
    Build a comparison table: current vs optimised weights with deltas.
    """
    rows = []
    for t in tickers:
        curr = current_weights.get(t, 0.0)
        opt = optimised["weights"].get(t, 0.0)
        rows.append({
            "Ticker": t,
            "Current Weight": curr,
            "Optimised Weight": opt,
            "Delta": opt - curr,
        })
    return pd.DataFrame(rows)