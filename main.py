import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from data_pipeline import (
    fetch_ticker_data,
    calculate_returns,
    fetch_risk_free_rate,
    fetch_exchange_rates,
)
from risk_analysis import (
    portfolio_return, 
    portfolio_volatility, 
    sharpe_ratio as compute_sharpe_ratio,
    downside_metrics
)
from expected_gl import compute_gl_table, format_gl_table
from expected_gl import compute_gl_table, format_gl_table
from lstm_model import load_forecaster
from optimizer import (
    min_variance_weights,
    max_sharpe_weights,
    risk_parity_weights,
    min_cvar_weights,
    compute_efficient_frontier,
    compare_portfolios
)
from rebalancer import (
    compute_drift,
    total_drift,
    needs_rebalancing,
    generate_trade_list,
    format_drift_table,
    format_trade_list,
)
from forecast_allocation import (
    forecast_all_vols,
    forecast_min_variance,
    forecast_risk_parity,
    volatility_target_weights,
    build_vol_comparison_table,
    format_vol_comparison,
    build_forecast_covariance,
)
from backtester import (
    simulate_portfolio,
    compute_backtest_stats,
    build_comparison_stats,
)
from asset_suggester import (
    build_candidate_returns,
    screen_candidates,
    allocate_suggestions,
    CANDIDATE_UNIVERSE,
)

@st.cache_resource
def get_forecaster():
    # Cache the pre-trained LSTM so the app only loads weights once per session.
    return load_forecaster(device="cpu")

SUPPORTED_CURRENCIES = ["USD", "CAD", "EUR", "GBP", "CHF", "AUD", "JPY", "HKD", "NZD", "SEK", "NOK", "DKK", "MXN", "SGD"]

@st.cache_data(ttl=3600)
def get_exchange_rates(currencies_tuple: tuple) -> dict:
    # Cache rates for 1 hour so repeated rerenders don't hammer Yahoo Finance.
    return fetch_exchange_rates(list(currencies_tuple))

@st.cache_data(ttl=3600)
def get_candidate_returns(existing_tickers_tuple: tuple) -> pd.DataFrame:
    # Cache candidate returns for 1 hour, keyed on the current portfolio's tickers.
    return build_candidate_returns(list(existing_tickers_tuple))

# Function to make gradient health score scale
def gradient_steps(n=100):
    steps = []
    for i, val in enumerate(np.linspace(0, 100, n+1)[:-1]):
        # interpolate color (red→yellow→green)
        if val < 50:
            # red → yellow
            ratio = val / 50
            r = 255
            g = int(255 * ratio)
            b = 0
        else:
            # yellow → green
            ratio = (val - 50) / 50
            r = int(255 * (1 - ratio))
            g = 255
            b = 0
        color = f"rgb({r},{g},{b})"
        steps.append({'range': [val, val + 100/n], 'color': color})
    return steps


st.title("Portfolio Risk Navigator")
st.subheader("Enter Portfolio Assets")

tickers = []
amounts = []
currencies = []

num_assets = st.number_input("How many assets are in your portfolio?", min_value=1, max_value=20, value=1, step=1)

# Column headers
h0, h1, h2 = st.columns([2, 1, 3])
with h0:
    st.caption("Ticker")
with h1:
    st.caption("Currency")
with h2:
    st.caption("Amount Invested")

for i in range(num_assets):
    cols = st.columns([2, 1, 3])

    with cols[0]:
        ticker = st.text_input(f"Ticker {i+1}", value="MSFT" if i == 0 else "", key=f"ticker_{i}", label_visibility="collapsed")
        tickers.append(ticker)

    with cols[1]:
        currency = st.selectbox(f"Currency {i+1}", SUPPORTED_CURRENCIES, key=f"currency_{i}", label_visibility="collapsed")
        currencies.append(currency)

    with cols[2]:
        amount = st.number_input(
            f"Amount {i+1}",
            min_value=0.00, value=1000.00, step=0.01, format="%.2f",
            key=f"amount_input_{i}",
            label_visibility="collapsed",
            help=f"Enter the amount invested in this asset in {currency}.",
        )
        amounts.append(amount)

# Validate tickers (no blanks, no duplicates)
normalized_tickers = []
duplicates = set()
seen = set()

for t in tickers:
    cleaned = t.strip().upper()
    if not cleaned:
        continue  # ignore blank entries
    if cleaned in seen:
        duplicates.add(cleaned)
    else:
        seen.add(cleaned)
    normalized_tickers.append(cleaned)

if duplicates:
    st.error(f"Duplicate tickers detected: {', '.join(sorted(duplicates))}. Please use each symbol only once.")
    st.stop()

# Fetch live exchange rates for any non-USD currencies in the portfolio
unique_foreign = tuple(sorted(set(c for c in currencies if c != "USD")))
if unique_foreign:
    with st.spinner("Fetching live exchange rates..."):
        rates = get_exchange_rates(("USD",) + unique_foreign)
    failed = rates.get("_failed", [])
    if failed:
        st.warning(
            f"Could not fetch live rates for: {', '.join(failed)}. "
            "Amounts in those currencies are treated as USD. Check your connection."
        )
else:
    rates = {"USD": 1.0}

# Convert every amount to USD for weight normalisation
usd_amounts = [amt * rates.get(curr, 1.0) for amt, curr in zip(amounts, currencies)]
total_usd = sum(usd_amounts)
weights = [usd / total_usd for usd in usd_amounts] if total_usd > 0 else [0.0] * len(usd_amounts)

# Show weights with currency conversion details
st.write("**Portfolio Weights (normalised to USD):**")
for t, w, amt, curr, usd_amt in zip(tickers, weights, amounts, currencies, usd_amounts):
    if t:
        if curr != "USD":
            rate = rates.get(curr, 1.0)
            st.write(f"**{t}:** {w:.2%}  —  {amt:,.2f} {curr} × {rate:.4f} = ${usd_amt:,.2f} USD")
        else:
            st.write(f"**{t}:** {w:.2%}  —  ${amt:,.2f} USD")

if unique_foreign:
    rate_strs = [f"1 {c} = {rates[c]:.4f} USD" for c in unique_foreign if c not in rates.get("_failed", [])]
    if rate_strs:
        st.caption("Live rates (cached 1 hr): " + "  |  ".join(rate_strs))

prices = fetch_ticker_data(tickers)
returns = calculate_returns(prices)
# Compute 30-day rolling volatility for each ticker (business days)
rolling_window = 30  # adjust if you want
historical_vol_df = returns.rolling(window=rolling_window).std().dropna()

st.text("")  # blank line for padding

st.write("**Portfolio Price Chart:**")
st.line_chart(prices)

st.text("")

st.write(f"**{rolling_window}-Day Historical Volatility (%):**")
if historical_vol_df.empty:
    st.warning("Not enough data to compute rolling volatility for the selected window.")
else:
    st.line_chart(historical_vol_df * 100)

st.write("Projected Annual Portfolio Return: ", portfolio_return(returns, weights))
st.write("Projected Annual Portfolio Volatility: ", portfolio_volatility(returns, weights))

# Show a correlation heat map and how each ticker pair is correlated
# Higher value means more correlated (if one goes up the other will go up as well)
# Lower means less correlated
# Negative values (closer to -1) mean inversely proportional (if one goes up in price, the other will fall)
corr = returns.corr()
corr_chart = px.imshow(corr, text_auto = True, aspect = "auto", title = "Correlation Heatmap")
st.plotly_chart(corr_chart)

# Pie chart shows visual distribution ratios of the portfolio
pie_chart = go.Figure(data = [go.Pie(labels = tickers, values = weights)])
pie_chart.update_layout(title = "Asset Distribution Chart")
st.plotly_chart(pie_chart)

# Sharpe Ratio relates a portfolio's returns to the risk associated with it
# A higher value (ex. > 1) means more return for the related risk
# A lower or negative number means more risk is taken than being returned
# Risk free rate is approximated by the U.S. Treasury yield (13 Week Treasury Bill Yield)
risk_free_rate = fetch_risk_free_rate()
p_return = portfolio_return(returns, weights)
p_volatility = portfolio_volatility(returns, weights)
sharpe_ratio = compute_sharpe_ratio(p_return, p_volatility, risk_free_rate)


# Display 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Expected Annual Return",
        f"{p_return:.2%}",
        help=(
            "The projected annual return based on historical daily returns, "
            "annualised over 252 trading days. "
            "For reference, the S&P 500 has historically averaged ~10% per year. "
            "Above 10% is strong; below 0% means the portfolio is losing money."
        ),
    )

with col2:
    st.metric(
        "Annualised Volatility",
        f"{p_volatility:.2%}",
        help=(
            "How much the portfolio's value fluctuates over a year. "
            "Lower is calmer. "
            "For reference, the S&P 500 typically has ~15–20% annual volatility. "
            "Below 10% is very stable (bond-like); above 30% is highly volatile."
        ),
    )

with col3:
    st.metric(
        "Annualised Sharpe Ratio",
        f"{sharpe_ratio:.2f}",
        help=(
            "Return earned per unit of risk taken, after subtracting the risk-free rate. "
            "Higher is better. "
            "Below 0.5 → poor risk-adjusted returns. "
            "0.5–1.0 → acceptable. "
            "1.0–2.0 → good (most well-managed funds aim here). "
            "Above 2.0 → excellent."
        ),
    )


st.text("")
st.text("")
st.subheader("Downside Risk Analysis")
st.write(
    "Standard volatility penalises upside and downside equally. "
    "These metrics focus specifically on losses — what investors "
    "actually experience as risk."
)

ds = downside_metrics(returns, weights, risk_free_rate=risk_free_rate)

ds_col1, ds_col2, ds_col3 = st.columns(3)
with ds_col1:
    sortino_val = ds["sortino_ratio"]
    sortino_display = f"{sortino_val:.2f}" if np.isfinite(sortino_val) else "∞"
    st.metric(
        "Sortino Ratio",
        sortino_display,
        help=(
            "Like Sharpe, but only penalises downside volatility — upside swings "
            "don't count against you. Higher is better. "
            "Below 1.0 → weak downside-adjusted returns. "
            "1.0–2.0 → good. "
            "Above 2.0 → excellent. "
            "Example: A Sortino of 1.5 means you earn 1.5 units of return for "
            "every unit of downside risk."
        ),
    )
with ds_col2:
    st.metric(
        "Daily VaR (95%)",
        f"{ds['var_daily']:.2%}",
        help=(
            "Value at Risk: with 95% confidence, your worst single-day loss "
            "won't exceed this amount. "
            "For a typical stock portfolio, daily VaR is usually 1–3%. "
            "Example: A VaR of 2.0% on a 10,000 dollar portfolio means on 95% of days "
            "you won't lose more than 200 dollars. On the worst 5% of days, losses can be larger."
        ),
    )
with ds_col3:
    st.metric(
        "Daily CVaR (95%)",
        f"{ds['cvar_daily']:.2%}",
        help=(
            "Conditional VaR (Expected Shortfall): the average loss on the worst "
            "5% of days. Always worse than VaR since it captures tail risk. "
            "Typically 1.5–2x your VaR value. "
            "Example: If CVaR is 3.0%, then on the worst 5% of trading days "
            "your portfolio loses 3% on average."
        ),
    )

ds_col4, ds_col5 = st.columns(2)
with ds_col4:
    st.metric(
        "Max Drawdown",
        f"{ds['max_drawdown']:.2%}",
        help=(
            "The largest peak-to-trough drop in portfolio value over the full "
            "historical period. Shown as a negative percentage. "
            "For reference, the S&P 500's max drawdown was about -34% during "
            "the 2020 COVID crash and -57% during the 2008 financial crisis. "
            "A max drawdown of -10% to -20% is typical for a diversified portfolio."
        ),
    )
with ds_col5:
    calmar_val = ds["calmar_ratio"]
    calmar_display = f"{calmar_val:.2f}" if np.isfinite(calmar_val) else "N/A"
    st.metric(
        "Calmar Ratio",
        calmar_display,
        help=(
            "Annual return divided by the absolute max drawdown. Measures how "
            "much return you earn per unit of worst-case pain. Higher is better. "
            "Below 0.5 → poor. "
            "0.5–1.0 → decent. "
            "1.0–3.0 → good. "
            "Above 3.0 → excellent (high return relative to drawdown risk)."
        ),
    )


# EXPECTED GAIN / EXPECTED LOSS
st.text("")
st.text("")
st.subheader("Expected Gain vs Expected Loss")
st.write(
    "From MIT OCW Lecture 13 (Prof. Jake Xia): instead of relying on "
    "volatility, assess each asset by its **Expected Gain** (average up-day "
    "return) versus Expected Loss (average down-day loss). The Skill "
    "Ratio maps directly to optimal position sizing — positive means "
    "the asset has positive expected value."
)

gl_table = compute_gl_table(returns, weights)
gl_display = format_gl_table(gl_table)
st.dataframe(gl_display, width='stretch', hide_index=True)

# Interpretation guidance
st.write("**How to read this table:**")
st.markdown(
    "- **G/L Ratio > 1.0** → the asset gains more on good days than it loses on bad days\n"
    "- **Skill Ratio > 0** → the asset has positive expected value; higher = stronger edge\n"
    "- **Skill Ratio < 0** → the asset is destroying capital on a risk-adjusted basis\n"
    "- **Kelly Fraction** → approximate maximum allocation suggested by the Kelly Criterion\n"
    "- **Win Rate** alone is misleading — a 40% win rate with G/L of 2.0 is highly profitable"
)


st.text("")
st.text("")
st.subheader("LSTM Volatility Forecast")

# Allow user to choose which ticker to use
selected_stock = st.selectbox("Select ticker for LSTM forecast", tickers)
# Let user pick how far into the future the model will predict
selected_forecast = st.selectbox("Select forecast horizon (days):", [1, 5, 10, 30], index = 0)

with st.spinner("Generating LSTM forecast... This may take a moment!"):
    # Grab returns for the selected ticker and reuse the cached universal LSTM for inference.
    return_series = returns[selected_stock].dropna()
    forecaster = get_forecaster()
    predicted_vol = forecaster.predict(return_series, horizon=selected_forecast)
    predicted_vol = np.array(predicted_vol).flatten()

st.success("Forecast complete!")

# Historical volatility
historical_vol = return_series.rolling(window = 30).std().dropna() # Turn returns into volatility for past 30 days
historical_vol = historical_vol[-30:]
if historical_vol.empty or len(predicted_vol) == 0:
    st.warning("Not enough data to compute volatility.")
else:
    # Combine
    # Forecast start date = the day after the last historical date
    forecast_start = historical_vol.index[-1] + pd.Timedelta(days=1)
    # Make forecast line start from the last historical point
    forecast_line = np.concatenate([[historical_vol.iloc[-1]], predicted_vol])
    # Make x-axis match: start from last historical date
    forecast_dates = pd.date_range(start=historical_vol.index[-1], periods=len(forecast_line), freq="B") # B = business days

    # Plot
    forecast_graph, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_vol.index, historical_vol * 100, label="Historical Volatility", color="gray", alpha=0.7)
    ax.plot(forecast_dates, forecast_line * 100, color="blue", linestyle="--", label="Forecasted Volatility")
    ax.set_title(f"{selected_stock} Volatility Forecast ({len(predicted_vol)}-Day Horizon)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Volatility (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax.legend()
    st.pyplot(forecast_graph)


st.text("")
st.text("")
st.subheader("Forecast-Driven Allocation")
st.write(
    "The LSTM forecast above is informational — this section makes it "
    "**prescriptive**. It runs the volatility forecast across every asset, "
    "builds a forward-looking covariance matrix, and feeds it into the "
    "optimiser so your allocation reacts to *predicted* risk, not just "
    "past risk."
)

# Run LSTM across all tickers
with st.spinner("Forecasting volatility for all assets..."):
    forecast_vols = forecast_all_vols(returns, forecaster, horizon=selected_forecast)

# Show historical vs predicted vol comparison
vol_comp = build_vol_comparison_table(returns, forecast_vols)
vol_comp_display = format_vol_comparison(vol_comp)
st.write("**Per-Asset Volatility: Historical vs Predicted**")
st.dataframe(vol_comp_display, width='stretch', hide_index=True)

# Let user pick forecast-driven strategy
fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    forecast_strategy = st.selectbox(
        "Forecast-driven strategy",
        [
            "Forecast Min-Variance",
            "Forecast Risk Parity",
            "Volatility Targeting",
        ],
        help=(
            "Min-Variance and Risk Parity use predicted vols in the covariance "
            "matrix. Volatility Targeting scales your current exposure to hit "
            "a target portfolio vol."
        ),
    )
with fc_col2:
    if forecast_strategy == "Volatility Targeting":
        target_vol_pct = st.slider(
            "Target annual volatility (%)",
            min_value=5, max_value=40, value=15, step=1,
            help=(
                "The engine scales your portfolio exposure so the predicted "
                "annual volatility matches this target. "
                "5–10% → conservative (bond-like). "
                "10–15% → moderate (balanced fund). "
                "15–20% → typical equity exposure (S&P 500 range). "
                "20%+ → aggressive."
            ),
        )
        target_vol = target_vol_pct / 100.0
    else:
        fc_max_weight = st.slider(
            "Max weight per asset (forecast)",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            key="fc_max_weight",
            help=(
                "Same as the optimiser cap, but applied to the forecast-driven allocation. "
                "Lower values force diversification. 0.30–0.50 is a sensible range "
                "for most portfolios."
            ),
        )

_FORECAST_STRATEGY_INFO = {
    "Forecast Min-Variance": (
        "**Forecast Min-Variance** works like the standard Minimum Variance optimiser, but replaces "
        "historical volatilities with the LSTM's predictions. If the model expects one asset's risk "
        "to rise, its weight is reduced *before* the spike happens. "
        "**Best for:** investors who want a forward-looking low-risk allocation and believe near-term "
        "volatility will differ meaningfully from the 5-year average."
    ),
    "Forecast Risk Parity": (
        "**Forecast Risk Parity** equalises risk contributions using *predicted* volatilities instead of "
        "historical ones. An asset the LSTM expects to become more volatile automatically loses weight, "
        "keeping the risk balance intact going forward. "
        "**Best for:** long-term balanced investors who also want the portfolio to adapt to upcoming "
        "risk changes rather than reacting after the fact."
    ),
    "Volatility Targeting": (
        "**Volatility Targeting** does not change *which* assets you hold — it scales your overall "
        "exposure up or down to keep predicted portfolio volatility near your chosen target. "
        "When the LSTM forecasts rising volatility, the engine reduces exposure and parks the remainder "
        "in cash. When it forecasts calm markets, it scales back up (up to 1.5×). "
        "**Best for:** investors with a specific risk budget (e.g. 'I want no more than 12% annual vol') "
        "who prefer to stay in their current assets rather than switch to a new allocation."
    ),
}
st.info(_FORECAST_STRATEGY_INFO[forecast_strategy])

# Run the selected forecast-driven strategy
try:
    if forecast_strategy == "Forecast Min-Variance":
        fc_result = forecast_min_variance(returns, forecast_vols, max_weight=fc_max_weight, risk_free_rate=risk_free_rate)
    elif forecast_strategy == "Forecast Risk Parity":
        fc_result = forecast_risk_parity(returns, forecast_vols, max_weight=fc_max_weight, risk_free_rate=risk_free_rate)
    else:
        base_weights_dict = {t: w for t, w in zip(tickers, weights) if t}
        fc_result = volatility_target_weights(
            base_weights_dict, returns, forecast_vols, target_vol=target_vol, risk_free_rate=risk_free_rate,
        )

    # Display results
    st.write(f"**{fc_result.get('method', forecast_strategy)} — Suggested Weights**")

    # Build comparison: current vs forecast-driven
    fc_tickers = [t for t in tickers if t]
    fc_rows = []
    for t in fc_tickers:
        curr = {tk: w for tk, w in zip(tickers, weights) if tk}.get(t, 0.0)
        fc_w = fc_result["weights"].get(t, 0.0)
        fc_rows.append({
            "Ticker": t,
            "Current Weight": f"{curr:.2%}",
            "Forecast Weight": f"{fc_w:.2%}",
            "Delta": f"{fc_w - curr:+.2%}",
        })
    # Include cash row for vol targeting
    if "Cash" in fc_result["weights"]:
        fc_rows.append({
            "Ticker": "Cash",
            "Current Weight": "0.00%",
            "Forecast Weight": f"{fc_result['weights']['Cash']:.2%}",
            "Delta": f"+{fc_result['weights']['Cash']:.2%}",
        })
    st.dataframe(pd.DataFrame(fc_rows), width='stretch', hide_index=True)

    # Performance metrics
    fc_perf1, fc_perf2, fc_perf3 = st.columns(3)
    with fc_perf1:
        st.metric(
            "Forecast Return",
            f"{fc_result['annual_return']:.2%}",
            delta=f"{fc_result['annual_return'] - p_return:+.2%}",
            help=(
                "Projected annual return using forecast-adjusted weights. "
                "A positive delta means the forecast-driven allocation "
                "is expected to outperform your current one."
            ),
        )
    with fc_perf2:
        st.metric(
            "Forecast Volatility",
            f"{fc_result['annual_volatility']:.2%}",
            delta=f"{fc_result['annual_volatility'] - p_volatility:+.2%}",
            delta_color="inverse",
            help=(
                "Predicted annual volatility using LSTM-adjusted covariance. "
                "A negative (green) delta means lower predicted risk than your "
                "current allocation."
            ),
        )
    with fc_perf3:
        st.metric(
            "Forecast Sharpe",
            f"{fc_result['sharpe']:.2f}",
            delta=f"{fc_result['sharpe'] - sharpe_ratio:+.2f}",
            help=(
                "Risk-adjusted return of the forecast-driven allocation. "
                "Above 1.0 is good, above 2.0 is excellent."
            ),
        )

    # Extra info for volatility targeting
    if "leverage" in fc_result:
        lev_col1, lev_col2, lev_col3 = st.columns(3)
        with lev_col1:
            st.metric("Leverage Factor", f"{fc_result['leverage']:.2f}x")
        with lev_col2:
            st.metric("Cash Allocation", f"{fc_result['cash_weight']:.1%}")
        with lev_col3:
            st.metric(
                "Predicted Port Vol",
                f"{fc_result['predicted_port_vol']:.2%}",
                help="The raw predicted vol before leverage scaling.",
            )
        if fc_result["leverage"] < 1.0:
            st.info(
                f"Predicted volatility ({fc_result['predicted_port_vol']:.1%}) exceeds your "
                f"target ({fc_result['target_vol']:.1%}), so the engine reduces exposure to "
                f"{fc_result['leverage']:.0%}x and parks {fc_result['cash_weight']:.1%} in cash."
            )
        elif fc_result["leverage"] > 1.01:
            st.info(
                f"Predicted volatility ({fc_result['predicted_port_vol']:.1%}) is below your "
                f"target ({fc_result['target_vol']:.1%}), so the engine scales up to "
                f"{fc_result['leverage']:.2f}x. Leverage above 1.0x requires margin."
            )

    # Risk contributions for forecast risk parity
    if "risk_contributions" in fc_result:
        st.write("**Forecast Risk Contributions (should be roughly equal):**")
        rc_df = pd.DataFrame({
            "Ticker": list(fc_result["risk_contributions"].keys()),
            "Risk Contribution": [f"{v:.2%}" for v in fc_result["risk_contributions"].values()],
        })
        st.dataframe(rc_df, width='stretch', hide_index=True)

except ValueError as e:
    st.warning(f"Forecast-driven allocation requires at least 2 assets. ({e})")
except Exception as e:
    st.warning(f"Could not compute forecast-driven allocation: {e}")


# Forecast volatility score
# Uses portfolio-level predicted annual vol (absolute, not relative z-score).
# Low predicted vol → high score; high predicted vol → low score.
# Anchored at 0 for ~67%+ annual vol, 100 for near-zero vol.
weights_arr = np.array(weights, dtype=float)
try:
    cov_fc = build_forecast_covariance(returns.dropna(), forecast_vols)
    pred_port_daily_var = float(np.dot(weights_arr, np.dot(cov_fc, weights_arr)))
    pred_port_annual_vol = np.sqrt(max(pred_port_daily_var, 0.0)) * np.sqrt(252)
except Exception:
    pred_port_annual_vol = p_volatility  # fallback to historical portfolio vol
fvs = float(np.clip(100 - pred_port_annual_vol * 150, 0, 100))


# Rolling volatility score
# Uses portfolio's current 30-day rolling vol (absolute annualised level).
aligned_returns = returns.dropna()
if aligned_returns.empty or np.isclose(weights_arr.sum(), 0.0):
    rvs = 50.0
else:
    portfolio_daily = aligned_returns @ weights_arr
    hist_port_vol = portfolio_daily.rolling(window=30).std().dropna()
    if hist_port_vol.empty:
        rvs = 50.0
    else:
        current_vol = float(hist_port_vol.iloc[-1])
        current_annual_vol = current_vol * np.sqrt(252)
        rvs = float(np.clip(100 - current_annual_vol * 150, 0, 100))


# Sharpe Ratio score
# tanh(sharpe) maps: -2→~0, -1→12, 0→50, 1→88, 2→~100
sharpe_clean = float(np.nan_to_num(sharpe_ratio, nan=0.0))
ss = float(np.clip(50 + 50 * np.tanh(sharpe_clean), 0, 100))

# Max Drawdown score
# 0% drawdown → 100; -67% drawdown → 0. Fixes the old formula's ceiling of 85.
weights_arr = np.array(weights, dtype=float)
aligned_returns = returns.dropna()
if aligned_returns.empty or np.isclose(weights_arr.sum(), 0.0):
    max_drawdown = 0.0
else:
    portfolio_daily = aligned_returns @ weights_arr
    cumulative = (1 + portfolio_daily).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())
dds = float(np.clip(100 + max_drawdown * 150, 0, 100))

# Concentration penalty (100 means fully diversified and 0 means fully concentrated)
# Rewards diversification (Modern Portfolio Theory encourages diversification)
weights_arr = np.array(weights, dtype=float)
if len(weights_arr) <= 1:
    cp = 0.0
else:
    entropy = -np.sum(weights_arr * np.log(weights_arr + 1e-9)) / np.log(len(weights_arr))
    cp = float(np.clip(entropy, 0, 1) * 100)


# Component weights:
# fvs/rvs: 20% each — absolute vol punishes genuinely risky portfolios
# ss: 20% — Sharpe matters but shouldn't override severe drawdown/vol signals
# dds: 25% — drawdown is the most visceral risk for investors
# cp: 15% — diversification bonus
health_score = 0.20*fvs + 0.20*rvs + 0.20*ss + 0.25*dds + 0.15*cp
health_score = float(np.nan_to_num(health_score, nan=0.0, posinf=100.0, neginf=0.0))


health_score_fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = health_score,
    domain = {'x': [0, 1], 'y': [0,1]},
    gauge = {
        'axis': {'range': [0,100]},
        'steps': gradient_steps(200),
        'bar': {'color': "rgba(0,0,0,0)"},
        'threshold': {
            'line': {'color': "black", 'width': 5},
            'thickness': 0.75,
            'value': health_score
        }
    }
))

# Add custom title with spacing above the gauge
health_score_fig.update_layout(
    annotations=[
        dict(
            text="Portfolio Health Score",
            x=0.5, y=1.3,  # position above gauge (y > 1 = above chart)
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=24)
        )
    ],
    margin=dict(t = 150, b = 20, l = 20, r = 20)  # a bit of padding around
)

st.plotly_chart(health_score_fig)


st.text("")
st.text("")


st.subheader("Portfolio Optimiser")
st.write(
    "Compare your current allocation against optimal "
    "portfolios. Each strategy solves a different objective — pick the "
    "one that matches your goals."
)

opt_col1, opt_col2 = st.columns(2)
with opt_col1:
    objective = st.selectbox(
        "Optimisation objective",
        [
            "Minimum Variance",
            "Maximum Sharpe Ratio",
            "Risk Parity (Equal Risk Contribution)",
            "Minimum CVaR (Tail-Risk)",
        ],
    )
with opt_col2:
    max_weight_cap = st.slider(
        "Max weight per asset",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help=(
            "Caps any single position to prevent over-concentration. "
            "At 1.0 (default), the optimiser can put 100% in one asset if it wants. "
            "At 0.5, no asset can exceed 50%. "
            "At 0.25, no asset exceeds 25% — forces diversification. "
            "A good starting point for most portfolios is 0.30–0.50."
        ),
    )

_OBJECTIVE_INFO = {
    "Minimum Variance": (
        "**Minimum Variance** finds the allocation with the lowest possible portfolio volatility, "
        "regardless of expected returns. It only needs the covariance matrix — no return forecasts — "
        "making it the most robust strategy. "
        "**Best for:** risk-averse investors whose primary goal is a smooth, low-volatility ride. "
        "It often over-weights bonds or low-vol assets heavily."
    ),
    "Maximum Sharpe Ratio": (
        "**Maximum Sharpe Ratio** maximises return earned per unit of risk taken (after subtracting "
        "the risk-free rate). It aims for the single best risk-adjusted allocation on the efficient frontier. "
        "**Best for:** investors who are confident historical return patterns will continue and want "
        "the most efficient trade-off between return and risk. "
        "**Caution:** most sensitive to return estimation errors — one historically strong asset can "
        "dominate the portfolio."
    ),
    "Risk Parity (Equal Risk Contribution)": (
        "**Risk Parity** allocates weights so that every asset contributes *equally* to total portfolio risk. "
        "Lower-volatility assets (e.g. bonds) receive more weight; higher-volatility assets receive less. "
        "Like Minimum Variance, it requires no return forecasts. "
        "**Best for:** long-term, balanced investors who want diversification in risk terms, not just dollar terms. "
        "A popular choice among institutional allocators."
    ),
    "Minimum CVaR (Tail-Risk)": (
        "**Minimum CVaR** minimises the *average loss* on the worst 5% of trading days (Expected Shortfall). "
        "Unlike the other strategies, it does not assume returns are normally distributed — it works directly "
        "from the historical loss distribution. "
        "**Best for:** investors who are especially sensitive to large, sudden drawdowns and want the "
        "portfolio that causes the least damage in a genuine market crisis."
    ),
}
st.info(_OBJECTIVE_INFO[objective])

# Run the selected optimiser
try:
    if objective == "Minimum Variance":
        opt_result = min_variance_weights(returns, max_weight=max_weight_cap, risk_free_rate=risk_free_rate)
    elif objective == "Maximum Sharpe Ratio":
        opt_result = max_sharpe_weights(
            returns, risk_free_rate=risk_free_rate, max_weight=max_weight_cap,
        )
    elif objective == "Risk Parity (Equal Risk Contribution)":
        opt_result = risk_parity_weights(returns, max_weight=max_weight_cap, risk_free_rate=risk_free_rate)
    else:
        opt_result = min_cvar_weights(returns, max_weight=max_weight_cap, risk_free_rate=risk_free_rate)

    # Build comparison table
    current_weights_dict = {t: w for t, w in zip(tickers, weights) if t}
    comparison_df = compare_portfolios(current_weights_dict, opt_result, [t for t in tickers if t])

    # Format for display
    display_df = comparison_df.copy()
    display_df["Current Weight"] = display_df["Current Weight"].map("{:.2%}".format)
    display_df["Optimised Weight"] = display_df["Optimised Weight"].map("{:.2%}".format)
    display_df["Delta"] = comparison_df["Delta"].map(lambda d: f"{d:+.2%}")
    st.dataframe(display_df, width='stretch', hide_index=True)

    # Performance comparison
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric(
            "Optimised Return",
            f"{opt_result['annual_return']:.2%}",
            delta=f"{opt_result['annual_return'] - p_return:+.2%}",
            help=(
                "The projected annual return of the optimised portfolio. "
                "A positive green delta means the optimiser found a higher-returning "
                "allocation than your current one."
            ),
        )
    with perf_col2:
        st.metric(
            "Optimised Volatility",
            f"{opt_result['annual_volatility']:.2%}",
            delta=f"{opt_result['annual_volatility'] - p_volatility:+.2%}",
            delta_color="inverse",  # lower vol is better
            help=(
                "The predicted annual volatility of the optimised portfolio. "
                "Here, a negative (green) delta is good — it means the optimiser "
                "found a less volatile allocation than your current one."
            ),
        )
    with perf_col3:
        st.metric(
            "Optimised Sharpe",
            f"{opt_result['sharpe']:.2f}",
            delta=f"{opt_result['sharpe'] - sharpe_ratio:+.2f}",
            help=(
                "The Sharpe ratio of the optimised portfolio. "
                "A positive green delta means better risk-adjusted returns. "
                "Target: above 1.0 is good, above 2.0 is excellent."
            ),
        )

    # Risk contributions (only for Risk Parity)
    if "risk_contributions" in opt_result:
        st.write("**Risk Contributions (should be roughly equal):**")
        rc_df = pd.DataFrame({
            "Ticker": list(opt_result["risk_contributions"].keys()),
            "Risk Contribution": [f"{v:.2%}" for v in opt_result["risk_contributions"].values()],
        })
        st.dataframe(rc_df, width='stretch', hide_index=True)

    # CVaR info (only for Min CVaR)
    if "cvar_daily" in opt_result:
        st.write(
            f"**Daily CVaR (95%):** {opt_result['cvar_daily']:.4%} — "
            f"On the worst 5% of days, the average loss is {opt_result['cvar_daily']:.2%}."
        )


    st.text("")
    st.write("**Efficient Frontier**")
    st.write(
        "The curve shows the best achievable return for every level of risk. "
        "Your current portfolio is marked in red; the optimised portfolio in green."
    )

    with st.spinner("Computing efficient frontier..."):
        frontier_df = compute_efficient_frontier(
            returns,
            risk_free_rate=risk_free_rate,
            n_points=40,
            max_weight=max_weight_cap,
        )

    if not frontier_df.empty:
        frontier_fig = go.Figure()

        # Frontier curve
        frontier_fig.add_trace(go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["target_return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="royalblue", width=2),
        ))

        # Current portfolio marker
        frontier_fig.add_trace(go.Scatter(
            x=[p_volatility],
            y=[p_return],
            mode="markers+text",
            name="Your Portfolio",
            marker=dict(color="red", size=12, symbol="circle"),
            text=["You"],
            textposition="top right",
        ))

        # Optimised portfolio marker
        frontier_fig.add_trace(go.Scatter(
            x=[opt_result["annual_volatility"]],
            y=[opt_result["annual_return"]],
            mode="markers+text",
            name="Optimised",
            marker=dict(color="limegreen", size=12, symbol="diamond"),
            text=["Optimal"],
            textposition="top right",
        ))

        frontier_fig.update_layout(
            xaxis_title="Annualised Volatility",
            yaxis_title="Annualised Return",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
            height=500,
            showlegend=True,
        )
        st.plotly_chart(frontier_fig, width='stretch')
    else:
        st.warning("Could not compute efficient frontier for the selected assets.")

    # ASSET SUGGESTIONS
    st.text("")
    st.text("")
    st.subheader("Asset Suggestions")
    st.write(
        "This section screens a universe of ~40 "
        "liquid ETFs across every major asset class and recommends the ones that would best "
        "complement your portfolio based on your chosen objective. "
    )

    _sugg_obj_col, _sugg_cap_col = st.columns([2, 1])
    with _sugg_obj_col:
        sugg_objective_label = st.selectbox(
            "Your objective",
            [
                "Best Risk-Adjusted Return",
                "Maximize Diversification",
                "Maximize Returns",
                "Maximize Stability",
            ],
            key="sugg_objective",
        )
    with _sugg_cap_col:
        _sugg_max_pct = st.slider(
            "Max weight per new asset",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            format="%d%%",
            key="sugg_max_weight",
            help="Caps how much of your portfolio the optimiser can assign to any single new asset.",
        )
        sugg_max_weight = _sugg_max_pct / 100

    _SUGG_OBJ_MAP = {
        "Best Risk-Adjusted Return": "sharpe",
        "Maximize Diversification":  "diversify",
        "Maximize Returns":          "returns",
        "Maximize Stability":        "stability",
    }
    _sugg_obj_key = _SUGG_OBJ_MAP[sugg_objective_label]

    # Detect whether stored results are stale (portfolio or objective changed)
    _sugg_state = st.session_state.get("sugg_results", None)
    _sugg_stale = (
        _sugg_state is None
        or _sugg_state.get("tickers") != tuple(sorted(tickers))
        or _sugg_state.get("objective") != _sugg_obj_key
        or _sugg_state.get("max_weight") != sugg_max_weight
    )

    if _sugg_stale and _sugg_state is not None:
        st.warning(
            "Portfolio, objective, or weight cap has changed — click **Find Suggestions** to refresh."
        )

    if st.button("Find Suggestions", key="run_sugg_btn"):
        with st.spinner(
            "Downloading ETF data and running joint optimisation… "
            "This may take up to 60 seconds on the first run."
        ):
            _cand_returns = get_candidate_returns(tuple(sorted(tickers)))

            if _cand_returns.empty:
                st.session_state["sugg_results"] = {
                    "tickers": tuple(sorted(tickers)),
                    "objective": _sugg_obj_key,
                    "max_weight": sugg_max_weight,
                    "df": pd.DataFrame(),
                    "n_candidates": 0,
                    "message": "Your portfolio already covers the full suggestion universe.",
                }
            else:
                _n_candidates = _cand_returns.shape[1]
                _shortlist, _trial_metrics = screen_candidates(
                    candidate_returns=_cand_returns,
                    portfolio_returns=returns.dropna(),
                    weights=weights,
                    risk_free_rate=risk_free_rate,
                    objective=_sugg_obj_key,
                )

                if not _shortlist:
                    _sugg_df = pd.DataFrame()
                    _msg = "No candidates showed meaningful improvement for this objective."
                else:
                    _sugg_df = allocate_suggestions(
                        shortlisted_tickers=_shortlist,
                        trial_metrics=_trial_metrics,
                        candidate_returns=_cand_returns,
                        portfolio_returns=returns.dropna(),
                        current_weights_dict=current_weights_dict,
                        risk_free_rate=risk_free_rate,
                        objective=_sugg_obj_key,
                        max_weight=sugg_max_weight,
                    )
                    _msg = "" if not _sugg_df.empty else "No candidates showed meaningful improvement for this objective."

                st.session_state["sugg_results"] = {
                    "tickers": tuple(sorted(tickers)),
                    "objective": _sugg_obj_key,
                    "max_weight": sugg_max_weight,
                    "df": _sugg_df,
                    "n_candidates": _n_candidates,
                    "message": _msg,
                }

    # Display stored results (persists across rerenders)
    _sugg_state = st.session_state.get("sugg_results", None)
    if _sugg_state is not None:
        if _sugg_state.get("message"):
            st.info(_sugg_state["message"])
        elif not _sugg_state["df"].empty:
            _sugg_display = _sugg_state["df"].copy()
            _sugg_display["Suggested Weight"] = _sugg_display["Suggested Weight"].map(
                lambda x: f"{x:.1%}"
            )
            _sugg_display["Δ Sharpe (trial)"] = _sugg_display["Δ Sharpe (trial)"].map(
                lambda x: f"{x:+.3f}" if np.isfinite(x) else "—"
            )
            _sugg_display["Δ Vol (trial)"] = _sugg_display["Δ Vol (trial)"].map(
                lambda x: f"{x:+.2%}" if np.isfinite(x) else "—"
            )
            st.dataframe(_sugg_display, width='stretch', hide_index=True)

            st.caption(
                f"Evaluated {_sugg_state['n_candidates']} candidate ETFs. "
                "Δ Sharpe and Δ Vol are computed by blending each asset at a 10% trial weight — "
                "they show directional impact, not the final joint-optimised impact."
            )
            st.info(
                "To invest in a suggested asset, add its ticker at the top of the page "
                "and re-run the Optimiser to see the full joint allocation."
            )
            st.caption(
                "Based on 5-year historical data. Past performance does not guarantee future results."
            )

    # REBALANCING ENGINE
    st.text("")
    st.text("")
    st.subheader("Rebalancing Engine")
    st.write(
        "The MIT lecture emphasises that diversification is only a free lunch "
        "if you rebalance — otherwise winners dominate, losers shrink, and "
        "the correlation benefit disappears. This section shows exactly what "
        "trades to execute to reach the optimised allocation above."
    )

    rebal_threshold = st.slider(
        "Drift threshold for rebalancing (%)",
        min_value=1.0, max_value=20.0, value=5.0, step=1.0,
        help=(
            "The percentage your portfolio must drift from target before rebalancing "
            "is triggered. Drift is measured as the total weight that needs to move. "
            "Example: if you hold 40% AAPL but the target is 25%, that's 15% drift "
            "on that one asset alone."
        ),
    ) / 100.0

    st.caption(
        "💡 **Recommended:** 3–5% for active investors who want tight tracking. "
        "5–10% for most people — balances accuracy with lower trading costs. "
        "10–20% for buy-and-hold investors who want to minimise trading."
    )

    # Compute drift: current weights vs optimizer target weights
    target_weights_dict = opt_result["weights"]
    drift_df = compute_drift(current_weights_dict, target_weights_dict)
    portfolio_drift = total_drift(drift_df)
    should_rebalance = needs_rebalancing(drift_df, threshold=rebal_threshold)

    # Drift summary
    if should_rebalance:
        st.warning(
            f"⚠️ Portfolio drift is **{portfolio_drift:.1%}** — exceeds your "
            f"{rebal_threshold:.0%} threshold. Rebalancing recommended."
        )
    else:
        st.success(
            f"✅ Portfolio drift is **{portfolio_drift:.1%}** — within your "
            f"{rebal_threshold:.0%} threshold. No rebalancing needed."
        )

    # Drift detail table
    st.write("**Drift Analysis**")
    drift_display = format_drift_table(drift_df)
    st.dataframe(drift_display, width='stretch', hide_index=True)

    # Trade list (always show, regardless of threshold)
    st.write("**Trade List**")
    st.write(
        f"Based on a total portfolio value of **${total_usd:,.0f}**, "
        "these are the trades needed to reach the optimised allocation:"
    )
    trade_df = generate_trade_list(drift_df, total_portfolio_value=total_usd)
    if trade_df.empty:
        st.info("No trades needed — your portfolio matches the target allocation.")
    else:
        trade_display = format_trade_list(trade_df)
        st.dataframe(trade_display, width='stretch', hide_index=True)

        # Total turnover
        total_traded = trade_df["Amount ($)"].sum() / 2  # each dollar moves once
        st.write(
            f"**Total turnover:** ${total_traded:,.2f} "
            f"({total_traded / total_usd:.1%} of portfolio)"
        )


    # BACKTEST
    st.text("")
    st.text("")
    st.subheader("Backtest: Your Weights vs Optimised")
    st.write(
        "This section simulates what would have happened if you had held "
        "your current allocation versus the optimised allocation. "
        "Set the start date to when you actually entered the market for "
        "personalised results. Both portfolios start at the same "
        "dollar value and are never rebalanced — pure buy-and-hold."
    )

    # Determine the valid date range from the available returns data
    _bt_index = returns.dropna().index
    if _bt_index.tz is not None:
        _bt_index = _bt_index.tz_localize(None)
    _min_bt_date = _bt_index.min().date()
    _max_bt_date = (_bt_index.max() - pd.Timedelta(days=30)).date()

    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        backtest_start_val = st.number_input(
            "Starting portfolio value ($)",
            min_value=100.00, value=10000.00, step=100.00, format="%.2f",
            help="Both portfolios start with this same dollar amount.",
        )
    with bt_col2:
        backtest_start_date = st.date_input(
            "Start date",
            value=_min_bt_date,
            min_value=_min_bt_date,
            max_value=_max_bt_date,
            help=(
                "The date you entered the market. The backtest runs from this "
                "date to the most recent trading day in the data. Defaults to "
                "the earliest available date (~5 years of history)."
            ),
        )

    # Slice returns to the selected window
    _start_ts = pd.Timestamp(backtest_start_date)
    if returns.index.tz is not None:
        _start_ts = _start_ts.tz_localize(returns.index.tz)
    returns_bt = returns.dropna()
    returns_bt = returns_bt[returns_bt.index >= _start_ts]

    if len(returns_bt) < 30:
        st.warning(
            f"Only {len(returns_bt)} trading days found after {backtest_start_date}. "
            "Please select an earlier start date for a meaningful backtest."
        )
        st.stop()

    _n_days = len(returns_bt)
    _n_years = _n_days / 252
    _actual_start = returns_bt.index[0].date()
    _actual_end = returns_bt.index[-1].date()
    st.caption(
        f"Simulating **{_actual_start}** → **{_actual_end}**  "
        f"({_n_days} trading days, {_n_years:.1f} years)"
    )

    # Simulate both portfolios over the selected window
    user_sim = simulate_portfolio(returns_bt, current_weights_dict, starting_value=backtest_start_val)
    opt_sim = simulate_portfolio(returns_bt, opt_result["weights"], starting_value=backtest_start_val)

    user_stats = compute_backtest_stats(user_sim, risk_free_rate=risk_free_rate)
    opt_stats = compute_backtest_stats(opt_sim, risk_free_rate=risk_free_rate)

    # Cumulative return chart
    st.write("**Portfolio Value Over Time**")
    value_chart = pd.DataFrame({
        "Your Weights": user_sim["portfolio_value"],
        "Optimised Weights": opt_sim["portfolio_value"],
    })
    fig_value = go.Figure()
    fig_value.add_trace(go.Scatter(
        x=value_chart.index, y=value_chart["Your Weights"],
        mode="lines", name="Your Weights",
        line=dict(color="red", width=2),
    ))
    fig_value.add_trace(go.Scatter(
        x=value_chart.index, y=value_chart["Optimised Weights"],
        mode="lines", name="Optimised Weights",
        line=dict(color="limegreen", width=2),
    ))
    fig_value.update_layout(
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=450,
        showlegend=True,
        hovermode="x unified",
    )
    st.plotly_chart(fig_value, width='stretch')

    # Final values
    user_final = user_sim["portfolio_value"].iloc[-1]
    opt_final = opt_sim["portfolio_value"].iloc[-1]
    dollar_diff = opt_final - user_final

    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        st.metric(
            "Your Final Value",
            f"${user_final:,.2f}",
            help="What your portfolio would be worth today with your original weights.",
        )
    with val_col2:
        st.metric(
            "Optimised Final Value",
            f"${opt_final:,.2f}",
            delta=f"${dollar_diff:+,.2f}",
            help="What the optimised portfolio would be worth today.",
        )
    with val_col3:
        st.metric(
            "Difference",
            f"${abs(dollar_diff):,.2f}",
            delta="Optimised wins" if dollar_diff > 0 else "Your weights win",
            delta_color="normal" if dollar_diff > 0 else "inverse",
            help="The dollar difference between the two strategies.",
        )

    # Drawdown comparison chart
    st.write("**Drawdown Comparison**")
    st.caption(
        "Drawdowns show how far each portfolio fell from its peak at any point. "
        "Shallower drawdowns mean less pain during downturns."
    )
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=user_sim.index, y=user_sim["drawdown"] * 100,
        mode="lines", name="Your Weights",
        line=dict(color="red", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
    ))
    fig_dd.add_trace(go.Scatter(
        x=opt_sim.index, y=opt_sim["drawdown"] * 100,
        mode="lines", name="Optimised Weights",
        line=dict(color="limegreen", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,255,0,0.1)",
    ))
    fig_dd.update_layout(
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        yaxis=dict(ticksuffix="%"),
        height=350,
        showlegend=True,
        hovermode="x unified",
    )
    st.plotly_chart(fig_dd, width='stretch')

    # Side-by-side stats table
    st.write("**Full Statistics Comparison**")
    comparison_table = build_comparison_stats(user_stats, opt_stats)
    st.dataframe(comparison_table, width='stretch', hide_index=True)

    # Verdict
    if dollar_diff > 0:
        st.success(
            f"Over this historical period, the optimised portfolio would have "
            f"earned **${dollar_diff:,.2f} more** than your current allocation "
            f"({opt_stats['total_return']:.1%} vs {user_stats['total_return']:.1%} total return)."
        )
    elif dollar_diff < 0:
        st.info(
            f"Over this historical period, your current allocation would have "
            f"outperformed the optimised portfolio by **${abs(dollar_diff):,.2f}** "
            f"({user_stats['total_return']:.1%} vs {opt_stats['total_return']:.1%} total return). "
            f"The optimiser may still offer better risk-adjusted returns — check the Sharpe and Sortino ratios."
        )
    else:
        st.info("Both portfolios produced identical returns over this period.")

    st.caption(
        "**Important:** This backtest uses historical data and assumes you held "
        "these exact weights from day one with no rebalancing. Past performance does "
        "not guarantee future results. The optimiser picks weights based on the full "
        "history, so it has a hindsight advantage — real-world results will differ."
    )

except ValueError as e:
    st.warning(f"Optimisation requires at least 2 assets with valid data. ({e})")

