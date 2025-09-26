import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from data_pipeline import fetch_ticker_data, calculate_returns, fetch_risk_free_rate
from risk_analysis import portfolio_return, portfolio_volatility, sharpe_ratio as compute_sharpe_ratio
from lstm_model import load_forecaster

@st.cache_resource
def get_forecaster():
    # Cache the pre-trained LSTM so the app only loads weights once per session.
    return load_forecaster(device="cpu")

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

num_assets = st.number_input("How many assets are in your portfolio?", min_value = 1, max_value = 20, value = 1, step = 1)

for i in range(num_assets):

    cols = st.columns([2, 3])  # Ticker input & Slider

    with cols[0]:
        ticker = st.text_input(f"Ticker {i+1}", value="MSFT" if i == 0 else "", key=f"ticker_{i}")
        tickers.append(ticker)

    with cols[1]:
        amount = st.slider(
            f"Amount in {ticker if ticker else 'Asset'} ($)",
            min_value=0, max_value=10000, value=1000, step=100,
            key=f"amount_slider_{i}"  # use index instead of ticker
        )
        amounts.append(amount)

total_amount = sum(amounts)
weights = [amt / total_amount for amt in amounts] if total_amount > 0 else [0] * len(amounts)

st.write("**Portfolio Weights:**")
for t, w in zip(tickers, weights):
    if t: # Skip empty slots
        st.write(f"{t}: {w:.2%}")

prices = fetch_ticker_data(tickers)
returns = calculate_returns(prices)

st.write("**Portfolio Price Chart:**")
st.line_chart(prices)

st.write("Portfolio Return: ", portfolio_return(returns, weights))
st.write("Portfolio Volatility: ", portfolio_volatility(returns, weights))

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
    st.metric("Expected Annual Return: ", f"{p_return:.2%}")

with col2:
    st.metric("Volatility: ", f"{p_volatility:.2%}")

with col3:
    st.metric("Sharpe Ratio: ", f"{sharpe_ratio:.2f}")


st.text("")  # blank line for padding
st.text("")
st.subheader("LSTM Voltaility Forecast")

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
    ax.plot(historical_vol.index, historical_vol, label="Historical Volatility", color="gray", alpha=0.7)
    ax.plot(forecast_dates, forecast_line, color="blue", linestyle="--", label="Forecasted Volatility")
    ax.set_title(f"{selected_stock} Volatility Forecast ({len(predicted_vol)}-Day Horizon)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Volatility")
    ax.legend()
    st.pyplot(forecast_graph)


# Forecast volatility score
# Based on LSTM model
# Take predicted volatility for the selected forecast horizon
hist_q90 = float(historical_vol.quantile(0.90))
max_vol_threshold = max(hist_q90, 1e-6)
pred_vol_forecast = float(np.nan_to_num(predicted_vol[-1], nan=0.0, posinf=max_vol_threshold, neginf=0.0))
fvs = 100 * (1 - pred_vol_forecast / (2 * max_vol_threshold))
fvs = float(np.clip(fvs, 0, 100))

# Rolling volatility score
# Based on last 30-day volatility
recent_vol = returns.tail(30).std().mean(skipna=True)
recent_vol = float(np.nan_to_num(recent_vol, nan=0.0, posinf=0.0, neginf=0.0))
rvs = float(np.clip(100 - recent_vol*1000, 0, 100))

# Sharpe Ratio score
sharpe_ratio = float(np.nan_to_num(sharpe_ratio, nan=0.0, posinf=2.0, neginf=0.0))
if sharpe_ratio <= 0:
    ss = 0.0
elif sharpe_ratio >= 2.0:
    ss = 100.0
else:
    ss = float(sharpe_ratio / 2.0 * 100.0)

# Max Drawdown score
# Drawdown measures the largest drop in value of portfolio from it's peak
# Max Drawdowns essentially let an investor know what the historically max amount of money
# they can lose from a peak (drawdown = peak-to-trough)
port_ret = (returns.fillna(0).values @ np.array(weights, dtype=float))
port_ret = pd.Series(port_ret, index=returns.index)
cum = (1 + port_ret).cumprod()
running_max = cum.cummax()
drawdown = (cum - running_max) / running_max
max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
dds = float(np.clip(100 + max_drawdown*100, 0, 100))

# Concentration penalty (100 means fully diversified and 0 means fully concentrated)
# Rewards diversification (Modern Portfolio Theory encourages diversification)
weights_arr = np.array(weights, dtype = float)
if len(weights_arr) <= 1:
    cp = 0.0
else:
    entropy = -np.sum(weights_arr * np.log(weights_arr + 1e-9)) / np.log(len(weights_arr))
    cp = float(np.clip(entropy, 0, 1) * 100)


# Gives more importance for different factors
# Forecast volume has most weight (most impact on health score) // 0.35
# Concentration penalty has least weight (least impact on health score) // 0.10
health_score = 0.35*fvs + 0.20*rvs + 0.20*ss + 0.15*dds + 0.10*cp
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

