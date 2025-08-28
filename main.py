import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_pipeline import fetch_ticker_data, calculate_returns
from risk_analysis import portfolio_return, portfolio_volatility


st.title("Portfolio Risk Navigator")

tickers = st.text_input("Enter tickers (comma seperated, no spaces)", "GOOGL,MSFT,TSLA,NVDA,V,MA").split(",")
weights = st.text_input("Enter weights (comma seperated, no spaces), must sum up to 1", "0.2,0.2,0.2,0.2,0.1,0.1").split(",")
weights = np.array([float(w) for w in weights])

prices = fetch_ticker_data(tickers)
returns = calculate_returns(prices)

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