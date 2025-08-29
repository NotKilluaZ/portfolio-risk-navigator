# 📊 Portfolio Risk Navigator

A Python-based portfolio risk analysis tool that helps investors analyze expected returns, volatility, and asset allocations.  
Built using **yfinance**, **NumPy**, **Plotly**, and **Streamlit**.

---

## 🌐 Live Demo

Try the app live on Streamlit:  
➡ [portfolio-risk-navigator.streamlit.app](https://portfolio-risk-navigator.streamlit.app/)

### What You’ll See
- Input your **ticker symbols** (e.g., `AAPL, TSLA, GOOGL`) and assign portfolio **weights**  
- Instantly view your:
  - 📈 Expected annual return  
  - 📉 Annualized portfolio volatility  
  - 🥧 Asset allocation pie chart  

*(Optional, if implemented)* You may also explore:  
- Rolling volatility over time  
- Return breakdowns per asset  
- Stress testing or scenario simulations  

### How to Use It
1. Enter your desired ticker symbols  
2. Assign weights (make sure they sum to 1)  
3. Click **Analyze** (or let Streamlit auto-update)  
4. Explore your portfolio’s risk/return profile through the generated metrics and visuals  

No setup required — just click, input, and analyze! 🚀  

---

## 🚀 Features
- Fetches historical price data from Yahoo Finance
- Calculates portfolio **annual expected return** and **volatility**
- Visualizes portfolio **asset allocation** with interactive pie charts
- 🔮 **LSTM Volatility Forecast**: Predicts future volatility for selected tickers using a trained LSTM model
- 📊 **30-Day Historical Volatility Chart**: Visualizes recent volatility trends
- 🏥 **Portfolio Health Gauge**: Provides an overall health score based on volatility, Sharpe ratio, drawdown, and diversification
- Modular design with separate data pipeline and risk analysis modules
- Lightweight, no database required (runs fresh every time)

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NotKilluaZ/portfolio-risk-navigator
   cd portfolio-risk-navigator
