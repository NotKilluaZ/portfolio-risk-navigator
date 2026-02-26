# 📊 Portfolio Risk Navigator

A Python-based portfolio risk analysis and optimisation tool that helps investors analyse expected returns, volatility, asset allocations, and make data-driven rebalancing decisions.

Built using **yfinance**, **NumPy**, **SciPy**, **PyTorch**, **Plotly**, and **Streamlit**.

Inspired by concepts from [MIT OCW Lecture 13 — Portfolio Management](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/) (Prof. Jake Xia).

---

## 🌐 Live Demo

Try the app live on Streamlit:
➡ [portfolio-risk-navigator.streamlit.app](https://portfolio-risk-navigator.streamlit.app/)

### How to Use It
1. Enter your desired ticker symbols (e.g., `AAPL`, `TSLA`, `GOOGL`)
2. Enter the exact dollar amount invested in each asset
3. Explore your portfolio's risk/return profile, optimisation suggestions, and rebalancing trades

No setup required — just click, input, and analyse! 🚀

---

## 🚀 Features

### Core Analytics
- **Portfolio Price Chart** — Historical price data fetched from Yahoo Finance via yfinance
- **30-Day Rolling Volatility** — Per-asset historical volatility displayed as percentages
- **Correlation Heatmap** — Interactive heatmap showing how each asset pair moves together
- **Asset Allocation Pie Chart** — Visual breakdown of portfolio weights
- **Key Metrics** — Expected annual return, annualised volatility, and Sharpe ratio with benchmark guidance in tooltips

### Downside Risk Analysis
Addresses the MIT lecture critique that standard volatility penalises upside and downside equally:
- **Sortino Ratio** — Like Sharpe, but only penalises downside volatility
- **Daily VaR (95%)** — Maximum expected single-day loss at 95% confidence
- **Daily CVaR (95%)** — Average loss on the worst 5% of days (Expected Shortfall)
- **Max Drawdown** — Largest peak-to-trough decline over the full history
- **Calmar Ratio** — Annual return per unit of drawdown risk

### Expected Gain vs Expected Loss
Direct implementation of the G/L framework from MIT Lecture 13:
- **Expected Gain (G)** and **Expected Loss (L)** per asset and for the portfolio
- **G/L Ratio** — Reward per unit of downside
- **Skill Ratio** — Ranges from -1 to +1; positive means positive expected value
- **Kelly Fraction** — Optimal position sizing derived from the Skill Ratio
- **Win Rate** — Fraction of positive-return days

### LSTM Volatility Forecast
- Pre-trained universal LSTM model (trained on S&P 500 rolling volatility)
- Selectable forecast horizons: 1, 5, 10, or 30 days
- Historical vs forecasted volatility chart with percentage y-axis

### Forecast-Driven Allocation
Turns the LSTM forecast from informational into **prescriptive**:
- Runs the LSTM across all portfolio assets automatically
- Builds a **forecast-adjusted covariance matrix** (LSTM-predicted vols × historical correlations)
- **Forecast Min-Variance** — Minimise predicted portfolio risk
- **Forecast Risk Parity** — Equal risk contribution using predicted vols
- **Volatility Targeting** — Scales exposure to hit a user-chosen target volatility, with leverage and cash allocation

### Portfolio Health Score
Composite gauge (0–100) combining:
- LSTM forecast volatility score
- Rolling volatility score
- Sharpe ratio score
- Max drawdown score
- Concentration/diversification penalty

### Portfolio Optimiser
Four mathematically-grounded optimisation strategies:
- **Minimum Variance** — Lowest possible portfolio risk (no return estimates needed)
- **Maximum Sharpe Ratio** — Best risk-adjusted return
- **Risk Parity** — Equal risk contribution from every asset
- **Minimum CVaR** — Minimise tail risk (Expected Shortfall)
- **Efficient Frontier** — Visual curve showing optimal return-for-risk tradeoffs, with current vs optimised portfolio markers
- Adjustable max weight constraint to prevent over-concentration

### Rebalancing Engine
Implements the MIT lecture principle that diversification only works if you rebalance:
- **Drift Analysis** — Per-asset drift between current and optimised weights
- **Rebalancing Threshold** — User-configurable drift trigger with recommended ranges
- **Trade List** — Concrete buy/sell orders with dollar amounts based on your total portfolio value
- **Turnover Calculation** — How much money needs to move, in dollars and as a percentage

### Guided Tooltips
Every metric includes a hover tooltip explaining:
- What the metric measures
- What good, acceptable, and poor values look like
- Real-world benchmarks (e.g., S&P 500 averages) for context

---

## 📁 Project Structure

```
portfolio-risk-navigator/
├── main.py                    # Streamlit UI and orchestration
├── data_pipeline.py           # Yahoo Finance data fetching and returns calculation
├── risk_analysis.py           # Portfolio math, Sharpe, and downside risk metrics
├── optimizer.py               # Min-variance, max-Sharpe, risk parity, CVaR, efficient frontier
├── rebalancer.py              # Drift tracking and trade list generation
├── expected_gl.py             # Expected Gain/Loss framework (MIT Lecture 13)
├── forecast_allocation.py     # LSTM → forecast covariance → allocation engine
├── lstm_model.py              # LSTM model definition and inference
├── train_universal_model.py   # Offline LSTM training script
├── models/
│   ├── universal_lstm.pth     # Pre-trained LSTM weights
│   ├── universal_scaler.pkl   # Fitted MinMaxScaler
│   └── universal_meta.json    # Training metadata
└── requirements.txt           # Python dependencies
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/NotKilluaZ/portfolio-risk-navigator
cd portfolio-risk-navigator
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run main.py
```

The app will open in your browser.

---

## 🧠 Universal LSTM Workflow

The app loads a single pre-trained LSTM rather than retraining per ticker.

### Retraining the model
```bash
# Full training (~500 S&P 500 tickers, 10 years of data, ~20-45 min)
python train_universal_model.py

# Quick test run (~2-3 min)
python train_universal_model.py --limit-tickers 20 --epochs 20
```

This will:
1. Fetch S&P 500 constituent tickers
2. Download historical price data
3. Compute rolling volatility and build training sequences
4. Train the LSTM with early stopping
5. Save weights to `models/universal_lstm.pth`, scaler to `models/universal_scaler.pkl`, and metadata to `models/universal_meta.json`

> **Note:** Install `lxml` before training (`pip install lxml`) — it's needed to scrape the S&P 500 ticker list from Wikipedia.

---

## 📦 Dependencies

```
streamlit
plotly
yfinance
numpy
pandas
matplotlib
torch
scikit-learn
scipy
```

All optimisation and risk modules use `scipy.optimize` and `numpy` linear algebra — no additional dependencies beyond the core stack.

---

## 📚 Theoretical Background

The app draws from several key concepts covered in MIT OCW 18.S096 Lecture 13:

- **"Portfolio management is fundamentally a sizing problem"** → Optimiser + forecast-driven allocation + Kelly fraction
- **"Volatility doesn't distinguish upside from downside"** → Sortino, CVaR, max drawdown
- **"Replace volatility with Expected Gain (G) and Expected Loss (L)"** → G/L framework with Skill Ratio
- **"Diversification only works if you rebalance"** → Rebalancing engine with drift tracking
- **"MPT is unstable — sensitive to return estimation error"** → Min-variance and risk parity prioritised over max-Sharpe
- **"Fat tails and power law distributions"** → CVaR-based optimisation (doesn't assume normality)