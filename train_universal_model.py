"""
Offline training entry point for the universal volatility LSTM.
Run this script on a beefier machine (or periodically) to refresh the model weights that the
Streamlit app consumes. The script pulls S&P 500 data, prepares the training dataset, trains the
LSTM, and stores the weights + scaler under ./models.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

from data_pipeline import fetch_ticker_data, calculate_returns
from lstm_model import (
    DEFAULT_FORECAST_LEN,
    DEFAULT_MODEL_HIDDEN_SIZE,
    DEFAULT_MODEL_LAYERS,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_SEQ_LEN,
    MultiStepLSTM,
    create_sequences_multi,
    set_seed,
)

# Configuration defaults keep the CLI easy to use but can be overridden by flags
DEFAULT_PERIOD = "10y"
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 80
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 15
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SP500_TABLE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CSV_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"


def _normalize_symbol(symbol: str) -> str | None:
    # Convert S&P listings (with dots) into yfinance-friendly symbols (dashes).
    if not isinstance(symbol, str):
        return None
    cleaned = symbol.strip()
    if not cleaned:
        return None
    return cleaned.replace(".", "-")

MODELS_DIR = Path("models")
META_PATH = MODELS_DIR / "universal_meta.json"


def get_sp500_tickers(limit: Optional[int] = None) -> List[str]:
    # Try multiple sources so older yfinance versions still give us the S&P 500 list.
    raw_tickers: List[str] = []

    yf_loader = getattr(yf, "tickers_sp500", None)
    if callable(yf_loader):
        try:
            raw_tickers = yf_loader() or []
        except Exception as exc:
            print(f"yfinance.tickers_sp500 failed: {exc}")

    if not raw_tickers:
        try:
            tables = pd.read_html(SP500_TABLE_URL)
            if tables:
                raw_tickers = tables[0]["Symbol"].tolist()
        except ImportError as exc:
            print(f"pandas.read_html unavailable (missing dependency): {exc}")
        except Exception as exc:
            print(f"Failed to scrape Wikipedia constituents table: {exc}")

    if not raw_tickers:
        try:
            csv_df = pd.read_csv(SP500_CSV_URL)
            raw_tickers = csv_df["Symbol"].tolist()
        except Exception as exc:
            raise RuntimeError("Could not determine S&P 500 tickers from any source.") from exc

    normalized: List[str] = []
    seen = set()
    for ticker in raw_tickers:
        symbol = _normalize_symbol(ticker)
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)

    if limit is not None:
        return normalized[:limit]
    return normalized


def prepare_volatility_dataset(
    tickers: List[str],
    period: str,
    seq_len: int,
    forecast_len: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    # Download prices, convert to rolling volatility, and slice into supervised sequences
    prices = fetch_ticker_data(tickers, period=period)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0] if tickers else "ticker")

    price_df = prices.copy()
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(-1)

    vol_series: List[np.ndarray] = []
    for column in price_df.columns:
        price_history = price_df[column].dropna()
        if price_history.empty:
            continue
        returns_series = price_history.pct_change().dropna()
        if returns_series.empty:
            continue
        rolling_vol = returns_series.rolling(window=seq_len).std().dropna()
        if len(rolling_vol) < seq_len + forecast_len:
            continue  # Skip assets without enough history
        vol_series.append(rolling_vol.values.astype(float))

    if not vol_series:
        raise RuntimeError("No tickers produced enough data to build training sequences.")

    # Fit scaler across every volatility sample so the distribution matches inference time.
    scaler = MinMaxScaler(feature_range=(0, 1))
    stacked = np.concatenate([vals.reshape(-1, 1) for vals in vol_series], axis=0)
    scaler.fit(stacked)

    X_blocks, Y_blocks = [], []
    for vols in vol_series:
        vols_norm = scaler.transform(vols.reshape(-1, 1)).flatten()
        X_part, Y_part = create_sequences_multi(vols_norm, seq_len, forecast_len)
        if len(X_part) > 0:
            X_blocks.append(X_part)
            Y_blocks.append(Y_part)

    if not X_blocks:
        raise RuntimeError("Scaling succeeded but no training windows were produced.")

    X_all = np.vstack(X_blocks)
    Y_all = np.vstack(Y_blocks)

    if max_samples is not None and len(X_all) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), size=max_samples, replace=False)
        X_all = X_all[idx]
        Y_all = Y_all[idx]

    return X_all, Y_all, scaler


def train_universal_model(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    forecast_len: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
) -> MultiStepLSTM:
    # Standard supervised training loop with early stopping on a validation split.
    device_t = torch.device(device)
    dataset_size = len(X_all)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    # Use 90% for training and hold out 10% for validation monitoring.
    split = max(int(dataset_size * 0.9), 1)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32).unsqueeze(-1)
    Y_train = torch.tensor(Y_all[train_idx], dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, drop_last=False
    )

    X_val = torch.tensor(X_all[val_idx], dtype=torch.float32).unsqueeze(-1) if len(val_idx) > 0 else None
    Y_val = torch.tensor(Y_all[val_idx], dtype=torch.float32) if len(val_idx) > 0 else None

    model = MultiStepLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=forecast_len,
        dropout=0.1,
    ).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val.to(device_t))
                val_loss = loss_fn(val_preds, Y_val.to(device_t)).item()
        else:
            val_loss = train_loss  # If validation split is empty, mirror train loss.

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (val_loss={val_loss:.6f}).")
                break

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the universal volatility LSTM.")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="Historical window to download from Yahoo Finance.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size used for training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Optimizer learning rate.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Number of bad epochs before early stop.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device for PyTorch (cpu, cuda, etc.).")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Historical window size the model sees.")
    parser.add_argument("--forecast-len", type=int, default=DEFAULT_FORECAST_LEN, help="Forecast horizon used during training.")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_MODEL_HIDDEN_SIZE, help="LSTM hidden units per layer.")
    parser.add_argument("--layers", type=int, default=DEFAULT_MODEL_LAYERS, help="Number of stacked LSTM layers.")
    parser.add_argument("--limit-tickers", type=int, default=None, help="Optional cap for tickers (useful for quick tests).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Where to store the trained weights.")
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH, help="Where to store the fitted scaler.")
    parser.add_argument("--meta-path", type=Path, default=META_PATH, help="Where to store metadata about the run.")
    parser.add_argument("--max-samples", type=int, default=200000, help="Optional cap on total training windows to control memory usage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = get_sp500_tickers(limit=args.limit_tickers)
    print(f"Fetched {len(tickers)} S&P 500 tickers.")

    X_all, Y_all, scaler = prepare_volatility_dataset(
        tickers=tickers,
        period=args.period,
        seq_len=args.seq_len,
        forecast_len=args.forecast_len,
        max_samples=args.max_samples,
    )
    print(f"Prepared {len(X_all)} training samples.")

    model = train_universal_model(
        X_all=X_all,
        Y_all=Y_all,
        forecast_len=args.forecast_len,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

    # Persist artifacts where the Streamlit app expects them.
    torch.save(model.state_dict(), args.model_path)
    with open(args.scaler_path, "wb") as handle:
        pickle.dump(scaler, handle)

    meta = {
        "tickers_used": len(tickers),
        "period": args.period,
        "seq_len": args.seq_len,
        "forecast_len": args.forecast_len,
        "hidden_size": args.hidden_size,
        "num_layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "device": args.device,
        "max_samples": args.max_samples,
        "actual_samples": len(X_all),
    }
    with open(args.meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Model saved to {args.model_path}")
    print(f"Scaler saved to {args.scaler_path}")
    print(f"Metadata written to {args.meta_path}")


if __name__ == "__main__":
    main()






