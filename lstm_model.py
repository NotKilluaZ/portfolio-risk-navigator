import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import random

# Reproducibility helper
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Multi-step LSTM (sequence -> forecast_len outputs)
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 2, output_size = 1, dropout = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)  # output_size = forecast_len

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden)
        last = out[:, -1, :]        # (batch, hidden)
        out = self.fc(last)         # (batch, output_size)
        return out

def create_sequences_multi(data, seq_len, forecast_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - forecast_len + 1):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + forecast_len])
    return np.array(X), np.array(Y)

def train_and_predict(returns_series: pd.Series,
                      seq_len: int = 30,
                      forecast_len: int = 30,
                      epochs: int = 150,
                      batch_size: int = 32,
                      hidden_size: int = 64,
                      lr: float = 1e-3,
                      device: str = "cpu",
                      verbose: bool = False):
    
    #Trains a direct multi-step LSTM on the 30-day rolling volatility computed
    #from the provided returns_series, then returns a forecast vector of length forecast_len (denormalized).
    # - returns_series: pandas Series of returns (pct or log returns) with datetime index.
    # - seq_len: how many past vol points to use.
    # - forecast_len: how many steps to predict in the future (multi-step).

    set_seed(42)
    device = torch.device(device)

    # Input checks
    returns_series = pd.Series(returns_series).dropna()
    if returns_series.empty:
        return np.array([])

    # Create rolling 30-day volatility (same measure you plot)
    rolling_vol = returns_series.rolling(window=30).std().dropna()
    if len(rolling_vol) <= seq_len + forecast_len:
        # not enough data
        if verbose:
            print("Not enough volatility points:", len(rolling_vol))
        return np.array([])

    # Convert to numpy (keep indices for plotting externally)
    vol_vals = rolling_vol.values.astype(float)

    # Train / Val split (fit scaler on training portion only to avoid leakage)
    total = len(vol_vals)
    # use earliest 80% for train
    train_cut = int(total * 0.8)
    train_vol = vol_vals[:train_cut]
    test_vol = vol_vals[train_cut - seq_len:]  # ensure sequences that start before cut can be used for test

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_vol.reshape(-1, 1))   # fit on train only
    vol_norm = scaler.transform(vol_vals.reshape(-1, 1)).flatten()

    # Create multi-step sequences on normalized series
    X_all, Y_all = create_sequences_multi(vol_norm, seq_len, forecast_len)
    if len(X_all) == 0:
        return np.array([])

    # Split into train/test by index on X_all
    train_count = int(len(X_all) * 0.8)
    X_train = X_all[:train_count]
    Y_train = Y_all[:train_count]
    X_val = X_all[train_count:]
    Y_val = Y_all[train_count:]

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (N, seq, 1)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)               # (N, forecast_len)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

    # Dataloaders
    train_ds = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model setup
    model = MultiStepLSTM(input_size=1, hidden_size=hidden_size, num_layers=2, output_size=forecast_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Optional scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_val = np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)          # (batch, forecast_len)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t.to(device))
            val_loss = loss_fn(val_out, Y_val_t.to(device)).item()

        scheduler.step(val_loss)
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs}  train_loss = {np.mean(train_losses):.6f}  val_loss={val_loss:.6f}")

        # Early stopping if no improvement in model
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve = locals().get("no_improve", 0) + 1
            if no_improve > 30:
                if verbose:
                    print("Early stopping at epoch", epoch)
                break

    # Restore best state
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    # Forecast (predict next forecast_len from last available window)
    model.eval()
    last_window = vol_norm[-seq_len:]                     # normalized last seq
    x_in = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
    with torch.no_grad():
        pred_norm = model(x_in).cpu().numpy().reshape(-1)  # (forecast_len,)

    # Inverse transform to original volatility units
    pred_vol = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()

    # Diagnostics in terminal
    if verbose:
        print("rolling_vol stats: min/mean/max:", float(vol_vals.min()), float(vol_vals.mean()), float(vol_vals.max()))
        print("pred_vol stats: min/mean/max:", float(pred_vol.min()), float(pred_vol.mean()), float(pred_vol.max()))
        print("last historical vol:", float(vol_vals[-1]))

    return pred_vol
