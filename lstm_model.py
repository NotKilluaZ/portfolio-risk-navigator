import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
from pathlib import Path
from typing import Optional


# Reproducibility helper
def set_seed(seed: int = 42) -> None:
    """Set every random seed we rely on so experiments can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Multi-step LSTM (sequence -> forecast_len outputs)
class MultiStepLSTM(nn.Module):
    """Standard LSTM stack that reads a sequence of scalars and forecasts multiple future steps."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 30,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x arrives with shape (batch, seq_len, 1). We only care about the last hidden state.
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last)
        return out


def create_sequences_multi(data: np.ndarray, seq_len: int, forecast_len: int):
    # Slice a single series into overlapping (history, future) pairs for supervised learning.
    X, Y = [], []
    for i in range(len(data) - seq_len - forecast_len + 1):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + forecast_len])
    return np.array(X), np.array(Y)


# Streamlit inference utilities
DEFAULT_MODEL_PATH = Path("models") / "universal_lstm.pth"
DEFAULT_SCALER_PATH = Path("models") / "universal_scaler.pkl"
DEFAULT_SEQ_LEN = 30
DEFAULT_FORECAST_LEN = 30
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL_HIDDEN_SIZE = 64
DEFAULT_MODEL_LAYERS = 2


class LSTMVolatilityForecaster:
    # Lightweight wrapper that turns the saved PyTorch weights + fitted scaler into a convenient
    # `predict` helper for the Streamlit app. The heavy training step happens offline.

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        scaler_path: Path = DEFAULT_SCALER_PATH,
        seq_len: int = DEFAULT_SEQ_LEN,
        forecast_len: int = DEFAULT_FORECAST_LEN,
        device: str = DEFAULT_DEVICE,
        hidden_size: int = DEFAULT_MODEL_HIDDEN_SIZE,
        num_layers: int = DEFAULT_MODEL_LAYERS,
        dropout: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Load artifacts immediately so failures show up while the app starts up, not mid-request.
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self) -> MultiStepLSTM:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing LSTM weights at {self.model_path}. Run train_universal_model.py first."
            )
        model = MultiStepLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.forecast_len,
            dropout=self.dropout,
        ).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_scaler(self) -> MinMaxScaler:
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Missing scaler at {self.scaler_path}. Run train_universal_model.py first."
            )
        with open(self.scaler_path, "rb") as handle:
            scaler = pickle.load(handle)
        if not isinstance(scaler, MinMaxScaler):
            raise TypeError("Loaded scaler is not a sklearn MinMaxScaler instance.")
        return scaler

    def predict(self, returns_series: pd.Series, horizon: int = DEFAULT_FORECAST_LEN) -> np.ndarray:
        # Forecast the next `horizon` days of volatility for the supplied returns series.
        # The scaler keeps the input distribution consistent with what the model saw offline.
        if horizon <= 0:
            raise ValueError("Forecast horizon must be a positive integer.")

        returns_series = pd.Series(returns_series).dropna()
        if returns_series.empty:
            return np.array([])

        # Compute rolling volatility in the same way the model was trained (30-day window).
        rolling_vol = returns_series.rolling(window=self.seq_len).std().dropna()
        if len(rolling_vol) < self.seq_len:
            return np.array([])

        # Normalise the last historical window so the model sees familiar scales.
        last_window = rolling_vol.values[-self.seq_len :].astype(float)
        last_window_norm = self.scaler.transform(last_window.reshape(-1, 1)).flatten()

        # Prepare tensor with shape (batch=1, seq_len, features=1).
        x_in = (
            torch.tensor(last_window_norm, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(self.device)
        )

        with torch.no_grad():
            pred_norm = self.model(x_in).cpu().numpy().reshape(-1)

        # Return only the portion the caller asked for, denormalised back to volatility units.
        pred_vol = self.scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        horizon = min(horizon, self.forecast_len)
        return pred_vol[:horizon]


def load_forecaster(
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    device: str = DEFAULT_DEVICE,
) -> LSTMVolatilityForecaster:
    # Convenience factory so Streamlit can cache a ready-to-use forecaster.
    return LSTMVolatilityForecaster(
        model_path=model_path or DEFAULT_MODEL_PATH,
        scaler_path=scaler_path or DEFAULT_SCALER_PATH,
        device=device,
    )


__all__ = [
    "DEFAULT_FORECAST_LEN",
    "DEFAULT_MODEL_HIDDEN_SIZE",
    "DEFAULT_MODEL_LAYERS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SCALER_PATH",
    "DEFAULT_SEQ_LEN",
    "LSTMVolatilityForecaster",
    "MultiStepLSTM",
    "create_sequences_multi",
    "load_forecaster",
    "set_seed",
]

