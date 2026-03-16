"""CNN-LSTM model definition and checkpoint I/O."""

import logging
import os

import numpy as np
import torch
import torch.nn as nn

from config import Config, DENSE_HIDDEN
from data_loader import StandardScaler

logger = logging.getLogger(__name__)


class CNNLSTMRegressor(nn.Module):
    """CNN-LSTM hybrid model for time-series regression.

    Architecture:
        2x Conv1d (cnn_channels, kernel) -> BatchNorm -> ReLU
        2-layer LSTM (lstm_hidden units)
        Dense head: Linear -> ReLU -> Dropout -> Linear(1)
    """

    def __init__(
        self,
        n_features: int,
        cnn_channels: int = 64,
        kernel: int = 5,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        pad = kernel // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, DENSE_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DENSE_HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)   # [B, T, F] -> [B, F, T]
        h = self.cnn(x)         # [B, C, T]
        h = h.transpose(1, 2)   # [B, T, C]
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    cfg: Config,
    tag: str,
    feature_cols: list,
    x_scaler: StandardScaler,
    y_mean: float,
    y_std: float,
    model_kwargs: dict,
    target_col: str,
    calibration: dict | None = None,
    history: dict | None = None,
) -> None:
    """Save model checkpoint with all metadata needed for inference."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    x_mean = x_scaler.mean_
    x_std = x_scaler.std_
    x_mean_list = x_mean.tolist() if hasattr(x_mean, "tolist") else list(x_mean)
    x_std_list = x_std.tolist() if hasattr(x_std, "tolist") else list(x_std)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "cfg": {
            "ticker": str(cfg.ticker),
            "start": str(cfg.start),
            "end": str(cfg.end),
            "lookback": int(cfg.lookback),
            "batch_size": int(cfg.batch_size),
        },
        "tag": str(tag),
        "feature_cols": [str(c) for c in feature_cols],
        "target_col": str(target_col),
        "x_scaler": {
            "mean": [float(v) for v in x_mean_list],
            "std": [float(v) for v in x_std_list],
        },
        "y_norm": {
            "mean": float(y_mean),
            "std": float(y_std),
        },
        "model_class": "CNNLSTMRegressor",
        "model_kwargs": {
            k: (
                int(v) if isinstance(v, (np.integer,))
                else float(v) if isinstance(v, (np.floating,))
                else v
            )
            for k, v in model_kwargs.items()
        },
        "calibration": {
            "slope": float((calibration or {}).get("slope", 1.0)),
            "intercept": float((calibration or {}).get("intercept", 0.0)),
        },
        "history": history,
    }

    torch.save(ckpt, save_path)
    logger.info("Saved checkpoint -> %s", save_path)


def load_checkpoint(
    load_path: str,
    device: str | None = None,
) -> tuple[nn.Module, StandardScaler, dict]:
    """Load model checkpoint and return (model, scaler, metadata)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(load_path, map_location=device, weights_only=True)

    if ckpt.get("model_class") != "CNNLSTMRegressor":
        raise ValueError(f"Unsupported model_class: {ckpt.get('model_class')}")

    model = CNNLSTMRegressor(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_scaler = StandardScaler()
    x_scaler.mean_ = np.array(ckpt["x_scaler"]["mean"], dtype=np.float32)
    x_scaler.std_ = np.array(ckpt["x_scaler"]["std"], dtype=np.float32)

    meta = {
        "cfg": ckpt["cfg"],
        "tag": ckpt.get("tag"),
        "feature_cols": ckpt["feature_cols"],
        "lookback": ckpt["cfg"]["lookback"],
        "y_mean": ckpt["y_norm"]["mean"],
        "y_std": ckpt["y_norm"]["std"],
        "target_col": ckpt.get("target_col", "y_next_logret"),
        "calibration": ckpt.get("calibration", {"slope": 1.0, "intercept": 0.0}),
        "has_calibration": "calibration" in ckpt,
    }
    return model, x_scaler, meta
