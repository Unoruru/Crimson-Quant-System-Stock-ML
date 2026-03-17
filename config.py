"""Global configuration for Crimson Quant System."""

import argparse
import json
import os

import torch
from dataclasses import dataclass

# Named constants replacing magic numbers
EPSILON = 1e-12
SCALER_EPSILON = 1e-8
RSI_WINDOW = 14
MACD_SIGNAL_SPAN = 9
DENSE_HIDDEN = 64
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 5
GRAD_CLIP_NORM = 1.0
HUBER_DELTA = 1.0
ANNUALIZED_FACTOR = 252.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


@dataclass
class Config:
    """Training and evaluation configuration."""

    ticker: str = "AAPL"
    start: str = "2019-04-01"
    end: str = "2022-11-01"
    quantile_level: float = 0.70

    lookback: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    epochs: int = 300
    patience: int = 30
    weight_decay: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 999

    _CONFIG_FIELDS = ("ticker", "start", "end", "quantile_level", "epochs", "patience")

    @classmethod
    def load(cls) -> "Config":
        """Load config from config.json if it exists, falling back to defaults."""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            valid = {k: v for k, v in overrides.items() if k in cls._CONFIG_FIELDS}
            return cls(**valid)
        return cls()


QUANTILE_LEVEL = Config.load().quantile_level

BASE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "ret", "logret",
    "hl_spread", "oc_change", "co_gap", "volume_chg",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "mom_3", "mom_5", "mom_10",
    "vol_5", "vol_10",
    "rsi_14", "macd", "macd_signal", "macd_hist",
]

SENTIMENT_FEATURES = [
    "sentiment",
    "has_news",
    "news_count",
    "sentiment_std",
    "sentiment_pos_ratio",
    "sentiment_neg_ratio",
    "sentiment_3d_mean",
    "sentiment_5d_mean",
    "sentiment_change_1d",
]


def _interactive_config():
    """Interactively prompt for configuration and save to config.json."""
    cfg = Config.load()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

    print("Configure Crimson Quant System (press Enter to keep current value)\n")
    ticker = input(f"  Ticker [{cfg.ticker}]: ").strip() or cfg.ticker
    start = input(f"  Start date [{cfg.start}]: ").strip() or cfg.start
    end = input(f"  End date [{cfg.end}]: ").strip() or cfg.end

    quantile_level = cfg.quantile_level
    while True:
        raw = input(f"  Quantile level [{quantile_level}]: ").strip()
        if not raw:
            break
        try:
            val = float(raw)
            if 0.0 < val < 1.0:
                quantile_level = val
                break
            print("  Must be a float strictly between 0 and 1. Try again.")
        except ValueError:
            print("  Invalid number. Try again.")

    updates = {"ticker": ticker, "start": start, "end": end, "quantile_level": quantile_level}

    # epochs
    current_epochs = cfg.epochs
    raw = input(f"  Max training epochs [{current_epochs}]: ").strip()
    if raw:
        try:
            val = int(raw)
            if val < 1:
                raise ValueError
            updates["epochs"] = val
        except ValueError:
            print(f"  Invalid value — keeping {current_epochs}")

    # patience
    current_patience = cfg.patience
    raw = input(f"  Early-stopping patience [{current_patience}]: ").strip()
    if raw:
        try:
            val = int(raw)
            if val < 1:
                raise ValueError
            updates["patience"] = val
        except ValueError:
            print(f"  Invalid value — keeping {current_patience}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(updates, f, indent=2)

    print(f"\nSaved to {config_path}")


def _show_config():
    """Display the current effective configuration."""
    cfg = Config.load()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    source = "config.json" if os.path.exists(config_path) else "defaults"

    print(f"Current configuration (source: {source}):\n")
    print(f"  Ticker:         {cfg.ticker}")
    print(f"  Start date:     {cfg.start}")
    print(f"  End date:       {cfg.end}")
    print(f"  Quantile level: {cfg.quantile_level}")
    print(f"  Max epochs:     {cfg.epochs}")
    print(f"  Patience:       {cfg.patience}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure Crimson Quant System")
    parser.add_argument("--config", action="store_true", help="interactively set configuration")
    parser.add_argument("--show", action="store_true", help="show current configuration")
    cli_args = parser.parse_args()

    if cli_args.config:
        _interactive_config()
    elif cli_args.show:
        _show_config()
    else:
        parser.print_help()
