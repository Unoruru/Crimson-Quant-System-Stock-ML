"""Feature engineering: technical indicators and sentiment loading."""

import os

import numpy as np
import pandas as pd

from .config import EPSILON, RSI_WINDOW, MACD_SIGNAL_SPAN


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and target column to OHLCV dataframe."""
    df = df.copy()

    df["ret"] = df["Close"].pct_change()
    df["logret"] = np.log(df["Close"] / (df["Close"].shift(1) + EPSILON))

    df["hl_spread"] = (df["High"] - df["Low"]) / (df["Close"] + EPSILON)
    df["oc_change"] = (df["Close"] - df["Open"]) / (df["Open"] + EPSILON)
    df["co_gap"] = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + EPSILON)
    df["volume_chg"] = df["Volume"].pct_change()

    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    for w in [3, 5, 10]:
        df[f"mom_{w}"] = df["Close"] / (df["Close"].shift(w) + EPSILON) - 1.0

    df["vol_5"] = df["logret"].rolling(5).std()
    df["vol_10"] = df["logret"].rolling(10).std()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(RSI_WINDOW).mean()
    roll_down = down.rolling(RSI_WINDOW).mean()
    rs = roll_up / (roll_down + EPSILON)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL_SPAN, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["sentiment"] = 0.0
    df["has_news"] = 0.0
    df["news_count"] = 0.0
    df["sentiment_std"] = 0.0
    df["sentiment_pos_ratio"] = 0.0
    df["sentiment_neg_ratio"] = 0.0
    df["sentiment_3d_mean"] = 0.0
    df["sentiment_5d_mean"] = 0.0
    df["sentiment_change_1d"] = 0.0

    df["y_next_logret"] = np.log(df["Close"].shift(-1) / (df["Close"] + EPSILON))

    return df


def load_sentiment_daily(path: str) -> pd.DataFrame:
    """Load daily sentiment scores from CSV."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Date", "sentiment"])

    s = pd.read_csv(path)
    if "Date" not in s.columns or "sentiment" not in s.columns:
        return pd.DataFrame(columns=["Date", "sentiment"])

    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    s = s.sort_values("Date").drop_duplicates("Date", keep="last")

    optional_cols = [
        "news_count",
        "sentiment_std",
        "sentiment_pos_ratio",
        "sentiment_neg_ratio",
    ]
    keep_cols = ["Date", "sentiment"] + [c for c in optional_cols if c in s.columns]
    return s[keep_cols]
