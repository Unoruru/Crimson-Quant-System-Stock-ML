"""Data loading, windowing, and scaling utilities."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import SCALER_EPSILON
from features import add_indicators, load_sentiment_daily
from stock_data_fetcher import get_stock_data


class WindowDataset(Dataset):
    """PyTorch dataset wrapping sliding windows of features and targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


def make_windows(
    df: pd.DataFrame,
    lookback: int,
    feature_cols: list,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create sliding windows of features and targets using stride tricks."""
    data = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    n_samples = len(df) - lookback
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows ({len(df)}) for lookback={lookback}."
        )

    # Use stride tricks to create overlapping windows without copying
    shape = (n_samples, lookback, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    X = np.ascontiguousarray(X)

    y = target[lookback - 1: lookback - 1 + n_samples]
    dates = pd.to_datetime(df.index[lookback: lookback + n_samples])
    return X, y, dates


class StandardScaler:
    """Z-score scaler for 3D arrays (samples x timesteps x features)."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X_3d: np.ndarray) -> None:
        flat = X_3d.reshape(-1, X_3d.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + SCALER_EPSILON

    def transform(self, X_3d: np.ndarray) -> np.ndarray:
        return (X_3d - self.mean_) / self.std_

    def fit_transform(self, X_3d: np.ndarray) -> np.ndarray:
        self.fit(X_3d)
        return self.transform(X_3d)


def _merge_sentiment(
    df: pd.DataFrame,
    ticker: str = "",
    clip_to_coverage: bool = False,
    sentiment_csv_path: str | None = None,
) -> pd.DataFrame:
    """Merge daily sentiment scores into the dataframe.

    Args:
        df: DataFrame with DatetimeIndex.
        ticker: Stock ticker used to locate the ticker-specific sentiment file.
        clip_to_coverage: If True, zero out sentiment for dates beyond
            the last available sentiment date (useful for inference on
            unseen data where future sentiment is unavailable).
        sentiment_csv_path: Explicit path to daily sentiment CSV. If provided
            and exists, used instead of the default lookup chain.
    """
    sentiment_cols = [
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
    existing_cols = [c for c in sentiment_cols if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    # Explicit path takes priority, then ticker-specific, then legacy
    if sentiment_csv_path and os.path.exists(sentiment_csv_path):
        sent_path = sentiment_csv_path
    elif ticker:
        sent_path = os.path.join("data", f"sentiment_daily_{ticker}.csv")
        if not os.path.exists(sent_path):
            sent_path = os.path.join("data", "sentiment_daily.csv")
    else:
        sent_path = os.path.join("data", "sentiment_daily.csv")
    sent = load_sentiment_daily(sent_path)

    df2 = df.reset_index()
    df2["Date"] = pd.to_datetime(df2["Date"]).dt.normalize()

    if sent is not None and len(sent) > 0 and "sentiment" in sent.columns:
        df2 = df2.merge(sent, on="Date", how="left")
        df2["has_news"] = df2["sentiment"].notna().astype(np.float32)

        if clip_to_coverage:
            sent_max_date = sent["Date"].max()
            clip_mask = df2["Date"] > sent_max_date
            for col in ["sentiment", "news_count", "sentiment_std", "sentiment_pos_ratio", "sentiment_neg_ratio"]:
                if col in df2.columns:
                    df2.loc[clip_mask, col] = np.nan

        df2["sentiment"] = df2["sentiment"].fillna(0.0)
        if "news_count" in df2.columns:
            df2["news_count"] = df2["news_count"].fillna(0.0)
        else:
            df2["news_count"] = df2["has_news"]

        if "sentiment_std" in df2.columns:
            df2["sentiment_std"] = df2["sentiment_std"].fillna(0.0)
        else:
            df2["sentiment_std"] = 0.0

        for col in ["sentiment_pos_ratio", "sentiment_neg_ratio"]:
            if col not in df2.columns:
                df2[col] = 0.0
            df2[col] = df2[col].fillna(0.0)
    else:
        df2["sentiment"] = 0.0
        df2["has_news"] = 0.0
        df2["news_count"] = 0.0
        df2["sentiment_std"] = 0.0
        df2["sentiment_pos_ratio"] = 0.0
        df2["sentiment_neg_ratio"] = 0.0

    df2["sentiment_3d_mean"] = df2["sentiment"].rolling(3, min_periods=1).mean()
    df2["sentiment_5d_mean"] = df2["sentiment"].rolling(5, min_periods=1).mean()
    df2["sentiment_change_1d"] = df2["sentiment"].diff().fillna(0.0)

    df2 = df2.sort_values("Date")
    df = df2.set_index("Date")
    df = df.dropna().copy()
    return df


def load_data(
    ticker: str,
    start: str,
    end: str,
    sentiment_csv_path: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV via yfinance, add indicators, merge sentiment."""
    df = get_stock_data(ticker, start, end)
    df = df.dropna().copy()
    df = add_indicators(df)
    return _merge_sentiment(df, ticker=ticker, sentiment_csv_path=sentiment_csv_path)


def load_data_from_csv(
    csv_path: str,
    start: str,
    end: str,
    auto_fetch_ticker: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV from CSV file, add indicators, merge sentiment.

    If the CSV does not exist and auto_fetch_ticker is provided,
    fetches data from yfinance instead.
    """
    if not os.path.isfile(csv_path) and auto_fetch_ticker:
        return load_data(auto_fetch_ticker, start, end)

    df = pd.read_csv(csv_path)

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")

    start_dt = pd.to_datetime(start).normalize()
    df = df[df["Date"] >= start_dt].copy()

    if len(df) == 0:
        raise ValueError(f"No rows left after date filter: [{start}, {end})")

    df = df.set_index("Date")
    df = df.dropna().copy()
    df = add_indicators(df)
    return _merge_sentiment(df, ticker=auto_fetch_ticker or "", clip_to_coverage=True)

