"""Stock data fetcher via yfinance."""

import logging
import os

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_OHLCV_DIR = "data"


def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance, with disk caching.

    Cache file: data/{ticker}_OHLCV_{start}_{end}.csv

    Args:
        ticker: Stock ticker symbol.
        start: Start date string.
        end: End date string.

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex.
    """
    cache_path = os.path.join(_OHLCV_DIR, f"{ticker}_OHLCV_{start}_{end}.csv")

    if os.path.exists(cache_path):
        logger.info("Loading %s OHLCV from cache: %s", ticker, cache_path)
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        df.index = pd.to_datetime(df.index).normalize()
        return df

    logger.info("Fetching %s data from yfinance [%s, %s]", ticker, start, end)

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker} [{start}, {end}]. "
            "Check ticker symbol and date range."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"

    os.makedirs(_OHLCV_DIR, exist_ok=True)
    df.to_csv(cache_path)
    logger.info("Saved OHLCV to %s", cache_path)

    return df
