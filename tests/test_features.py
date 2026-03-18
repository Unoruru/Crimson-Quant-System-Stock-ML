"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from crimson_quant.features import add_indicators


@pytest.fixture
def sample_ohlcv():
    """Create a minimal OHLCV dataframe for testing."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(100) * 0.2,
            "High": close + np.abs(np.random.randn(100) * 0.5),
            "Low": close - np.abs(np.random.randn(100) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 100),
        },
        index=dates,
    )


class TestAddIndicators:
    def test_adds_expected_columns(self, sample_ohlcv):
        result = add_indicators(sample_ohlcv)
        expected_cols = [
            "ret", "logret", "hl_spread", "oc_change", "co_gap", "volume_chg",
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_12", "ema_26",
            "mom_3", "mom_5", "mom_10",
            "vol_5", "vol_10",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "sentiment", "has_news", "news_count", "sentiment_std",
            "sentiment_pos_ratio", "sentiment_neg_ratio",
            "sentiment_3d_mean", "sentiment_5d_mean", "sentiment_change_1d",
            "y_next_logret",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self, sample_ohlcv):
        original_cols = list(sample_ohlcv.columns)
        add_indicators(sample_ohlcv)
        assert list(sample_ohlcv.columns) == original_cols

    def test_rsi_in_range(self, sample_ohlcv):
        result = add_indicators(sample_ohlcv)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_sentiment_default_zero(self, sample_ohlcv):
        result = add_indicators(sample_ohlcv)
        assert (result["sentiment"] == 0.0).all()
        assert (result["has_news"] == 0.0).all()
        assert (result["news_count"] == 0.0).all()
        assert (result["sentiment_pos_ratio"] == 0.0).all()
        assert (result["sentiment_neg_ratio"] == 0.0).all()

    def test_row_count_preserved(self, sample_ohlcv):
        result = add_indicators(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
