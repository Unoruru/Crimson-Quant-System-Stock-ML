"""Tests for stock_data_fetcher module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_data_fetcher import get_stock_data


class TestGetStockData:
    def _make_mock_df(self, periods=50):
        dates = pd.bdate_range("2023-01-02", periods=periods)
        return pd.DataFrame(
            {
                "Open": np.random.uniform(140, 160, periods),
                "High": np.random.uniform(155, 170, periods),
                "Low": np.random.uniform(135, 150, periods),
                "Close": np.random.uniform(140, 165, periods),
                "Volume": np.random.randint(50_000_000, 100_000_000, periods),
            },
            index=dates,
        )

    @patch("stock_data_fetcher.yf.download")
    def test_returns_dataframe(self, mock_download):
        mock_download.return_value = self._make_mock_df()
        df = get_stock_data("AAPL", "2023-01-01", "2023-06-30")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "Close" in df.columns
        assert df.index.name == "Date"

    @patch("stock_data_fetcher.yf.download")
    def test_raises_on_empty(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="no data"):
            get_stock_data("INVALID", "2099-01-01", "2099-06-30")

    @patch("stock_data_fetcher.yf.download")
    def test_flattens_multiindex_columns(self, mock_download):
        df = self._make_mock_df()
        df.columns = pd.MultiIndex.from_tuples(
            [(c, "AAPL") for c in df.columns]
        )
        mock_download.return_value = df
        result = get_stock_data("AAPL", "2023-01-01", "2023-06-30")
        assert not isinstance(result.columns, pd.MultiIndex)
