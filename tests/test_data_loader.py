"""Tests for data loading, windowing, and scaling utilities."""

import numpy as np
import pandas as pd
import pytest

from data_loader import WindowDataset, StandardScaler, make_windows


@pytest.fixture
def sample_df():
    """Create a minimal dataframe with indicators for window tests."""
    n = 80
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "feat_c": np.random.randn(n),
            "target": np.random.randn(n) * 0.01,
        },
        index=dates,
    )
    return df


class TestMakeWindows:
    def test_output_shapes(self, sample_df):
        lookback = 10
        feature_cols = ["feat_a", "feat_b", "feat_c"]
        X, y, dates = make_windows(sample_df, lookback, feature_cols, "target")

        expected_samples = len(sample_df) - lookback
        assert X.shape == (expected_samples, lookback, len(feature_cols))
        assert y.shape == (expected_samples,)
        assert len(dates) == expected_samples

    def test_window_content(self, sample_df):
        lookback = 5
        feature_cols = ["feat_a"]
        X, y, dates = make_windows(sample_df, lookback, feature_cols, "target")

        expected_first_window = sample_df["feat_a"].values[:lookback].reshape(lookback, 1)
        np.testing.assert_array_almost_equal(X[0], expected_first_window)

    def test_lookback_too_large_raises(self, sample_df):
        with pytest.raises(ValueError, match="Not enough rows"):
            make_windows(sample_df, lookback=len(sample_df) + 1, feature_cols=["feat_a"], target_col="target")

    def test_dates_alignment(self, sample_df):
        lookback = 10
        X, y, dates = make_windows(sample_df, lookback, ["feat_a"], "target")
        assert dates[0] == sample_df.index[lookback]
        assert dates[-1] == sample_df.index[-1]


class TestStandardScaler:
    def test_fit_transform_shape(self):
        X = np.random.randn(50, 10, 3).astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape

    def test_zero_mean_unit_std(self):
        np.random.seed(42)
        X = np.random.randn(100, 20, 5).astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        flat = X_scaled.reshape(-1, 5)
        np.testing.assert_array_almost_equal(flat.mean(axis=0), np.zeros(5), decimal=5)
        np.testing.assert_array_almost_equal(flat.std(axis=0), np.ones(5), decimal=2)

    def test_transform_uses_fit_params(self):
        X_fit = np.random.randn(50, 10, 3).astype(np.float32)
        X_new = np.random.randn(20, 10, 3).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(X_fit)
        X_scaled = scaler.transform(X_new)

        assert X_scaled.shape == X_new.shape
        expected = (X_new - scaler.mean_) / scaler.std_
        np.testing.assert_array_almost_equal(X_scaled, expected)


class TestWindowDataset:
    def test_length(self):
        X = np.random.randn(30, 10, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        ds = WindowDataset(X, y)
        assert len(ds) == 30

    def test_getitem_shapes(self):
        X = np.random.randn(30, 10, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        ds = WindowDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (10, 5)
        assert y_item.shape == ()
