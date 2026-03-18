"""Tests for shared metric functions."""

import math

import numpy as np
import pytest

from crimson_quant.metrics import (
    rmse,
    fit_affine_calibration,
    apply_affine_calibration,
    is_safe_affine_calibration,
    max_drawdown,
    sharpe_ratio,
    logret_to_next_close,
    compute_logret_metrics,
    compute_price_metrics,
    compute_direction_metrics,
    compute_trading_metrics,
)


class TestRmse:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = math.sqrt(1.0 / 3.0)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert abs(rmse(y_true, y_pred) - rmse(y_pred, y_true)) < 1e-10


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = np.array([1.0, 1.1, 1.2, 1.3])
        assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        equity = np.array([1.0, 1.2, 0.9, 1.1])
        mdd = max_drawdown(equity)
        assert mdd < 0
        assert mdd == pytest.approx(0.9 / 1.2 - 1.0, abs=1e-6)

    def test_total_loss(self):
        equity = np.array([1.0, 0.5, 0.25])
        mdd = max_drawdown(equity)
        assert mdd == pytest.approx(-0.75, abs=1e-6)


class TestSharpeRatio:
    def test_empty_returns(self):
        assert math.isnan(sharpe_ratio(np.array([])))

    def test_zero_volatility(self):
        returns = np.array([0.01, 0.01, 0.01])
        assert math.isnan(sharpe_ratio(returns))

    def test_positive_sharpe(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        sharpe = sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not math.isnan(sharpe)


class TestLogretToNextClose:
    def test_zero_logret(self):
        today = np.array([100.0, 200.0])
        logret = np.array([0.0, 0.0])
        result = logret_to_next_close(today, logret)
        np.testing.assert_array_almost_equal(result, today)

    def test_known_logret(self):
        today = np.array([100.0])
        logret = np.array([np.log(1.05)])
        result = logret_to_next_close(today, logret)
        np.testing.assert_array_almost_equal(result, [105.0])


class TestComputePriceMetrics:
    def test_perfect_prediction(self):
        close = np.array([100.0, 101.0, 102.0])
        metrics = compute_price_metrics(close, close)
        assert metrics["MAE"] == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["MAPE_%"] == 0.0
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_all_keys(self):
        true_c = np.array([100.0, 110.0, 105.0])
        pred_c = np.array([101.0, 109.0, 106.0])
        metrics = compute_price_metrics(true_c, pred_c)
        assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE_%", "R2"}


class TestCalibrationHelpers:
    def test_affine_calibration_reduces_bias(self):
        y_true = np.array([0.00, 0.01, 0.02, 0.03])
        y_pred = np.array([-0.02, -0.01, 0.00, 0.01])
        calibration = fit_affine_calibration(y_true, y_pred)
        calibrated = apply_affine_calibration(y_pred, calibration)
        assert abs(np.mean(calibrated - y_true)) < abs(np.mean(y_pred - y_true))

    def test_compute_logret_metrics_returns_expected_keys(self):
        y_true = np.array([0.00, 0.01, -0.01])
        y_pred = np.array([0.00, 0.02, -0.02])
        metrics = compute_logret_metrics(y_true, y_pred)
        expected = {
            "LogRet_MAE", "LogRet_RMSE", "LogRet_Bias", "LogRet_Corr",
            "True_LogRet_Mean", "Pred_LogRet_Mean", "True_LogRet_Std", "Pred_LogRet_Std",
        }
        assert set(metrics.keys()) == expected

    def test_unsafe_calibration_falls_back_to_identity(self):
        y_true = np.array([-0.01, 0.00, 0.01, 0.02])
        y_pred = np.array([0.02, 0.021, 0.022, 0.023])
        calibration = fit_affine_calibration(y_true, y_pred)
        assert calibration == {"slope": 1.0, "intercept": 0.0}
        assert is_safe_affine_calibration(calibration)


class TestComputeDirectionMetrics:
    def test_perfect_direction(self):
        today = np.array([100.0, 100.0, 100.0])
        true_c = np.array([105.0, 95.0, 110.0])
        pred_c = np.array([103.0, 97.0, 108.0])
        metrics = compute_direction_metrics(today, true_c, pred_c)
        assert metrics["DirAcc_%"] == 100.0

    def test_returns_all_keys(self):
        today = np.array([100.0, 100.0])
        true_c = np.array([105.0, 95.0])
        pred_c = np.array([103.0, 97.0])
        metrics = compute_direction_metrics(today, true_c, pred_c)
        expected_keys = {"DirAcc_%", "UpPrecision_%", "DownPrecision_%", "PredUpRatio_%", "PredDownRatio_%"}
        assert set(metrics.keys()) == expected_keys


class TestComputeTradingMetrics:
    def test_returns_correct_shape(self):
        today = np.array([100.0] * 20)
        true_c = np.array([101.0] * 20)
        pred_c = np.array([100.5 + i * 0.1 for i in range(20)])
        metrics, strat_eq, bh_eq = compute_trading_metrics(today, true_c, pred_c)
        assert isinstance(metrics, dict)
        assert len(strat_eq) == 20
        assert len(bh_eq) == 20
        assert "StrategyReturn_%" in metrics
        assert "Sharpe" in metrics

    def test_custom_quantile(self):
        today = np.array([100.0] * 20)
        true_c = np.array([101.0] * 20)
        pred_c = np.array([100.5 + i * 0.1 for i in range(20)])
        metrics, _, _ = compute_trading_metrics(today, true_c, pred_c, quantile_level=0.90)
        assert metrics["Threshold_Quantile_%"] == 90.0

    def test_negative_predictions_do_not_open_long_positions(self):
        today = np.array([100.0, 100.0, 100.0])
        true_c = np.array([99.0, 101.0, 98.0])
        pred_c = np.array([99.5, 99.2, 99.8])
        metrics, strat_eq, _ = compute_trading_metrics(today, true_c, pred_c)
        assert metrics["TradeCount"] == 0
        assert metrics["Exposure_%"] == 0.0
        np.testing.assert_array_almost_equal(strat_eq, np.ones_like(strat_eq))
