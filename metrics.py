"""Shared metric functions for training and evaluation."""

import logging
import math

import numpy as np

from config import EPSILON, SCALER_EPSILON, ANNUALIZED_FACTOR, QUANTILE_LEVEL

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def fit_affine_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_slope: float = 0.25,
    max_slope: float = 2.0,
    max_abs_intercept: float = 0.03,
) -> dict:
    """Fit an affine calibration y ~= slope * pred + intercept on validation data."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    pred_var = float(np.var(y_pred))
    true_mean = float(np.mean(y_true))
    pred_mean = float(np.mean(y_pred))

    if pred_var < EPSILON:
        slope = 1.0
    else:
        cov = float(np.mean((y_pred - pred_mean) * (y_true - true_mean)))
        slope = cov / pred_var
        if not np.isfinite(slope) or slope <= 0:
            pred_std = float(np.std(y_pred))
            true_std = float(np.std(y_true))
            slope = true_std / pred_std if pred_std > EPSILON else 1.0

    intercept = float(true_mean - slope * pred_mean)
    calibration = {
        "slope": float(slope),
        "intercept": intercept,
    }
    if not is_safe_affine_calibration(
        calibration,
        min_slope=min_slope,
        max_slope=max_slope,
        max_abs_intercept=max_abs_intercept,
    ):
        return {"slope": 1.0, "intercept": 0.0}
    return calibration


def is_safe_affine_calibration(
    calibration: dict | None,
    min_slope: float = 0.25,
    max_slope: float = 2.0,
    max_abs_intercept: float = 0.03,
) -> bool:
    """Check whether a fitted calibration is conservative enough to apply."""
    if not calibration:
        return False

    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    return (
        np.isfinite(slope)
        and np.isfinite(intercept)
        and min_slope <= slope <= max_slope
        and abs(intercept) <= max_abs_intercept
    )


def apply_affine_calibration(y_pred: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Apply affine calibration to predictions."""
    if not calibration:
        return np.asarray(y_pred, dtype=np.float64)

    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return y_pred * slope + intercept


def compute_logret_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute diagnostics on the raw regression target (log return)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    return {
        "LogRet_MAE": float(np.mean(np.abs(err))),
        "LogRet_RMSE": float(rmse(y_true, y_pred)),
        "LogRet_Bias": float(np.mean(err)),
        "LogRet_Corr": corr,
        "True_LogRet_Mean": float(np.mean(y_true)),
        "Pred_LogRet_Mean": float(np.mean(y_pred)),
        "True_LogRet_Std": float(np.std(y_true)),
        "Pred_LogRet_Std": float(np.std(y_pred)),
    }


KEY_METRIC_ORDER = [
    "MAE",
    "RMSE",
    "MAPE_%",
    "R2",
    "LogRet_MAE",
    "LogRet_Bias",
    "DirAcc_%",
    "StrategyReturn_%",
    "ExcessReturn_%",
    "Sharpe",
    "MaxDrawdown_%",
    "WinRate_%",
    "TradeCount",
    "Exposure_%",
]


def _format_metric_value(value) -> str:
    """Format metric values consistently for text and markdown outputs."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.6f}"
    return str(value)


def write_metrics_report(
    metrics_txt_path: str,
    header_lines: list[str],
    sections: list[tuple[str, dict]],
    key_metric_order: list[str] | None = None,
) -> None:
    """Write a scan-friendly plain-text metrics report."""
    key_metric_order = key_metric_order or KEY_METRIC_ORDER
    merged_metrics = {}
    for _, section_metrics in sections:
        merged_metrics.update(section_metrics)

    summary_items = [(key, merged_metrics[key]) for key in key_metric_order if key in merged_metrics]

    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"{line}\n")
        if summary_items:
            f.write("\n[Key Metrics - Scan Here]\n")
            for key, value in summary_items:
                f.write(f"!!! {key}: {_format_metric_value(value)}\n")
        f.write("\n")
        for section_name, section_metrics in sections:
            f.write(f"[{section_name}]\n")
            for key, value in section_metrics.items():
                f.write(f"{key}: {_format_metric_value(value)}\n")
            f.write("\n")


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown from peak."""
    running_max = np.maximum.accumulate(equity_curve)
    dd = equity_curve / (running_max + EPSILON) - 1.0
    return float(dd.min())


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio."""
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    if len(daily_returns) == 0:
        return float("nan")
    std = daily_returns.std()
    if std < EPSILON:
        return float("nan")
    return float((daily_returns.mean() / std) * np.sqrt(ANNUALIZED_FACTOR))


def logret_to_next_close(today_close: np.ndarray, pred_logret: np.ndarray) -> np.ndarray:
    """Convert log return prediction to next-day close price."""
    return today_close * np.exp(pred_logret)


def compute_price_metrics(true_close: np.ndarray, pred_close: np.ndarray) -> dict:
    """Compute price-level metrics: MAE, RMSE, MAPE, R2."""
    mae = float(np.mean(np.abs(pred_close - true_close)))
    rmse_v = float(rmse(true_close, pred_close))

    denom = np.clip(np.abs(true_close), SCALER_EPSILON, None)
    mape = float(np.mean(np.abs((pred_close - true_close) / denom)) * 100.0)

    ss_res = float(np.sum((true_close - pred_close) ** 2))
    ss_tot = float(np.sum((true_close - np.mean(true_close)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + EPSILON))

    return {
        "MAE": mae,
        "RMSE": rmse_v,
        "MAPE_%": mape,
        "R2": r2,
    }


def compute_direction_metrics(
    today_close: np.ndarray,
    true_close: np.ndarray,
    pred_close: np.ndarray,
) -> dict:
    """Compute directional accuracy and precision metrics."""
    true_dir = np.sign(true_close - today_close)
    pred_dir = np.sign(pred_close - today_close)

    valid = true_dir != 0
    dir_acc = float(np.mean((true_dir[valid] == pred_dir[valid]).astype(float)) * 100.0) if np.any(valid) else float("nan")

    pred_up = pred_dir > 0
    pred_down = pred_dir < 0
    true_up = true_dir > 0
    true_down = true_dir < 0

    up_precision = float(np.mean(true_up[pred_up]) * 100.0) if np.any(pred_up) else float("nan")
    down_precision = float(np.mean(true_down[pred_down]) * 100.0) if np.any(pred_down) else float("nan")

    pred_up_ratio = float(np.mean(pred_up) * 100.0)
    pred_down_ratio = float(np.mean(pred_down) * 100.0)

    return {
        "DirAcc_%": dir_acc,
        "UpPrecision_%": up_precision,
        "DownPrecision_%": down_precision,
        "PredUpRatio_%": pred_up_ratio,
        "PredDownRatio_%": pred_down_ratio,
    }


def compute_trading_metrics(
    today_close: np.ndarray,
    true_close: np.ndarray,
    pred_close: np.ndarray,
    quantile_level: float = QUANTILE_LEVEL,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Quantile-based long-only strategy metrics."""
    true_ret = (true_close / (today_close + EPSILON)) - 1.0
    pred_ret = (pred_close / (today_close + EPSILON)) - 1.0

    logger.debug(
        "pred_ret distribution: min=%.6f max=%.6f mean=%.6f q50=%.6f q75=%.6f q90=%.6f q95=%.6f",
        pred_ret.min(), pred_ret.max(), pred_ret.mean(),
        np.quantile(pred_ret, 0.50), np.quantile(pred_ret, 0.75),
        np.quantile(pred_ret, 0.90), np.quantile(pred_ret, 0.95),
    )

    threshold = max(float(np.quantile(pred_ret, quantile_level)), 0.0)
    signal = (pred_ret > threshold).astype(float)

    strat_daily = signal * true_ret
    bh_daily = true_ret

    strat_equity = np.cumprod(1.0 + strat_daily)
    bh_equity = np.cumprod(1.0 + bh_daily)

    strat_return = float(strat_equity[-1] - 1.0) if len(strat_equity) > 0 else float("nan")
    bh_return = float(bh_equity[-1] - 1.0) if len(bh_equity) > 0 else float("nan")
    excess_return = strat_return - bh_return

    active_mask = signal > 0
    win_rate = float(np.mean(strat_daily[active_mask] > 0) * 100.0) if np.any(active_mask) else float("nan")
    sharpe = sharpe_ratio(strat_daily)
    mdd = max_drawdown(strat_equity) if len(strat_equity) > 0 else float("nan")

    trade_count = int(np.sum(active_mask))
    exposure = float(np.mean(signal) * 100.0)

    metrics_dict = {
        "StrategyReturn_%": strat_return * 100.0,
        "BuyHoldReturn_%": bh_return * 100.0,
        "ExcessReturn_%": excess_return * 100.0,
        "Sharpe": sharpe,
        "MaxDrawdown_%": mdd * 100.0,
        "WinRate_%": win_rate,
        "TradeCount": trade_count,
        "Exposure_%": exposure,
        "Threshold_Value": float(threshold),
        "Threshold_Quantile_%": quantile_level * 100.0,
    }
    return metrics_dict, strat_equity, bh_equity
