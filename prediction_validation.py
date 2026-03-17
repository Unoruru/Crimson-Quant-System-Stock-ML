"""Inference entry point for Crimson Quant System.

Loads trained checkpoints and evaluates on unseen data fetched via yfinance.
Evaluates both no_sentiment and with_sentiment experiments when available.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from torch.utils.data import DataLoader

from model import load_checkpoint
from data_loader import load_data, make_windows, WindowDataset
from metrics import (
    compute_price_metrics,
    compute_direction_metrics,
    compute_trading_metrics,
    compute_logret_metrics,
    apply_affine_calibration,
    fit_affine_calibration,
    is_safe_affine_calibration,
    logret_to_next_close,
    write_metrics_report,
)
from plotting import plot_forecast_eval, plot_strategy_equity
from config import QUANTILE_LEVEL, Config
from train import predict_all_logret
from sentiment_evaluation import evaluate_and_save_sentiment
from fetch_news import fetch_news_for_period

# =========================================================
# Config
# =========================================================
OUT_ROOT = "eval_outputs"


# =========================================================
# Range parsing
# =========================================================
def parse_eval_range(range_str: str, train_end: str) -> str:
    """Convert a range string to an absolute end date.

    Supports relative durations (30d, 4w, 3m) and absolute dates (YYYY-MM-DD).
    The duration is added starting from the day after *train_end*.
    """
    train_end_dt = pd.to_datetime(train_end).normalize()
    eval_start = train_end_dt + pd.Timedelta(days=1)

    match = re.match(r"^(\d+)([dwm])$", range_str.lower())
    if match:
        n, unit = int(match.group(1)), match.group(2)
        if unit == "d":
            return (eval_start + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
        elif unit == "w":
            return (eval_start + pd.Timedelta(weeks=n)).strftime("%Y-%m-%d")
        elif unit == "m":
            return (eval_start + relativedelta(months=n)).strftime("%Y-%m-%d")

    try:
        return pd.to_datetime(range_str).strftime("%Y-%m-%d")
    except (ValueError, pd.errors.ParserError):
        raise ValueError(
            f"Invalid --range '{range_str}'. "
            f"Use Nd/Nw/Nm (e.g. 30d, 4w, 3m) or YYYY-MM-DD"
        )


# =========================================================
# Utilities
# =========================================================
def pick_latest_file(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _resolve_calibration(meta: dict) -> dict:
    """Use checkpoint calibration when available, otherwise fit from saved val predictions."""
    if meta.get("has_calibration", False):
        calibration = meta.get("calibration", {"slope": 1.0, "intercept": 0.0})
        if is_safe_affine_calibration(calibration):
            return calibration
        print(f"[WARN] Ignoring unsafe checkpoint calibration: {calibration}")

    tag = meta.get("tag")
    if tag:
        val_pred_path = os.path.join(f"my_fig_{tag}", "val_predictions.csv")
        if os.path.exists(val_pred_path):
            try:
                pred_df = pd.read_csv(val_pred_path)
                if {"True_LogRet", "Pred_LogRet"}.issubset(pred_df.columns):
                    calibration = fit_affine_calibration(
                        pred_df["True_LogRet"].to_numpy(),
                        pred_df["Pred_LogRet"].to_numpy(),
                    )
                    print(
                        "[INFO] Using fallback calibration from "
                        f"{val_pred_path}: slope={calibration['slope']:.4f}, "
                        f"intercept={calibration['intercept']:.5f}"
                    )
                    return calibration
            except Exception as exc:
                print(f"[WARN] Failed to derive fallback calibration from {val_pred_path}: {exc}")

    calibration = meta.get("calibration", {"slope": 1.0, "intercept": 0.0})
    if is_safe_affine_calibration(calibration):
        return calibration
    return {"slope": 1.0, "intercept": 0.0}


def _ensure_sentiment_data(ticker: str, eval_start: str, eval_end: str) -> str | None:
    """Fetch news for the prediction window and produce a daily sentiment CSV.

    Calls the Alpha Vantage API for the prediction period only (eval_start → eval_end).
    Returns path to daily sentiment CSV, or None on failure.
    """
    try:
        raw_csv = fetch_news_for_period(ticker, eval_start, eval_end)
    except Exception as exc:
        print(f"[WARN] Failed to fetch prediction news: {exc}")
        return None

    try:
        return evaluate_and_save_sentiment(raw_csv, ticker, eval_start, eval_end, prefix="prediction")
    except Exception as exc:
        print(f"[WARN] Failed to build prediction sentiment: {exc}")
        return None


# =========================================================
# Evaluation
# =========================================================
def evaluate_unseen_period(model, x_scaler, meta, df: pd.DataFrame, device: str,
                           eval_start_date=None, eval_end_date=None):
    """Evaluate checkpoint on unseen dates after the training end date."""
    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    lookback = int(meta["lookback"])
    y_mean = float(meta["y_mean"])
    y_std = float(meta["y_std"])
    calibration = _resolve_calibration(meta)

    train_end_date = pd.to_datetime(meta["cfg"]["end"]).normalize()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    X, y, dates = make_windows(df, lookback, feature_cols, target_col)
    dates = pd.to_datetime(dates)

    eval_mask = dates > train_end_date

    if eval_start_date is not None:
        eval_start = pd.to_datetime(eval_start_date).normalize()
        eval_mask = eval_mask & (dates >= eval_start)
    if eval_end_date is not None:
        eval_end = pd.to_datetime(eval_end_date).normalize()
        eval_mask = eval_mask & (dates <= eval_end)

    if not np.any(eval_mask):
        raise ValueError(
            f"No evaluation dates selected.\n"
            f"  Train end date: {train_end_date.date()}\n"
            f"  Data date range: {dates.min().date()} -> {dates.max().date()} ({len(dates)} windows)\n"
            f"  eval_start_date: {eval_start_date}\n"
            f"  eval_end_date: {eval_end_date}\n"
            f"  Dates after train end: {np.sum(dates > train_end_date)}"
        )

    first_eval_date = dates[eval_mask].min()
    hist_mask = dates < first_eval_date

    X_eval = X[eval_mask]
    y_eval = y[eval_mask]
    dates_eval = dates[eval_mask]

    dates_hist = dates[hist_mask]

    X_eval_s = x_scaler.transform(X_eval)
    y_eval_s = (y_eval - y_mean) / y_std

    batch_size = int(meta["cfg"]["batch_size"])
    eval_loader = DataLoader(WindowDataset(X_eval_s, y_eval_s), batch_size=batch_size, shuffle=False)

    y_true_raw, y_pred_raw = predict_all_logret(model, eval_loader, device, y_mean, y_std)
    y_pred_raw_uncal = y_pred_raw.copy()
    y_pred_raw = apply_affine_calibration(y_pred_raw, calibration)

    if target_col == "y_next_logret":
        today_close_eval = df["Close"].shift(1).reindex(dates_eval).values.astype(np.float64)
        true_close_eval = logret_to_next_close(today_close_eval, y_true_raw)
        pred_close_eval = logret_to_next_close(today_close_eval, y_pred_raw)
    else:
        today_close_eval = df["Close"].shift(1).reindex(dates_eval).values.astype(np.float64)
        true_close_eval = y_true_raw
        pred_close_eval = y_pred_raw

    price_metrics = compute_price_metrics(true_close_eval, pred_close_eval)
    logret_metrics = compute_logret_metrics(y_true_raw, y_pred_raw)
    direction_metrics = compute_direction_metrics(today_close_eval, true_close_eval, pred_close_eval)
    trading_metrics, strat_equity, bh_equity = compute_trading_metrics(
        today_close_eval, true_close_eval, pred_close_eval, quantile_level=QUANTILE_LEVEL
    )

    metrics = {}
    metrics.update(price_metrics)
    metrics.update(logret_metrics)
    metrics.update(direction_metrics)
    metrics.update(trading_metrics)

    history_dates = dates_hist
    history_close = df["Close"].reindex(history_dates).values

    return {
        "metrics": metrics,
        "history_dates": history_dates,
        "history_close": history_close,
        "eval_dates": dates_eval,
        "today_close_eval": today_close_eval,
        "true_close_eval": true_close_eval,
        "pred_close_eval": pred_close_eval,
        "true_raw_eval": y_true_raw,
        "pred_raw_eval": y_pred_raw,
        "pred_raw_eval_uncal": y_pred_raw_uncal,
        "strat_equity": strat_equity,
        "bh_equity": bh_equity,
        "target_col": target_col,
        "train_end_date": train_end_date,
        "n_eval": len(dates_eval),
        "n_total_windows": len(dates),
        "calibration": calibration,
    }


# =========================================================
# Single-checkpoint evaluation
# =========================================================
def _evaluate_single_checkpoint(cfg, ckpt_path: str, eval_end: str, out_dir: str,
                                sentiment_csv_path: str | None = None):
    """Load one checkpoint, evaluate on unseen data, save outputs."""

    model, x_scaler, meta = load_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using ckpt: {ckpt_path}")
    print(f"[INFO] ckpt ticker: {meta.get('cfg', {}).get('ticker', 'UNKNOWN')}")

    feature_cols = meta.get("feature_cols", [])
    print(f"[INFO] ckpt features: {feature_cols}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output folder: {out_dir}")

    start = meta.get("cfg", {}).get("start")
    train_end = meta.get("cfg", {}).get("end")
    ticker = meta.get("cfg", {}).get("ticker", "AAPL")

    train_end_dt = pd.to_datetime(train_end).normalize()
    eval_end_dt = pd.to_datetime(eval_end).normalize()
    if eval_end_dt <= train_end_dt:
        eval_end = pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"[WARN] eval_end ({eval_end_dt.date()}) <= train end ({train_end}), using {eval_end}")

    df = load_data(ticker, start=start, end=eval_end, sentiment_csv_path=sentiment_csv_path)

    tag = f"{ticker}_final_eval"
    print(f"[INFO] Train end date: {train_end}")
    print(f"[INFO] Eval end date: {eval_end}")
    print(f"[INFO] Data range: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} rows)")

    result = evaluate_unseen_period(model, x_scaler, meta, df, device=device,
                                    eval_end_date=eval_end)

    metrics = result["metrics"]

    print(
        f"[{tag}] "
        f"FinalEval MAE={metrics['MAE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"LogRetMAE={metrics['LogRet_MAE']:.5f} | "
        f"Bias={metrics['LogRet_Bias']:.5f} | "
        f"MAPE={metrics['MAPE_%']:.2f}% | "
        f"R2={metrics['R2']:.4f} | "
        f"DirAcc={metrics['DirAcc_%']:.2f}% | "
        f"PredUp={metrics['PredUpRatio_%']:.2f}% | "
        f"PredDown={metrics['PredDownRatio_%']:.2f}% | "
        f"StratRet={metrics['StrategyReturn_%']:.2f}% | "
        f"BuyHold={metrics['BuyHoldReturn_%']:.2f}% | "
        f"Excess={metrics['ExcessReturn_%']:.2f}% | "
        f"Sharpe={metrics['Sharpe']:.4f} | "
        f"Trades={metrics['TradeCount']} | "
        f"Exposure={metrics['Exposure_%']:.2f}% | "
        f"Q={metrics['Threshold_Quantile_%']:.0f}% | "
        f"Thr={metrics['Threshold_Value']:.5f}"
    )

    metrics_text = (
        f"FINAL EVAL -> "
        f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
        f"LogRetMAE={metrics['LogRet_MAE']:.5f} | Bias={metrics['LogRet_Bias']:.5f} | "
        f"MAPE={metrics['MAPE_%']:.2f}% | R2={metrics['R2']:.4f} | "
        f"DirAcc={metrics['DirAcc_%']:.2f}% | "
        f"UpPrec={metrics['UpPrecision_%']:.2f}% | DownPrec={metrics['DownPrecision_%']:.2f}% | "
        f"PredUp={metrics['PredUpRatio_%']:.2f}% | PredDown={metrics['PredDownRatio_%']:.2f}% | "
        f"StratRet={metrics['StrategyReturn_%']:.2f}% | BuyHold={metrics['BuyHoldReturn_%']:.2f}% | "
        f"Excess={metrics['ExcessReturn_%']:.2f}% | Sharpe={metrics['Sharpe']:.4f} | "
        f"MDD={metrics['MaxDrawdown_%']:.2f}% | WinRate={metrics['WinRate_%']:.2f}% | "
        f"Trades={metrics['TradeCount']} | Exposure={metrics['Exposure_%']:.2f}% | "
        f"Q={metrics['Threshold_Quantile_%']:.0f}% | Thr={metrics['Threshold_Value']:.5f}"
    )

    plot_forecast_eval(
        history_dates=result["history_dates"],
        history_close=result["history_close"],
        eval_dates=result["eval_dates"],
        eval_true=result["true_close_eval"],
        eval_pred=result["pred_close_eval"],
        metrics_text=metrics_text,
        out_dir=out_dir,
    )

    plot_strategy_equity(
        dates=result["eval_dates"],
        strat_equity=result["strat_equity"],
        bh_equity=result["bh_equity"],
        out_dir=out_dir,
    )

    pred_df = pd.DataFrame({
        "Date": result["eval_dates"],
        "Today_Close": result["today_close_eval"],
        "True_Close_next_day": result["true_close_eval"],
        "Pred_Close_next_day": result["pred_close_eval"],
    })

    if result["target_col"] == "y_next_logret":
        pred_df["True_LogRet"] = result["true_raw_eval"]
        pred_df["Pred_LogRet"] = result["pred_raw_eval"]
        pred_df["Pred_LogRet_Raw"] = result["pred_raw_eval_uncal"]

    pred_df["Abs_Error"] = np.abs(pred_df["Pred_Close_next_day"] - pred_df["True_Close_next_day"])
    pred_df["True_Direction"] = np.sign(pred_df["True_Close_next_day"] - pred_df["Today_Close"])
    pred_df["Pred_Direction"] = np.sign(pred_df["Pred_Close_next_day"] - pred_df["Today_Close"])
    pred_df.to_csv(os.path.join(out_dir, "eval_predictions.csv"), index=False)

    write_metrics_report(
        metrics_txt_path=os.path.join(out_dir, "metrics.txt"),
        header_lines=[
            f"Checkpoint: {os.path.basename(ckpt_path)}",
            f"Ticker: {ticker}",
            f"Date range used: [{start}, {eval_end}]",
            f"Target: {result['target_col']}",
            f"Train end date from checkpoint: {result['train_end_date'].date()}",
            f"Evaluation start date: {result['eval_dates'].min().date()}",
            f"Evaluation end date: {result['eval_dates'].max().date()}",
            f"Total windows: {result['n_total_windows']}",
            f"Evaluation windows: {result['n_eval']}",
            f"Calibration_Slope: {result['calibration']['slope']:.6f}",
            f"Calibration_Intercept: {result['calibration']['intercept']:.6f}",
        ],
        sections=[
            ("Final Evaluation Metrics", metrics),
        ],
    )

    print(f"[OK] Saved outputs to: {out_dir}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Crimson Quant System - Inference on unseen data",
    )
    parser.add_argument(
        "--range", "-r", default=None,
        help=(
            "prediction range from training end date. "
            "Examples: 30d (30 days), 4w (4 weeks), 3m (3 months), "
            "or an absolute end date YYYY-MM-DD"
        ),
    )
    args = parser.parse_args()

    cfg = Config.load()
    ticker = cfg.ticker

    if args.range:
        eval_end = parse_eval_range(args.range, cfg.end)
        print(f"[INFO] Prediction range: {cfg.end} -> {eval_end}")
    else:
        eval_end = parse_eval_range("1m", cfg.end)
        print(f"[INFO] Prediction range (default 1m): {cfg.end} -> {eval_end}")

    today = pd.Timestamp.today().normalize()
    eval_end_dt = pd.to_datetime(eval_end).normalize()
    if eval_end_dt > today:
        raise SystemExit(
            f"[ERROR] --range end date {eval_end} is in the future ({today.date()} is today).\n"
            f"prediction_validation.py requires ground-truth close prices and cannot evaluate\n"
            f"future dates. Use 'python predict.py' for a forward-looking signal instead."
        )

    experiments = [
        {"tag": "no_sentiment",   "ckpt": f"{ticker}_no_sentiment_best.pt",   "needs_sentiment": False},
        {"tag": "with_sentiment", "ckpt": f"{ticker}_with_sentiment_best.pt", "needs_sentiment": True},
    ]

    evaluated = 0

    for exp in experiments:
        ckpt_path = os.path.join("checkpoints", exp["ckpt"])
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {exp['tag']}")
        print(f"{'=' * 60}")

        sentiment_csv_path = None
        if exp["needs_sentiment"]:
            eval_start = (pd.to_datetime(cfg.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            sentiment_csv_path = _ensure_sentiment_data(ticker, eval_start=eval_start, eval_end=eval_end)

        out_dir = os.path.join(OUT_ROOT, exp["tag"])
        _evaluate_single_checkpoint(cfg, ckpt_path, eval_end, out_dir, sentiment_csv_path=sentiment_csv_path)
        evaluated += 1

    if evaluated == 0:
        raise FileNotFoundError(
            f"No checkpoints found for ticker {ticker}. "
            f"Expected files in checkpoints/ matching {ticker}_*_best.pt"
        )

    print(f"\n[DONE] Evaluated {evaluated} checkpoint(s).")


if __name__ == "__main__":
    main()
