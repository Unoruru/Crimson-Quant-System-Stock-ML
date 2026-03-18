"""predict.py — Daily next-day trading signal for Crimson Quant System.

Run post-market-close to produce a BUY/HOLD or SELL/CASH signal for the
next trading day using the trained CNN-LSTM checkpoint.

Usage:
    python predict.py                 # use with_sentiment checkpoint
    python predict.py --no-sentiment  # use no_sentiment checkpoint
"""

import argparse
import os
from datetime import date, timedelta
from math import exp

import numpy as np
import torch

from crimson_quant.config import Config, QUANTILE_LEVEL
from crimson_quant.data_loader import _merge_sentiment
from crimson_quant.features import add_indicators
from crimson_quant.fetch_news import fetch_news_for_period
from crimson_quant.metrics import apply_affine_calibration
from crimson_quant.model import load_checkpoint
from crimson_quant.sentiment_evaluation import evaluate_and_save_sentiment
from crimson_quant.stock_data_fetcher import get_stock_data

# Extra CALENDAR days of OHLCV history fetched beyond the lookback window so
# rolling indicators have enough history to warm up before the inference window.
#
# Budget:  lookback=60 trading days  +  sma_50 warmup=50 trading days  = 110 td
# Calendar conversion: 110 td × (7/5) + ~10 holidays ≈ 164 cal days
# We use 150 to stay well above that floor with a safety margin.
WARM_UP_DAYS = 150


# =========================================================
# Helpers
# =========================================================

def _next_trading_day(ref: date) -> date:
    """Return the next weekday after *ref* (skips Saturday and Sunday)."""
    nxt = ref + timedelta(days=1)
    while nxt.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
        nxt += timedelta(days=1)
    return nxt


def _load_threshold(out_dir: str, quantile_level: float) -> float:
    """Derive the signal threshold from historical eval predictions.

    Loads eval_outputs/{tag}/eval_predictions.csv, takes the *quantile_level*
    quantile of the calibrated predicted log-returns, and floors it at 0.0 so
    the threshold is never negative (we only go long on positive predictions).

    Falls back to 0.0 when the file is missing or unreadable.
    """
    import pandas as pd  # local import to keep module-level imports minimal

    csv_path = os.path.join(out_dir, "eval_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] No eval_predictions.csv found at '{csv_path}'; threshold = 0.0")
        return 0.0

    try:
        df = pd.read_csv(csv_path)
        if "Pred_LogRet" not in df.columns:
            print(f"[WARN] 'Pred_LogRet' column missing in '{csv_path}'; threshold = 0.0")
            return 0.0
        vals = df["Pred_LogRet"].dropna().values
        if len(vals) == 0:
            return 0.0
        return float(max(np.quantile(vals, quantile_level), 0.0))
    except Exception as exc:
        print(f"[WARN] Could not load threshold from '{csv_path}': {exc}; threshold = 0.0")
        return 0.0


def _fetch_sentiment(ticker: str, start: str, end: str) -> str | None:
    """Fetch and score news articles for *ticker* over [start, end].

    Returns the path to the daily sentiment CSV, or None on any failure
    (missing API key, network error, etc.).  A None return causes the caller
    to use zeroed-out sentiment features rather than crashing.
    """
    try:
        raw_csv = fetch_news_for_period(ticker, start, end)
    except Exception as exc:
        print(f"[WARN] Failed to fetch news ({ticker} {start}→{end}): {exc}")
        return None

    try:
        return evaluate_and_save_sentiment(raw_csv, ticker, start, end, prefix="live")
    except Exception as exc:
        print(f"[WARN] Failed to score sentiment: {exc}")
        return None


# =========================================================
# Core prediction
# =========================================================

def run_predict(use_sentiment: bool = True) -> None:
    """Load checkpoint, fetch live data, and print tomorrow's trading signal."""

    # ----------------------------------------------------------
    # Step 1: Config + checkpoint
    # ----------------------------------------------------------
    cfg = Config.load()
    ticker = cfg.ticker
    tag = "with_sentiment" if use_sentiment else "no_sentiment"
    ckpt_path = os.path.join("checkpoints", f"{ticker}_{tag}_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run 'python train.py' first to train the model."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, x_scaler, meta = load_checkpoint(ckpt_path, device)

    lookback     = int(meta["lookback"])
    feature_cols = meta["feature_cols"]
    y_mean       = float(meta["y_mean"])
    y_std        = float(meta["y_std"])
    calibration  = meta.get("calibration", {"slope": 1.0, "intercept": 0.0})

    # ----------------------------------------------------------
    # Step 2: Dates
    # ----------------------------------------------------------
    today        = date.today()
    # yfinance's `end` parameter is exclusive, so add 1 day to include today
    yf_end       = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    buffer_start = (today - timedelta(days=lookback + WARM_UP_DAYS)).strftime("%Y-%m-%d")
    tomorrow     = _next_trading_day(today)

    # ----------------------------------------------------------
    # Step 3: OHLCV
    # ----------------------------------------------------------
    print(f"[INFO] Fetching OHLCV  : {ticker}  {buffer_start} → {today}")
    df = get_stock_data(ticker, buffer_start, yf_end)

    # ----------------------------------------------------------
    # Step 4: Sentiment (optional)
    # ----------------------------------------------------------
    sentiment_csv_path = None
    if use_sentiment:
        print(f"[INFO] Fetching news   : {ticker}  {buffer_start} → {today}")
        sentiment_csv_path = _fetch_sentiment(
            ticker, buffer_start, today.strftime("%Y-%m-%d")
        )

    # ----------------------------------------------------------
    # Step 5: Feature engineering
    # ----------------------------------------------------------
    df = add_indicators(df)

    # CRITICAL: y_next_logret is NaN on the last (today's) row because
    # tomorrow's close is unknown.  Dropping it now prevents _merge_sentiment's
    # internal dropna() from silently removing today's row.
    df = df.drop(columns=["y_next_logret"])

    df = _merge_sentiment(
        df,
        ticker=ticker,
        clip_to_coverage=True,
        sentiment_csv_path=sentiment_csv_path,
    )

    if len(df) < lookback:
        raise ValueError(
            f"Not enough rows after feature engineering: got {len(df)}, need {lookback}.\n"
            f"Try increasing WARM_UP_DAYS or verifying market data availability."
        )

    # ----------------------------------------------------------
    # Step 6: Build inference window  → (1, lookback, n_features)
    # ----------------------------------------------------------
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from dataframe: {missing}")

    window    = df[feature_cols].values[-lookback:].astype(np.float32)
    window_3d = window[np.newaxis, :, :]   # (1, lookback, n_features)

    # ----------------------------------------------------------
    # Step 7: Scale → infer → denormalize → calibrate
    # ----------------------------------------------------------
    window_scaled = x_scaler.transform(window_3d)
    tensor        = torch.tensor(window_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_norm = model(tensor).item()

    pred_logret = pred_norm * y_std + y_mean
    pred_logret = float(
        apply_affine_calibration(np.array([pred_logret]), calibration)[0]
    )

    # ----------------------------------------------------------
    # Step 8: Signal
    # ----------------------------------------------------------
    out_dir   = os.path.join("eval_outputs", tag)
    threshold = _load_threshold(out_dir, QUANTILE_LEVEL)
    signal    = "BUY / HOLD" if pred_logret > threshold else "SELL / CASH"

    # ----------------------------------------------------------
    # Step 9: Print formatted report
    # ----------------------------------------------------------
    today_close = float(df["Close"].iloc[-1])
    pred_close  = today_close * exp(pred_logret)
    today_str   = df.index[-1].strftime("%Y-%m-%d")
    q_pct       = int(round(QUANTILE_LEVEL * 100))
    sign        = "+" if pred_logret >= 0 else ""
    thr_sign    = "+" if threshold    >= 0 else ""

    print()
    print("=" * 54)
    print("   CRIMSON QUANT — NEXT-DAY TRADING SIGNAL")
    print("=" * 54)
    print(f"   Ticker           : {ticker}")
    print(f"   Checkpoint       : {tag}")
    print(f"   Today's Close    : ${today_close:.2f}  ({today_str})")
    print(f"   Target Date      : {tomorrow.strftime('%Y-%m-%d')}")
    print(f"   Predicted Close  : ${pred_close:.2f}")
    print(f"   Predicted LogRet : {sign}{pred_logret:.5f}")
    print(f"   Threshold (Q{q_pct}%) : {thr_sign}{threshold:.5f}")
    print("   " + "-" * 50)
    print(f"   SIGNAL           : [ {signal} ]")
    print("=" * 54)
    print()


# =========================================================
# CLI entry point
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crimson Quant System — next-day trading signal"
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Use the no_sentiment checkpoint instead of with_sentiment.",
    )
    args = parser.parse_args()
    run_predict(use_sentiment=not args.no_sentiment)


if __name__ == "__main__":
    main()
