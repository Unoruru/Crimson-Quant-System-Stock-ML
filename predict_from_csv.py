import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from torch_main import (
    load_checkpoint,
    add_indicators,
    make_windows,
    WindowDataset,
    load_sentiment_daily,
)

# =========================================================
# 0) Default config
# =========================================================
DEFAULT_CKPT_GLOB = os.path.join("checkpoints", "AAPL_no_sentiment_best.pt")
DEFAULT_CSV_GLOB = "AAPL_2018-01-01_2025-12-31.csv"

OUT_DIR = "eval_outputs"
TAG = "AAPL_final_eval"

# ===== 你可以手动控制评估日期范围 =====
EVAL_START_DATE = "2022-11-03"   # None 表示从训练结束后第一天开始
EVAL_END_DATE   = "2023-6-30"   # None 表示直到 CSV 最后一天

# 如果你不想手动指定日期，也可以用最后 N 天
USE_EXPLICIT_DATE_RANGE = True

# 备用方案：最后 N 天 / 最后比例
EVAL_DAYS = 126
EVAL_RATIO = 0.20

QUANTILE_LEVEL = 0.70


# =========================================================
# 1) Utilities
# =========================================================
def pick_latest_file(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    if len(daily_returns) == 0:
        return float("nan")
    std = daily_returns.std()
    if std < 1e-12:
        return float("nan")
    return float((daily_returns.mean() / std) * np.sqrt(252.0))


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    dd = equity_curve / (running_max + 1e-12) - 1.0
    return float(dd.min())


def logret_to_next_close(today_close: np.ndarray, pred_logret: np.ndarray) -> np.ndarray:
    return today_close * np.exp(pred_logret)


# =========================================================
# 2) Load CSV and align preprocessing with torch_main.py
# =========================================================
def load_data_from_csv(csv_path: str, start: str, end: str) -> pd.DataFrame:
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

    # same indicators as training
    df = add_indicators(df)

    # remove placeholder sentiment then merge real daily sentiment
    if "sentiment" in df.columns:
        df = df.drop(columns=["sentiment"])

    sent_path = os.path.join("data", "sentiment_daily.csv")
    sent = load_sentiment_daily(sent_path)

    df2 = df.reset_index()
    df2["Date"] = pd.to_datetime(df2["Date"]).dt.normalize()

    if sent is not None and len(sent) > 0 and "sentiment" in sent.columns:
        df2 = df2.merge(sent, on="Date", how="left")

        sent_max_date = sent["Date"].max()
        df2["sentiment"] = df2["sentiment"].ffill()
        df2.loc[df2["Date"] > sent_max_date, "sentiment"] = np.nan
        df2["sentiment"] = df2["sentiment"].fillna(0.0)
    else:
        df2["sentiment"] = 0.0

    df2 = df2.sort_values("Date")
    df = df2.set_index("Date")
    df = df.dropna().copy()
    return df


# =========================================================
# 3) Predict raw model outputs
# =========================================================
@torch.no_grad()
def predict_all(model, loader, device, y_mean, y_std):
    model.eval()
    ys, ps = [], []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true_s = np.concatenate(ys)
    y_pred_s = np.concatenate(ps)

    y_true = y_true_s * y_std + y_mean
    y_pred = y_pred_s * y_std + y_mean
    return y_true, y_pred


# =========================================================
# 4) Metrics
# =========================================================
def compute_price_metrics(true_close: np.ndarray, pred_close: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(pred_close - true_close)))
    rmse_v = float(rmse(true_close, pred_close))
    denom = np.clip(np.abs(true_close), 1e-8, None)
    mape = float(np.mean(np.abs((pred_close - true_close) / denom)) * 100.0)

    ss_res = float(np.sum((true_close - pred_close) ** 2))
    ss_tot = float(np.sum((true_close - np.mean(true_close)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {
        "MAE": mae,
        "RMSE": rmse_v,
        "MAPE_%": mape,
        "R2": r2,
    }


def compute_direction_metrics(today_close: np.ndarray, true_close: np.ndarray, pred_close: np.ndarray) -> dict:
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


def compute_trading_metrics(today_close: np.ndarray, true_close: np.ndarray, pred_close: np.ndarray):
    true_ret = (true_close / (today_close + 1e-12)) - 1.0
    pred_ret = (pred_close / (today_close + 1e-12)) - 1.0

    print("\n[DEBUG pred_ret distribution]")
    print(f"min   = {pred_ret.min():.6f}")
    print(f"max   = {pred_ret.max():.6f}")
    print(f"mean  = {pred_ret.mean():.6f}")
    print(f"q50   = {np.quantile(pred_ret, 0.50):.6f}")
    print(f"q75   = {np.quantile(pred_ret, 0.75):.6f}")
    print(f"q90   = {np.quantile(pred_ret, 0.90):.6f}")
    print(f"q95   = {np.quantile(pred_ret, 0.95):.6f}")

    threshold = np.quantile(pred_ret, QUANTILE_LEVEL)
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

    metrics = {
        "StrategyReturn_%": strat_return * 100.0,
        "BuyHoldReturn_%": bh_return * 100.0,
        "ExcessReturn_%": excess_return * 100.0,
        "Sharpe": sharpe,
        "MaxDrawdown_%": mdd * 100.0,
        "WinRate_%": win_rate,
        "TradeCount": trade_count,
        "Exposure_%": exposure,
        "Threshold_Value": float(threshold),
        "Threshold_Quantile_%": QUANTILE_LEVEL * 100.0,
    }
    return metrics, strat_equity, bh_equity


# =========================================================
# 5) Plotting
# =========================================================
def plot_forecast(history_dates, history_close, eval_dates, eval_true, eval_pred, metrics_text, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(history_dates, history_close, label="History / Context")
    plt.plot(eval_dates, eval_true, label="Evaluation Ground Truth")
    plt.plot(eval_dates, eval_pred, label="Evaluation Predictions")
    plt.title("Stock Prediction Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.legend()
    plt.figtext(0.01, 0.005, metrics_text, ha="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_full.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(eval_dates, eval_true, label="Evaluation Ground Truth")
    plt.plot(eval_dates, eval_pred, label="Evaluation Predictions")
    plt.title("Final Evaluation Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.legend()
    plt.figtext(0.01, 0.005, metrics_text, ha="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_zoom.png"), dpi=200)
    plt.close()


def plot_strategy_equity(eval_dates, strat_equity, bh_equity, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(eval_dates, strat_equity, label="Strategy Equity")
    plt.plot(eval_dates, bh_equity, label="Buy & Hold Equity")
    plt.title("Strategy vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_vs_buyhold.png"), dpi=200)
    plt.close()


# =========================================================
# 6) Final evaluation only
# =========================================================
def evaluate_unseen_period(model, x_scaler, meta, df: pd.DataFrame, device: str):
    """
    只评估 checkpoint 训练结束日期之后的 unseen dates。
    允许再手动指定一个更窄的 evaluation 日期范围。
    """
    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    lookback = int(meta["lookback"])
    y_mean = float(meta["y_mean"])
    y_std = float(meta["y_std"])

    train_end_date = pd.to_datetime(meta["cfg"]["end"]).normalize()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    X, y, dates = make_windows(df, lookback, feature_cols, target_col)
    dates = pd.to_datetime(dates)

    # 先保证只评估训练结束之后的日期
    eval_mask = dates > train_end_date

    # 再根据你手动指定的日期范围缩小评估区间
    if USE_EXPLICIT_DATE_RANGE:
        if EVAL_START_DATE is not None:
            eval_start = pd.to_datetime(EVAL_START_DATE).normalize()
            eval_mask = eval_mask & (dates >= eval_start)
        if EVAL_END_DATE is not None:
            eval_end = pd.to_datetime(EVAL_END_DATE).normalize()
            eval_mask = eval_mask & (dates <= eval_end)
    else:
        unseen_dates = dates[eval_mask]
        if len(unseen_dates) == 0:
            raise ValueError(
                f"No unseen dates found after checkpoint end date {train_end_date.date()}."
            )

        # 在 unseen_dates 内部，再取最后 N 天或最后比例
        if EVAL_DAYS is not None:
            n_eval = min(EVAL_DAYS, len(unseen_dates))
        else:
            n_eval = max(1, int(len(unseen_dates) * EVAL_RATIO))

        chosen_dates = unseen_dates[-n_eval:]
        eval_mask = np.isin(dates, chosen_dates)

    if not np.any(eval_mask):
        raise ValueError(
            "No evaluation dates selected. "
            "Check checkpoint end date, CSV coverage, and your EVAL_START_DATE / EVAL_END_DATE."
        )

    # 历史上下文：训练结束前 + 评估开始前的数据都可以作为 history
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

    y_true_raw, y_pred_raw = predict_all(model, eval_loader, device, y_mean, y_std)

    if target_col == "y_next_logret":
        today_close_eval = df["Close"].shift(1).reindex(dates_eval).values.astype(np.float64)
        true_close_eval = logret_to_next_close(today_close_eval, y_true_raw)
        pred_close_eval = logret_to_next_close(today_close_eval, y_pred_raw)
    else:
        today_close_eval = df["Close"].shift(1).reindex(dates_eval).values.astype(np.float64)
        true_close_eval = y_true_raw
        pred_close_eval = y_pred_raw

    price_metrics = compute_price_metrics(true_close_eval, pred_close_eval)
    direction_metrics = compute_direction_metrics(today_close_eval, true_close_eval, pred_close_eval)
    trading_metrics, strat_equity, bh_equity = compute_trading_metrics(
        today_close_eval, true_close_eval, pred_close_eval
    )

    metrics = {}
    metrics.update(price_metrics)
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
        "strat_equity": strat_equity,
        "bh_equity": bh_equity,
        "target_col": target_col,
        "train_end_date": train_end_date,
        "n_eval": len(dates_eval),
        "n_total_windows": len(dates),
    }


# =========================================================
# 7) Main
# =========================================================
def main():
    ckpt_path = pick_latest_file(DEFAULT_CKPT_GLOB)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found: {DEFAULT_CKPT_GLOB}")

    csv_path = pick_latest_file(DEFAULT_CSV_GLOB)
    if csv_path is None:
        raise FileNotFoundError(f"No csv found: {DEFAULT_CSV_GLOB}")

    model, x_scaler, meta = load_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using ckpt: {ckpt_path}")
    print(f"[INFO] Using csv : {csv_path}")
    print(f"[INFO] ckpt ticker: {meta.get('cfg', {}).get('ticker', 'UNKNOWN')}")

    start = meta.get("cfg", {}).get("start")
    end = meta.get("cfg", {}).get("end")

    df = load_data_from_csv(csv_path, start=start, end=end)
    result = evaluate_unseen_period(model, x_scaler, meta, df, device=device)

    metrics = result["metrics"]

    print(
        f"[{TAG}] "
        f"FinalEval MAE={metrics['MAE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
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

    os.makedirs(OUT_DIR, exist_ok=True)

    plot_forecast(
        history_dates=result["history_dates"],
        history_close=result["history_close"],
        eval_dates=result["eval_dates"],
        eval_true=result["true_close_eval"],
        eval_pred=result["pred_close_eval"],
        metrics_text=metrics_text,
        out_dir=OUT_DIR,
    )

    plot_strategy_equity(
        eval_dates=result["eval_dates"],
        strat_equity=result["strat_equity"],
        bh_equity=result["bh_equity"],
        out_dir=OUT_DIR,
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

    pred_df["Abs_Error"] = np.abs(pred_df["Pred_Close_next_day"] - pred_df["True_Close_next_day"])
    pred_df["True_Direction"] = np.sign(pred_df["True_Close_next_day"] - pred_df["Today_Close"])
    pred_df["Pred_Direction"] = np.sign(pred_df["Pred_Close_next_day"] - pred_df["Today_Close"])
    pred_df.to_csv(os.path.join(OUT_DIR, "eval_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {os.path.basename(ckpt_path)}\n")
        f.write(f"CSV: {os.path.basename(csv_path)}\n")
        f.write(f"Date range used: [{start}, {end})\n")
        f.write(f"Target: {result['target_col']}\n")
        f.write(f"Train end date from checkpoint: {result['train_end_date'].date()}\n")
        f.write(f"Evaluation start date: {result['eval_dates'].min().date()}\n")
        f.write(f"Evaluation end date: {result['eval_dates'].max().date()}\n")
        f.write(f"Total windows: {result['n_total_windows']}\n")
        f.write(f"Evaluation windows: {result['n_eval']}\n\n")

        f.write("[Final Evaluation Metrics]\n")
        for k, v in metrics.items():
            if isinstance(v, (int, np.integer)):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")

    print(f"[OK] Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()