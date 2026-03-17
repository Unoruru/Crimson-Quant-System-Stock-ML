"""Training entry point for Crimson Quant System.

Merges the training loop (train_model, predict_all_logret) with the
experiment runner (run_experiment) and CLI (main).
"""

import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from config import (
    Config, BASE_FEATURES, SENTIMENT_FEATURES, TRAIN_RATIO, VAL_RATIO,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, GRAD_CLIP_NORM, HUBER_DELTA, QUANTILE_LEVEL,
)
from data_loader import load_data, make_windows, WindowDataset, StandardScaler
from model import CNNLSTMRegressor, save_checkpoint
from metrics import (
    compute_price_metrics,
    compute_direction_metrics,
    compute_trading_metrics,
    compute_logret_metrics,
    fit_affine_calibration,
    apply_affine_calibration,
    logret_to_next_close,
    write_metrics_report,
)
from plotting import plot_forecasting_close, plot_strategy_equity, plot_losses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Seed
# ------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def _use_amp(device: str) -> bool:
    """Check if automatic mixed precision is available."""
    return device == "cuda" and torch.cuda.is_available()


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> float:
    """Compute average loss on a data loader."""
    model.eval()
    total = 0.0
    use_amp = _use_amp(device)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(X)
            loss = loss_fn(pred, y)
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> tuple[nn.Module, dict]:
    """Train model with early stopping, AMP, and return (model, history)."""
    device = cfg.device
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
    )
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    use_amp = _use_amp(device)
    scaler = GradScaler(enabled=use_amp)

    if use_amp:
        logger.info("Using automatic mixed precision (AMP) training.")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    bad_epochs = 0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(X)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            total += loss.item() * X.size(0)

        train_loss = total / len(train_loader.dataset)
        val_loss = _eval_loss(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = opt.param_groups[0]["lr"]
        if epoch % 5 == 0:
            logger.info(
                "Epoch %03d | loss=%.6f | val_loss=%.6f | lr=%.2e",
                epoch, train_loss, val_loss, current_lr,
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info(
                    "Epoch %03d | loss=%.6f | val_loss=%.6f | lr=%.2e",
                    epoch, train_loss, val_loss, current_lr,
                )
                logger.info("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def predict_all_logret(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and denormalize predictions."""
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


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------

def _find_existing_news_csv(ticker: str, data_dir: str = "data") -> str | None:
    """Look for existing local news CSVs for a ticker."""
    import glob as _glob
    files = _glob.glob(os.path.join(data_dir, f"{ticker}*News_raw*.csv"))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def run_experiment(cfg: Config, use_sentiment: bool, tag: str):
    logger.info("========== Running: %s ==========", tag)

    sentiment_csv_path = None
    if use_sentiment:
        from fetch_news import fetch_news_for_period
        from sentiment_evaluation import evaluate_and_save_sentiment

        news_csv = None
        try:
            news_csv = fetch_news_for_period(cfg.ticker, cfg.start, cfg.end)
        except (ValueError, RuntimeError) as exc:
            logger.warning("News fetch failed (%s) — falling back to local cache", exc)
            news_csv = _find_existing_news_csv(cfg.ticker)
            if news_csv is None:
                logger.warning("No local news data found — sentiment will be zeros")

        if news_csv:
            try:
                sentiment_csv_path = evaluate_and_save_sentiment(
                    news_csv, cfg.ticker, cfg.start, cfg.end
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning("Sentiment evaluation failed: %s — sentiment will be zeros", exc)

    df = load_data(cfg.ticker, cfg.start, cfg.end, sentiment_csv_path=sentiment_csv_path)

    feature_cols = BASE_FEATURES + (SENTIMENT_FEATURES if use_sentiment else [])
    target_col = "y_next_logret"

    X, y, dates = make_windows(df, cfg.lookback, feature_cols, target_col)

    n = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    X_train, y_train, dates_train = X[:n_train], y[:n_train], dates[:n_train]
    X_val, y_val, dates_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val], dates[n_train:n_train + n_val]
    X_test, y_test, dates_test = X[n_train + n_val:], y[n_train + n_val:], dates[n_train + n_val:]

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    train_loader = DataLoader(WindowDataset(X_train, y_train_s), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val_s), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(X_test, y_test_s), batch_size=cfg.batch_size, shuffle=False)

    n_features = X_train.shape[-1]

    model_kwargs = dict(
        n_features=n_features,
        cnn_channels=64,
        kernel=5,
        lstm_hidden=96,
        lstm_layers=2,
        dropout=0.2,
    )
    model = CNNLSTMRegressor(**model_kwargs)
    model, history = train_model(model, train_loader, val_loader, cfg)

    ckpt_path = os.path.join("checkpoints", f"{cfg.ticker}_{tag}_best.pt")
    val_true_logret, val_pred_logret = predict_all_logret(model, val_loader, cfg.device, y_mean, y_std)
    test_true_logret, test_pred_logret = predict_all_logret(model, test_loader, cfg.device, y_mean, y_std)
    calibration = fit_affine_calibration(val_true_logret, val_pred_logret)
    val_pred_logret_raw = val_pred_logret.copy()
    test_pred_logret_raw = test_pred_logret.copy()
    val_pred_logret = apply_affine_calibration(val_pred_logret, calibration)
    test_pred_logret = apply_affine_calibration(test_pred_logret, calibration)

    save_checkpoint(
        save_path=ckpt_path,
        model=model,
        cfg=cfg,
        tag=tag,
        feature_cols=feature_cols,
        x_scaler=x_scaler,
        y_mean=y_mean,
        y_std=y_std,
        model_kwargs=model_kwargs,
        target_col=target_col,
        calibration=calibration,
        history=history,
    )

    today_close_val = df["Close"].shift(1).reindex(pd.to_datetime(dates_val)).values.astype(np.float64)
    true_close_val = logret_to_next_close(today_close_val, val_true_logret)
    pred_close_val = logret_to_next_close(today_close_val, val_pred_logret)

    today_close_test = df["Close"].shift(1).reindex(pd.to_datetime(dates_test)).values.astype(np.float64)
    true_close_test = logret_to_next_close(today_close_test, test_true_logret)
    pred_close_test = logret_to_next_close(today_close_test, test_pred_logret)

    price_metrics_val = compute_price_metrics(true_close_val, pred_close_val)
    logret_metrics_val = compute_logret_metrics(val_true_logret, val_pred_logret)
    direction_metrics_val = compute_direction_metrics(today_close_val, true_close_val, pred_close_val)
    trading_metrics_val, strat_equity, bh_equity = compute_trading_metrics(today_close_val, true_close_val, pred_close_val, quantile_level=QUANTILE_LEVEL)

    price_metrics_test = compute_price_metrics(true_close_test, pred_close_test)
    logret_metrics_test = compute_logret_metrics(test_true_logret, test_pred_logret)

    metrics = {}
    metrics.update(price_metrics_val)
    metrics.update(logret_metrics_val)
    metrics.update(direction_metrics_val)
    metrics.update(trading_metrics_val)

    logger.info(
        "[%s] Val MAE=%.4f | Val RMSE=%.4f | Val LogRetMAE=%.5f | "
        "Val Bias=%.5f | Cal(slope=%.3f,int=%.5f) | Val MAPE=%.2f%% | "
        "Val DirAcc=%.2f%% | PredUp=%.2f%% | PredDown=%.2f%% | "
        "Val StratRet=%.2f%% | Val Excess=%.2f%% | Sharpe=%.4f | "
        "Trades=%d | Exposure=%.2f%% | Q=%.0f%% | Thr=%.5f | "
        "Test MAE=%.4f | Test RMSE=%.4f",
        tag,
        price_metrics_val["MAE"], price_metrics_val["RMSE"],
        logret_metrics_val["LogRet_MAE"], logret_metrics_val["LogRet_Bias"],
        calibration["slope"], calibration["intercept"],
        price_metrics_val["MAPE_%"],
        direction_metrics_val["DirAcc_%"],
        direction_metrics_val["PredUpRatio_%"], direction_metrics_val["PredDownRatio_%"],
        trading_metrics_val["StrategyReturn_%"], trading_metrics_val["ExcessReturn_%"],
        trading_metrics_val["Sharpe"],
        trading_metrics_val["TradeCount"], trading_metrics_val["Exposure_%"],
        trading_metrics_val["Threshold_Quantile_%"], trading_metrics_val["Threshold_Value"],
        price_metrics_test["MAE"], price_metrics_test["RMSE"],
    )

    train_close = df["Close"].loc[pd.to_datetime(dates_train)].values
    val_close = true_close_val

    metrics_text = (
        f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
        f"MAPE={metrics['MAPE_%']:.2f}% | DirAcc={metrics['DirAcc_%']:.2f}% | "
        f"UpPrec={metrics['UpPrecision_%']:.2f}% | DownPrec={metrics['DownPrecision_%']:.2f}% | "
        f"PredUp={metrics['PredUpRatio_%']:.2f}% | PredDown={metrics['PredDownRatio_%']:.2f}% | "
        f"StratRet={metrics['StrategyReturn_%']:.2f}% | BuyHold={metrics['BuyHoldReturn_%']:.2f}% | "
        f"Excess={metrics['ExcessReturn_%']:.2f}% | Sharpe={metrics['Sharpe']:.4f} | "
        f"MDD={metrics['MaxDrawdown_%']:.2f}% | WinRate={metrics['WinRate_%']:.2f}% | "
        f"Trades={metrics['TradeCount']} | Exposure={metrics['Exposure_%']:.2f}% | "
        f"Q={metrics['Threshold_Quantile_%']:.0f}% | Thr={metrics['Threshold_Value']:.5f}"
    )

    out_dir = f"my_fig_{tag}"
    plot_forecasting_close(
        dates_train=dates_train,
        train_close=train_close,
        dates_val=dates_val,
        val_close=val_close,
        val_pred=pred_close_val,
        dates_test=dates_test,
        test_close=true_close_test,
        test_pred=pred_close_test,
        metrics_text=metrics_text,
        out_dir=out_dir,
    )
    plot_strategy_equity(
        dates=dates_val,
        strat_equity=strat_equity,
        bh_equity=bh_equity,
        out_dir=out_dir,
    )
    plot_losses(history, out_dir=out_dir)

    os.makedirs(out_dir, exist_ok=True)

    write_metrics_report(
        metrics_txt_path=os.path.join(out_dir, "metrics.txt"),
        header_lines=[
            f"Experiment: {tag}",
            f"Ticker: {cfg.ticker}",
            f"Date range: [{cfg.start}, {cfg.end}]",
            f"Lookback: {cfg.lookback}",
            f"Target: {target_col}",
            f"Calibration_Slope: {calibration['slope']:.6f}",
            f"Calibration_Intercept: {calibration['intercept']:.6f}",
        ],
        sections=[
            ("Validation Price Metrics", price_metrics_val),
            ("Validation LogRet Metrics", logret_metrics_val),
            ("Validation Direction Metrics", direction_metrics_val),
            ("Validation Trading Metrics", trading_metrics_val),
            ("Test LogRet Metrics", logret_metrics_test),
            ("Test Price Metrics", price_metrics_test),
        ],
    )

    pred_df = pd.DataFrame({
        "Date": pd.to_datetime(dates_val),
        "Today_Close": today_close_val,
        "True_Close_next_day": true_close_val,
        "Pred_Close_next_day": pred_close_val,
        "True_LogRet": val_true_logret,
        "Pred_LogRet": val_pred_logret,
        "Pred_LogRet_Raw": val_pred_logret_raw,
    })
    pred_df["Abs_Error"] = np.abs(pred_df["Pred_Close_next_day"] - pred_df["True_Close_next_day"])
    pred_df["True_Direction"] = np.sign(pred_df["True_Close_next_day"] - pred_df["Today_Close"])
    pred_df["Pred_Direction"] = np.sign(pred_df["Pred_Close_next_day"] - pred_df["Today_Close"])
    pred_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    cfg = Config.load()
    set_seed(cfg.seed)

    run_experiment(cfg, use_sentiment=False, tag="no_sentiment")
    run_experiment(cfg, use_sentiment=True, tag="with_sentiment")


if __name__ == "__main__":
    main()
