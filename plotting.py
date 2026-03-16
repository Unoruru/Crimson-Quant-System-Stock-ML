"""Visualization functions for forecasts, strategy, and training."""

import os

import matplotlib.pyplot as plt


def plot_forecasting_close(
    dates_train,
    train_close,
    dates_val,
    val_close,
    val_pred,
    dates_test,
    test_close,
    test_pred,
    metrics_text: str,
    out_dir: str = "my_fig",
) -> None:
    """Plot full and zoomed forecast charts."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, train_close, label="Train")
    plt.plot(dates_val, val_close, label="Validation")
    plt.plot(dates_val, val_pred, label="Val Predictions")
    plt.plot(dates_test, test_close, label="Test")
    plt.plot(dates_test, test_pred, label="Test Predictions")
    plt.title("Stock Prediction Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.legend()
    plt.figtext(0.01, 0.005, metrics_text, ha="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_full.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(dates_val, val_close, label="Validation")
    plt.plot(dates_val, val_pred, label="Val Predictions")
    plt.plot(dates_test, test_close, label="Test")
    plt.plot(dates_test, test_pred, label="Test Predictions")
    plt.title("Validation + Test Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.legend()
    plt.figtext(0.01, 0.005, metrics_text, ha="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_zoom.png"), dpi=200)
    plt.close()


def plot_strategy_equity(dates, strat_equity, bh_equity, out_dir: str = "my_fig") -> None:
    """Plot strategy equity curve vs buy-and-hold."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(dates, strat_equity, label="Strategy Equity")
    plt.plot(dates, bh_equity, label="Buy & Hold Equity")
    plt.title("Strategy vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_vs_buyhold.png"), dpi=200)
    plt.close()


def plot_losses(history: dict, out_dir: str = "my_fig") -> None:
    """Plot training and validation loss curves."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_losses.png"), dpi=200)
    plt.close()


def plot_forecast_eval(
    history_dates,
    history_close,
    eval_dates,
    eval_true,
    eval_pred,
    metrics_text: str,
    out_dir: str,
) -> None:
    """Plot evaluation forecast charts (full with history + zoom)."""
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
