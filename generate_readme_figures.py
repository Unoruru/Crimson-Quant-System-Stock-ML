"""
generate_readme_figures.py
==========================
Generates docs/figures/performance_overview.png — a 2x2 composite figure
for the README Results section.

  Panel [0,0]: NoS  — True vs Predicted Close Price (eval period)
  Panel [0,1]: WithS — True vs Predicted Close Price (eval period)
  Panel [1,0]: Strategy equity curves (NoS, WithS, Buy-and-Hold)
  Panel [1,1]: Training loss curves (NoS & WithS train/val)

Run from the project root:
    python generate_readme_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT       = os.path.dirname(os.path.abspath(__file__))
NOS_CSV    = os.path.join(ROOT, "eval_outputs", "no_sentiment",   "eval_predictions.csv")
WITHS_CSV  = os.path.join(ROOT, "eval_outputs", "with_sentiment", "eval_predictions.csv")
NOS_LOSS   = os.path.join(ROOT, "training_outputs", "no_sentiment",   "training_history.csv")
WITHS_LOSS = os.path.join(ROOT, "training_outputs", "with_sentiment", "training_history.csv")
OUT_DIR    = os.path.join(ROOT, "docs", "figures")
OUT_PATH   = os.path.join(OUT_DIR, "performance_overview.png")

C_NOS   = "#2166ac"
C_WITHS = "#d6604d"
C_BH    = "#333333"
C_TRUE  = "#1f77b4"

plt.rcParams.update({
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "grid.linestyle":  "--",
})

QUANTILE_LEVEL = 0.70
EPSILON = 1e-12


def load_eval(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_equity_curves(df: pd.DataFrame) -> tuple:
    today  = df["Today_Close"].values.astype(float)
    true_c = df["True_Close_next_day"].values.astype(float)
    pred_c = df["Pred_Close_next_day"].values.astype(float)

    true_ret = (true_c / (today + EPSILON)) - 1.0
    pred_ret = (pred_c / (today + EPSILON)) - 1.0

    threshold = max(float(np.quantile(pred_ret, QUANTILE_LEVEL)), 0.0)
    signal    = (pred_ret > threshold).astype(float)

    strat_equity = np.cumprod(1.0 + signal * true_ret)
    bh_equity    = np.cumprod(1.0 + true_ret)
    return strat_equity, bh_equity


def load_loss(path: str):
    if not os.path.exists(path):
        return None, None
    h = pd.read_csv(path)
    return h["train_loss"].values, h["val_loss"].values


def main() -> None:
    for p in (NOS_CSV, WITHS_CSV):
        if not os.path.exists(p):
            sys.exit(
                f"ERROR: required file not found: {p}\n"
                "Run `python train.py` then `python prediction_validation.py` first."
            )

    nos   = load_eval(NOS_CSV)
    withs = load_eval(WITHS_CSV)

    strat_nos,   bh_nos   = compute_equity_curves(nos)
    strat_withs, _        = compute_equity_curves(withs)

    train_nos,   val_nos   = load_loss(NOS_LOSS)
    train_withs, val_withs = load_loss(WITHS_LOSS)

    dates = nos["Date"].values

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150, constrained_layout=True)
    fig.suptitle(
        "AMZN CNN-LSTM Performance Overview  |  Eval: Jan\u2013Mar 2026  |  Train: 2023\u20132025",
        fontsize=13, fontweight="bold",
    )

    # [0,0] NoS price forecast
    ax = axes[0, 0]
    ax.plot(dates, nos["True_Close_next_day"].values,
            color=C_TRUE, lw=1.8, label="True Close")
    ax.plot(dates, nos["Pred_Close_next_day"].values,
            color=C_NOS, lw=1.5, linestyle="--", label="Predicted (NoS)")
    ax.set_title("No-Sentiment Model \u2014 Price Forecast")
    ax.set_ylabel("AMZN Close Price (USD)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # [0,1] WithS price forecast
    ax = axes[0, 1]
    ax.plot(dates, withs["True_Close_next_day"].values,
            color=C_TRUE, lw=1.8, label="True Close")
    ax.plot(dates, withs["Pred_Close_next_day"].values,
            color=C_WITHS, lw=1.5, linestyle="--", label="Predicted (WithS)")
    ax.set_title("With-Sentiment Model \u2014 Price Forecast")
    ax.set_ylabel("AMZN Close Price (USD)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # [1,0] Strategy equity curves
    ax = axes[1, 0]
    ax.plot(dates, strat_nos,   color=C_NOS,   lw=1.8, label="Strategy (NoS)")
    ax.plot(dates, strat_withs, color=C_WITHS, lw=1.8, label="Strategy (WithS)")
    ax.plot(dates, bh_nos,      color=C_BH,    lw=1.4, linestyle=":", label="Buy-and-Hold")
    ax.axhline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Strategy Equity Curves (70th-pct threshold, long-only)")
    ax.set_ylabel("Portfolio Value (start = 1.0)")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # [1,1] Training loss curves
    ax = axes[1, 1]
    if train_nos is None and train_withs is None:
        ax.text(
            0.5, 0.5,
            "Training history not available.\nRun `python train.py` to generate.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#555",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc"),
        )
        ax.set_axis_off()
    else:
        if train_nos is not None:
            e = np.arange(1, len(train_nos) + 1)
            ax.plot(e, train_nos, color=C_NOS,  lw=1.6, label="NoS train")
            ax.plot(e, val_nos,   color=C_NOS,  lw=1.6, linestyle="--", alpha=0.75, label="NoS val")
        if train_withs is not None:
            e = np.arange(1, len(train_withs) + 1)
            ax.plot(e, train_withs, color=C_WITHS, lw=1.6, label="WithS train")
            ax.plot(e, val_withs,   color=C_WITHS, lw=1.6, linestyle="--", alpha=0.75, label="WithS val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("HuberLoss")
        ax.legend(loc="upper right")
    ax.set_title("Training & Validation Loss (HuberLoss)")

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
