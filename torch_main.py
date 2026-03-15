# torch_main.py
import os
import math
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class CFG:
    ticker: str = "AAPL"
    start: str = "2019-04-01"
    end: str = "2022-11-01"

    lookback: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    epochs: int = 300
    patience: int = 30
    weight_decay: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 999


cfg = CFG()
0.8

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(cfg.seed)


# -----------------------------
# 2) Feature engineering
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # basic returns
    df["ret"] = df["Close"].pct_change()
    df["logret"] = np.log(df["Close"] / (df["Close"].shift(1) + 1e-12))

    # OHLC-derived features
    df["hl_spread"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)
    df["oc_change"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-12)
    df["co_gap"] = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-12)
    df["volume_chg"] = df["Volume"].pct_change()

    # moving averages
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # momentum
    for w in [3, 5, 10]:
        df[f"mom_{w}"] = df["Close"] / (df["Close"].shift(w) + 1e-12) - 1.0

    # volatility
    df["vol_5"] = df["logret"].rolling(5).std()
    df["vol_10"] = df["logret"].rolling(10).std()

    # RSI(14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # placeholder sentiment
    df["sentiment"] = 0.0

    # target: next-day log return
    df["y_next_logret"] = np.log(df["Close"].shift(-1) / (df["Close"] + 1e-12))

    return df


def load_sentiment_daily(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Date", "sentiment"])

    s = pd.read_csv(path)
    if "Date" not in s.columns or "sentiment" not in s.columns:
        return pd.DataFrame(columns=["Date", "sentiment"])

    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    s = s.sort_values("Date").drop_duplicates("Date", keep="last")
    return s[["Date", "sentiment"]]


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df = add_indicators(df)

    # remove placeholder sentiment before merging real sentiment
    if "sentiment" in df.columns:
        df = df.drop(columns=["sentiment"])

    sent_path = os.path.join("data", "sentiment_daily.csv")
    sent = load_sentiment_daily(sent_path)

    df2 = df.reset_index()
    df2["Date"] = pd.to_datetime(df2["Date"]).dt.normalize()

    if sent is not None and len(sent) > 0 and "sentiment" in sent.columns:
        df2 = df2.merge(sent, on="Date", how="left")
        df2["sentiment"] = df2["sentiment"].ffill().fillna(0.0)
    else:
        df2["sentiment"] = 0.0

    df2 = df2.sort_values("Date")
    df = df2.set_index("Date")
    df = df.dropna().copy()
    return df


# -----------------------------
# 3) Dataset / windows / scaler
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_windows(df: pd.DataFrame, lookback: int, feature_cols: list, target_col: str):
    data = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    X_list, y_list, d_list = [], [], []

    for t in range(lookback - 1, len(df) - 1):
        X_list.append(data[t - lookback + 1: t + 1])
        y_list.append(target[t])
        d_list.append(df.index[t + 1])  # predicted date = next day

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    dates = pd.to_datetime(d_list)
    return X, y, dates


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X_3d: np.ndarray):
        flat = X_3d.reshape(-1, X_3d.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8

    def transform(self, X_3d: np.ndarray):
        return (X_3d - self.mean_) / self.std_

    def fit_transform(self, X_3d: np.ndarray):
        self.fit(X_3d)
        return self.transform(X_3d)


# -----------------------------
# 3.5) Checkpoint IO
# -----------------------------
def save_checkpoint(
    save_path: str,
    model: nn.Module,
    cfg: CFG,
    tag: str,
    feature_cols: list,
    x_scaler: StandardScaler,
    y_mean: float,
    y_std: float,
    model_kwargs: dict,
    target_col: str,
    history: dict = None,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    x_mean = x_scaler.mean_
    x_std = x_scaler.std_
    x_mean_list = x_mean.tolist() if hasattr(x_mean, "tolist") else list(x_mean)
    x_std_list = x_std.tolist() if hasattr(x_std, "tolist") else list(x_std)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "cfg": {
            "ticker": str(cfg.ticker),
            "start": str(cfg.start),
            "end": str(cfg.end),
            "lookback": int(cfg.lookback),
            "batch_size": int(cfg.batch_size),
        },
        "tag": str(tag),
        "feature_cols": [str(c) for c in feature_cols],
        "target_col": str(target_col),
        "x_scaler": {
            "mean": [float(v) for v in x_mean_list],
            "std": [float(v) for v in x_std_list],
        },
        "y_norm": {
            "mean": float(y_mean),
            "std": float(y_std),
        },
        "model_class": "CNNLSTMRegressor",
        "model_kwargs": {
            k: (
                int(v) if isinstance(v, (np.integer,))
                else float(v) if isinstance(v, (np.floating,))
                else v
            )
            for k, v in model_kwargs.items()
        },
        "history": history,
    }

    torch.save(ckpt, save_path)
    print(f"[OK] Saved checkpoint -> {save_path}")


def load_checkpoint(load_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        ckpt = torch.load(load_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(load_path, map_location=device)

    if ckpt.get("model_class") != "CNNLSTMRegressor":
        raise ValueError(f"Unsupported model_class: {ckpt.get('model_class')}")

    model = CNNLSTMRegressor(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_scaler = StandardScaler()
    x_scaler.mean_ = np.array(ckpt["x_scaler"]["mean"], dtype=np.float32)
    x_scaler.std_ = np.array(ckpt["x_scaler"]["std"], dtype=np.float32)

    meta = {
        "cfg": ckpt["cfg"],
        "tag": ckpt.get("tag"),
        "feature_cols": ckpt["feature_cols"],
        "lookback": ckpt["cfg"]["lookback"],
        "y_mean": ckpt["y_norm"]["mean"],
        "y_std": ckpt["y_norm"]["std"],
        "target_col": ckpt.get("target_col", "y_next_logret"),
    }
    return model, x_scaler, meta


# -----------------------------
# 4) Model
# -----------------------------
class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        cnn_channels: int = 64,
        kernel: int = 5,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        pad = kernel // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, T, F] -> [B, F, T]
        h = self.cnn(x)         # [B, C, T]
        h = h.transpose(1, 2)   # [B, T, C]
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        y = self.head(last).squeeze(-1)
        return y


# -----------------------------
# 5) Metrics / prediction utils
# -----------------------------
def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    dd = equity_curve / (running_max + 1e-12) - 1.0
    return float(dd.min())


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    if len(daily_returns) == 0:
        return float("nan")
    std = daily_returns.std()
    if std < 1e-12:
        return float("nan")
    return float((daily_returns.mean() / std) * np.sqrt(252.0))


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

    # 新增：模型信号分布
    pred_up_ratio = float(np.mean(pred_up) * 100.0)
    pred_down_ratio = float(np.mean(pred_down) * 100.0)

    return {
        "DirAcc_%": dir_acc,
        "UpPrecision_%": up_precision,
        "DownPrecision_%": down_precision,
        "PredUpRatio_%": pred_up_ratio,
        "PredDownRatio_%": pred_down_ratio,
    }


def compute_trading_metrics(today_close: np.ndarray, true_close: np.ndarray, pred_close: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Quantile-based long-only strategy:
    - only go long on the strongest predicted upside days
    - otherwise stay flat

    这比固定阈值更稳，因为不同实验下 pred_ret 尺度会变化。
    """
    true_ret = (true_close / (today_close + 1e-12)) - 1.0
    pred_ret = (pred_close / (today_close + 1e-12)) - 1.0

    # ===== 调试分布：建议保留，方便看模型输出尺度 =====
    print("\n[DEBUG pred_ret distribution]")
    print(f"min   = {pred_ret.min():.6f}")
    print(f"max   = {pred_ret.max():.6f}")
    print(f"mean  = {pred_ret.mean():.6f}")
    print(f"q50   = {np.quantile(pred_ret, 0.50):.6f}")
    print(f"q75   = {np.quantile(pred_ret, 0.75):.6f}")
    print(f"q90   = {np.quantile(pred_ret, 0.90):.6f}")
    print(f"q95   = {np.quantile(pred_ret, 0.95):.6f}")

    # ===== 最终推荐：分位数阈值 =====
    quantile_level = 0.70   # 可先试 0.70 / 0.80 / 0.90
    threshold = np.quantile(pred_ret, quantile_level)

    # 只在预测最强的那部分日子做多，否则空仓
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
        "Threshold_Quantile_%": quantile_level * 100.0,
    }
    return metrics, strat_equity, bh_equity


@torch.no_grad()
def predict_all_logret(model, loader, device, y_mean, y_std):
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


def logret_to_next_close(today_close: np.ndarray, pred_logret: np.ndarray) -> np.ndarray:
    return today_close * np.exp(pred_logret)


# -----------------------------
# 6) Plotting
# -----------------------------
def plot_forecasting_close(
    dates_train, train_close,
    dates_val, val_close, val_pred,
    dates_test, test_close, test_pred,
    metrics_text,
    out_dir="my_fig",
):
    os.makedirs(out_dir, exist_ok=True)

    # Full plot: Train + Val + ValPred + Test + TestPred
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

    # Zoom plot: Val + Test + Predictions
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


def plot_strategy_equity(dates, strat_equity, bh_equity, out_dir="my_fig"):
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


def plot_losses(history, out_dir="my_fig"):
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


# -----------------------------
# 7) Train
# -----------------------------
def train(model, train_loader, val_loader, cfg: CFG):
    device = cfg.device
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    history = {"train_loss": [], "val_loss": []}

    @torch.no_grad()
    def eval_loss(model_, loader_):
        model_.eval()
        tot = 0.0
        for X, y in loader_:
            X = X.to(device)
            y = y.to(device)
            pred = model_(X)
            loss = loss_fn(pred, y)
            tot += loss.item() * X.size(0)
        return tot / len(loader_.dataset)

    best_val = float("inf")
    bad = 0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += loss.item() * X.size(0)

        train_loss = tot / len(train_loader.dataset)
        val_loss = eval_loss(model, val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = opt.param_groups[0]["lr"]
        if epoch%5 == 0:
            print(f"Epoch {epoch:03d} | loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Epoch {epoch:03d} | loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}")
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# -----------------------------
# 8) Main experiment
# -----------------------------
def run_experiment(use_sentiment: bool, tag: str):
    print(f"\n========== Running: {tag} ==========")

    df = load_data(cfg.ticker, cfg.start, cfg.end)

    if use_sentiment:
        # 只保留真实 sentiment 可用的时间段
        sent_path = os.path.join("data", "sentiment_daily.csv")
        sent = load_sentiment_daily(sent_path)
        if len(sent) > 0:
            sent_max_date = sent["Date"].max()
            df = df[df.index <= sent_max_date].copy()
            print(f"[INFO] Sentiment coverage end date: {sent_max_date.date()}")

    base_features = [
        "Open", "High", "Low", "Close", "Volume",
        "ret", "logret",
        "hl_spread", "oc_change", "co_gap", "volume_chg",
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26",
        "mom_3", "mom_5", "mom_10",
        "vol_5", "vol_10",
        "rsi_14", "macd", "macd_signal", "macd_hist",
    ]

    feature_cols = base_features + (["sentiment"] if use_sentiment else [])
    target_col = "y_next_logret"

    X, y, dates = make_windows(df, cfg.lookback, feature_cols, target_col)

    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

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
    model, history = train(model, train_loader, val_loader, cfg)

    ckpt_path = os.path.join("checkpoints", f"{cfg.ticker}_{tag}_best.pt")
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
        history=history,
    )

    val_true_logret, val_pred_logret = predict_all_logret(model, val_loader, cfg.device, y_mean, y_std)
    test_true_logret, test_pred_logret = predict_all_logret(model, test_loader, cfg.device, y_mean, y_std)

    # recover close prices for evaluation
    today_close_val = df["Close"].shift(1).reindex(pd.to_datetime(dates_val)).values.astype(np.float64)
    true_close_val = logret_to_next_close(today_close_val, val_true_logret)
    pred_close_val = logret_to_next_close(today_close_val, val_pred_logret)

    today_close_test = df["Close"].shift(1).reindex(pd.to_datetime(dates_test)).values.astype(np.float64)
    true_close_test = logret_to_next_close(today_close_test, test_true_logret)
    pred_close_test = logret_to_next_close(today_close_test, test_pred_logret)

    price_metrics_val = compute_price_metrics(true_close_val, pred_close_val)
    direction_metrics_val = compute_direction_metrics(today_close_val, true_close_val, pred_close_val)
    trading_metrics_val, strat_equity, bh_equity = compute_trading_metrics(today_close_val, true_close_val, pred_close_val)

    price_metrics_test = compute_price_metrics(true_close_test, pred_close_test)

    metrics = {}
    metrics.update(price_metrics_val)
    metrics.update(direction_metrics_val)
    metrics.update(trading_metrics_val)

    print(
    f"[{tag}] "
    f"Val MAE={price_metrics_val['MAE']:.4f} | "
    f"Val RMSE={price_metrics_val['RMSE']:.4f} | "
    f"Val MAPE={price_metrics_val['MAPE_%']:.2f}% | "
    f"Val DirAcc={direction_metrics_val['DirAcc_%']:.2f}% | "
    f"PredUp={direction_metrics_val['PredUpRatio_%']:.2f}% | "
    f"PredDown={direction_metrics_val['PredDownRatio_%']:.2f}% | "
    f"Val StratRet={trading_metrics_val['StrategyReturn_%']:.2f}% | "
    f"Val Excess={trading_metrics_val['ExcessReturn_%']:.2f}% | "
    f"Sharpe={trading_metrics_val['Sharpe']:.4f} | "
    f"Trades={trading_metrics_val['TradeCount']} | "
    f"Exposure={trading_metrics_val['Exposure_%']:.2f}% | "
    f"Q={trading_metrics_val['Threshold_Quantile_%']:.0f}% | "
    f"Thr={trading_metrics_val['Threshold_Value']:.5f} | "
    f"Test MAE={price_metrics_test['MAE']:.4f} | "
    f"Test RMSE={price_metrics_test['RMSE']:.4f}"
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

    # save metrics
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {tag}\n")
        f.write(f"Ticker: {cfg.ticker}\n")
        f.write(f"Date range: {cfg.start} -> {cfg.end}\n")
        f.write(f"Lookback: {cfg.lookback}\n")
        f.write(f"Target: {target_col}\n")
        f.write("\n[Validation Price Metrics]\n")
        for k, v in price_metrics_val.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\n[Validation Direction Metrics]\n")
        for k, v in direction_metrics_val.items():
            if isinstance(v, (int, np.integer)):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")

        f.write("\n[Validation Trading Metrics]\n")
        for k, v in trading_metrics_val.items():
            if isinstance(v, (int, np.integer)):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")
        f.write("\n[Test Price Metrics]\n")
        for k, v in price_metrics_test.items():
            f.write(f"{k}: {v:.6f}\n")

    # save validation predictions
    pred_df = pd.DataFrame({
        "Date": pd.to_datetime(dates_val),
        "Today_Close": today_close_val,
        "True_Close_next_day": true_close_val,
        "Pred_Close_next_day": pred_close_val,
        "True_LogRet": val_true_logret,
        "Pred_LogRet": val_pred_logret,
    })
    pred_df["Abs_Error"] = np.abs(pred_df["Pred_Close_next_day"] - pred_df["True_Close_next_day"])
    pred_df["True_Direction"] = np.sign(pred_df["True_Close_next_day"] - pred_df["Today_Close"])
    pred_df["Pred_Direction"] = np.sign(pred_df["Pred_Close_next_day"] - pred_df["Today_Close"])
    pred_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)


def main():
    run_experiment(use_sentiment=False, tag="no_sentiment")
    run_experiment(use_sentiment=True, tag="with_sentiment")


if __name__ == "__main__":
    main()