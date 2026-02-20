# torch_main.py (LSTM-only, predict next-day CLOSE, report-style plots)
import os
import math
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
    start: str = "2019-04-01"      # 对齐你 report 的时间段更直观
    end: str = "2023-05-01"

    lookback: int =  30           # report 里常用 window=20
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 200             # report 风格：训练久一点并画 loss
    patience: int = 20            # 早停更宽松，避免太早停导致曲线太短

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 999

cfg = CFG()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# -----------------------------
# 2) Feature engineering
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 你报告里核心用 Close + Volume + Sentiment(可选)
    # 这里保留一部分常用指标，后续你要对齐 report 可再精简
    df["ret"] = df["Close"].pct_change()
    df["logret"] = np.log(df["Close"] / df["Close"].shift(1))

    # SMA/EMA
    for w in [5, 10, 20]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

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

    # Placeholder sentiment（下一步再接你真实情绪）
    df["sentiment"] = 0.0

    # ✅ 目标：预测 next-day Close（对齐你 report 的预测曲线）
    df["y_next_close"] = df["Close"].shift(-1)

    return df

def load_sentiment_daily(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Date", "sentiment"])
    s = pd.read_csv(path)
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    s = s.sort_values("Date").drop_duplicates("Date", keep="last")
    return s[["Date", "sentiment"]]


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV from yfinance, add indicators, then merge daily sentiment (if available)
    from data/sentiment_daily.csv.

    Fixes:
    1) yfinance MultiIndex columns -> flattened to 1 level
    2) avoid sentiment column collision (placeholder vs merged sentiment)
    """
    # --- 1) download price data ---
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    # ✅ FIX 1: flatten MultiIndex columns (e.g., ('Close','AAPL') -> 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # --- 2) add indicators / target ---
    df = add_indicators(df)

    # ✅ FIX 2: drop placeholder sentiment to avoid merge collision
    # add_indicators() creates df["sentiment"]=0.0; remove it before merging real sentiment
    if "sentiment" in df.columns:
        df = df.drop(columns=["sentiment"])

    # --- 3) load daily sentiment (optional) ---
    sent_path = os.path.join("data", "sentiment_daily.csv")
    sent = load_sentiment_daily(sent_path)

    # ensure Date column exists for merge
    df2 = df.reset_index()  # yfinance index is DatetimeIndex named "Date"
    df2["Date"] = pd.to_datetime(df2["Date"]).dt.normalize()

    # --- 4) merge sentiment ---
    if sent is not None and len(sent) > 0 and "sentiment" in sent.columns:
        df2 = df2.merge(sent, on="Date", how="left")

        # fill missing sentiment for trading days without news
        df2["sentiment"] = df2["sentiment"].ffill().fillna(0.0)
    else:
        # no sentiment file -> use 0
        df2["sentiment"] = 0.0

    # set back to datetime index
    df2 = df2.sort_values("Date")
    df = df2.set_index("Date")

    # final cleanup (drop rows with any NaNs from indicators/target shifting)
    df = df.dropna().copy()
    return df


# -----------------------------
# 3) Dataset (windowed) + dates
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

    # t 对应输入窗口最后一天索引；目标是 t+1 的 close（已 shift(-1)）
    for t in range(lookback - 1, len(df) - 1):
        X_list.append(data[t - lookback + 1: t + 1])
        y_list.append(target[t])
        d_list.append(df.index[t + 1])  # 预测的是 next-day close，所以日期用 t+1

    X = np.stack(X_list, axis=0)     # [N, lookback, features]
    y = np.array(y_list)             # [N]
    dates = pd.to_datetime(d_list)   # [N]
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
# 4) Model (LSTM-only)
# -----------------------------

# class LSTMRegressor(nn.Module):
#     def __init__(self, n_features: int, hidden: int = 64, layers: int = 3, dropout: float = 0.2):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=n_features,
#             hidden_size=hidden,
#             num_layers=layers,
#             batch_first=True,
#             dropout=dropout if layers > 1 else 0.0
#         )
#         self.fc = nn.Linear(hidden, 1)

#     def forward(self, x):
#         out, _ = self.lstm(x)        # [B, T, H]
#         last = out[:, -1, :]         # [B, H]
#         y = self.fc(last).squeeze(-1)
#         return y
    
class CNNLSTMRegressor(nn.Module):
    """
    Input:  x [B, T, F]
    CNN:    conv over time (T), channels=F
    LSTM:   models longer dependencies on CNN features
    Output: y [B]
    """
    def __init__(self, n_features: int, cnn_channels: int = 64, kernel: int = 5,
                 lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.0):
        super().__init__()

        pad = kernel // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=cnn_channels, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=kernel, padding=pad),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        h = self.cnn(x)             # [B, C, T]
        h = h.transpose(1, 2)       # [B, T, C]
        out, _ = self.lstm(h)       # [B, T, H]
        last = out[:, -1, :]        # [B, H]
        y = self.fc(last).squeeze(-1)
        return y

# -----------------------------
# 5) Metrics / Pred
# -----------------------------
def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


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

    # 反标准化回 Close 价格尺度
    y_true = y_true_s * y_std + y_mean
    y_pred = y_pred_s * y_std + y_mean
    return y_true, y_pred


# -----------------------------
# 6) Plotting (report-style)
# -----------------------------
def plot_forecasting_close(
    dates_train, train_close,
    dates_val, val_close,
    val_pred,
    out_dir="my_fig"
):
    os.makedirs(out_dir, exist_ok=True)

    # Full plot: Train + Val + Pred
    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, train_close, label="Train")
    plt.plot(dates_val,   val_close,   label="Validation")
    plt.plot(dates_val,   val_pred,    label="Predictions")
    plt.title("Stock Prediction Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_full.png"), dpi=200)
    plt.close()

    # Zoom plot: Val + Pred only
    plt.figure(figsize=(14, 6))
    plt.plot(dates_val, val_close, label="Validation")
    plt.plot(dates_val, val_pred,  label="Predictions")
    plt.title("Stock Prediction Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_zoom.png"), dpi=200)
    plt.close()


def plot_losses(history, out_dir="my_fig"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(history["train_loss"], label="loss")
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

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()  # report 里常见 MSE loss 曲线

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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# -----------------------------
# 8) Main
# -----------------------------
def run_experiment(use_sentiment: bool, tag: str):
    print(f"\n========== Running: {tag} ==========")

    df = load_data(cfg.ticker, cfg.start, cfg.end)

    # ---------- feature selection ----------
    base_features = [
        "Close", "Volume",
        "ret", "logret",
        "sma_5", "sma_10", "sma_20",
        "ema_12", "ema_26",
        "rsi_14", "macd", "macd_signal",
    ]

    if use_sentiment:
        feature_cols = base_features + ["sentiment"]
    else:
        feature_cols = base_features

    target_col = "y_next_close"

    X, y, dates = make_windows(df, cfg.lookback, feature_cols, target_col)

    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    X_train, y_train, dates_train = X[:n_train], y[:n_train], dates[:n_train]
    X_val,   y_val,   dates_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val], dates[n_train:n_train + n_val]
    X_test,  y_test,  dates_test  = X[n_train + n_val:], y[n_train + n_val:], dates[n_train + n_val:]

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    y_train_s = (y_train - y_mean) / y_std
    y_val_s   = (y_val   - y_mean) / y_std
    y_test_s  = (y_test  - y_mean) / y_std

    train_loader = DataLoader(WindowDataset(X_train, y_train_s), batch_size=cfg.batch_size, shuffle=False)
    val_loader   = DataLoader(WindowDataset(X_val,   y_val_s),   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(WindowDataset(X_test,  y_test_s),  batch_size=cfg.batch_size, shuffle=False)

    n_features = X_train.shape[-1]

    model = CNNLSTMRegressor(
        n_features=n_features,
        cnn_channels=64,
        kernel=5,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.0
    )
    model, history = train(model, train_loader, val_loader, cfg)

    val_true, val_pred = predict_all(model, val_loader, cfg.device, y_mean, y_std)
    test_true, test_pred = predict_all(model, test_loader, cfg.device, y_mean, y_std)

    test_mae = np.mean(np.abs(test_true - test_pred))
    test_rmse = rmse(test_true, test_pred)

    print(f"[{tag}] MAE={test_mae:.4f} | RMSE={test_rmse:.4f}")

    close_series = df["Close"]
    train_close = close_series.loc[pd.to_datetime(dates_train)].values
    val_close   = close_series.loc[pd.to_datetime(dates_val)].values

    out_dir = f"my_fig_{tag}"

    plot_forecasting_close(
        dates_train=dates_train, train_close=train_close,
        dates_val=dates_val,     val_close=val_close,
        val_pred=val_pred,
        out_dir=out_dir
    )

    plot_losses(history, out_dir=out_dir)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {tag}\n")
        f.write(f"MAE (Close): {test_mae:.4f}\n")
        f.write(f"RMSE (Close): {test_rmse:.4f}\n")


def main():
    run_experiment(use_sentiment=False, tag="no_sentiment")
    run_experiment(use_sentiment=True,  tag="with_sentiment")


if __name__ == "__main__":
    main()