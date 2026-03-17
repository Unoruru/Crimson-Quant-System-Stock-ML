# Crimson Quant System

CNN-LSTM stock price prediction with optional news sentiment, quantile-based trading strategy.

## Project Structure

```
Config
  config.py             Configuration dataclass, CLI config tool, feature list
  config.json           Persistent overrides (ticker, start, end, quantile_level, lookback, epochs, patience)

Training pipeline
  train.py              Training entry point — training loop, early stopping, prediction helpers
  model.py              CNNLSTMRegressor (Conv1d → LSTM → Dense)
  data_loader.py        Windowed dataset, scaler, train/val/test split
  features.py           Technical indicator computation, sentiment loader

Data fetching
  stock_data_fetcher.py Yahoo Finance OHLCV fetcher via yfinance
  fetch_news.py         News fetching from Alpha Vantage API (pagination, chunked date ranges)

Sentiment
  sentiment_evaluation.py  VADER sentiment scoring, daily aggregation, training/prediction CSV output

Evaluation & Signals
  metrics.py            Price, direction, and trading strategy metrics
  plotting.py           Forecast, equity curve, and loss plots
  prediction_validation.py  Historical back-test on held-out dates — requires ground-truth closes, end date must be ≤ today
  predict.py            Daily live signal — fetches today's data, runs inference, prints BUY/HOLD or SELL/CASH for tomorrow

Directories
  checkpoints/          Saved model weights (.pt)
  data/                 Raw CSVs, sentiment scores, news articles
  eval_outputs/         Evaluation results on held-out period
  my_fig_no_sentiment/  Plots from no-sentiment experiment
  my_fig_with_sentiment/ Plots from sentiment experiment
  tests/                pytest test suite
```

## Workflow

### One-time setup

1. **Configure** — set ticker, dates, quantile level, lookback window, epochs, and patience via `config.py --config` or edit `config.json`
2. **Fetch news** (optional, for `with_sentiment` model) — `fetch_news.py` pulls articles from Alpha Vantage; requires `NEWSAPI_KEY`
3. **Train** — `train.py` fetches OHLCV automatically, scores sentiment, and trains both `no_sentiment` and `with_sentiment` checkpoints
4. **Back-test** — `prediction_validation.py` evaluates both models on held-out historical dates and writes `eval_outputs/{tag}/eval_predictions.csv`; `predict.py` reads this file to calibrate its signal threshold

> **Note:** Step 4 is required before running `predict.py` for a meaningful threshold. Without it, the threshold falls back to `0.0` (any positive prediction → BUY).

### Daily (post-market-close)

5. **Live signal** — `predict.py` fetches today's OHLCV and news, runs inference, and prints a BUY/HOLD or SELL/CASH signal for the next trading day

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure ticker, date range, quantile level, lookback window, epochs, and early-stopping patience (interactive or view current)
python config.py --config
python config.py --show

# Train (reads ticker and date range from config.json)
# To change ticker or date range, run: python config.py --config
python train.py

# Back-test both models on held-out historical data (end date must be ≤ today)
# Outputs to eval_outputs/no_sentiment/ and eval_outputs/with_sentiment/
python prediction_validation.py                          # defaults to 1 month after training end
python prediction_validation.py --range 30d              # 30 days from training end
python prediction_validation.py --range 4w               # 4 weeks from training end
python prediction_validation.py --range 3m               # 3 months from training end
python prediction_validation.py --range 2024-06-15       # up to a specific past date

# Generate next-day trading signal (run after market close each day)
python predict.py                   # uses with_sentiment checkpoint
python predict.py --no-sentiment    # uses no_sentiment checkpoint

# Fetch news from Alpha Vantage (requires NEWSAPI_KEY)
python fetch_news.py --ticker AAPL --time-from 20190401T0000 --time-to 20221101T0000

# Score articles and build daily sentiment CSV
python -c "from sentiment_evaluation import evaluate_and_save_sentiment; evaluate_and_save_sentiment('data/AAPL_News_AlphaVantage_....csv', 'AAPL', '2019-04-01', '2022-11-01')"

# Run tests
python -m pytest tests/ -v
```

## Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your Alpha Vantage API key:
   ```
   NEWSAPI_KEY=your_actual_api_key
   ```

> `.env` is git-ignored and will not be pushed to the repository.

## Configuration Priority

`config.json` > dataclass defaults in `config.py`

> **Note:** `train.py` reads from `config.json` only — use `python config.py --config` to set ticker, dates, lookback, epochs, and patience. `prediction_validation.py` accepts `--range` to control the back-test window; the end date must be ≤ today since it requires real close prices for comparison. For a forward-looking signal use `predict.py` instead.

| Field | Type | Default | Description |
|---|---|---|---|
| `ticker` | string | `AAPL` | Stock ticker symbol |
| `start` | string | `2019-04-01` | Training period start date (YYYY-MM-DD) |
| `end` | string | `2022-11-01` | Training period end date (YYYY-MM-DD) |
| `quantile_level` | float | `0.70` | Percentile threshold for the long-only trading strategy (0 < x < 1) |
| `lookback` | int | `60` | Sliding window size in days used to build each training sample |
| `epochs` | int | `300` | Maximum training epochs |
| `patience` | int | `30` | Early-stopping patience (epochs without val-loss improvement before stopping) |
