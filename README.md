# Crimson Quant System

CNN-LSTM stock price prediction with optional news sentiment, quantile-based trading strategy.

## Project Structure

```
Config
  config.py             Configuration dataclass, CLI config tool, feature list
  config.json           Persistent overrides (ticker, start, end)

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

Evaluation
  metrics.py            Price, direction, and trading strategy metrics
  plotting.py           Forecast, equity curve, and loss plots
  predicate.py          Inference on unseen dates — evaluates both checkpoints, auto-rebuilds sentiment

Directories
  checkpoints/          Saved model weights (.pt)
  data/                 Raw CSVs, sentiment scores, news articles
  eval_outputs/         Evaluation results on held-out period
  my_fig_no_sentiment/  Plots from no-sentiment experiment
  my_fig_with_sentiment/ Plots from sentiment experiment
  tests/                pytest test suite
```

## Workflow

1. **Configure** — set ticker, dates, and quantile level via `config.py --config` or edit `config.json`
2. **Fetch data** — `train.py` pulls OHLCV from Yahoo Finance automatically
3. **Fetch news** (optional) — `fetch_news.py` pulls articles from Alpha Vantage
4. **Score sentiment** (optional) — `sentiment_evaluation.py` scores articles with VADER and aggregates daily
5. **Build features** — technical indicators computed in `features.py` during training
6. **Train** — CNN-LSTM trains on 60-day windows predicting next-day log return; produces `training_sentiment_*.csv`
7. **Evaluate** — `predicate.py` evaluates on held-out data; produces `prediction_sentiment_*.csv` for sentiment runs

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure ticker, date range, and quantile level (interactive or view current)
python config.py --config
python config.py --show

# Train (uses config.json defaults, or override via CLI)
python train.py
python train.py --ticker MSFT --start 2020-01-01 --end 2023-01-01

# Evaluate both models on unseen data (outputs to eval_outputs/no_sentiment/ and eval_outputs/with_sentiment/)
python predicate.py                          # defaults to 1 month after training end
python predicate.py --range 30d              # predict 30 days from training end
python predicate.py --range 4w               # predict 4 weeks from training end
python predicate.py --range 3m               # predict 3 months from training end
python predicate.py --range 2026-06-15       # predict until specific date

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

CLI args > `config.json` > dataclass defaults in `config.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `ticker` | string | `AAPL` | Stock ticker symbol |
| `start` | string | `2019-04-01` | Training period start date (YYYY-MM-DD) |
| `end` | string | `2022-11-01` | Training period end date (YYYY-MM-DD) |
| `quantile_level` | float | `0.70` | Percentile threshold for the long-only trading strategy (0 < x < 1) |