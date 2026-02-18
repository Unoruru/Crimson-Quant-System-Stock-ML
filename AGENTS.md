# Agent Instructions

## Project

AAPL stock price prediction using LSTM and CNN-LSTM hybrid neural networks combined with VADER sentiment analysis on financial news headlines (New York Times, Financial Times). Data range: 2019-04 to 2023-04.

## Tech Stack

- **Language:** Python
- **ML/DL:** TensorFlow/Keras (Sequential LSTM, CNN-LSTM hybrid), scikit-learn (MinMaxScaler, metrics)
- **NLP:** NLTK VADER sentiment analyzer
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Statistics:** statsmodels (seasonal decomposition, ADF test)
- **Web scraping:** requests, BeautifulSoup
- **Database:** MongoDB Atlas (pymongo)
- **API server:** Flask (REST API for CRUD on stock data)

## Project Structure

```
├── main.py                          # Full pipeline orchestrator
├── code/
│   ├── data_acquire.py              # Data acquisition facade (stock + news + sentiment)
│   ├── model.py                     # LSTM and CNN-LSTM model inference/training wrappers
│   ├── functions/
│   │   ├── data_gathering_and_storage/
│   │   │   ├── api_requests.py      # IEX Cloud stock API, NYT API, FT web scraping
│   │   │   ├── sentiment_analysis.py # VADER sentiment scoring on headlines
│   │   │   └── storage.py           # MongoDB Atlas connect/insert/fetch
│   │   ├── preprocessing.py         # Missing value interpolation, outlier detection
│   │   ├── indicator.py             # Monthly returns, ADF test, OBV, seasonality
│   │   ├── model_training.py        # LSTMStockModel class (LSTM + CNN-LSTM architectures, train, predict, viz)
│   │   └── visualization.py         # Plotting functions (area, line, scatter, covid, OBV)
│   ├── aapl data gathering.ipynb    # Interactive data acquisition notebook
│   ├── data processing.ipynb        # Data processing notebook
│   └── forcast copy.ipynb           # Forecasting notebook
├── model/
│   ├── model1.h5                    # LSTM trained on 2019-04 to 2023-03 (price only)
│   ├── model2.h5                    # LSTM trained on 2021-01 to 2023-03 (price only)
│   ├── model3.h5                    # LSTM trained on 2021-01 to 2023-03 (price + sentiment)
│   ├── model4.h5                    # CNN-LSTM trained on 2021-01 to 2023-03 (price only)
│   └── model5.h5                    # CNN-LSTM trained on 2021-01 to 2023-03 (price + sentiment)
├── data/
│   ├── aapl_News_All.csv            # Combined news headlines
│   ├── aapl_News_FT.csv             # Financial Times headlines
│   └── aapl_News_NYTimes_original.csv # New York Times headlines
├── fig/                             # Generated plots and figures
└── rest-api-server/
    └── app.py                       # Flask CRUD API for MongoDB stock data
```

## How to Run

- **Full pipeline:** `python main.py` (acquires data, stores in MongoDB, preprocesses, runs EDA, loads pre-trained models for prediction)
- **Notebooks:** Run Jupyter notebooks in `code/` for interactive exploration
- **Flask API:** Set `MONGO_DB_CONN_STRING` env var, then `python rest-api-server/app.py`

## Data Pipeline

1. **Acquire stock data** from IEX Cloud API (AAPL historical prices)
2. **Scrape news headlines** from NYT API and FT website
3. **Compute sentiment scores** using NLTK VADER on headlines
4. **Merge** stock data with aggregated daily sentiment scores
5. **Store** combined dataset in MongoDB Atlas
6. **Preprocess:** interpolate missing dates, detect outliers (z-score)
7. **EDA:** COVID impact analysis, monthly returns, ADF test, seasonal decomposition, OBV signals
8. **Forecast:** load pre-trained LSTM models, predict, visualize, compute RMSE

## Pre-trained Models

Five `.h5` Keras models in `model/` (3 pure-LSTM, 2 CNN-LSTM hybrid). The pipeline loads these for inference; training code exists in `LSTMStockModel` class but is commented out in `model.py`.

## Known Code Issues (Resolved 2026-02-18)

The following issues were identified and fixed in the integrity audit:

- ~~**Hardcoded credentials:** MongoDB URI, IEX API key, NYT API key~~ -> Moved to env vars
- ~~**Typo:** `oulier_detection`~~ -> Renamed to `outlier_detection`
- ~~**Missing `__init__.py`** in `code/functions/` and `code/functions/data_gathering_and_storage/`~~ -> Created
- ~~**Broken import:** `statsmodels.tsa.arima_model` (removed in 0.13)~~ -> Removed unused import
- ~~**Silent bugs:** discarded `sort_values()` and `drop_duplicates()` results~~ -> Assigned back
- ~~**Infinite loop:** KeyError in NYT retry loop skipped increment~~ -> Fixed
- ~~**4x redundant sentiment scoring**~~ -> Single `polarity_scores()` call, unpack values
- ~~**Figure memory leaks**~~ -> Added `plt.close()` after `savefig()`
- ~~**REST API:** No input validation~~ -> Added schema validation and ObjectId checks
- **Relative path inconsistency:** CSV save paths in `api_requests.py` use `../../../data/` which may break depending on working directory

## Recent Changes (2026-02-18)

### CNN-LSTM Model Tuning

**Problem:** CNN-LSTM models (4-5) performed significantly worse than pure LSTM models.

**Initial Results:**
- Model 4 (CNN-LSTM, Close): RMSE = 55.60
- Model 5 (CNN-LSTM, Sentiment): RMSE = 46.62

**Tuning Actions Applied:**

1. **Reduced CNN filters:** Changed from (64, 32) to (32, 16) - less aggressive feature extraction
2. **Increased kernel size:** Changed from 3 to 5 - better capture of local patterns
3. **Added MaxPooling1D:** After Conv layers for dimensionality reduction
4. **Increased LSTM units:** From 50 to 100 (first 2 layers) + 50
5. **Increased dropout:** From 0.2 to 0.3 - better regularization
6. **Reduced batch size:** From 64 to 32 - more stable training
7. **Added early stopping:** patience=15, restore_best_weights=True
8. **Increased epochs:** From 50 to 100 with early stopping

**Tuned Results:**
- Model 4 (CNN-LSTM, Close): RMSE = 46.87 (**+15.7% improvement**)
- Model 5 (CNN-LSTM, Sentiment): RMSE = 50.38 (-8.1%)

**Model 5 Investigation:**
Multiple architectures tested for 2D input (close + sentiment):
- Reduced filter architecture: RMSE = 63.41 (worse)
- Higher dropout (0.5): RMSE = 55.36 (worse)
- Same as Model 4 architecture: RMSE = 55.36 (still worse)

**Root Cause:** Sentiment data is **synthetic** (derived from price returns), not real news sentiment. This adds noise rather than signal. The sentiment feature hurts performance in all cases.

**Files Modified:**
- `code/functions/model_training.py` - CNN-LSTM architecture changes

## Issues

No open GitHub issues.
