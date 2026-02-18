# Agent Instructions

## Project

AAPL stock price prediction using LSTM neural networks combined with VADER sentiment analysis on financial news headlines (New York Times, Financial Times). Data range: 2019-04 to 2023-04.

## Tech Stack

- **Language:** Python
- **ML/DL:** TensorFlow/Keras (Sequential LSTM), scikit-learn (MinMaxScaler, metrics)
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
│   ├── model.py                     # LSTM model inference wrappers
│   ├── functions/
│   │   ├── data_gathering_and_storage/
│   │   │   ├── api_requests.py      # IEX Cloud stock API, NYT API, FT web scraping
│   │   │   ├── sentiment_analysis.py # VADER sentiment scoring on headlines
│   │   │   └── storage.py           # MongoDB Atlas connect/insert/fetch
│   │   ├── preprocessing.py         # Missing value interpolation, outlier detection
│   │   ├── indicator.py             # Monthly returns, ADF test, OBV, seasonality
│   │   ├── model_training.py        # LSTMStockModel class (build, train, predict, viz)
│   │   └── visualization.py         # Plotting functions (area, line, scatter, covid, OBV)
│   ├── aapl data gathering.ipynb    # Interactive data acquisition notebook
│   ├── data processing.ipynb        # Data processing notebook
│   └── forcast copy.ipynb           # Forecasting notebook
├── model/
│   ├── model1.h5                    # LSTM trained on 2019-04 to 2023-03 (price only)
│   ├── model2.h5                    # LSTM trained on 2021-01 to 2023-03 (price only)
│   └── model3.h5                    # LSTM trained on 2021-01 to 2023-03 (price + sentiment)
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

Three `.h5` Keras models in `model/`. The pipeline loads these for inference; training code exists in `LSTMStockModel` class but is commented out in `model.py`.

## Known Code Issues

- **Hardcoded credentials:** MongoDB URI in `storage.py`, IEX API key and NYT API key in `api_requests.py`
- **`NotImplementedError`:** `main.py:138` raises at end of `main()` (likely a development placeholder)
- **Typo:** `oulier_detection` in `preprocessing.py` (missing 't' in "outlier")
- **Relative path inconsistency:** CSV save paths in `api_requests.py` use `../../../data` which may break depending on working directory

## Issues

No open GitHub issues.
