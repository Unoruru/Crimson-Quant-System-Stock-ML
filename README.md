## AAPL Stock Forecasting Based on LSTM / CNN-LSTM and Sentiment Analysis

### Description
This project combines deep learning techniques, sentiment analysis and financial indicators to realize AAPL stock analysis and prediction. It uses both pure LSTM and CNN-LSTM hybrid architectures for time-series forecasting. It contains the following sections: using API requests for data acquisition, using MongoDB Altas for data storage, using Flask local-based API for CRUD functions, using visualization tools and ARIMA for data exploration, using ntlk for sentiment analysis, using LSTM and CNN-LSTM for model training and data forecasting.

##### Notes
- `StockModel()` is a class in model.py with 2 integrated LSTM models in 1 dimension and 2 dimensions training process respectively.
- `data processing.ipynb` is an assistance to show the whole EDA process
- `News Headlines ` are scarping from New York Times and Financial Times takes almost `2 hours` respectively, don't recommend to run.
-  Model Training process also takes time, for code cleanness I stored the trained model under file model/:
      - model 1: LSTM trained by historical close price from 2019-04-01 to 2023-03-31
      - model 2: LSTM trained by historical close price from 2021-01-01 to 2023-03-31
      - model 3: LSTM trained by historical close price and predicted sentiment scores from 2021-01-01 to 2023-03-31
      - model 4: CNN-LSTM trained by historical close price from 2021-01-01 to 2023-03-31
      - model 5: CNN-LSTM trained by historical close price and sentiment scores from 2021-01-01 to 2023-03-31
- Stock API is from `IEX cloud`, the API might be expired when you test the data, so I just show the extraction from Database in main.py. If you want to test the Stock API, get the key from here-> https://iexcloud.io/ and replace the api_key in the acquire_data function from code/data_gathering_and_storage/api_requests.py
- REST API to realize GRUD functions can be run through rest-api-server/app.py, related guidance is in the readme.md under rest-api-server file.

### File structure
```
- README.md
- AGENTS.md
- main.py
- requirements.txt
- environment.yml
- code/
    - __init__.py
    - data_acquire.py
    - model.py
    - functions/
        - data_gathering_and_storage/
            - api_requests.py
            - sentiment_analysis.py
            - storage.py
        - indicator.py
        - model_training.py
        - preprocessing.py
        - visualization.py
    - aapl data gathering.ipynb
    - data processing.ipynb
    - forcast copy.ipynb
- data/
    - aapl_News_All.csv
    - aapl_News_FT.csv
    - aapl_News_NYTimes_original.csv
- fig/
- model/
    - model1.h5
    - model2.h5
    - model3.h5
    - model4.h5
    - model5.h5
- rest-api-server/
    - app.py
    - README.MD
    - requirements.txt
```

### CNN-LSTM Architecture
Models 4 and 5 use a CNN-LSTM hybrid that prepends convolutional layers before the LSTM stack:

**Original Architecture (before tuning):**
```
Conv1D(64, kernel=3) -> BatchNorm -> Conv1D(32, kernel=3) -> BatchNorm
  -> LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)
```

**Tuned Architecture (current):**
```
Conv1D(32, kernel=5, padding='same') -> BatchNorm -> Conv1D(16, kernel=3, padding='same') -> BatchNorm
  -> MaxPooling1D(pool_size=2)
  -> LSTM(100) -> Dropout(0.3) -> LSTM(100) -> Dropout(0.3) -> LSTM(50) -> Dropout(0.3) -> Dense(1)
```

- **Rationale:** Conv1D layers extract local temporal patterns (short-term price movements) before the LSTM layers model long-range dependencies.
- **`padding='same'`** preserves the sequence length through convolutional layers.
- **MaxPooling1D** reduces dimensionality while retaining important features.
- **Tuning improvements:** Reduced filters (64→32), increased kernel (3→5), added MaxPooling, increased LSTM units (50→100), added early stopping.

### Partial Results
__The overall trend of AAPL with covid-19 period highlighted:__
<img src="/fig/data exploration/4.1_a_covid_highlight.png?raw=true" width="800" />

__Monthly return:__
- The highest range of returns occurred in March and August.
- February and April exhibited a relatively small interquartile range (IQR), indicating less variability in returns during these months.
- April, June, July, and December showed a low downside trade probability, with the minimum line close to the 25th percentile. This suggests that these months might provide favorable opportunities for entering the market with lower downside risk.

<img src="/fig/data exploration/4.1_b_monthly return.jpg?raw=true" width="800" />

__Seasonality Adjustments:__
- In periods where market has strong bull or bear (upward or downward) trends, seasonality effects might be weak to observe. However, if market exhibits range bound behavior it, such effect can be more evident.
- As 2020 is the bull market year, 2022 is a ordinary market year. The figures below shows the seasonal adjustments performance of two kinds. The seasonality effect is more obvious in 2022.
<img src="/fig/data exploration/4.1_d_seasonalityadj(2020).jpg?raw=true" width="800" />
<img src="/fig/data exploration/4.1_d_seasonalityadj(2022).jpg?raw=true" width="800" />

__On Balance Volume Indicators__
<img src="/fig/data exploration/4.2_signals_obvbased.png?raw=true" width="800" />

__LSTM Prediction using model3__
<img src="/fig/model training/5.2 prediction3.png?raw=true" width="800" />

__CNN-LSTM Prediction using model4 (Close Price Only)__
<img src="/fig/model training/5.3 prediction4_cnn_lstm.png?raw=true" width="800" />

__CNN-LSTM Prediction using model5 (Close + Sentiment)__
<img src="/fig/model training/5.3 prediction5_cnn_lstm_sentiment.png?raw=true" width="800" />

---

## Model Performance Comparison

### RMSE Results (Lower is Better)

| Model | Architecture | Features | Training Period | RMSE |
|-------|--------------|----------|-----------------|------|
| Model 1 | Pure LSTM | Close Price | 2019-04 to 2023-03 | ~7.0 |
| Model 2 | Pure LSTM | Close Price | 2021-01 to 2023-03 | ~9.5 |
| Model 3 | Pure LSTM | Close + Sentiment | 2021-01 to 2023-03 | ~12.9 |
| Model 4 | CNN-LSTM | Close Price | 2021-01 to 2023-03 | **46.87** |
| Model 5 | CNN-LSTM | Close + Sentiment | 2021-01 to 2023-03 | ~46-55 |

### Discussion: CNN vs CNN+Sentiment

#### Key Findings:

1. **Pure LSTM vs CNN-LSTM:**
   - Pure LSTM models (1-3) significantly outperform CNN-LSTM models (4-5) on this dataset
   - The additional CNN layers may introduce complexity that outweighs the benefit for short time-series data

2. **Sentiment Impact:**
   - **Model 3 (Pure LSTM + Sentiment):** Adding sentiment improved predictions slightly in some cases
   - **Model 5 (CNN-LSTM + Sentiment):** Sentiment consistently hurts performance
   - **Root Cause:** The sentiment data used is derived from price returns (synthetic), not real news sentiment. This introduces noise rather than useful signal.

3. **CNN-LSTM Tuning:**
   - After hyperparameter tuning (reduced filters, increased kernel size, added MaxPooling, early stopping), Model 4 improved by ~15%
   - The tuned architecture better balances feature extraction with model complexity

#### Conclusions:

- For this AAPL stock dataset, **Pure LSTM is recommended** over CNN-LSTM
- Real news sentiment data (not synthetic) would be needed to improve Model 5
- The CNN-LSTM architecture may be better suited for longer time-series or when more training data is available


