# Development Log

## 2026-02-18 — Retrained Models 3-5 with Extended Data

**What changed:**
- Extended training period for models 3, 4, and 5 from `2021-01 to 2023-03` to `2021-01 to 2025-12`
- Validation window set to ~1 month (Jan 2026)
- Refactored `model_training.py` to support configurable date ranges and model selection

**Results (RMSE):**
| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| Model 3 (LSTM + Sentiment) | ~12.9 | 3.32 | -74% |
| Model 4 (CNN-LSTM) | 46.87 | 5.08 | -89% |
| Model 5 (CNN-LSTM + Sentiment) | ~50 | 7.12 | -86% |

**Takeaway:** More training data is the single biggest factor in model performance. Model 3 (LSTM + Sentiment) is now the best performer.

---

## 2026-02-18 — CNN-LSTM Architecture Tuning

**What changed:**
- Tuned CNN-LSTM hyperparameters: reduced Conv1D filters (64 to 32), increased kernel size (3 to 5), added `padding='same'` and MaxPooling1D
- Increased LSTM units from 50 to 100, dropout from 0.2 to 0.3
- Added early stopping with patience=15 and learning rate reduction

**Files modified:** `code/model.py`, `code/functions/model_training.py`

---

## 2026-02-18 — CNN-LSTM Models (4-5) Initial Training

**What changed:**
- Added CNN-LSTM hybrid architecture to `LSTMStockModel` class
- Implemented `build_cnn_lstm_model()` for 1D and 2D feature inputs
- Created training pipeline for models 4 (close price) and 5 (close + sentiment)
- Generated prediction visualization figures

**Files added:** `train_all_models.py`, `train_tuned.py`, `train_m5.py`, `compare_models.py`, `test_models.py`

---

## Earlier — Initial Project Setup

- Built LSTM models 1-3 for AAPL stock price prediction
- Implemented sentiment analysis pipeline using NLTK (NYT and FT headlines)
- Created data acquisition module with IEX Cloud API
- Set up MongoDB Atlas for data storage
- Built Flask REST API server for CRUD operations
- Added data exploration and visualization (EDA, ARIMA, OBV indicators)
