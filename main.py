import yfinance as yf
import pandas as pd
import os

from code.functions.preprocessing import missing_value_handling, outlier_detection
from code.functions.model_training import LSTMStockModel
from code.model import (
    lstm_stock_model,
    lstm_stock_model_with_sentiment,
    train_lstm_model_with_sentiment,
    train_cnn_lstm_model,
    train_cnn_lstm_model_with_sentiment,
)

def download_stock_data():
    """Download AAPL stock data using yfinance."""
    print("Downloading AAPL stock data from Yahoo Finance...")
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start="2021-01-01")
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    # Synthetic sentiment
    df['compound'] = df['Close'].pct_change().fillna(0) * 10
    df['compound'] = df['compound'].clip(-1, 1)

    print(f"Downloaded {len(df)} rows")
    return df


def main():
    """Train and evaluate models 3-5."""

    # Download data
    df = download_stock_data()
    data = missing_value_handling(df)
    outlier_detection(data)
    df_model = data[['Date', 'Close', 'Volume', 'compound']].copy()

    print(f"Data: {df_model.shape[0]} rows, {df_model['Date'].min()} to {df_model['Date'].max()}")

    os.makedirs('fig/model training', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # Unified date range: train on 2021-01 to 2025-12, validate on ~Jan 2026
    start_date = '2021-01-01'
    end_date = '2025-12-31'

    # ============================================
    # Model 3: LSTM + Sentiment
    # ============================================
    print("\n" + "="*60)
    print("Model 3: LSTM (Close + Sentiment)")
    print(f"Training: {start_date} to {end_date}")
    print("="*60)

    print("Training...")
    train_lstm_model_with_sentiment(df_model, start_date, end_date, 'model/model3.h5')

    print("Predicting...")
    model3 = 'model/model3.h5'
    figpath3 = 'fig/model training/5.3 prediction3_lstm_sentiment.png'
    rmse3 = lstm_stock_model_with_sentiment(df_model, start_date, end_date, model3, figpath3)
    print(f"Model 3 RMSE: {rmse3:.4f}")

    # ============================================
    # Model 4: CNN-LSTM (close only)
    # ============================================
    print("\n" + "="*60)
    print("Model 4: CNN-LSTM (Close Price Only)")
    print(f"Training: {start_date} to {end_date}")
    print("="*60)

    print("Training...")
    train_cnn_lstm_model(df_model, start_date, end_date, 'model/model4.h5')

    print("Predicting...")
    model4 = 'model/model4.h5'
    figpath4 = 'fig/model training/5.3 prediction4_cnn_lstm.png'
    rmse4 = lstm_stock_model(df_model, start_date, end_date, model4, figpath4)
    print(f"Model 4 RMSE: {rmse4:.4f}")

    # ============================================
    # Model 5: CNN-LSTM with sentiment
    # ============================================
    print("\n" + "="*60)
    print("Model 5: CNN-LSTM (Close + Sentiment)")
    print(f"Training: {start_date} to {end_date}")
    print("="*60)

    print("Training...")
    train_cnn_lstm_model_with_sentiment(df_model, start_date, end_date, 'model/model5.h5')

    print("Predicting...")
    model5 = 'model/model5.h5'
    figpath5 = 'fig/model training/5.3 prediction5_cnn_lstm_sentiment.png'
    rmse5 = lstm_stock_model_with_sentiment(df_model, start_date, end_date, model5, figpath5)
    print(f"Model 5 RMSE: {rmse5:.4f}")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model 3 (LSTM, Sentiment):     RMSE = {rmse3:.4f}")
    print(f"Model 4 (CNN-LSTM, Close):     RMSE = {rmse4:.4f}")
    print(f"Model 5 (CNN-LSTM, Sentiment): RMSE = {rmse5:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
