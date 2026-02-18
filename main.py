import yfinance as yf
import pandas as pd
import os

from code.functions.preprocessing import missing_value_handling, oulier_detection
from code.functions.model_training import LSTMStockModel
from code.model import lstm_stock_model, lstm_stock_model_with_sentiment, train_cnn_lstm_model, train_cnn_lstm_model_with_sentiment

def download_stock_data():
    """Download AAPL stock data using yfinance."""
    print("Downloading AAPL stock data from Yahoo Finance...")
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="5y")
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    # Synthetic sentiment
    df['compound'] = df['Close'].pct_change().fillna(0) * 10
    df['compound'] = df['compound'].clip(-1, 1)

    print(f"Downloaded {len(df)} rows")
    return df


def main():
    """Train and evaluate models 4-5 (CNN-LSTM)."""

    # Download data
    df = download_stock_data()
    data = missing_value_handling(df)
    oulier_detection(data)
    df_model = data[['Date', 'Close', 'Volume', 'compound']].copy()

    print(f"Data: {df_model.shape[0]} rows, {df_model['Date'].min()} to {df_model['Date'].max()}")

    os.makedirs('fig/model training', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # Date range
    start_date = '2021-02-19'
    end_date = '2023-03-31'

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
    print(f"Model 4 (CNN-LSTM, Close):     RMSE = {rmse4:.4f}")
    print(f"Model 5 (CNN-LSTM, Sentiment): RMSE = {rmse5:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
