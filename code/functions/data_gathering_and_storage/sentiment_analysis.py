import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

file_path = 'data/aapl_News_All.csv'

def preprocess_news_data():
    news_data = pd.read_csv(file_path)
    news_data = news_data[['Date', 'Headline']].sort_values(by='Date').reset_index(drop=True)
    news_data['Date'] = pd.to_datetime(news_data['Date'])
    
    # sentiment analysis components
    news_data['compound'] = ''
    news_data['negative'] = ''
    news_data['neutral'] = ''
    news_data['positive'] = ''
    
    return news_data

def sentiment_score_calculation(news_data):
    sid = SentimentIntensityAnalyzer()

    scores = news_data['Headline'].apply(lambda x: sid.polarity_scores(x))
    news_data['compound'] = scores.apply(lambda x: x['compound'])
    news_data['negative'] = scores.apply(lambda x: x['neg'])
    news_data['neutral'] = scores.apply(lambda x: x['neu'])
    news_data['positive'] = scores.apply(lambda x: x['pos'])

    return news_data

def aggregate_same_day_news(news_data):
    news_with_sentiment = news_data.groupby('Date').agg({
        'Headline': ' '.join,
        'compound': 'mean',
        'negative': 'mean',
        'neutral': 'mean',
        'positive': 'mean'
    }).reset_index()
    return news_with_sentiment




