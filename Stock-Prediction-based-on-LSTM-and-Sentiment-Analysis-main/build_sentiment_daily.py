import os
import re
import numpy as np
import pandas as pd

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 第一次运行需要下载词典
def ensure_vader():
    try:
        _ = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")

def guess_date_col(cols):
    cols_l = [c.lower() for c in cols]
    candidates = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["date", "time", "published", "pub", "datetime", "created", "timestamp"]):
            candidates.append(c)
    return candidates[0] if candidates else None

def guess_text_col(cols):
    cols_l = [c.lower() for c in cols]
    candidates = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["title", "headline", "summary", "content", "text", "description", "article"]):
            candidates.append(c)
    return candidates[0] if candidates else None

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def build_daily_sentiment(news_csv_path: str, out_csv_path: str):
    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    df = pd.read_csv(news_csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {news_csv_path}")

    date_col = guess_date_col(df.columns)
    text_col = guess_text_col(df.columns)

    if date_col is None:
        raise ValueError(f"Cannot find a date-like column in: {list(df.columns)}")
    if text_col is None:
        # 兜底：如果没有明显的文本列，就把所有 object 列拼起来
        obj_cols = [c for c in df.columns if df[c].dtype == "object" and c != date_col]
        if not obj_cols:
            raise ValueError(f"Cannot find a text-like column in: {list(df.columns)}")
        df["__text__"] = df[obj_cols].astype(str).agg(" ".join, axis=1)
        text_col = "__text__"

    # parse datetime -> date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df["Date"] = df[date_col].dt.normalize()

    # compute VADER compound sentiment per row
    texts = df[text_col].apply(clean_text)
    df["sentiment_row"] = texts.apply(lambda t: sia.polarity_scores(t)["compound"])

    # daily aggregation (mean); 也可以改成 median
    daily = df.groupby("Date", as_index=False)["sentiment_row"].mean()
    daily = daily.rename(columns={"sentiment_row": "sentiment"}).sort_values("Date")

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    daily.to_csv(out_csv_path, index=False)
    print(f"[OK] Saved daily sentiment: {out_csv_path}")
    print(daily.head(5))
    print("... rows:", len(daily))

if __name__ == "__main__":
    # 你可以在这里选择用哪个新闻源
    news_csv = os.path.join("data", "aapl_News_NYTimes_original.csv")
    out_csv  = os.path.join("data", "sentiment_daily.csv")
    build_daily_sentiment(news_csv, out_csv)