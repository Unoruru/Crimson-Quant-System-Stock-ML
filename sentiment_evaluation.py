"""Sentiment scoring and daily aggregation using VADER.

Scores news articles and produces daily sentiment CSVs for the training pipeline.
"""

import os
import re

import numpy as np
import pandas as pd

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# ============================================================
# VADER setup
# ============================================================

def ensure_vader():
    try:
        _ = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")


# ============================================================
# Column detection
# ============================================================

def guess_date_col(cols):
    candidates = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["date", "time", "published", "pub", "datetime", "created", "timestamp"]):
            candidates.append(c)
    return candidates[0] if candidates else None


def guess_text_col(cols):
    candidates = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["title", "headline", "headlines", "summary", "content", "text", "description", "article"]):
            candidates.append(c)
    return candidates[0] if candidates else None


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# ============================================================
# Scoring
# ============================================================

def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``sentiment`` column with per-row VADER compound scores.

    If *Summary* and *Headline* columns both exist, VADER scores
    ``Headline + " " + Summary`` for richer context.

    Parameters
    ----------
    df : DataFrame
        Must contain a text column detectable by :func:`guess_text_col`
        (e.g. ``Headline``, ``title``, ``summary``).

    Returns
    -------
    DataFrame
        The input frame with an additional ``sentiment`` column.
    """
    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    df = df.copy()

    # Combine Headline + Summary for better VADER accuracy when both exist
    if "Summary" in df.columns and "Headline" in df.columns:
        texts = (
            df["Headline"].fillna("").apply(clean_text)
            + " "
            + df["Summary"].fillna("").apply(clean_text)
        )
    else:
        text_col = guess_text_col(df.columns)
        if text_col is None:
            date_col = guess_date_col(df.columns)
            obj_cols = [
                c for c in df.columns if df[c].dtype == "object" and c != date_col
            ]
            if not obj_cols:
                raise ValueError(
                    f"Cannot find a text-like column in: {list(df.columns)}"
                )
            df["__text__"] = df[obj_cols].astype(str).agg(" ".join, axis=1)
            text_col = "__text__"
        texts = df[text_col].apply(clean_text)

    df["sentiment"] = texts.apply(lambda t: sia.polarity_scores(t)["compound"])
    return df


# ============================================================
# Daily aggregation
# ============================================================

def _aggregate_daily_sentiment(
    df: pd.DataFrame, has_av: bool, has_ticker: bool
) -> pd.DataFrame:
    """Aggregate per-article sentiment to daily values.

    Priority: ticker_sentiment (weighted by relevance) > av_sentiment > VADER.
    """
    score_col = "sentiment"
    if has_ticker:
        weights = df["ticker_relevance"].fillna(0.5)
        score_col = "_score"
        df = df.assign(
            _score=df["ticker_sentiment"].fillna(df.get("av_sentiment", 0.0)),
            _weight=weights,
        )
    elif has_av:
        score_col = "av_sentiment"

    if has_ticker:
        def _daily_stats(group: pd.DataFrame) -> pd.Series:
            scores = group[score_col].to_numpy(dtype=np.float64)
            weights = group["_weight"].to_numpy(dtype=np.float64)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                weights = np.ones_like(scores, dtype=np.float64)
                weight_sum = weights.sum()
            mean_score = float(np.average(scores, weights=weights))
            variance = float(np.average((scores - mean_score) ** 2, weights=weights))
            return pd.Series({
                "sentiment": mean_score,
                "news_count": float(len(group)),
                "sentiment_std": float(np.sqrt(max(variance, 0.0))),
                "sentiment_pos_ratio": float(np.mean(scores > 0)),
                "sentiment_neg_ratio": float(np.mean(scores < 0)),
            })

        daily = df.groupby("Date").apply(_daily_stats, include_groups=False).reset_index()
    else:
        grouped = df.groupby("Date")[score_col]
        daily = grouped.agg(["mean", "count", "std"]).reset_index()
        daily = daily.rename(columns={
            "mean": "sentiment",
            "count": "news_count",
            "std": "sentiment_std",
        })
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
        daily["sentiment_pos_ratio"] = grouped.apply(lambda s: float(np.mean(s > 0))).values
        daily["sentiment_neg_ratio"] = grouped.apply(lambda s: float(np.mean(s < 0))).values

    return daily


def build_daily_sentiment(news_csv_path: str, out_csv_path: str):
    try:
        df = pd.read_csv(news_csv_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty CSV: {news_csv_path}")
    if df.empty:
        raise ValueError(f"Empty CSV: {news_csv_path}")

    date_col = guess_date_col(df.columns)
    if date_col is None:
        raise ValueError(f"Cannot find a date-like column in: {list(df.columns)}")

    # Use AV sentiment when available, fall back to VADER
    has_av = "av_sentiment" in df.columns and df["av_sentiment"].notna().any()
    has_ticker = (
        "ticker_sentiment" in df.columns and df["ticker_sentiment"].notna().any()
    )

    if not has_av:
        df = score_articles(df)

    # Save per-article CSV with sentiment column
    article_out = news_csv_path
    df.to_csv(article_out, index=False, encoding="utf-8-sig")

    # Parse dates and aggregate daily
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df["Date"] = df[date_col].dt.normalize()

    daily = _aggregate_daily_sentiment(df, has_av, has_ticker)
    daily = daily.sort_values("Date")

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    daily.to_csv(out_csv_path, index=False)
    print(f"[OK] Saved daily sentiment: {out_csv_path}")
    print(daily.head(5))
    print("... rows:", len(daily))


# ============================================================
# High-level evaluate + save
# ============================================================

def evaluate_and_save_sentiment(
    news_csv_path: str, ticker: str, start: str, end: str,
    output_dir: str = "data", prefix: str = "training",
) -> str:
    """Score articles and produce daily sentiment CSV.

    Returns path to: data/{prefix}_sentiment_{ticker}_{start}_{end}.csv
    """
    out_filename = f"{prefix}_sentiment_{ticker}_{start}_{end}.csv"
    out_path = os.path.join(output_dir, out_filename)

    if os.path.exists(out_path):
        if os.path.getmtime(news_csv_path) > os.path.getmtime(out_path):
            print(f"[INFO] News data updated — rebuilding sentiment")
        else:
            try:
                df = pd.read_csv(out_path)
                if len(df) > 0:
                    print(f"[INFO] Using cached sentiment: {out_path} ({len(df)} days)")
                    return out_path
            except Exception:
                pass

    build_daily_sentiment(news_csv_path, out_path)
    return out_path
