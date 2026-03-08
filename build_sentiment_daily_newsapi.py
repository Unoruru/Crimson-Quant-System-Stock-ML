import os
import re
from typing import List

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# ============================================================
# Editable settings
# Change these values here, then run: python build_sentiment_daily.py
# ============================================================
DATA_DIR = "data"
INPUT_FILENAME = "Apple_OR_AAPL_2026-02-08_2026-03-07_newsapi.csv"  # or e.g. "apple_news.csv"
OUTPUT_FILENAME = "sentiment_daily.csv"

# Which text columns to combine for sentiment scoring
USE_TITLE = True
USE_DESCRIPTION = True
USE_CONTENT = True

# Row-level aggregation across the selected text columns
# "mean" = average score across title/description/content
# "sum"  = sum scores across title/description/content
TEXT_AGG = "mean"

# Daily aggregation across all articles on the same day
# "mean" / "median"
DAILY_AGG = "mean"

# Drop rows where all selected text fields are empty
DROP_EMPTY_TEXT = True


# ============================================================
# Helpers
# ============================================================
def ensure_vader() -> None:
    """Download VADER lexicon if needed."""
    try:
        _ = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")


def clean_text(x) -> str:
    """Normalize whitespace and handle missing values."""
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def score_text(sia: SentimentIntensityAnalyzer, text: str) -> float:
    """Return VADER compound score for a text string."""
    text = clean_text(text)
    if not text:
        return 0.0
    return float(sia.polarity_scores(text)["compound"])


def build_text_list(row: pd.Series, cols: List[str]) -> List[str]:
    """Collect non-empty text fields from one article row."""
    parts = []
    for c in cols:
        if c in row.index:
            t = clean_text(row[c])
            if t:
                parts.append(t)
    return parts


def aggregate_scores(values: List[float], mode: str) -> float:
    """Aggregate multiple sentiment scores into one score."""
    if not values:
        return 0.0
    if mode == "mean":
        return float(sum(values) / len(values))
    if mode == "sum":
        return float(sum(values))
    raise ValueError("TEXT_AGG must be 'mean' or 'sum'.")


def build_daily_sentiment(news_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    """Read fetched news CSV and save daily sentiment CSV."""
    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    if not os.path.exists(news_csv_path):
        raise FileNotFoundError(f"Input news CSV not found: {news_csv_path}")

    df = pd.read_csv(news_csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {news_csv_path}")

    # This script is designed to match fetch_news.py output.
    required_date_col = "publishedAt"
    if required_date_col not in df.columns:
        raise ValueError(
            f"Cannot find '{required_date_col}' in input CSV. Found columns: {list(df.columns)}"
        )

    selected_text_cols = []
    if USE_TITLE:
        selected_text_cols.append("title")
    if USE_DESCRIPTION:
        selected_text_cols.append("description")
    if USE_CONTENT:
        selected_text_cols.append("content")

    existing_text_cols = [c for c in selected_text_cols if c in df.columns]
    if not existing_text_cols:
        raise ValueError(
            f"None of the selected text columns exist in the CSV. "
            f"Selected={selected_text_cols}, Found={list(df.columns)}"
        )

    # Parse time and derive trading-date-style day key.
    df[required_date_col] = pd.to_datetime(df[required_date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[required_date_col]).copy()
    if df.empty:
        raise ValueError("All publishedAt values failed to parse.")

    df["Date"] = df[required_date_col].dt.tz_convert(None).dt.normalize()

    # Combine title / description / content into one row-level sentiment score.
    row_scores = []
    text_used = []
    for _, row in df.iterrows():
        parts = build_text_list(row, existing_text_cols)
        text_used.append(" | ".join(parts))
        scores = [score_text(sia, p) for p in parts]
        row_scores.append(aggregate_scores(scores, TEXT_AGG))

    df["text_used"] = text_used
    df["sentiment_row"] = row_scores

    if DROP_EMPTY_TEXT:
        df = df[df["text_used"].str.len() > 0].copy()
        if df.empty:
            raise ValueError("No valid text left after filtering empty rows.")

    # Daily aggregation.
    if DAILY_AGG == "mean":
        daily = (
            df.groupby("Date", as_index=False)
            .agg(
                sentiment=("sentiment_row", "mean"),
                article_count=("sentiment_row", "size"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
    elif DAILY_AGG == "median":
        daily = (
            df.groupby("Date", as_index=False)
            .agg(
                sentiment=("sentiment_row", "median"),
                article_count=("sentiment_row", "size"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        raise ValueError("DAILY_AGG must be 'mean' or 'median'.")

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    daily.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved daily sentiment CSV: {out_csv_path}")
    print(f"[INFO] Input rows used      : {len(df)}")
    print(f"[INFO] Output day rows      : {len(daily)}")
    print(f"[INFO] Text columns used    : {existing_text_cols}")
    print(f"[INFO] Row aggregation      : {TEXT_AGG}")
    print(f"[INFO] Daily aggregation    : {DAILY_AGG}")
    print("\n[INFO] Preview:")
    print(daily.head(10).to_string(index=False))

    return daily


# ============================================================
# Main
# ============================================================
def main() -> None:
    news_csv_path = os.path.join(DATA_DIR, INPUT_FILENAME)
    out_csv_path = os.path.join(DATA_DIR, OUTPUT_FILENAME)
    build_daily_sentiment(news_csv_path, out_csv_path)


if __name__ == "__main__":
    main()
