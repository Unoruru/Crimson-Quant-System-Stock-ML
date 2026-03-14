import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# ============================================================
# Editable settings
# Change these values here, then run:
#     python fetch_news_alphavantage.py
# ============================================================
API_KEY = "YI76ZVZQORBEPOEHX"

# Examples:
# "AAPL"
# "AAPL,MSFT"
# "COIN,CRYPTO:BTC,FOREX:USD"
TICKERS = "AAPL"

# Optional topics, examples:
# "technology"
# "technology,ipo"
# "financial_markets"
TOPICS = None

# Alpha Vantage time format: YYYYMMDDTHHMM
# Example: 20260301T0000
TIME_FROM = "20260208T0000"
TIME_TO = "20260307T2359"

SORT = "EARLIEST"   # LATEST / EARLIEST / RELEVANCE
LIMIT = 1000        # up to 1000
OUTPUT_FILENAME = None   # e.g. "aapl_news_alphavantage.csv"
DATA_DIR = "data"

BASE_URL = "https://www.alphavantage.co/query"


def validate_time_str(time_str: str) -> str:
    """Validate Alpha Vantage news time format: YYYYMMDDTHHMM"""
    try:
        datetime.strptime(time_str, "%Y%m%dT%H%M")
    except ValueError as exc:
        raise ValueError(
            f"Invalid time '{time_str}'. Use format YYYYMMDDTHHMM, e.g. 20260301T0000"
        ) from exc
    return time_str


def fetch_news(
    api_key: str,
    tickers: Optional[str] = None,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "EARLIEST",
    limit: int = 1000,
) -> Dict[str, Any]:
    """Fetch news from Alpha Vantage NEWS_SENTIMENT endpoint."""
    if not api_key:
        raise ValueError("API_KEY is empty.")

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "sort": sort,
        "limit": limit,
    }

    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to

    response = requests.get(BASE_URL, params=params, timeout=60)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(f"Alpha Vantage request failed: {detail}") from exc

    data = response.json()

    # Alpha Vantage often returns "Note", "Information", or "Error Message"
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage information: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage note: {data['Note']}")

    return data


def parse_ticker_sentiment(item: Dict[str, Any], target_tickers: List[str]) -> Dict[str, Any]:
    """
    Extract ticker-sentiment fields related to the target tickers if available.
    """
    ticker_sentiments = item.get("ticker_sentiment", []) or []

    matched = []
    for ts in ticker_sentiments:
        ticker = str(ts.get("ticker", "")).upper()
        if ticker in target_tickers:
            matched.append(ts)

    if not matched:
        return {
            "matched_tickers": None,
            "ticker_relevance_score": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
        }

    # If multiple matched tickers, join them; numeric fields take the first match
    first = matched[0]
    return {
        "matched_tickers": ",".join(str(x.get("ticker", "")) for x in matched),
        "ticker_relevance_score": first.get("relevance_score"),
        "ticker_sentiment_score": first.get("ticker_sentiment_score"),
        "ticker_sentiment_label": first.get("ticker_sentiment_label"),
    }


def feed_to_dataframe(feed: List[Dict[str, Any]], tickers: Optional[str]) -> pd.DataFrame:
    """
    Convert Alpha Vantage feed into a dataframe.
    Output columns are designed to be similar to your NewsAPI CSV style.
    """
    target_tickers = []
    if tickers:
        target_tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]

    rows = []
    for item in feed:
        source = item.get("source")
        authors = item.get("authors")
        summary = item.get("summary")
        title = item.get("title")
        url = item.get("url")
        banner_image = item.get("banner_image")
        time_published = item.get("time_published")
        overall_sentiment_score = item.get("overall_sentiment_score")
        overall_sentiment_label = item.get("overall_sentiment_label")
        category_within_source = item.get("category_within_source")
        source_domain = item.get("source_domain")
        topics_list = item.get("topics", [])

        topic_names = []
        for t in topics_list:
            if isinstance(t, dict):
                topic_names.append(str(t.get("topic", "")))
            else:
                topic_names.append(str(t))

        ticker_info = parse_ticker_sentiment(item, target_tickers)

        rows.append(
            {
                # Keep similar naming to your NewsAPI output
                "source_id": None,
                "source_name": source,
                "author": ", ".join(authors) if isinstance(authors, list) else authors,
                "title": title,
                "description": summary,
                "content": summary,   # duplicate summary here for easier downstream use
                "url": url,
                "urlToImage": banner_image,
                "publishedAt": time_published,
                "query": tickers,

                # Extra Alpha Vantage-specific useful fields
                "overall_sentiment_score": overall_sentiment_score,
                "overall_sentiment_label": overall_sentiment_label,
                "category_within_source": category_within_source,
                "source_domain": source_domain,
                "topics": ",".join([x for x in topic_names if x]),
                "matched_tickers": ticker_info["matched_tickers"],
                "ticker_relevance_score": ticker_info["ticker_relevance_score"],
                "ticker_sentiment_score": ticker_info["ticker_sentiment_score"],
                "ticker_sentiment_label": ticker_info["ticker_sentiment_label"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Alpha Vantage gives time as YYYYMMDDTHHMMSS
    df["publishedAt"] = pd.to_datetime(
        df["publishedAt"],
        format="%Y%m%dT%H%M%S",
        errors="coerce",
        utc=True,
    )

    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    df = df.sort_values("publishedAt", ascending=True).reset_index(drop=True)
    return df


def default_filename(tickers: Optional[str], time_from: Optional[str], time_to: Optional[str]) -> str:
    base = tickers or "news"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ",") else "_" for ch in base)
    safe = safe.replace(",", "_")
    safe = "_".join(safe.split())
    safe = safe.strip("_") or "news"

    tf = time_from or "start"
    tt = time_to or "end"
    return f"{safe}_{tf}_{tt}_alphavantage_news.csv"


def main() -> None:
    if TIME_FROM:
        validate_time_str(TIME_FROM)
    if TIME_TO:
        validate_time_str(TIME_TO)

    if LIMIT < 1 or LIMIT > 1000:
        raise ValueError("LIMIT must be between 1 and 1000.")

    if TIME_FROM and TIME_TO:
        dt_from = datetime.strptime(TIME_FROM, "%Y%m%dT%H%M")
        dt_to = datetime.strptime(TIME_TO, "%Y%m%dT%H%M")
        if dt_to < dt_from:
            raise ValueError("TIME_TO must be the same as or later than TIME_FROM.")

    os.makedirs(DATA_DIR, exist_ok=True)

    output_filename = OUTPUT_FILENAME or default_filename(TICKERS, TIME_FROM, TIME_TO)
    out_path = os.path.join(DATA_DIR, output_filename)

    print("[INFO] Fetching Alpha Vantage news...")
    print(f"[INFO] Tickers     : {TICKERS}")
    print(f"[INFO] Topics      : {TOPICS}")
    print(f"[INFO] Time range  : {TIME_FROM} -> {TIME_TO}")
    print(f"[INFO] Sort        : {SORT}")
    print(f"[INFO] Limit       : {LIMIT}")
    print(f"[INFO] Output CSV  : {out_path}")

    data = fetch_news(
        api_key=API_KEY,
        tickers=TICKERS,
        topics=TOPICS,
        time_from=TIME_FROM,
        time_to=TIME_TO,
        sort=SORT,
        limit=LIMIT,
    )

    feed = data.get("feed", [])
    df = feed_to_dataframe(feed, TICKERS)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved {len(df)} articles to: {out_path}")

    if not df.empty:
        print("\n[INFO] Articles per day:")
        per_day = df["publishedAt"].dt.strftime("%Y-%m-%d").value_counts().sort_index()
        print(per_day.to_string())

        print("\n[INFO] Preview:")
        print(df[["publishedAt", "source_name", "title"]].head(10).to_string(index=False))
    else:
        print("[WARN] No articles returned for this query/time range.")


if __name__ == "__main__":
    main()