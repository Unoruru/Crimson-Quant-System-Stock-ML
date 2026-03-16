"""News fetching from Alpha Vantage API.

Handles API calls, pagination, and raw article CSV output.
"""

import os
import time
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from config import Config

# ============================================================
# Settings
# ============================================================

load_dotenv()
API_KEY = os.environ.get("NEWSAPI_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"

SORT = "EARLIEST"   # LATEST / EARLIEST / RELEVANCE
LIMIT = 1000
DATA_DIR = "data"


# ============================================================
# Validation
# ============================================================

def validate_time_str(time_str: str) -> str:
    try:
        datetime.strptime(time_str, "%Y%m%dT%H%M")
    except ValueError as exc:
        raise ValueError(
            f"Invalid time '{time_str}'. Use format YYYYMMDDTHHMM, e.g. 20260301T0000"
        ) from exc
    return time_str


# ============================================================
# API calls
# ============================================================

def fetch_news(
    api_key: str,
    tickers: Optional[str] = None,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "EARLIEST",
    limit: int = 1000,
) -> Dict[str, Any]:
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

    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage information: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage note: {data['Note']}")

    return data


def feed_to_simple_dataframe(
    feed: List[Dict[str, Any]],
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    rows = []

    for item in feed:
        title = item.get("title")
        summary = item.get("summary", "")
        time_published = item.get("time_published")
        av_sentiment = _safe_float(item.get("overall_sentiment_score"))

        dt = pd.to_datetime(
            time_published,
            format="%Y%m%dT%H%M%S",
            errors="coerce",
            utc=True,
        )

        ticker_relevance = np.nan
        ticker_sentiment = np.nan
        if ticker:
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    ticker_relevance = _safe_float(ts.get("relevance_score"))
                    ticker_sentiment = _safe_float(
                        ts.get("ticker_sentiment_score"),
                    )
                    break

        rows.append(
            {
                "Date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else None,
                "Headline": title,
                "Summary": summary or "",
                "av_sentiment": av_sentiment,
                "ticker_relevance": ticker_relevance,
                "ticker_sentiment": ticker_sentiment,
            }
        )

    EXPECTED_COLUMNS = ["Date", "Headline", "Summary", "av_sentiment", "ticker_relevance", "ticker_sentiment"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = df.dropna(subset=["Date", "Headline"]).drop_duplicates().reset_index(drop=True)
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    return df


def _extract_raw(feed: list) -> pd.DataFrame:
    """Extract minimal (date, headlines) columns from Alpha Vantage feed."""
    rows = []
    for article in feed:
        pub = article.get("time_published", "")
        title = article.get("title", "")
        if pub and title:
            rows.append({"date": pub[:8], "headlines": title})  # pub[:8] = YYYYMMDD
    df = pd.DataFrame(rows, columns=["date", "headlines"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df.dropna(subset=["date", "headlines"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values("date", inplace=True)
    return df.reset_index(drop=True)


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning NaN on failure."""
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def default_filename(tickers: Optional[str], time_from: Optional[str], time_to: Optional[str], raw: bool = False) -> str:
    base = tickers or "news"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ",") else "_" for ch in base)
    safe = safe.replace(",", "_")
    safe = "_".join(safe.split())
    safe = safe.strip("_") or "news"

    tf = time_from[:8] if time_from else "start"
    tt = time_to[:8] if time_to else "end"
    tag = "raw" if raw else "AlphaVantage"
    return f"{safe}_News_{tag}_{tf}_{tt}.csv"


# ============================================================
# Pagination helper for long date ranges
# ============================================================

def fetch_news_chunked(
    api_key: str,
    tickers: Optional[str] = None,
    topics: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "EARLIEST",
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch news in monthly chunks to work around the 1000-article limit."""
    if not time_from or not time_to:
        data = fetch_news(api_key, tickers, topics, time_from, time_to, sort, limit)
        return data.get("feed", [])

    dt_from = datetime.strptime(time_from, "%Y%m%dT%H%M")
    dt_to = datetime.strptime(time_to, "%Y%m%dT%H%M")

    all_feed: List[Dict[str, Any]] = []
    chunk_start = dt_from

    # Compute total chunks for progress display
    total_chunks = 0
    tmp = dt_from
    while tmp < dt_to:
        tmp = min(tmp + relativedelta(months=1), dt_to)
        total_chunks += 1

    chunk_num = 0
    while chunk_start < dt_to:
        chunk_end = min(chunk_start + relativedelta(months=1), dt_to)
        tf = chunk_start.strftime("%Y%m%dT%H%M")
        tt = chunk_end.strftime("%Y%m%dT%H%M")
        chunk_num += 1

        print(f"[INFO] Fetching chunk {chunk_num}/{total_chunks}: {tf} -> {tt}")
        try:
            data = fetch_news(api_key, tickers, topics, tf, tt, sort, limit)
        except RuntimeError as exc:
            print(f"[WARN] Stopping fetch early: {exc}")
            break
        feed = data.get("feed", [])
        all_feed.extend(feed)
        print(f"[INFO]   Got {len(feed)} articles")

        chunk_start = chunk_end
        if chunk_start < dt_to:
            time.sleep(13)  # Rate limit: free tier allows 5 calls/min

    return all_feed


# ============================================================
# High-level fetch for a date range
# ============================================================

def fetch_news_for_period(ticker: str, start: str, end: str, output_dir: str = "data") -> str:
    """Fetch news for a date range in a single API call, save minimal CSV, return path.

    Output CSV columns: date (YYYY-MM-DD), headlines (article title).
    Skips API call if output CSV already exists and covers the full range (caching).
    """
    time_from = start.replace("-", "") + "T0000"
    time_to   = end.replace("-", "") + "T2359"

    validate_time_str(time_from)
    validate_time_str(time_to)

    output_filename = default_filename(ticker, time_from, time_to, raw=True)
    out_path = os.path.join(output_dir, output_filename)

    # --- Caching: skip fetch if complete data already on disk ---
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path)
            if len(existing_df) > 0:
                max_date = pd.to_datetime(existing_df["date"]).max()
                end_date  = pd.to_datetime(end)
                if max_date >= end_date - pd.Timedelta(days=1):
                    print(f"[INFO] News data complete: {out_path} ({len(existing_df)} articles)")
                    return out_path
        except Exception:
            pass

    if not API_KEY:
        raise ValueError("No API key configured. Set NEWSAPI_KEY environment variable.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Fetching Alpha Vantage news for {ticker}: {time_from[:8]} -> {end} (single request)")
    data = fetch_news(
        api_key=API_KEY,
        tickers=ticker,
        time_from=time_from,
        time_to=time_to,
        sort=SORT,
        limit=LIMIT,
    )
    feed = data.get("feed", [])
    df = _extract_raw(feed)

    if df.empty:
        print(f"[WARN] No articles returned for {ticker} in {start} -> {end}")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {len(df)} articles to: {out_path}")
    return out_path


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch news articles from Alpha Vantage",
    )
    parser.add_argument(
        "--ticker", "-t", default=None,
        help="Stock ticker symbol (default: from config)",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: from config)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: from config)",
    )
    parser.add_argument(
        "--extend", type=int, default=2, metavar="MONTHS",
        help="Extend end date by N months to cover eval period (default: 2)",
    )
    args = parser.parse_args()

    cfg = Config.load()
    ticker = args.ticker or cfg.ticker
    start = args.start or cfg.start
    end = args.end or cfg.end

    end_dt = datetime.strptime(end, "%Y-%m-%d") + relativedelta(months=args.extend)
    end_extended = end_dt.strftime("%Y-%m-%d")

    out_path = fetch_news_for_period(ticker, start, end_extended)
    print(f"\n[DONE] News articles saved to: {out_path}")


if __name__ == "__main__":
    main()
