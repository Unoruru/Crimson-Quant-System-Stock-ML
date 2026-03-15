import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# ============================================================
# Editable settings
# Change these values here, then run:
#     python fetch_news_alphavantage.py
# ============================================================
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"

TICKERS = "AAPL"
TOPICS = None

# Format: YYYYMMDDTHHMM
TIME_FROM = "20221102T0000"
TIME_TO = "20221202T2359"

SORT = "EARLIEST"   # LATEST / EARLIEST / RELEVANCE
LIMIT = 1000
OUTPUT_FILENAME = None
DATA_DIR = "data"

BASE_URL = "https://www.alphavantage.co/query"


def validate_time_str(time_str: str) -> str:
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


def feed_to_simple_dataframe(feed: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for item in feed:
        title = item.get("title")
        time_published = item.get("time_published")

        dt = pd.to_datetime(
            time_published,
            format="%Y%m%dT%H%M%S",
            errors="coerce",
            utc=True,
        )

        rows.append(
            {
                "Date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else None,
                "Headline": title,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["Date", "Headline"]).drop_duplicates().reset_index(drop=True)
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    return df


def default_filename(tickers: Optional[str], time_from: Optional[str], time_to: Optional[str]) -> str:
    base = tickers or "news"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ",") else "_" for ch in base)
    safe = safe.replace(",", "_")
    safe = "_".join(safe.split())
    safe = safe.strip("_") or "news"

    tf = time_from[:8] if time_from else "start"
    tt = time_to[:8] if time_to else "end"
    return f"{safe}_News_AlphaVantage_{tf}_{tt}.csv"


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
    df = feed_to_simple_dataframe(feed)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved {len(df)} articles to: {out_path}")

    if not df.empty:
        print("\n[INFO] Articles per day:")
        print(df["Date"].value_counts().sort_index().to_string())

        print("\n[INFO] Preview:")
        print(df.head(10).to_string(index=False))
    else:
        print("[WARN] No articles returned for this query/time range.")


if __name__ == "__main__":
    main()