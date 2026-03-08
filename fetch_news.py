import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import requests


# ============================================================
# Editable settings
# Change these values here, then run: python fetch_news.py
# ============================================================
API_KEY = "bb1e60da84c047edb63d3fa4005ab9f6"
QUERY = "Apple OR AAPL"
START_DATE = "2026-02-08"
END_DATE = "2026-03-07"
LANGUAGE = "en"
SORT_BY = "publishedAt"            # relevancy / popularity / publishedAt
PAGE_SIZE = 100                    # max 100
MAX_PAGES = 10
DOMAINS = None                     # e.g. "reuters.com,bbc.com"
EXCLUDE_DOMAINS = None             # e.g. "yahoo.com"
SEARCH_IN = None                   # e.g. "title,description,content"
OUTPUT_FILENAME = None             # e.g. "apple_news.csv" ; None = auto filename
DATA_DIR = "data"

BASE_URL = "https://newsapi.org/v2/everything"


def validate_date(date_str: str) -> str:
    """Validate YYYY-MM-DD date format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{date_str}'. Use YYYY-MM-DD format.") from exc
    return date_str


def fetch_news(
    query: str,
    start_date: str,
    end_date: str,
    api_key: str,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 100,
    max_pages: int = 10,
    domains: Optional[str] = None,
    exclude_domains: Optional[str] = None,
    search_in: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch news articles from NewsAPI Everything endpoint."""
    if not api_key:
        raise ValueError("API_KEY is empty.")

    headers = {"X-Api-Key": api_key}
    all_articles: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "from": start_date,
            "to": end_date,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "page": page,
        }

        if domains:
            params["domains"] = domains
        if exclude_domains:
            params["excludeDomains"] = exclude_domains
        if search_in:
            params["searchIn"] = search_in

        response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = response.json()
            except Exception:
                detail = response.text

            msg = str(detail)

            if "too far in the past" in msg:
                raise RuntimeError(
                    "NewsAPI 当前套餐不支持这么早的历史新闻。"
                    "请把 START_DATE 改近一些，或升级套餐。"
                ) from exc

            raise RuntimeError(f"NewsAPI request failed on page {page}: {detail}") from exc

        data = response.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"NewsAPI returned error: {data}")

        articles = data.get("articles", [])
        total_results = data.get("totalResults", 0)
        print(f"[INFO] Page {page}: fetched {len(articles)} articles (totalResults={total_results})")

        if not articles:
            break

        all_articles.extend(articles)

        if len(articles) < page_size:
            break

        time.sleep(0.3)

    return all_articles


def articles_to_dataframe(articles: List[Dict[str, Any]], query: str) -> pd.DataFrame:
    """Convert NewsAPI article list to a clean dataframe."""
    rows = []
    for item in articles:
        source = item.get("source") or {}
        rows.append(
            {
                "source_id": source.get("id"),
                "source_name": source.get("name"),
                "author": item.get("author"),
                "title": item.get("title"),
                "description": item.get("description"),
                "content": item.get("content"),
                "url": item.get("url"),
                "urlToImage": item.get("urlToImage"),
                "publishedAt": item.get("publishedAt"),
                "query": query,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    df = df.sort_values("publishedAt", ascending=True).reset_index(drop=True)
    return df


def default_filename(query: str, start_date: str, end_date: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in query.strip())
    safe = "_".join(safe.split())
    safe = safe.strip("_") or "news"
    return f"{safe}_{start_date}_{end_date}_newsapi.csv"


def main() -> None:
    validate_date(START_DATE)
    validate_date(END_DATE)

    if PAGE_SIZE < 1 or PAGE_SIZE > 100:
        raise ValueError("PAGE_SIZE must be between 1 and 100.")
    if MAX_PAGES < 1:
        raise ValueError("MAX_PAGES must be >= 1.")
    if pd.to_datetime(END_DATE) < pd.to_datetime(START_DATE):
        raise ValueError("END_DATE must be the same as or later than START_DATE.")

    os.makedirs(DATA_DIR, exist_ok=True)

    output_filename = OUTPUT_FILENAME or default_filename(QUERY, START_DATE, END_DATE)
    out_path = os.path.join(DATA_DIR, output_filename)

    print("[INFO] Fetching news...")
    print(f"[INFO] Query       : {QUERY}")
    print(f"[INFO] Date range  : {START_DATE} -> {END_DATE}")
    print(f"[INFO] Language    : {LANGUAGE}")
    print(f"[INFO] Sort by     : {SORT_BY}")
    print(f"[INFO] Output CSV  : {out_path}")

    articles = fetch_news(
        query=QUERY,
        start_date=START_DATE,
        end_date=END_DATE,
        api_key=API_KEY,
        language=LANGUAGE,
        sort_by=SORT_BY,
        page_size=PAGE_SIZE,
        max_pages=MAX_PAGES,
        domains=DOMAINS,
        exclude_domains=EXCLUDE_DOMAINS,
        search_in=SEARCH_IN,
    )

    df = articles_to_dataframe(articles, QUERY)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved {len(df)} articles to: {out_path}")
    if not df.empty:
        print("\n[INFO] Preview:")
        print(df[["publishedAt", "source_name", "title"]].head(10).to_string(index=False))
    else:
        print("[WARN] No articles returned for this query/date range.")


if __name__ == "__main__":
    main()