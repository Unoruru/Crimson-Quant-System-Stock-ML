"""Tests for sentiment_evaluation module."""

import numpy as np
import pandas as pd
import pytest

from crimson_quant.sentiment_evaluation import (
    _aggregate_daily_sentiment,
    build_daily_sentiment,
    score_articles,
)


class TestScoreArticles:
    def test_adds_sentiment_column(self):
        df = pd.DataFrame({
            "Date": ["2022-11-02", "2022-11-03"],
            "Headline": ["Great earnings beat expectations", "Stock crashes badly"],
        })
        result = score_articles(df)
        assert "sentiment" in result.columns
        assert len(result) == 2

    def test_uses_summary_when_available(self):
        df = pd.DataFrame({
            "Date": ["2022-11-02"],
            "Headline": ["Neutral title"],
            "Summary": ["This is absolutely wonderful amazing great fantastic news"],
        })
        result = score_articles(df)
        # With positive summary, sentiment should be positive
        assert result["sentiment"].iloc[0] > 0

    def test_sentiment_values_in_range(self):
        df = pd.DataFrame({
            "Date": ["2022-11-02", "2022-11-03", "2022-11-04"],
            "Headline": [
                "Wonderful amazing great news",
                "Terrible horrible bad news",
                "Neutral statement about something",
            ],
        })
        result = score_articles(df)
        assert all(-1 <= v <= 1 for v in result["sentiment"])

    def test_handles_empty_text(self):
        df = pd.DataFrame({
            "Date": ["2022-11-02"],
            "Headline": [""],
        })
        result = score_articles(df)
        assert "sentiment" in result.columns
        assert len(result) == 1


class TestAggregateDailySentiment:
    def test_uses_ticker_sentiment_when_available(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2022-11-02", "2022-11-02"]),
            "ticker_sentiment": [0.3, 0.5],
            "ticker_relevance": [0.9, 0.1],
            "av_sentiment": [0.1, 0.2],
        })
        daily = _aggregate_daily_sentiment(df, has_av=True, has_ticker=True)
        assert len(daily) == 1
        assert "sentiment" in daily.columns
        assert "news_count" in daily.columns
        # Weighted average: (0.3*0.9 + 0.5*0.1) / (0.9+0.1) = 0.32
        assert daily["sentiment"].iloc[0] == pytest.approx(0.32)
        assert daily["news_count"].iloc[0] == pytest.approx(2.0)

    def test_uses_av_sentiment_when_no_ticker(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2022-11-02", "2022-11-02"]),
            "av_sentiment": [0.1, 0.3],
        })
        daily = _aggregate_daily_sentiment(df, has_av=True, has_ticker=False)
        assert daily["sentiment"].iloc[0] == pytest.approx(0.2)

    def test_falls_back_to_vader_sentiment(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2022-11-02", "2022-11-02"]),
            "sentiment": [0.5, 0.7],
        })
        daily = _aggregate_daily_sentiment(df, has_av=False, has_ticker=False)
        assert daily["sentiment"].iloc[0] == pytest.approx(0.6)


class TestBuildDailySentiment:
    def test_produces_daily_aggregated_file(self, tmp_path):
        news_csv = tmp_path / "news.csv"
        news_csv.write_text(
            "Date,Headline\n"
            "2022-11-02,Apple reports great earnings\n"
            "2022-11-03,Stock drops sharply\n"
        )
        out_csv = tmp_path / "sentiment_daily.csv"
        build_daily_sentiment(str(news_csv), str(out_csv))

        result = pd.read_csv(str(out_csv))
        assert {"Date", "sentiment", "news_count", "sentiment_std"}.issubset(result.columns)
        assert len(result) == 2

    def test_uses_av_sentiment_when_present(self, tmp_path):
        news_csv = tmp_path / "news.csv"
        news_csv.write_text(
            "Date,Headline,av_sentiment\n"
            "2022-11-02,Apple earnings,0.3\n"
            "2022-11-02,More news,0.5\n"
            "2022-11-03,Neutral day,0.0\n"
        )
        out_csv = tmp_path / "sentiment_daily.csv"
        build_daily_sentiment(str(news_csv), str(out_csv))

        result = pd.read_csv(str(out_csv))
        assert len(result) == 2
        # Nov 2 should be mean of 0.3 and 0.5 = 0.4
        nov2 = result[result["Date"].str.contains("2022-11-02")]
        assert nov2["sentiment"].iloc[0] == pytest.approx(0.4)

    def test_aggregates_multiple_articles_per_day(self, tmp_path):
        news_csv = tmp_path / "news.csv"
        news_csv.write_text(
            "Date,Headline\n"
            "2022-11-02,Great earnings beat\n"
            "2022-11-02,Stock drops sharply\n"
            "2022-11-03,Neutral day\n"
        )
        out_csv = tmp_path / "sentiment_daily.csv"
        build_daily_sentiment(str(news_csv), str(out_csv))

        result = pd.read_csv(str(out_csv))
        assert len(result) == 2

    def test_raises_on_empty_csv(self, tmp_path):
        news_csv = tmp_path / "empty.csv"
        news_csv.write_text("Date,Headline\n")
        out_csv = tmp_path / "sentiment_daily.csv"

        with pytest.raises(ValueError, match="Empty CSV"):
            build_daily_sentiment(str(news_csv), str(out_csv))
