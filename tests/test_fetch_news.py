"""Tests for fetch_news module."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fetch_news import (
    _safe_float,
    default_filename,
    feed_to_simple_dataframe,
    fetch_news,
    fetch_news_chunked,
    fetch_news_for_period,
    validate_time_str,
)


class TestValidateTimeStr:
    def test_valid_format_passes(self):
        assert validate_time_str("20221102T0000") == "20221102T0000"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid time"):
            validate_time_str("2022-11-02")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid time"):
            validate_time_str("")


class TestFeedToSimpleDataframe:
    def test_converts_feed_items(self):
        feed = [
            {
                "title": "Apple earnings",
                "time_published": "20221102T093000",
                "summary": "Apple reported Q3 earnings.",
                "overall_sentiment_score": "0.15",
            },
            {
                "title": "AAPL drops",
                "time_published": "20221103T140000",
                "summary": "AAPL fell 2%.",
                "overall_sentiment_score": "-0.3",
            },
        ]
        df = feed_to_simple_dataframe(feed)
        assert len(df) == 2
        assert "Date" in df.columns
        assert "Headline" in df.columns
        assert "Summary" in df.columns
        assert "av_sentiment" in df.columns
        assert df["Date"].iloc[0] == "2022-11-02"
        assert df["av_sentiment"].iloc[0] == pytest.approx(0.15)

    def test_extracts_ticker_sentiment(self):
        feed = [
            {
                "title": "Apple earnings",
                "time_published": "20221102T093000",
                "overall_sentiment_score": "0.15",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.9",
                        "ticker_sentiment_score": "0.25",
                    },
                    {
                        "ticker": "MSFT",
                        "relevance_score": "0.1",
                        "ticker_sentiment_score": "-0.1",
                    },
                ],
            },
        ]
        df = feed_to_simple_dataframe(feed, ticker="AAPL")
        assert len(df) == 1
        assert df["ticker_relevance"].iloc[0] == pytest.approx(0.9)
        assert df["ticker_sentiment"].iloc[0] == pytest.approx(0.25)

    def test_ticker_sentiment_nan_when_no_match(self):
        feed = [
            {
                "title": "Something",
                "time_published": "20221102T093000",
                "overall_sentiment_score": "0.1",
                "ticker_sentiment": [
                    {"ticker": "MSFT", "relevance_score": "0.9", "ticker_sentiment_score": "0.5"},
                ],
            },
        ]
        df = feed_to_simple_dataframe(feed, ticker="AAPL")
        assert np.isnan(df["ticker_relevance"].iloc[0])
        assert np.isnan(df["ticker_sentiment"].iloc[0])

    def test_handles_empty_feed(self):
        df = feed_to_simple_dataframe([])
        assert df.empty

    def test_drops_rows_with_missing_date(self):
        feed = [
            {"title": "Article", "time_published": None},
            {"title": "Good article", "time_published": "20221102T093000"},
        ]
        df = feed_to_simple_dataframe(feed)
        assert len(df) == 1

    def test_drops_rows_with_missing_headline(self):
        feed = [
            {"title": None, "time_published": "20221102T093000"},
            {"title": "Good article", "time_published": "20221103T093000"},
        ]
        df = feed_to_simple_dataframe(feed)
        assert len(df) == 1

    def test_deduplicates(self):
        feed = [
            {"title": "Same article", "time_published": "20221102T093000"},
            {"title": "Same article", "time_published": "20221102T093000"},
        ]
        df = feed_to_simple_dataframe(feed)
        assert len(df) == 1


class TestDefaultFilename:
    def test_generates_expected_filename(self):
        result = default_filename("AAPL", "20221102T0000", "20221202T2359")
        assert result == "AAPL_News_AlphaVantage_20221102_20221202.csv"

    def test_handles_none_tickers(self):
        result = default_filename(None, "20221102T0000", "20221202T2359")
        assert result == "news_News_AlphaVantage_20221102_20221202.csv"

    def test_handles_none_time_range(self):
        result = default_filename("AAPL", None, None)
        assert result == "AAPL_News_AlphaVantage_start_end.csv"


class TestFetchNews:
    @patch("fetch_news.requests.get")
    def test_successful_api_call(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"feed": [{"title": "Test"}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = fetch_news(api_key="test_key", tickers="AAPL")
        assert "feed" in result

    def test_raises_on_empty_api_key(self):
        with pytest.raises(ValueError, match="API_KEY is empty"):
            fetch_news(api_key="")

    @patch("fetch_news.requests.get")
    def test_raises_on_api_error_message(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Error Message": "Invalid API call"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Alpha Vantage error"):
            fetch_news(api_key="test_key")

    @patch("fetch_news.requests.get")
    def test_raises_on_http_error(self, mock_get):
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_resp.text = "Internal Server Error"
        mock_resp.json.side_effect = Exception("no json")
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Alpha Vantage request failed"):
            fetch_news(api_key="test_key")


class TestFetchNewsChunked:
    @patch("fetch_news.fetch_news")
    def test_single_chunk_for_short_range(self, mock_fetch):
        mock_fetch.return_value = {"feed": [{"title": "A"}]}
        result = fetch_news_chunked(
            api_key="key",
            time_from="20221101T0000",
            time_to="20221115T2359",
        )
        assert len(result) == 1
        assert mock_fetch.call_count == 1

    @patch("fetch_news.fetch_news")
    def test_multiple_chunks_for_long_range(self, mock_fetch):
        mock_fetch.return_value = {"feed": [{"title": "A"}]}
        result = fetch_news_chunked(
            api_key="key",
            time_from="20220101T0000",
            time_to="20220401T2359",
        )
        # 3 full months + partial day remainder = 4 chunks
        assert mock_fetch.call_count == 4
        assert len(result) == 4

    @patch("fetch_news.fetch_news")
    def test_no_time_range_single_call(self, mock_fetch):
        mock_fetch.return_value = {"feed": [{"title": "A"}]}
        result = fetch_news_chunked(api_key="key")
        assert mock_fetch.call_count == 1


class TestSafeFloat:
    def test_valid_string(self):
        assert _safe_float("0.5") == pytest.approx(0.5)

    def test_none_returns_nan(self):
        assert np.isnan(_safe_float(None))

    def test_invalid_string_returns_nan(self):
        assert np.isnan(_safe_float("not_a_number"))

    def test_float_passthrough(self):
        assert _safe_float(0.75) == pytest.approx(0.75)


class TestFetchNewsForPeriod:
    """Tests for fetch_news_for_period (single-request, minimal CSV)."""

    @patch("fetch_news.fetch_news")
    @patch("fetch_news.API_KEY", "test_key")
    def test_single_api_call(self, mock_fetch, tmp_path):
        """Calls fetch_news once for the full range (not chunked)."""
        out_dir = str(tmp_path)
        mock_fetch.return_value = {
            "feed": [
                {"title": "New article", "time_published": "20220701T093000"},
            ]
        }

        fetch_news_for_period("AAPL", "2022-04-01", "2022-11-01", output_dir=out_dir)

        assert mock_fetch.call_count == 1
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["time_from"] == "20220401T0000"
        assert call_kwargs["time_to"] == "20221101T2359"

    @patch("fetch_news.fetch_news")
    @patch("fetch_news.API_KEY", "test_key")
    def test_complete_data_no_api_call(self, mock_fetch, tmp_path):
        """Full-range raw CSV exists -> no API call made."""
        out_dir = str(tmp_path)
        complete_df = pd.DataFrame({
            "date": ["2022-10-31"],
            "headlines": ["Last day article"],
        })
        out_filename = "AAPL_News_raw_20220401_20221101.csv"
        complete_path = os.path.join(out_dir, out_filename)
        complete_df.to_csv(complete_path, index=False)

        result = fetch_news_for_period("AAPL", "2022-04-01", "2022-11-01", output_dir=out_dir)

        mock_fetch.assert_not_called()
        assert result == complete_path

    @patch("fetch_news.fetch_news")
    @patch("fetch_news.API_KEY", "test_key")
    def test_output_columns_are_date_and_headlines(self, mock_fetch, tmp_path):
        """Output CSV has 'date' and 'headlines' columns (not Date/Headline)."""
        out_dir = str(tmp_path)
        mock_fetch.return_value = {
            "feed": [
                {"title": "Article A", "time_published": "20220615T093000"},
                {"title": "Article B", "time_published": "20220701T120000"},
            ]
        }

        out_path = fetch_news_for_period("AAPL", "2022-04-01", "2022-11-01", output_dir=out_dir)

        result_df = pd.read_csv(out_path)
        assert "date" in result_df.columns
        assert "headlines" in result_df.columns
        assert len(result_df) == 2

    @patch("fetch_news.fetch_news")
    @patch("fetch_news.API_KEY", "test_key")
    def test_output_filename_uses_raw_tag(self, mock_fetch, tmp_path):
        """Output filename contains '_News_raw_' not '_News_AlphaVantage_'."""
        out_dir = str(tmp_path)
        mock_fetch.return_value = {"feed": []}

        out_path = fetch_news_for_period("AAPL", "2022-04-01", "2022-11-01", output_dir=out_dir)

        assert "_News_raw_" in os.path.basename(out_path)
        assert "AlphaVantage" not in os.path.basename(out_path)
