"""Tests for prediction_validation.py range parsing and sentiment helpers."""

import pytest
from unittest.mock import patch, MagicMock

from prediction_validation import parse_eval_range, _ensure_sentiment_data


class TestParseEvalRange:
    """Tests for parse_eval_range()."""

    def test_days(self):
        result = parse_eval_range("30d", "2022-11-01")
        assert result == "2022-12-02"

    def test_weeks(self):
        result = parse_eval_range("4w", "2022-11-01")
        assert result == "2022-11-30"

    def test_months(self):
        result = parse_eval_range("3m", "2022-11-01")
        assert result == "2023-02-02"

    def test_single_month(self):
        result = parse_eval_range("1m", "2022-01-31")
        assert result == "2022-03-01"

    def test_absolute_date(self):
        result = parse_eval_range("2023-06-15", "2022-11-01")
        assert result == "2023-06-15"

    def test_uppercase_unit(self):
        result = parse_eval_range("30D", "2022-11-01")
        assert result == "2022-12-02"

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid --range"):
            parse_eval_range("30x", "2022-11-01")

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid --range"):
            parse_eval_range("not-a-date", "2022-11-01")

    def test_zero_days(self):
        result = parse_eval_range("0d", "2022-11-01")
        assert result == "2022-11-02"


class TestEnsureSentimentData:
    """Tests for _ensure_sentiment_data()."""

    @patch("prediction_validation.evaluate_and_save_sentiment")
    @patch("prediction_validation.fetch_news_for_period")
    def test_happy_path_uses_prediction_window(self, mock_fetch, mock_eval):
        """fetch and evaluate are called with the prediction window dates, not training dates."""
        mock_fetch.return_value = "data/AMZN_News_raw_20260102_20260302.csv"
        mock_eval.return_value = "data/prediction_sentiment_AMZN_2026-01-02_2026-03-02.csv"

        result = _ensure_sentiment_data("AMZN", "2026-01-02", "2026-03-02")

        mock_fetch.assert_called_once_with("AMZN", "2026-01-02", "2026-03-02")
        mock_eval.assert_called_once_with(
            "data/AMZN_News_raw_20260102_20260302.csv",
            "AMZN", "2026-01-02", "2026-03-02",
            prefix="prediction",
        )
        assert result == "data/prediction_sentiment_AMZN_2026-01-02_2026-03-02.csv"

    @patch("prediction_validation.fetch_news_for_period")
    def test_fetch_failure_returns_none(self, mock_fetch):
        """Returns None gracefully when fetch_news_for_period raises."""
        mock_fetch.side_effect = RuntimeError("API error")

        result = _ensure_sentiment_data("AMZN", "2026-01-02", "2026-03-02")

        assert result is None

    @patch("prediction_validation.evaluate_and_save_sentiment")
    @patch("prediction_validation.fetch_news_for_period")
    def test_sentiment_failure_returns_none(self, mock_fetch, mock_eval):
        """Returns None gracefully when evaluate_and_save_sentiment raises."""
        mock_fetch.return_value = "data/AMZN_News_raw_20260102_20260302.csv"
        mock_eval.side_effect = ValueError("bad CSV")

        result = _ensure_sentiment_data("AMZN", "2026-01-02", "2026-03-02")

        assert result is None
