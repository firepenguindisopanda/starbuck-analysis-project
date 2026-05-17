"""Tests for the data ingestion and validation module (src/data/load_data.py)."""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.load_data import (
    load_json_to_dataframe,
    validate_portfolio_schema,
    validate_profile_schema,
    validate_transcript_schema,
    load_all_datasets,
    generate_data_quality_report,
)


class TestLoadJsonToDataFrame:
    def test_load_portfolio_returns_dataframe(self, portfolio_json):
        """Verify JSONL portfolio is loaded as a DataFrame."""
        path = Path(__file__).resolve().parent.parent / "portfolio.json"
        df = load_json_to_dataframe(str(path))
        assert isinstance(df, pd.DataFrame)

    def test_load_portfolio_has_correct_shape(self, portfolio_json):
        path = Path(__file__).resolve().parent.parent / "portfolio.json"
        df = load_json_to_dataframe(str(path))
        assert len(df) == len(portfolio_json), f"Expected {len(portfolio_json)} rows"
        # At minimum: id, offer_type, difficulty, reward, duration, channels
        assert df.shape[1] >= 6

    def test_load_profile_returns_dataframe(self):
        path = Path(__file__).resolve().parent.parent / "profile.json"
        df = load_json_to_dataframe(str(path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 17000

    def test_load_transcript_returns_dataframe(self):
        path = Path(__file__).resolve().parent.parent / "transcript.json"
        df = load_json_to_dataframe(str(path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 306534

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_json_to_dataframe("/nonexistent/path.json")


class TestValidatePortfolioSchema:
    def test_valid_portfolio_passes(self, portfolio_df):
        result = validate_portfolio_schema(portfolio_df)
        assert result["valid_schema"] is True
        assert len(result["issues"]) == 0

    def test_required_columns_present(self, portfolio_df):
        result = validate_portfolio_schema(portfolio_df)
        for col in ["id", "offer_type", "difficulty", "reward", "duration", "channels"]:
            assert col in result["columns"]

    def test_offer_types_are_valid(self, portfolio_df):
        result = validate_portfolio_schema(portfolio_df)
        types = result["offer_types"]
        assert types is not None
        for t in types:
            assert t in ["bogo", "discount", "informational"]

    def test_offer_type_normalized_added(self, portfolio_df):
        df = portfolio_df.copy()
        validate_portfolio_schema(df)
        assert "offer_type_normalized" in df.columns
        assert df["offer_type_normalized"].iloc[0] == df["offer_type"].iloc[0].upper()

    def test_negative_difficulty_detected(self):
        bad = pd.DataFrame({
            "id": ["1"], "offer_type": ["bogo"], "difficulty": [-5],
            "reward": [5], "duration": [7], "channels": [["email"]],
        })
        result = validate_portfolio_schema(bad)
        issues = [i for i in result["issues"] if "negative" in i.lower()]
        assert len(issues) > 0


class TestValidateProfileSchema:
    def test_valid_profile_passes(self, profile_df):
        result = validate_profile_schema(profile_df)
        assert result["valid_schema"] is True

    def test_required_columns_present(self, profile_df):
        result = validate_profile_schema(profile_df)
        for col in ["age", "became_member_on", "gender", "id", "income"]:
            assert col in result["columns"]

    def test_detects_age_118_sentinel(self, profile_df):
        result = validate_profile_schema(profile_df)
        assert "age_118_count" in result
        assert result["age_118_count"] > 0

    def test_detects_missing_gender(self, profile_df):
        result = validate_profile_schema(profile_df)
        assert result["missing_gender_count"] > 0

    def test_detects_missing_income(self, profile_df):
        result = validate_profile_schema(profile_df)
        assert result["missing_income_count"] > 0

    def test_age_missing_flag_added(self):
        df = pd.DataFrame({
            "id": ["a"], "age": [118], "gender": ["M"],
            "income": [50000], "became_member_on": [20180101],
        })
        validate_profile_schema(df)
        assert "age_missing" in df.columns
        assert df["age_missing"].iloc[0] == 1


class TestValidateTranscriptSchema:
    def test_valid_transcript_passes(self, transcript_df):
        result = validate_transcript_schema(transcript_df)
        assert result["valid_schema"] is True

    def test_required_columns_present(self, transcript_df):
        result = validate_transcript_schema(transcript_df)
        for col in ["event", "person", "time", "value"]:
            assert col in result["columns"]

    def test_event_types_are_valid(self, transcript_df):
        result = validate_transcript_schema(transcript_df)
        events = result["event_types"]
        assert events is not None
        for e in events:
            assert e in ["transaction", "offer received", "offer viewed", "offer completed"]

    def test_time_range_is_reasonable(self, transcript_df):
        result = validate_transcript_schema(transcript_df)
        t = result["time_range"]
        assert t["min"] >= 0
        assert t["max"] > 0
        assert t["duration_days"] == 29  # known: 714 hours ~ 29.75 days


class TestLoadAllDatasets:
    def test_returns_tuple_of_three(self, data_dir):
        p, pr, t = load_all_datasets(str(data_dir))
        assert isinstance(p, pd.DataFrame)
        assert isinstance(pr, pd.DataFrame)
        assert isinstance(t, pd.DataFrame)

    def test_correct_row_counts(self, data_dir):
        p, pr, t = load_all_datasets(str(data_dir))
        assert len(p) == 10
        assert len(pr) == 17000
        assert len(t) == 306534


class TestGenerateDataQualityReport:
    def test_returns_dict_with_expected_keys(self, portfolio_df, profile_df, transcript_df):
        report = generate_data_quality_report(portfolio_df, profile_df, transcript_df)
        for key in ["portfolio_validation", "profile_validation", "transcript_validation", "summary"]:
            assert key in report

    def test_summary_has_basic_counts(self, portfolio_df, profile_df, transcript_df):
        report = generate_data_quality_report(portfolio_df, profile_df, transcript_df)
        s = report["summary"]
        assert s["total_offers"] == 10
        assert s["total_customers"] == 17000
        assert s["total_events"] == 306534

    def test_detects_cross_dataset_consistency(self, portfolio_df, profile_df, transcript_df):
        report = generate_data_quality_report(portfolio_df, profile_df, transcript_df)
        s = report["summary"]
        assert "customers_not_in_transcript" in s
        assert "offers_not_in_transcript" in s
