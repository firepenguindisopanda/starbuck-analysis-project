"""Tests for the feature engineering module (src/data/feature_engineering.py)."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.feature_engineering import (
    create_customer_demographic_features,
    create_offer_features,
    normalize_features,
)


class TestCreateCustomerDemographicFeatures:
    def test_returns_dataframe(self):
        profile = pd.DataFrame({
            "id": ["a", "b"],
            "age": [25, 118],
            "gender": ["M", None],
            "income": [50000.0, None],
            "became_member_on": pd.to_datetime(["20180101", "20170101"], format="%Y%m%d"),
        })
        profile["age_missing"] = (profile["age"] == 118).astype(int)
        profile["age_clean"] = profile["age"].replace(118, np.nan)
        profile["tenure_days"] = (pd.Timestamp("2018-07-26") - profile["became_member_on"]).dt.days
        profile["tenure_months"] = profile["tenure_days"] / 30.44

        result = create_customer_demographic_features(profile)
        assert isinstance(result, pd.DataFrame)
        assert "id" in result.columns
        assert len(result) == 2

    def test_creates_age_missing_flag(self):
        profile = pd.DataFrame({
            "id": ["a", "b"],
            "age": [25, 118],
            "gender": ["M", None],
            "income": [50000.0, None],
            "became_member_on": pd.to_datetime(["20180101", "20170101"], format="%Y%m%d"),
        })
        profile["age_missing"] = (profile["age"] == 118).astype(int)
        profile["age_clean"] = profile["age"].replace(118, np.nan)
        profile["tenure_days"] = (pd.Timestamp("2018-07-26") - profile["became_member_on"]).dt.days
        profile["tenure_months"] = profile["tenure_days"] / 30.44

        result = create_customer_demographic_features(profile)
        assert "age_missing" in result.columns
        assert result["age_missing"].iloc[1] == 1

    def test_creates_gender_onehot(self):
        profile = pd.DataFrame({
            "id": ["a", "b"],
            "age": [25, 30],
            "gender": ["M", "F"],
            "income": [50000.0, 60000.0],
            "became_member_on": pd.to_datetime(["20180101", "20170101"], format="%Y%m%d"),
        })
        profile["age_missing"] = 0
        profile["age_clean"] = profile["age"]
        profile["tenure_days"] = (pd.Timestamp("2018-07-26") - profile["became_member_on"]).dt.days
        profile["tenure_months"] = profile["tenure_days"] / 30.44

        result = create_customer_demographic_features(profile)
        assert "gender_M" in result.columns
        assert "gender_F" in result.columns
        assert result["gender_M"].iloc[0] == 1
        assert result["gender_F"].iloc[1] == 1


class TestCreateOfferFeatures:
    def test_returns_dataframe(self):
        portfolio = pd.DataFrame({
            "id": ["1", "2"],
            "offer_type": ["bogo", "discount"],
            "difficulty": [5, 10],
            "reward": [5, 5],
            "duration": [7, 10],
            "channels": [["email"], ["email", "mobile"]],
        })
        result = create_offer_features(portfolio)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_creates_onehot_offer_types(self):
        portfolio = pd.DataFrame({
            "id": ["1", "2", "3"],
            "offer_type": ["bogo", "discount", "informational"],
            "difficulty": [5, 10, 0],
            "reward": [5, 5, 0],
            "duration": [7, 10, 3],
            "channels": [["email"], ["email"], ["web"]],
        })
        result = create_offer_features(portfolio)
        assert "offer_type_bogo" in result.columns
        assert "offer_type_discount" in result.columns
        assert "offer_type_informational" in result.columns

    def test_creates_interaction_features(self):
        portfolio = pd.DataFrame({
            "id": ["1"],
            "offer_type": ["bogo"],
            "difficulty": [5],
            "reward": [5],
            "duration": [7],
            "channels": [["email", "mobile"]],
        })
        result = create_offer_features(portfolio)
        assert "difficulty_x_reward" in result.columns
        assert "reward_per_day" in result.columns

    def test_creates_channel_dummies(self):
        portfolio = pd.DataFrame({
            "id": ["1"],
            "offer_type": ["bogo"],
            "difficulty": [5],
            "reward": [5],
            "duration": [7],
            "channels": [["email", "mobile", "social", "web"]],
        })
        result = create_offer_features(portfolio)
        for ch in ["channel_email", "channel_mobile", "channel_social", "channel_web"]:
            assert ch in result.columns


class TestNormalizeFeatures:
    def test_returns_normalized_values(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        norm_df, scaler = normalize_features(df, ["a", "b"])
        assert abs(norm_df["a"].mean()) < 1e-10
        assert abs(norm_df["a"].std(ddof=0) - 1.0) < 1e-10

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        norm_df, scaler = normalize_features(df, ["a", "nonexistent"])
        assert "a" in norm_df.columns

    def test_returns_fitted_scaler(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        _, scaler = normalize_features(df, ["a"])
        transformed = scaler.transform([[3.0]])
        assert abs(transformed[0][0]) < 1.0
