"""Shared fixtures for Starbucks project tests."""

import json
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def portfolio_json(data_dir: Path) -> list[dict]:
    with open(data_dir / "portfolio.json") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def profile_json(data_dir: Path) -> list[dict]:
    with open(data_dir / "profile.json") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def transcript_json(data_dir: Path) -> list[dict]:
    with open(data_dir / "transcript.json") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def portfolio_df(portfolio_json: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(portfolio_json)


@pytest.fixture(scope="session")
def profile_df(profile_json: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(profile_json)


@pytest.fixture(scope="session")
def transcript_df(transcript_json: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(transcript_json)
