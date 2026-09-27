"""Tests for the send-time offer response model (src/models/offer_response.py)."""

import numpy as np
import pandas as pd
import pytest

from src.models.offer_response import (HONEST_COLS, add_send_time_features,
                                       build_send_table)

BOGO = 'bogo1'      # 5-day window (120 h)
INFO = 'info1'


@pytest.fixture
def portfolio():
    return pd.DataFrame({
        'id': [BOGO, INFO],
        'offer_type': ['bogo', 'informational'],
        'difficulty': [10, 0],
        'reward': [10, 0],
        'duration': [5, 3],
        'channels': [['email', 'mobile'], ['email']],
    })


@pytest.fixture
def profile():
    return pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'age': [30, 118, 50],
        'gender': ['F', None, 'M'],
        'income': [60000.0, np.nan, 80000.0],
        'became_member_on': [20170101, 20180101, 20160601],
    })


def _events(rows):
    df = pd.DataFrame(rows, columns=['person', 'event', 'time', 'offer_id', 'transaction_amount'])
    return df


@pytest.fixture
def transcript():
    return _events([
        # a: received at 0, completed at 100 (inside 0..120) -> completed
        ('a', 'offer received', 0, BOGO, None),
        ('a', 'transaction', 50, None, 12.0),
        ('a', 'offer completed', 100, BOGO, None),
        # a: received again at 168, a transaction at the same hour, nothing completed -> 0
        ('a', 'offer received', 168, BOGO, None),
        ('a', 'transaction', 168, None, 3.0),
        # b: received at 0, completed at 200 (after the window closed) -> 0
        ('b', 'offer received', 0, BOGO, None),
        ('b', 'offer completed', 200, BOGO, None),
        # b: informational offers are out of scope
        ('b', 'offer received', 168, INFO, None),
        # c: received at 0 and again at 48, one completion at 60 -> credited to the 48 send
        ('c', 'offer received', 0, BOGO, None),
        ('c', 'offer received', 48, BOGO, None),
        ('c', 'offer completed', 60, BOGO, None),
        # c: received at 600, window ends at 720 > 714 -> censored and dropped
        ('c', 'offer received', 600, BOGO, None),
        ('c', 'transaction', 714, None, 1.0),
    ])


def _target(sends, person, sent_at):
    row = sends[(sends['person'] == person) & (sends['sent_at'] == sent_at)]
    assert len(row) == 1
    return int(row['completed'].iloc[0])


class TestBuildSendTable:
    def test_completion_inside_window_counts(self, portfolio, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        assert _target(sends, 'a', 0) == 1

    def test_completion_after_window_does_not(self, portfolio, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        assert _target(sends, 'b', 0) == 0

    def test_repeat_send_credits_only_the_latest(self, portfolio, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        assert _target(sends, 'c', 0) == 0
        assert _target(sends, 'c', 48) == 1

    def test_second_send_without_completion_is_zero(self, portfolio, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        assert _target(sends, 'a', 168) == 0

    def test_informational_offers_excluded(self, portfolio, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        assert INFO not in set(sends['offer_id'])

    def test_censored_window_dropped(self, portfolio, transcript):
        sends, counts = build_send_table(portfolio, transcript)
        assert not ((sends['person'] == 'c') & (sends['sent_at'] == 600)).any()
        assert counts['sends_censored_dropped'] == 1
        assert counts['sends_kept'] == len(sends) == 5


class TestSendTimeFeatures:
    def test_history_is_strictly_before_the_send(self, portfolio, profile, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        df = add_send_time_features(sends, portfolio, profile, transcript)
        a168 = df[(df['person'] == 'a') & (df['sent_at'] == 168)].iloc[0]
        assert a168['prior_sends'] == 1
        assert a168['prior_completions'] == 1
        assert a168['prior_completion_rate'] == 1.0
        # The transaction at hour 168 happens at the send, so it is not known yet.
        assert a168['prior_transactions'] == 1
        assert a168['prior_spend'] == 12.0
        assert a168['hours_since_last_transaction'] == 118

    def test_first_send_has_no_history(self, portfolio, profile, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        df = add_send_time_features(sends, portfolio, profile, transcript)
        a0 = df[(df['person'] == 'a') & (df['sent_at'] == 0)].iloc[0]
        assert a0['prior_sends'] == 0
        assert np.isnan(a0['prior_completion_rate'])
        assert np.isnan(a0['hours_since_last_transaction'])

    def test_age_sentinel_becomes_missing(self, portfolio, profile, transcript):
        sends, _ = build_send_table(portfolio, transcript)
        df = add_send_time_features(sends, portfolio, profile, transcript)
        b0 = df[df['person'] == 'b'].iloc[0]
        assert np.isnan(b0['age']) and b0['age_missing'] == 1

    def test_no_feature_sees_the_future(self, portfolio, profile, transcript):
        """Deleting every event at or after a send must not change its features."""
        sends, _ = build_send_table(portfolio, transcript)
        full = add_send_time_features(sends, portfolio, profile, transcript).set_index('send_id')
        for _, send in sends.iterrows():
            past = transcript[(transcript['time'] < send['sent_at'])
                              | ((transcript['event'] == 'offer received')
                                 & (transcript['person'] == send['person'])
                                 & (transcript['offer_id'] == send['offer_id'])
                                 & (transcript['time'] == send['sent_at']))]
            one = sends[sends['send_id'] == send['send_id']]
            truncated = add_send_time_features(one, portfolio, profile, past).set_index('send_id')
            pd.testing.assert_series_equal(
                full.loc[send['send_id'], HONEST_COLS].astype(float),
                truncated.loc[send['send_id'], HONEST_COLS].astype(float),
                check_names=False)
