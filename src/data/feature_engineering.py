"""
Feature Engineering module for Starbucks customer segmentation project.

This module creates features for customer segmentation and predictive modeling:

1. Customer demographic features (age, income, gender, tenure)
2. Behavioral features (transaction frequency, average amount, offer response rates)
3. Offer features (encoded categorical variables, interaction terms)
4. RFM features (Recency, Frequency, Monetary with quintile-based scoring)
5. Time-decay features (recent spend patterns, offer recency)
6. Channel interaction features (per-channel engagement, total channels used)
7. Offer completion timing features (avg time to view/complete)
8. Customer lifetime value proxy
9. Missing data handling with appropriate imputation strategies

Follows Data Scientist principles: reproducible, documented, validated.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(base_path: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and perform initial processing of the three datasets.
    
    Args:
        base_path: Base directory containing the JSON files
        
    Returns:
        Tuple of processed (portfolio_df, profile_df, transcript_df)
    """
    # Load data
    portfolio = pd.read_json(f"{base_path}/portfolio.json", lines=True)
    profile = pd.read_json(f"{base_path}/profile.json", lines=True)
    transcript = pd.read_json(f"{base_path}/transcript.json", lines=True)
    
    # Add normalized offer_type (uppercase for consistency)
    portfolio['offer_type_normalized'] = portfolio['offer_type'].str.upper()
    
    # Process profile: handle missing data sentinel values
    profile['age_missing'] = (profile['age'] == 118).astype(int)
    profile['age_clean'] = profile['age'].replace(118, np.nan)
    
    # Convert became_member_on to datetime and calculate tenure
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
    # Assume data ends on 2018-07-26 (max time in transcript is 714 hours ~ 29.75 days)
    profile['tenure_days'] = (pd.Timestamp('2018-07-26') - profile['became_member_on']).dt.days
    profile['tenure_months'] = profile['tenure_days'] / 30.44
    
    # Extract offer_id and transaction amounts from transcript
    transcript['offer_id'] = transcript['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    transcript['transaction_amount'] = transcript['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) and 'amount' in x else None
    )
    
    return portfolio, profile, transcript


def create_customer_demographic_features(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer demographic features from profile data.
    
    Args:
        profile: Processed profile DataFrame
        
    Returns:
        DataFrame with customer demographic features
    """
    print("\n" + "="*60)
    print("CREATING CUSTOMER DEMOGRAPHIC FEATURES")
    print("="*60)
    
    # Start with customer IDs
    features = profile[['id']].copy()
    
    # Age features (handle missing with flag)
    features['age'] = profile['age_clean']
    features['age_missing'] = profile['age_missing']
    features['age_imputed'] = features['age'].fillna(features['age'].median())
    
    # Age bins for categorical analysis
    features['age_bin'] = pd.cut(
        features['age_imputed'], 
        bins=[0, 25, 35, 45, 55, 65, 100], 
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    ).astype(str)
    
    # Income features (handle missing with flag)
    features['income'] = profile['income']
    features['income_missing'] = profile['income'].isna().astype(int)
    features['income_imputed'] = features['income'].fillna(features['income'].median())
    
    # Income bins
    features['income_bin'] = pd.cut(
        features['income_imputed'],
        bins=[0, 40000, 60000, 80000, 100000, 200000],
        labels=['<40K', '40-60K', '60-80K', '80-100K', '100K+']
    ).astype(str)
    
    # Gender features (handle missing as separate category)
    features['gender'] = profile['gender'].fillna('Unknown')
    features['gender_M'] = (features['gender'] == 'M').astype(int)
    features['gender_F'] = (features['gender'] == 'F').astype(int)
    features['gender_O'] = (features['gender'] == 'O').astype(int)
    features['gender_Unknown'] = (features['gender'] == 'Unknown').astype(int)
    
    # Tenure features
    features['tenure_days'] = profile['tenure_days']
    features['tenure_months'] = profile['tenure_months']
    features['tenure_bin'] = pd.cut(
        features['tenure_months'],
        bins=[0, 6, 12, 24, 60, 1000],
        labels=['0-6M', '6-12M', '1-2Y', '2-5Y', '5Y+']
    ).astype(str)
    
    # Membership year (seasonality feature)
    features['member_year'] = profile['became_member_on'].dt.year
    features['member_month'] = profile['became_member_on'].dt.month
    
    print(f" Created {len(features.columns)-1} demographic features for {len(features)} customers")
    
    return features


def create_customer_behavioral_features(profile: pd.DataFrame, transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer behavioral features from transaction and offer event data.
    
    Args:
        profile: Processed profile DataFrame
        transcript: Processed transcript DataFrame
        
    Returns:
        DataFrame with customer behavioral features
    """
    print("\n" + "="*60)
    print("CREATING CUSTOMER BEHAVIORAL FEATURES")
    print("="*60)
    
    # Start with customer IDs
    features = profile[['id']].copy()
    
    # ---- Transaction Behavior ----
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['transaction_amount']
    
    # Transaction stats per customer
    trans_stats = transactions.groupby('person').agg({
        'amount': ['count', 'sum', 'mean', 'std', 'min', 'max']
    }).reset_index()
    trans_stats.columns = ['customer_id', 'trans_count', 'trans_total', 'trans_avg', 'trans_std', 'trans_min', 'trans_max']
    trans_stats['trans_std'] = trans_stats['trans_std'].fillna(0)  # Fill NaN std for single transactions
    
    # Merge with features
    features = features.merge(trans_stats, left_on='id', right_on='customer_id', how='left')
    features.drop('customer_id', axis=1, inplace=True)
    
    # Fill NaN for customers with no transactions
    trans_cols = ['trans_count', 'trans_total', 'trans_avg', 'trans_std', 'trans_min', 'trans_max']
    features[trans_cols] = features[trans_cols].fillna(0)
    
    # ---- Offer Response Behavior ----
    offer_events = transcript[transcript['offer_id'].notna()].copy()
    
    # Offer received count by customer
    received = offer_events[offer_events['event'] == 'offer received'].groupby('person').size().reset_index()
    received.columns = ['customer_id', 'offers_received']
    features = features.merge(received, left_on='id', right_on='customer_id', how='left')
    features['offers_received'] = features['offers_received'].fillna(0)
    features.drop('customer_id', axis=1, inplace=True)
    
    # Offer viewed count by customer
    viewed = offer_events[offer_events['event'] == 'offer viewed'].groupby('person').size().reset_index()
    viewed.columns = ['customer_id', 'offers_viewed']
    features = features.merge(viewed, left_on='id', right_on='customer_id', how='left')
    features['offers_viewed'] = features['offers_viewed'].fillna(0)
    features.drop('customer_id', axis=1, inplace=True)
    
    # Offer completed count by customer
    completed = offer_events[offer_events['event'] == 'offer completed'].groupby('person').size().reset_index()
    completed.columns = ['customer_id', 'offers_completed']
    features = features.merge(completed, left_on='id', right_on='customer_id', how='left')
    features['offers_completed'] = features['offers_completed'].fillna(0)
    features.drop('customer_id', axis=1, inplace=True)
    
    # Derived metrics
    features['view_rate'] = np.where(
        features['offers_received'] > 0,
        features['offers_viewed'] / features['offers_received'],
        0
    )
    features['completion_rate'] = np.where(
        features['offers_received'] > 0,
        features['offers_completed'] / features['offers_received'],
        0
    )
    features['view_to_completion_rate'] = np.where(
        features['offers_viewed'] > 0,
        features['offers_completed'] / features['offers_viewed'],
        0
    )
    
    # ---- Offer Type Preferences ----
    # Merge offer type information
    offer_events_merged = offer_events.merge(
        pd.read_json('portfolio.json', lines=True)[['id', 'offer_type']], 
        left_on='offer_id', right_on='id', how='left'
    )
    
    # BOGO response rates
    bogo_events = offer_events_merged[offer_events_merged['offer_type'] == 'bogo']
    for event_type in ['offer received', 'offer viewed', 'offer completed']:
        col_name = f'bogo_{event_type.split()[1]}'
        stats = bogo_events[bogo_events['event'] == event_type].groupby('person').size().reset_index()
        stats.columns = ['customer_id', col_name]
        features = features.merge(stats, left_on='id', right_on='customer_id', how='left')
        features[col_name] = features[col_name].fillna(0)
        features.drop('customer_id', axis=1, inplace=True)
    
    # Discount response rates
    discount_events = offer_events_merged[offer_events_merged['offer_type'] == 'discount']
    for event_type in ['offer received', 'offer viewed', 'offer completed']:
        col_name = f'discount_{event_type.split()[1]}'
        stats = discount_events[discount_events['event'] == event_type].groupby('person').size().reset_index()
        stats.columns = ['customer_id', col_name]
        features = features.merge(stats, left_on='id', right_on='customer_id', how='left')
        features[col_name] = features[col_name].fillna(0)
        features.drop('customer_id', axis=1, inplace=True)
    
    # Informational offer interactions (no completion event, so just received/viewed)
    info_events = offer_events_merged[offer_events_merged['offer_type'] == 'informational']
    for event_type in ['offer received', 'offer viewed']:
        col_name = f'info_{event_type.split()[1]}'
        stats = info_events[info_events['event'] == event_type].groupby('person').size().reset_index()
        stats.columns = ['customer_id', col_name]
        features = features.merge(stats, left_on='id', right_on='customer_id', how='left')
        features[col_name] = features[col_name].fillna(0)
        features.drop('customer_id', axis=1, inplace=True)
    
    # Calculate offer-type-specific response rates
    features['bogo_view_rate'] = np.where(
        features['bogo_received'] > 0,
        features['bogo_viewed'] / features['bogo_received'],
        0
    )
    features['bogo_completion_rate'] = np.where(
        features['bogo_received'] > 0,
        features['bogo_completed'] / features['bogo_received'],
        0
    )
    features['discount_view_rate'] = np.where(
        features['discount_received'] > 0,
        features['discount_viewed'] / features['discount_received'],
        0
    )
    features['discount_completion_rate'] = np.where(
        features['discount_received'] > 0,
        features['discount_completed'] / features['discount_received'],
        0
    )
    features['info_view_rate'] = np.where(
        features['info_received'] > 0,
        features['info_viewed'] / features['info_received'],
        0
    )
    
    print(f" Created {len(features.columns)-1} behavioral features for {len(features)} customers")
    
    return features


def create_offer_features(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Create offer features for predictive modeling.
    
    Args:
        portfolio: Portfolio DataFrame
        
    Returns:
        DataFrame with offer features
    """
    print("\n" + "="*60)
    print("CREATING OFFER FEATURES")
    print("="*60)
    
    features = portfolio.copy()
    
    # One-hot encode offer_type
    features['offer_type_bogo'] = (features['offer_type'] == 'bogo').astype(int)
    features['offer_type_discount'] = (features['offer_type'] == 'discount').astype(int)
    features['offer_type_informational'] = (features['offer_type'] == 'informational').astype(int)
    
    # Channel features (explode channels list into binary columns)
    channels_exploded = features.explode('channels')
    channel_dummies = pd.get_dummies(channels_exploded['channels'], prefix='channel')
    channel_agg = channel_dummies.groupby(channels_exploded.index).max()
    
    # Merge back
    features = features.drop('channels', axis=1)
    features = features.join(channel_agg)
    
    # Fill any missing channel columns (in case some offers don't have all channels)
    for channel in ['channel_email', 'channel_mobile', 'channel_social', 'channel_web']:
        if channel not in features.columns:
            features[channel] = 0
    
    # Interaction features
    features['difficulty_x_reward'] = features['difficulty'] * features['reward']
    features['difficulty_x_duration'] = features['difficulty'] * features['duration']
    features['reward_per_day'] = features['reward'] / features['duration']
    features['difficulty_per_day'] = features['difficulty'] / features['duration']
    
    print(f" Created {len(features.columns)-6} offer features for {len(features)} offers")
    
    return features


def create_customer_offer_interaction_features(customer_features: pd.DataFrame, 
                                                offer_features: pd.DataFrame,
                                                transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-offer interaction features for predictive modeling.
    
    These features represent how a specific customer might respond to a specific offer.
    
    Args:
        customer_features: Customer demographic and behavioral features
        offer_features: Offer features
        transcript: Transcript data
        
    Returns:
        DataFrame with customer-offer interaction features
    """
    print("\n" + "="*60)
    print("CREATING CUSTOMER-OFFER INTERACTION FEATURES")
    print("="*60)
    
    # This will be used for the predictive model (Phase 5)
    # We need to create a row for each customer-offer pair
    
    # Get unique customers and offers
    customers = customer_features['id'].unique()
    offers = offer_features['id'].unique()
    
    # Create all combinations
    interactions = []
    for customer in customers:
        for offer in offers:
            interactions.append({'customer_id': customer, 'offer_id': offer})
    
    interactions = pd.DataFrame(interactions)
    
    # Merge customer features (drop id and rename to customer_id)
    customer_merge = customer_features.copy()
    customer_merge = customer_merge.rename(columns={'id': 'customer_id'})
    interactions = interactions.merge(customer_merge, on='customer_id', how='left')
    
    # Merge offer features (drop id and rename to offer_id)
    offer_merge = offer_features.copy()
    offer_merge = offer_merge.rename(columns={'id': 'offer_id'})
    interactions = interactions.merge(offer_merge, on='offer_id', how='left')
    
    # Add target variable: did the customer complete this offer?
    # Get completed offers from transcript
    completed = transcript[
        (transcript['event'] == 'offer completed') & 
        (transcript['offer_id'].notna())
    ][['person', 'offer_id']].copy()
    completed['completed'] = 1
    
    interactions = interactions.merge(
        completed, 
        left_on=['customer_id', 'offer_id'], 
        right_on=['person', 'offer_id'], 
        how='left'
    )
    interactions['completed'] = interactions['completed'].fillna(0).astype(int)
    interactions.drop(['person'], axis=1, inplace=True)
    
    print(f" Created {len(interactions)} customer-offer interactions with {len(interactions.columns)-3} features")
    
    return interactions


def create_rfm_features(profile: pd.DataFrame, transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary) features for each customer.
    
    Args:
        profile: Processed profile DataFrame
        transcript: Processed transcript DataFrame
        
    Returns:
        DataFrame with RFM features indexed by customer id
    """
    print("\n" + "="*60)
    print("CREATING RFM FEATURES")
    print("="*60)
    
    features = profile[['id']].copy()
    DATA_END_HOURS = transcript['time'].max()
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    
    # Recency: days since last transaction (relative to data end date)
    last_tx = transactions.groupby('person')['time'].max().reset_index()
    last_tx.columns = ['id', 'last_tx_time']
    features = features.merge(last_tx, on='id', how='left')
    max_recency = (DATA_END_HOURS - transactions['time'].min()) / 24
    features['recency_days'] = (DATA_END_HOURS - features['last_tx_time']) / 24
    features['recency_days'] = features['recency_days'].fillna(max_recency)
    features.drop('last_tx_time', axis=1, inplace=True)
    
    # Frequency: number of transactions
    freq = transactions.groupby('person').size().reset_index()
    freq.columns = ['id', 'frequency']
    features = features.merge(freq, on='id', how='left')
    features['frequency'] = features['frequency'].fillna(0)
    
    # Monetary: total spend
    monetary = transactions.groupby('person')['transaction_amount'].sum().reset_index()
    monetary.columns = ['id', 'monetary']
    features = features.merge(monetary, on='id', how='left')
    features['monetary'] = features['monetary'].fillna(0)
    
    # RFM Score: quintile-based (1-5 scale for each dimension)
    # Recency: lower recency_days = more recent = higher score
    features['R_quintile'] = pd.qcut(
        features['recency_days'].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    # Frequency: higher = better
    features['F_quintile'] = pd.qcut(
        features['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    # Monetary: higher = better
    features['M_quintile'] = pd.qcut(
        features['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    
    features['rfm_score'] = features['R_quintile'] + features['F_quintile'] + features['M_quintile']
    features.drop(['R_quintile', 'F_quintile', 'M_quintile'], axis=1, inplace=True)
    
    print(f" Created {len(features.columns)-1} RFM features for {len(features)} customers")
    
    return features


def create_time_decay_features(profile: pd.DataFrame, transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-decay features measuring recent activity patterns.
    
    Args:
        profile: Processed profile DataFrame
        transcript: Processed transcript DataFrame
        
    Returns:
        DataFrame with time-decay features indexed by customer id
    """
    print("\n" + "="*60)
    print("CREATING TIME-DECAY FEATURES")
    print("="*60)
    
    features = profile[['id']].copy()
    DATA_END_HOURS = transcript['time'].max()
    CUTOFF_7D = DATA_END_HOURS - 7 * 24
    CUTOFF_14D = DATA_END_HOURS - 14 * 24
    
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    
    # Spend last 7 days
    recent_7d = transactions[transactions['time'] >= CUTOFF_7D].groupby('person')['transaction_amount'].sum().reset_index()
    recent_7d.columns = ['id', 'spend_last_7d']
    features = features.merge(recent_7d, on='id', how='left')
    features['spend_last_7d'] = features['spend_last_7d'].fillna(0)
    
    # Spend last 14 days
    recent_14d = transactions[transactions['time'] >= CUTOFF_14D].groupby('person')['transaction_amount'].sum().reset_index()
    recent_14d.columns = ['id', 'spend_last_14d']
    features = features.merge(recent_14d, on='id', how='left')
    features['spend_last_14d'] = features['spend_last_14d'].fillna(0)
    
    # Spend trend: ratio of recent average daily spend to overall average transaction
    avg_trans = transactions.groupby('person')['transaction_amount'].mean().reset_index()
    avg_trans.columns = ['id', '_avg_trans']
    features = features.merge(avg_trans, on='id', how='left')
    features['spend_trend'] = np.where(
        features['_avg_trans'] > 0,
        features['spend_last_14d'] / 14 / features['_avg_trans'],
        0
    )
    features.drop('_avg_trans', axis=1, inplace=True)
    
    # Offer recency: days since last offer received
    received = transcript[transcript['event'] == 'offer received'].copy()
    last_offer = received.groupby('person')['time'].max().reset_index()
    last_offer.columns = ['id', 'last_offer_time']
    features = features.merge(last_offer, on='id', how='left')
    max_offer_recency = (DATA_END_HOURS - received['time'].min()) / 24
    features['offer_recency_days'] = (DATA_END_HOURS - features['last_offer_time']) / 24
    features['offer_recency_days'] = features['offer_recency_days'].fillna(max_offer_recency)
    features.drop('last_offer_time', axis=1, inplace=True)
    
    print(f" Created {len(features.columns)-1} time-decay features for {len(features)} customers")
    
    return features


def create_channel_interaction_features(transcript: pd.DataFrame,
                                        portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Create channel interaction features for customer-offer pairs.
    
    For each customer-offer pair where the offer was viewed, flags which
    channels the offer was distributed through. Also computes total distinct
    channels the customer engages with at the customer level.
    
    Args:
        transcript: Processed transcript DataFrame
        portfolio: Portfolio DataFrame
        
    Returns:
        Tuple of (customer_level_features, interaction_level_features)
    """
    print("\n" + "="*60)
    print("CREATING CHANNEL INTERACTION FEATURES")
    print("="*60)
    
    all_channels = set()
    for channels in portfolio['channels']:
        all_channels.update(channels)
    all_channels = sorted(all_channels)
    
    # Build offer-channel mapping
    offer_channels = portfolio[['id', 'channels']].copy()
    for channel in all_channels:
        offer_channels[f'channel_{channel}'] = offer_channels['channels'].apply(
            lambda x, ch=channel: 1 if ch in x else 0
        )
    offer_channels.drop('channels', axis=1, inplace=True)
    offer_channels.rename(columns={'id': 'offer_id'}, inplace=True)
    
    # Customer-level: total_channels_used
    received = transcript[transcript['event'] == 'offer received'][['person', 'offer_id']].drop_duplicates()
    received = received.merge(offer_channels, on='offer_id', how='left')
    channel_cols = [f'channel_{c}' for c in all_channels]
    customer_channels = received.groupby('person')[channel_cols].max().reset_index()
    customer_channels['total_channels_used'] = customer_channels[channel_cols].sum(axis=1)
    customer_channels.rename(columns={'person': 'id'}, inplace=True)
    customer_level = customer_channels[['id', 'total_channels_used']].copy()
    
    # Interaction-level: viewed_via_<channel> flags
    viewed = transcript[transcript['event'] == 'offer viewed'][['person', 'offer_id']].drop_duplicates()
    viewed_with_channels = viewed.merge(offer_channels, on='offer_id', how='left')
    for channel in all_channels:
        viewed_with_channels.rename(
            columns={f'channel_{channel}': f'viewed_via_{channel}'},
            inplace=True
        )
    viewed_with_channels.rename(columns={'person': 'customer_id'}, inplace=True)
    
    # Add total_channels_used to interaction level
    viewed_with_channels = viewed_with_channels.merge(
        customer_level, left_on='customer_id', right_on='id', how='left'
    )
    viewed_with_channels.drop('id', axis=1, inplace=True)
    
    print(f" Created {len(customer_level.columns)-1} customer-level and "
          f"{len(viewed_with_channels.columns)-2} interaction-level channel features")
    
    return customer_level, viewed_with_channels


def create_offer_timing_features(profile: pd.DataFrame, transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Create offer completion and viewing timing features per customer.
    
    Args:
        profile: Processed profile DataFrame
        transcript: Processed transcript DataFrame
        
    Returns:
        DataFrame with timing features indexed by customer id
    """
    print("\n" + "="*60)
    print("CREATING OFFER TIMING FEATURES")
    print("="*60)
    
    features = profile[['id']].copy()
    offer_events = transcript[transcript['offer_id'].notna()].copy()
    
    # Time to view: hours between receiving and viewing
    received_times = offer_events[offer_events['event'] == 'offer received'][
        ['person', 'offer_id', 'time']
    ].copy()
    received_times.rename(columns={'time': 'received_time'}, inplace=True)
    viewed_times = offer_events[offer_events['event'] == 'offer viewed'][
        ['person', 'offer_id', 'time']
    ].copy()
    viewed_times.rename(columns={'time': 'viewed_time'}, inplace=True)
    
    # Use first received and first viewed per (person, offer_id) for clean pairing
    received_first = received_times.groupby(['person', 'offer_id'])['received_time'].min().reset_index()
    viewed_first = viewed_times.groupby(['person', 'offer_id'])['viewed_time'].min().reset_index()
    view_diff = received_first.merge(viewed_first, on=['person', 'offer_id'], how='inner')
    view_diff = view_diff[view_diff['viewed_time'] >= view_diff['received_time']]
    view_diff['hours_to_view'] = view_diff['viewed_time'] - view_diff['received_time']
    
    avg_view = view_diff.groupby('person')['hours_to_view'].mean().reset_index()
    avg_view.columns = ['id', 'avg_time_to_view']
    features = features.merge(avg_view, on='id', how='left')
    features['avg_time_to_view'] = features['avg_time_to_view'].fillna(0)
    
    # Time to complete: hours between receiving and completing
    completed_times = offer_events[offer_events['event'] == 'offer completed'][
        ['person', 'offer_id', 'time']
    ].copy()
    completed_times.rename(columns={'time': 'completed_time'}, inplace=True)
    
    received_for_complete = received_times.groupby(['person', 'offer_id'])['received_time'].min().reset_index()
    completed_first = completed_times.groupby(['person', 'offer_id'])['completed_time'].min().reset_index()
    complete_diff = received_for_complete.merge(completed_first, on=['person', 'offer_id'], how='inner')
    complete_diff = complete_diff[complete_diff['completed_time'] >= complete_diff['received_time']]
    complete_diff['hours_to_complete'] = complete_diff['completed_time'] - complete_diff['received_time']
    
    avg_complete = complete_diff.groupby('person')['hours_to_complete'].mean().reset_index()
    avg_complete.columns = ['id', 'avg_time_to_complete']
    features = features.merge(avg_complete, on='id', how='left')
    features['avg_time_to_complete'] = features['avg_time_to_complete'].fillna(0)
    
    print(f" Created {len(features.columns)-1} offer timing features for {len(features)} customers")
    
    return features


def create_clv_feature(customer_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer lifetime value proxy feature.
    
    CLV proxy = trans_total * (1 + completion_rate)
    
    Args:
        customer_features: Merged customer features DataFrame (must contain
            trans_total and completion_rate columns)
            
    Returns:
        DataFrame with clv_proxy feature indexed by customer id
    """
    print("\n" + "="*60)
    print("CREATING CLV PROXY FEATURE")
    print("="*60)
    
    features = customer_features[['id']].copy()
    features['clv_proxy'] = customer_features['trans_total'] * (1 + customer_features['completion_rate'])
    
    print(f" Created 1 CLV proxy feature for {len(features)} customers")
    
    return features


def normalize_features(features: pd.DataFrame, columns_to_normalize: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize specified features using StandardScaler.
    
    Args:
        features: DataFrame with features
        columns_to_normalize: List of column names to normalize
        
    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    scaler = StandardScaler()
    features_norm = features.copy()
    
    # Only normalize columns that exist and are numeric
    cols_to_norm = [col for col in columns_to_normalize if col in features.columns]
    
    if cols_to_norm:
        features_norm[cols_to_norm] = scaler.fit_transform(features_norm[cols_to_norm])
    
    return features_norm, scaler


def save_features(customer_features: pd.DataFrame, offer_features: pd.DataFrame,
                   interactions: pd.DataFrame,
                   rfm_features: pd.DataFrame = None,
                   time_decay_features: pd.DataFrame = None,
                   channel_customer_features: pd.DataFrame = None,
                   channel_interaction_features: pd.DataFrame = None,
                   offer_timing_features: pd.DataFrame = None,
                   base_path: str = '.') -> None:
    """
    Save engineered features to disk.
    
    Args:
        customer_features: Customer features DataFrame
        offer_features: Offer features DataFrame
        interactions: Customer-offer interaction features DataFrame
        rfm_features: RFM features DataFrame (optional)
        time_decay_features: Time-decay features DataFrame (optional)
        channel_customer_features: Customer-level channel features DataFrame (optional)
        channel_interaction_features: Customer-offer channel features DataFrame (optional)
        offer_timing_features: Offer timing features DataFrame (optional)
        base_path: Base directory for saving
    """
    output_dir = Path(base_path) / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    customer_features.to_csv(output_dir / 'customer_features.csv', index=False)
    offer_features.to_csv(output_dir / 'offer_features.csv', index=False)
    interactions.to_csv(output_dir / 'interaction_features.csv', index=False)
    
    if rfm_features is not None:
        rfm_features.to_csv(output_dir / 'rfm_features.csv', index=False)
    if time_decay_features is not None:
        time_decay_features.to_csv(output_dir / 'time_decay_features.csv', index=False)
    if channel_customer_features is not None:
        channel_customer_features.to_csv(output_dir / 'channel_customer_features.csv', index=False)
    if channel_interaction_features is not None:
        channel_interaction_features.to_csv(output_dir / 'channel_interaction_features.csv', index=False)
    if offer_timing_features is not None:
        offer_timing_features.to_csv(output_dir / 'offer_timing_features.csv', index=False)
    
    print(f"\n Features saved to {output_dir}")


def generate_feature_summary(customer_features: pd.DataFrame, offer_features: pd.DataFrame,
                              interactions: pd.DataFrame,
                              rfm_features: pd.DataFrame = None,
                              time_decay_features: pd.DataFrame = None,
                              channel_customer_features: pd.DataFrame = None,
                              channel_interaction_features: pd.DataFrame = None,
                              offer_timing_features: pd.DataFrame = None,
                              output_path: str = 'reports/feature_summary.json') -> Dict:
    """
    Generate a summary report of engineered features.
    
    Args:
        customer_features: Customer features DataFrame
        offer_features: Offer features DataFrame
        interactions: Interaction features DataFrame
        rfm_features: RFM features DataFrame (optional)
        time_decay_features: Time-decay features DataFrame (optional)
        channel_customer_features: Customer-level channel features DataFrame (optional)
        channel_interaction_features: Customer-offer channel features DataFrame (optional)
        offer_timing_features: Offer timing features DataFrame (optional)
        output_path: Path to save the summary report
        
    Returns:
        Dictionary with feature summary
    """
    summary = {
        'customer_features': {
            'count': len(customer_features),
            'feature_count': len(customer_features.columns) - 1,
            'features': [col for col in customer_features.columns if col != 'id']
        },
        'offer_features': {
            'count': len(offer_features),
            'feature_count': len(offer_features.columns) - 6,
            'features': [col for col in offer_features.columns if col not in ['id', 'reward', 'channels', 'difficulty', 'duration', 'offer_type']]
        },
        'interaction_features': {
            'count': len(interactions),
            'feature_count': len(interactions.columns) - 3,
            'target_distribution': interactions['completed'].value_counts().to_dict()
        }
    }
    
    if rfm_features is not None:
        summary['rfm_features'] = {
            'count': len(rfm_features),
            'feature_count': len(rfm_features.columns) - 1,
            'features': [col for col in rfm_features.columns if col != 'id']
        }
    if time_decay_features is not None:
        summary['time_decay_features'] = {
            'count': len(time_decay_features),
            'feature_count': len(time_decay_features.columns) - 1,
            'features': [col for col in time_decay_features.columns if col != 'id']
        }
    if channel_customer_features is not None:
        summary['channel_customer_features'] = {
            'count': len(channel_customer_features),
            'feature_count': len(channel_customer_features.columns) - 1,
            'features': [col for col in channel_customer_features.columns if col != 'id']
        }
    if channel_interaction_features is not None:
        summary['channel_interaction_features'] = {
            'count': len(channel_interaction_features),
            'feature_count': len(channel_interaction_features.columns) - 2,
            'features': [col for col in channel_interaction_features.columns
                         if col not in ['customer_id', 'offer_id']]
        }
    if offer_timing_features is not None:
        summary['offer_timing_features'] = {
            'count': len(offer_timing_features),
            'feature_count': len(offer_timing_features.columns) - 1,
            'features': [col for col in offer_timing_features.columns if col != 'id']
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n Feature summary saved to: {output_path}")
    
    return summary


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - FEATURE ENGINEERING")
    print("="*60)
    
    # Load processed data
    print("\nLoading and processing data...")
    portfolio, profile, transcript = load_processed_data()
    print(f" Loaded {len(portfolio)} offers, {len(profile)} customers, {len(transcript)} events")
    
    # Create features
    customer_demo = create_customer_demographic_features(profile)
    customer_behavior = create_customer_behavioral_features(profile, transcript)
    
    # Merge customer features
    customer_features = customer_demo.merge(customer_behavior, on='id', how='outer')
    
    # Create RFM features and merge
    rfm_features = create_rfm_features(profile, transcript)
    customer_features = customer_features.merge(rfm_features, on='id', how='left')
    
    # Create time-decay features and merge
    time_decay_features = create_time_decay_features(profile, transcript)
    customer_features = customer_features.merge(time_decay_features, on='id', how='left')
    
    # Create offer timing features and merge
    offer_timing_features = create_offer_timing_features(profile, transcript)
    customer_features = customer_features.merge(offer_timing_features, on='id', how='left')
    
    # Create CLV proxy (depends on trans_total and completion_rate from behavioral features)
    clv_features = create_clv_feature(customer_features)
    customer_features = customer_features.merge(clv_features, on='id', how='left')
    
    print(f"\n Total customer features: {len(customer_features.columns)-1}")
    
    # Create offer features
    offer_features = create_offer_features(portfolio)
    
    # Create interaction features for predictive modeling
    interactions = create_customer_offer_interaction_features(customer_features, offer_features, transcript)
    
    # Create channel interaction features
    channel_customer, channel_interaction = create_channel_interaction_features(transcript, portfolio)
    
    # Merge channel customer-level features into customer features
    customer_features = customer_features.merge(channel_customer, on='id', how='left')
    customer_features['total_channels_used'] = customer_features['total_channels_used'].fillna(0)
    
    # Merge channel interaction-level features into interactions
    interactions = interactions.merge(
        channel_interaction, on=['customer_id', 'offer_id'], how='left'
    )
    channel_view_cols = [c for c in channel_interaction.columns if c.startswith('viewed_via_')]
    for col in channel_view_cols:
        if col in interactions.columns:
            interactions[col] = interactions[col].fillna(0)
    if 'total_channels_used' in interactions.columns:
        # Already merged via customer_features into interactions, skip duplicate
        pass
    
    print(f"\n Total customer features (with channel): {len(customer_features.columns)-1}")
    print(f" Total interaction features: {len(interactions.columns)-3}")
    
    # Save features
    save_features(
        customer_features, offer_features, interactions,
        rfm_features=rfm_features,
        time_decay_features=time_decay_features,
        channel_customer_features=channel_customer,
        channel_interaction_features=channel_interaction,
        offer_timing_features=offer_timing_features
    )
    
    # Generate summary
    summary = generate_feature_summary(
        customer_features, offer_features, interactions,
        rfm_features=rfm_features,
        time_decay_features=time_decay_features,
        channel_customer_features=channel_customer,
        channel_interaction_features=channel_interaction,
        offer_timing_features=offer_timing_features
    )
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f" Customer features: {summary['customer_features']['count']} customers × {summary['customer_features']['feature_count']} features")
    print(f" Offer features: {summary['offer_features']['count']} offers × {summary['offer_features']['feature_count']} features")
    print(f" Interaction features: {summary['interaction_features']['count']} interactions")
    print(f" Target distribution: {summary['interaction_features']['target_distribution']}")
    if 'rfm_features' in summary:
        print(f" RFM features: {summary['rfm_features']['count']} customers × {summary['rfm_features']['feature_count']} features")
    if 'time_decay_features' in summary:
        print(f" Time-decay features: {summary['time_decay_features']['count']} customers × {summary['time_decay_features']['feature_count']} features")
    if 'offer_timing_features' in summary:
        print(f" Offer timing features: {summary['offer_timing_features']['count']} customers × {summary['offer_timing_features']['feature_count']} features")
    print("="*60)
