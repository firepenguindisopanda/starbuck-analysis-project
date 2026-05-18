"""
Power BI Export Module for Starbucks Customer Segmentation Project.

This module exports all relevant data to CSV files optimized for Power BI import,
using a proper star schema design with fact and dimension tables.

Why Power BI?
- Demonstrates business intelligence skills (not just Python)
- Creates executive-ready dashboards
- Shows ability to work with enterprise BI tools
- Complements Python/ML work with visual analytics

Star Schema Design:
- Fact tables: fact_offer_interactions, fact_kpi_metrics, fact_segment_performance
- Dimension tables: dim_customers, dim_offers, dim_date
- Supporting tables: customers (legacy), offers, transactions, offer_events,
  cluster_profiles, model_performance, ate_results, recommendation_results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

CLUSTER_LABELS = {
    0: 'Unknown Demographics',
    1: 'Discount Responders',
    2: 'BOGO Responders',
    3: 'Low Engagement'
}


def export_customer_data(base_path: str = '.') -> pd.DataFrame:
    """
    Export customer master data with cluster assignments.
    
    Returns:
        Customer DataFrame ready for Power BI
    """
    print("\n" + "="*60)
    print("EXPORTING CUSTOMER DATA FOR POWER BI")
    print("="*60)
    
    # Load customer features
    customers = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_features.csv')
    
    # Load cluster assignments
    clusters = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_clusters.csv')
    
    # Merge
    customers_export = customers.merge(clusters, on='id', how='left')
    
    # Add cluster labels for readability
    cluster_labels = {
        0: 'Unknown Demographics',
        1: 'Discount Responders',
        2: 'BOGO Responders',
        3: 'Low Engagement'
    }
    customers_export['cluster_label'] = customers_export['cluster'].map(cluster_labels)
    
    # Select and rename columns for Power BI
    export_cols = {
        'id': 'customer_id',
        'age_imputed': 'age',
        'income_imputed': 'income',
        'tenure_months': 'tenure_months',
        'gender_M': 'is_male',
        'gender_F': 'is_female',
        'gender_O': 'is_other_gender',
        'gender_Unknown': 'is_unknown_gender',
        'trans_count': 'transaction_count',
        'trans_total': 'total_spend',
        'trans_avg': 'avg_transaction',
        'offers_received': 'offers_received',
        'offers_viewed': 'offers_viewed',
        'offers_completed': 'offers_completed',
        'view_rate': 'view_rate',
        'completion_rate': 'completion_rate',
        'cluster': 'cluster_id',
        'cluster_label': 'cluster_label'
    }
    
    # Only keep columns that exist
    existing_cols = {k: v for k, v in export_cols.items() if k in customers_export.columns}
    customers_export = customers_export[list(existing_cols.keys())].rename(columns=existing_cols)
    
    # Add calculated columns for Power BI
    customers_export['customer_segment'] = customers_export['cluster_label']
    customers_export['engagement_level'] = pd.cut(
        customers_export['completion_rate'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    print(f" Exported {len(customers_export)} customers with {len(customers_export.columns)} columns")
    
    return customers_export


def export_offer_data(base_path: str = '.') -> pd.DataFrame:
    """
    Export offer master data.
    
    Returns:
        Offer DataFrame ready for Power BI
    """
    print("\nExporting offer data...")
    
    # Load portfolio
    portfolio = pd.read_json(Path(base_path) / 'portfolio.json', lines=True)
    
    # Explode channels into separate rows for Power BI (creates one row per offer-channel)
    portfolio_expanded = portfolio.explode('channels')
    
    # Rename columns
    portfolio_expanded = portfolio_expanded.rename(columns={
        'id': 'offer_id',
        'offer_type': 'offer_type',
        'difficulty': 'difficulty',
        'reward': 'reward',
        'duration': 'duration_days',
        'channels': 'channel'
    })
    
    # Add derived columns
    portfolio_expanded['offer_type_upper'] = portfolio_expanded['offer_type'].str.upper()
    portfolio_expanded['reward_per_day'] = portfolio_expanded['reward'] / portfolio_expanded['duration_days']
    portfolio_expanded['difficulty_per_day'] = portfolio_expanded['difficulty'] / portfolio_expanded['duration_days']
    
    print(f" Exported {len(portfolio_expanded)} offer-channel combinations")
    
    return portfolio_expanded


def export_transaction_data(base_path: str = '.') -> pd.DataFrame:
    """
    Export transaction data for Power BI.
    
    Returns:
        Transaction DataFrame ready for Power BI
    """
    print("\nExporting transaction data...")
    
    # Load transcript
    transcript = pd.read_json(Path(base_path) / 'transcript.json', lines=True)
    
    # Filter to transactions only
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    
    # Extract amount
    transactions['amount'] = transactions['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transactions = transactions[transactions['amount'].notna()].copy()
    
    # Rename columns
    transactions = transactions.rename(columns={
        'person': 'customer_id',
        'time': 'time_hours',
        'amount': 'transaction_amount'
    })
    
    # Add derived columns
    transactions['transaction_date'] = pd.to_datetime('2018-07-26') - pd.to_timedelta(transactions['time_hours'], unit='h')
    transactions['day_of_week'] = transactions['transaction_date'].dt.day_name()
    transactions['hour_of_day'] = transactions['transaction_date'].dt.hour
    
    # Select columns
    export = transactions[['customer_id', 'transaction_amount', 'time_hours', 
                           'transaction_date', 'day_of_week', 'hour_of_day']].copy()
    
    print(f" Exported {len(export)} transactions")
    
    return export


def export_offer_events(base_path: str = '.') -> pd.DataFrame:
    """
    Export offer events (received, viewed, completed) for Power BI.
    
    Returns:
        Offer events DataFrame ready for Power BI
    """
    print("\nExporting offer events...")
    
    # Load transcript
    transcript = pd.read_json(Path(base_path) / 'transcript.json', lines=True)
    
    # Filter to offer events only
    offer_events = transcript[transcript['event'].isin(['offer received', 'offer viewed', 'offer completed'])].copy()
    
    # Extract offer_id
    offer_events['offer_id'] = offer_events['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    offer_events = offer_events[offer_events['offer_id'].notna()].copy()
    
    # Rename columns
    offer_events = offer_events.rename(columns={
        'person': 'customer_id',
        'event': 'event_type',
        'time': 'time_hours'
    })
    
    # Add derived columns
    offer_events['event_date'] = pd.to_datetime('2018-07-26') - pd.to_timedelta(offer_events['time_hours'], unit='h')
    offer_events['day_of_week'] = offer_events['event_date'].dt.day_name()
    
    # Select columns
    export = offer_events[['customer_id', 'offer_id', 'event_type', 'time_hours', 
                           'event_date', 'day_of_week']].copy()
    
    print(f" Exported {len(export)} offer events")
    
    return export


def export_cluster_profiles(base_path: str = '.') -> pd.DataFrame:
    """
    Export cluster profiles for Power BI.
    
    Returns:
        Cluster profiles DataFrame
    """
    print("\nExporting cluster profiles...")
    
    # Load cluster profiles
    profiles = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'cluster_profiles.csv')
    
    # Add cluster labels
    cluster_labels = {
        0: 'Unknown Demographics',
        1: 'Discount Responders',
        2: 'BOGO Responders',
        3: 'Low Engagement'
    }
    profiles['cluster_label'] = profiles['cluster_id'].map(cluster_labels)
    
    # Reshape for Power BI (long format)
    metrics = ['completion_rate_mean', 'view_rate_mean', 'trans_avg_mean', 
               'income_imputed_mean', 'age_imputed_mean']
    
    profiles_long = []
    for _, row in profiles.iterrows():
        for metric in metrics:
            if metric in row.index:
                profiles_long.append({
                    'cluster_id': int(row['cluster_id']),
                    'cluster_label': row['cluster_label'],
                    'metric': metric.replace('_mean', '').replace('_', ' ').title(),
                    'value': row[metric]
                })
    
    profiles_export = pd.DataFrame(profiles_long)
    
    print(f" Exported {len(profiles_export)} cluster profile metrics")
    
    return profiles_export


def export_model_performance(base_path: str = '.') -> pd.DataFrame:
    """
    Export model performance metrics for Power BI.
    
    Returns:
        Model performance DataFrame
    """
    print("\nExporting model performance metrics...")
    
    # Load model comparison
    comparison = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'model_comparison.csv')
    
    # Reshape for Power BI (long format)
    metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    
    performance_long = []
    for _, row in comparison.iterrows():
        for metric in metrics:
            if metric in row.index:
                performance_long.append({
                    'model_name': row['Model'],
                    'metric': metric,
                    'value': row[metric]
                })
    
    performance_export = pd.DataFrame(performance_long)
    
    print(f" Exported {len(performance_export)} model performance metrics")
    
    return performance_export


def export_ate_results(base_path: str = '.') -> pd.DataFrame:
    """
    Export Average Treatment Effect results for Power BI.
    
    Returns:
        ATE results DataFrame
    """
    print("\nExporting ATE results...")
    
    # Load causal report
    with open(Path(base_path) / 'reports' / 'causal_report.json', 'r') as f:
        causal_data = json.load(f)
    
    ate_results = causal_data.get('ate_results', {})
    
    # Convert to DataFrame
    ate_list = []
    for offer_type, values in ate_results.items():
        ate_list.append({
            'offer_type': offer_type.replace('_', ' ').title(),
            'control_mean': values.get('control_mean', 0),
            'treatment_mean': values.get('treatment_mean', 0),
            'ate_dollars': values.get('ate', 0),
            'ate_percent': values.get('ate_percent', 0)
        })
    
    ate_export = pd.DataFrame(ate_list)
    
    print(f" Exported {len(ate_export)} ATE results")
    
    return ate_export


def export_recommendation_results(base_path: str = '.') -> pd.DataFrame:
    """
    Export recommendation system results for Power BI.
    
    Returns:
        Recommendation results DataFrame
    """
    print("\nExporting recommendation results...")
    
    # Load recommendation results
    with open(Path(base_path) / 'data' / 'processed' / 'recommendation_results.json', 'r') as f:
        rec_data = json.load(f)
    
    # Create summary
    summary = {
        'method': ['Random Targeting', 'Rule-Based Targeting'],
        'completion_rate': [
            rec_data.get('random_completion_rate', 0),
            rec_data.get('rule_based_completion_rate', 0)
        ],
        'lift_percent': [0, rec_data.get('lift_percent', 0)]
    }
    
    rec_export = pd.DataFrame(summary)
    
    print(f" Exported recommendation system results")
    
    return rec_export


def export_fact_offer_interactions(base_path: str = '.') -> pd.DataFrame:
    """
    Create the core fact table for the star schema.

    Each row represents one offer interaction event (received, viewed, or completed)
    with proper foreign keys linking to dim_customers, dim_offers, and dim_date.

    Returns:
        Fact offer interactions DataFrame
    """
    print("\nExporting fact_offer_interactions (star schema fact table)...")

    transcript = pd.read_json(Path(base_path) / 'transcript.json', lines=True)

    offer_events = transcript[transcript['event'].isin(['offer received', 'offer viewed', 'offer completed'])].copy()

    offer_events['offer_id'] = offer_events['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    offer_events = offer_events[offer_events['offer_id'].notna()].copy()

    offer_events = offer_events.rename(columns={
        'person': 'customer_id',
        'event': 'event_type',
        'time': 'time_hours'
    })

    offer_events['event_type'] = offer_events['event_type'].str.replace('offer ', '')

    offer_events['event_date'] = pd.to_datetime('2018-07-26') - pd.to_timedelta(offer_events['time_hours'], unit='h')

    offer_events['event_date_key'] = offer_events['event_date'].dt.strftime('%Y-%m-%d')

    offer_events['day_since_start'] = offer_events['time_hours'] // 24

    fact = offer_events[['customer_id', 'offer_id', 'event_type',
                          'time_hours', 'event_date', 'event_date_key',
                          'day_since_start']].copy()

    fact = fact.drop_duplicates().reset_index(drop=True)

    print(f" Exported {len(fact)} fact offer interactions")
    return fact


def export_dim_customers(base_path: str = '.') -> pd.DataFrame:
    """
    Create the customer dimension table for the star schema.

    Rows: one per customer, with demographics, cluster info,
    engagement level, RFM segment, and CLV proxy.

    Returns:
        Customer dimension DataFrame
    """
    print("\nExporting dim_customers (star schema dimension)...")

    customers = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_features.csv')
    clusters = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_clusters.csv')

    dim = customers.merge(clusters, on='id', how='left')

    # RFM score is already in customer_features.csv from feature engineering
    # No need to merge from a separate file

    dim['cluster_label'] = dim['cluster'].map(CLUSTER_LABELS)

    dim['engagement_level'] = pd.cut(
        dim['completion_rate'],
        bins=[-0.001, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    def _rfm_segment(score):
        if pd.isna(score):
            return 'Unknown'
        if score >= 9:
            return 'Champions'
        if score >= 7:
            return 'Loyal'
        if score >= 5:
            return 'Potential'
        if score >= 3:
            return 'At Risk'
        return 'Lost'

    dim['rfm_segment'] = dim['rfm_score'].apply(_rfm_segment)

    dim = dim.rename(columns={'id': 'customer_id'})

    cols = [
        'customer_id', 'age_imputed', 'income_imputed', 'gender',
        'tenure_months', 'cluster', 'cluster_label',
        'engagement_level', 'rfm_segment', 'clv_proxy',
        'completion_rate', 'view_rate', 'trans_total', 'trans_avg', 'trans_count'
    ]
    existing = [c for c in cols if c in dim.columns]
    dim = dim[existing].copy()

    print(f" Exported {len(dim)} customer dimension rows with {len(dim.columns)} columns")
    return dim


def export_dim_offers(base_path: str = '.') -> pd.DataFrame:
    """
    Create the offer dimension table for the star schema.

    Rows: one per offer (not exploded by channel), with
    channel flags, derived metrics, and offer_type_category.

    Returns:
        Offer dimension DataFrame
    """
    print("\nExporting dim_offers (star schema dimension)...")

    portfolio = pd.read_json(Path(base_path) / 'portfolio.json', lines=True)

    dim = portfolio.rename(columns={
        'id': 'offer_id',
        'duration': 'duration_days'
    })

    channel_lists = dim['channels'].copy()
    dim['channel_email'] = channel_lists.apply(lambda x: 'email' in x if isinstance(x, list) else False)
    dim['channel_mobile'] = channel_lists.apply(lambda x: 'mobile' in x if isinstance(x, list) else False)
    dim['channel_social'] = channel_lists.apply(lambda x: 'social' in x if isinstance(x, list) else False)
    dim['channel_web'] = channel_lists.apply(lambda x: 'web' in x if isinstance(x, list) else False)

    dim['reward_per_day'] = dim['reward'] / dim['duration_days'].replace(0, np.nan)
    dim['difficulty_per_day'] = dim['difficulty'] / dim['duration_days'].replace(0, np.nan)

    dim['channel_count'] = channel_lists.apply(len)

    def _offer_category(offer_type):
        if offer_type in ('bogo', 'discount'):
            return 'Promotional'
        return 'Informational'

    dim['offer_type_category'] = dim['offer_type'].apply(_offer_category)

    dim = dim.drop(columns=['channels'])

    print(f" Exported {len(dim)} offer dimension rows with {len(dim.columns)} columns")
    return dim


def export_fact_kpi_metrics(base_path: str = '.') -> pd.DataFrame:
    """
    Create a pre-computed KPI metrics table for Power BI card visuals.

    Returns:
        KPI metrics DataFrame (one row per metric)
    """
    print("\nExporting fact_kpi_metrics (pre-computed KPIs)...")

    customers = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_features.csv')

    with open(Path(base_path) / 'data' / 'processed' / 'model_metrics.json', 'r') as f:
        model_metrics = json.load(f)

    with open(Path(base_path) / 'reports' / 'causal_report.json', 'r') as f:
        causal_data = json.load(f)

    with open(Path(base_path) / 'data' / 'processed' / 'recommendation_results.json', 'r') as f:
        rec_data = json.load(f)

    ate_results = causal_data.get('ate_results', {})

    rows = [
        {'metric_name': 'Total Customers', 'metric_value': len(customers), 'metric_category': 'Customer', 'format': 'Whole Number'},
        {'metric_name': 'Total Revenue', 'metric_value': customers['trans_total'].sum(), 'metric_category': 'Financial', 'format': 'Currency'},
        {'metric_name': 'Average Completion Rate', 'metric_value': customers['completion_rate'].mean(), 'metric_category': 'Engagement', 'format': 'Percentage'},
        {'metric_name': 'Average View Rate', 'metric_value': customers['view_rate'].mean(), 'metric_category': 'Engagement', 'format': 'Percentage'},
        {'metric_name': 'Average Transaction Amount', 'metric_value': customers['trans_avg'].mean(), 'metric_category': 'Financial', 'format': 'Currency'},
        {'metric_name': 'Model AUC-ROC', 'metric_value': model_metrics.get('auc_roc', 0), 'metric_category': 'Model', 'format': 'Decimal'},
        {'metric_name': 'Recommendation Lift %', 'metric_value': rec_data.get('lift_percent', 0), 'metric_category': 'Business', 'format': 'Percentage'},
    ]

    for offer_type in ('bogo', 'discount', 'informational'):
        key = offer_type
        if key in ate_results:
            rows.append({
                'metric_name': f'ATE {offer_type.title()}',
                'metric_value': ate_results[key].get('ate', 0),
                'metric_category': 'Causal Impact',
                'format': 'Currency'
            })

    for offer_type in ('bogo', 'discount', 'informational'):
        key = offer_type
        if key in ate_results:
            rows.append({
                'metric_name': f'ATE {offer_type.title()} %',
                'metric_value': ate_results[key].get('ate_percent', 0),
                'metric_category': 'Causal Impact',
                'format': 'Percentage'
            })

    kpi = pd.DataFrame(rows)

    print(f" Exported {len(kpi)} KPI metrics")
    return kpi


def export_fact_segment_performance(base_path: str = '.') -> pd.DataFrame:
    """
    Create pre-aggregated segment performance metrics for Power BI.

    Returns:
        Segment performance DataFrame (one row per cluster)
    """
    print("\nExporting fact_segment_performance (pre-aggregated segments)...")

    customers = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_features.csv')
    clusters = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_clusters.csv')

    dim = customers.merge(clusters, on='id', how='left')

    with open(Path(base_path) / 'data' / 'processed' / 'recommendation_results.json', 'r') as f:
        rec_data = json.load(f)

    rules = rec_data.get('rules', {})

    completion_by_type = rec_data.get('completion_by_type', {})

    seg = dim.groupby('cluster').agg(
        segment_size=('id', 'count'),
        avg_completion_rate=('completion_rate', 'mean'),
        avg_view_rate=('view_rate', 'mean'),
        avg_transaction_amount=('trans_avg', 'mean'),
        avg_income=('income_imputed', 'mean'),
        avg_age=('age_imputed', 'mean'),
        total_revenue=('trans_total', 'sum'),
    ).reset_index()

    seg['revenue_per_customer'] = seg['total_revenue'] / seg['segment_size']

    seg['cluster_label'] = seg['cluster'].map(CLUSTER_LABELS)

    def _best_offer(cluster_id):
        rule = rules.get(str(int(cluster_id)), {})
        return rule.get('primary', 'unknown')

    seg['best_offer_type'] = seg['cluster'].apply(_best_offer)

    def _best_completion(cluster_id):
        if cluster_id == 0:
            return completion_by_type.get('informational', 0)
        elif cluster_id == 1:
            return completion_by_type.get('discount', 0)
        elif cluster_id == 2:
            return completion_by_type.get('bogo', 0)
        else:
            return max(completion_by_type.values()) if completion_by_type else 0

    seg['best_offer_completion_rate'] = seg['cluster'].apply(_best_completion)

    seg = seg.rename(columns={'cluster': 'cluster_id'})

    cols = [
        'cluster_id', 'cluster_label', 'segment_size',
        'avg_completion_rate', 'avg_view_rate', 'avg_transaction_amount',
        'avg_income', 'avg_age',
        'total_revenue', 'revenue_per_customer',
        'best_offer_type', 'best_offer_completion_rate'
    ]
    seg = seg[cols]

    print(f" Exported {len(seg)} segment performance rows")
    return seg


def export_calendar_table() -> pd.DataFrame:
    """
    Export an enhanced calendar/date dimension table for Power BI time intelligence.

    Includes fiscal quarter, is_holiday_period flags, and day number since start.

    Returns:
        Calendar DataFrame with date attributes
    """
    print("\nExporting enhanced calendar table (dim_date)...")

    start_date = pd.Timestamp('2018-06-01')
    end_date = pd.Timestamp('2018-07-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    campaign_start = pd.Timestamp('2018-06-27')

    calendar = pd.DataFrame({
        'date': dates,
        'date_key': dates.strftime('%Y-%m-%d'),
        'year': dates.year,
        'quarter': dates.quarter,
        'fiscal_quarter': 'FQ' + dates.quarter.astype(str),
        'month': dates.month,
        'month_name': dates.strftime('%B'),
        'day': dates.day,
        'day_of_week': dates.dayofweek,
        'day_name': dates.strftime('%A'),
        'week_of_year': dates.isocalendar().week.astype(int),
        'is_weekend': dates.dayofweek >= 5,
        'is_weekday': dates.dayofweek < 5,
        'is_holiday_period': ((dates >= pd.Timestamp('2018-07-01')) &
                              (dates <= pd.Timestamp('2018-07-07'))),
        'day_since_start': (dates - campaign_start).days
    })

    print(f" Exported {len(calendar)} calendar dates ({start_date.date()} to {end_date.date()})")

    return calendar


def save_all_exports(base_path: str = '.') -> Tuple[Path, Dict[str, pd.DataFrame]]:
    """
    Save all exported DataFrames to CSV files for Power BI.
    
    Args:
        base_path: Base directory
    """
    print("\n" + "="*60)
    print("SAVING ALL EXPORTS FOR POWER BI")
    print("="*60)
    
    export_dir = Path(base_path) / 'powerbi_data'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Star schema tables (primary)
    star_schema_datasets = {
        'fact_offer_interactions': export_fact_offer_interactions(base_path),
        'fact_kpi_metrics': export_fact_kpi_metrics(base_path),
        'fact_segment_performance': export_fact_segment_performance(base_path),
        'dim_customers': export_dim_customers(base_path),
        'dim_offers': export_dim_offers(base_path),
        'dim_date': export_calendar_table(),
    }

    # Legacy / supporting tables (kept for backward compatibility)
    legacy_datasets = {
        'customers': export_customer_data(base_path),
        'offers': export_offer_data(base_path),
        'transactions': export_transaction_data(base_path),
        'offer_events': export_offer_events(base_path),
        'cluster_profiles': export_cluster_profiles(base_path),
        'model_performance': export_model_performance(base_path),
        'ate_results': export_ate_results(base_path),
        'recommendation_results': export_recommendation_results(base_path),
    }

    datasets = {**star_schema_datasets, **legacy_datasets}
    
    # Save to CSV
    for name, df in datasets.items():
        output_path = export_dir / f'{name}.csv'
        df.to_csv(output_path, index=False)
        print(f"   Saved: {output_path.name}")
    
    # Create a summary file
    summary = pd.DataFrame({
        'dataset': list(datasets.keys()),
        'rows': [len(df) for df in datasets.values()],
        'columns': [len(df.columns) for df in datasets.values()]
    })
    summary.to_csv(export_dir / 'export_summary.csv', index=False)
    
    print(f"\n All datasets saved to: {export_dir}")
    print(f" Total datasets: {len(datasets)}")
    print(f" Total rows: {summary['rows'].sum():,}")
    
    return export_dir, datasets


def create_powerbi_instructions(export_dir: Path, datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Create instructions for importing data into Power BI, including star schema,
    new fact/dimension tables, and updated DAX measures.

    Args:
        export_dir: Directory where CSV files are saved
        datasets: Dict of name -> DataFrame for dynamic row/col counts
    """
    d = datasets
    instructions = f"""# Power BI Dashboard Setup Instructions

## Overview
This folder contains CSV files exported from the Starbucks Customer Segmentation project,
organized as a **star schema** optimized for Power BI import and DAX time-intelligence.

### Star Schema Architecture
```
                    ┌──────────────────┐
                    │   dim_date        │
                    │ (date dimension)  │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────┐
│ dim_customers │  │fact_offer_inter-  │  │  dim_offers   │
│(customer dim) │◄─┤   actions         │─►│ (offer dim)   │
└───────────────┘  │ (core fact table) │  └───────────────┘
                   └──────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ fact_kpi_metrics      │
                │ (pre-computed KPIs)   │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │fact_segment_performance│
                │ (pre-aggregated segs) │
                └───────────────────────┘
```

## Files Included

### Star Schema Tables (Primary - use these for dashboards)

#### Fact Tables
1. **fact_offer_interactions.csv** - Core fact table: one row per offer event
   - {len(d['fact_offer_interactions']):,} rows × {len(d['fact_offer_interactions'].columns)} columns
   - Key fields: customer_id (FKtodim_customers), offer_id (FKtodim_offers), event_date (FKtodim_date), event_type, time_hours, day_since_start

2. **fact_kpi_metrics.csv** - Pre-computed KPI card values
   - {len(d['fact_kpi_metrics']):,} rows × {len(d['fact_kpi_metrics'].columns)} columns
   - Key fields: metric_name, metric_value, metric_category, format

3. **fact_segment_performance.csv** - Pre-aggregated segment KPIs
   - {len(d['fact_segment_performance']):,} rows × {len(d['fact_segment_performance'].columns)} columns
   - Key fields: cluster_id, cluster_label, segment_size, avg_completion_rate, total_revenue, best_offer_type

#### Dimension Tables
4. **dim_customers.csv** - Customer dimension (one row per customer)
   - {len(d['dim_customers']):,} rows × {len(d['dim_customers'].columns)} columns
   - Key fields: customer_id (PK), age, income, gender, tenure_months, cluster, cluster_label, engagement_level, rfm_segment, clv_proxy

5. **dim_offers.csv** - Offer dimension (one row per offer, not exploded)
   - {len(d['dim_offers']):,} rows × {len(d['dim_offers'].columns)} columns
   - Key fields: offer_id (PK), offer_type, difficulty, reward, duration_days, reward_per_day, difficulty_per_day, channel_email/mobile/social/web, offer_type_category

6. **dim_date.csv** - Enhanced calendar dimension for time-intelligence
   - {len(d['dim_date']):,} rows × {len(d['dim_date'].columns)} columns
   - Key fields: date, date_key, year, quarter, fiscal_quarter, month, day_of_week, is_weekend, is_holiday_period, day_since_start

### Legacy / Supporting Tables
7. **customers.csv** - Flat customer data (backward-compatible)
   - {len(d['customers']):,} rows × {len(d['customers'].columns)} columns

8. **offers.csv** - Offer data with channels (exploded, backward-compatible)
   - {len(d['offers']):,} rows × {len(d['offers'].columns)} columns

9. **transactions.csv** - Transaction events
   - {len(d['transactions']):,} rows × {len(d['transactions'].columns)} columns

10. **offer_events.csv** - Offer events (flat, backward-compatible)
    - {len(d['offer_events']):,} rows × {len(d['offer_events'].columns)} columns

11. **cluster_profiles.csv** - Cluster profile metrics (long format)
    - {len(d['cluster_profiles']):,} rows × {len(d['cluster_profiles'].columns)} columns

12. **model_performance.csv** - Model comparison metrics (long format)
    - {len(d['model_performance']):,} rows × {len(d['model_performance'].columns)} columns

13. **ate_results.csv** - Causal inference (Average Treatment Effect)
    - {len(d['ate_results']):,} rows × {len(d['ate_results'].columns)} columns

14. **recommendation_results.csv** - Recommendation system performance
    - {len(d['recommendation_results']):,} rows × {len(d['recommendation_results'].columns)} columns

## Power BI Import Steps

### Option A: Get Data to Text/CSV (one at a time)
- Click "Get Data" to "Text/CSV"
- Navigate to `{export_dir}`, select a CSV, click "Load"

### Option B: Get Data to Folder (all at once, recommended)
- Click "Get Data" to "Folder"
- Navigate to `{export_dir}`, click "OK"
- In Power Query: Click "Combine & Load" to "Combine & Load To..."
- This auto-merges all CSVs into separate query tables

### Power Query Data Type Fixes
| Table | Column | Set Type To |
|---|---|---|
| dim_customers | age_imputed, income_imputed, clv_proxy | Decimal Number |
| dim_customers | customer_id | Text |
| dim_customers | engagement_level, rfm_segment, cluster_label | Text |
| dim_offers | offer_id | Text |
| dim_offers | channel_email, channel_mobile, channel_social, channel_web | True/False |
| dim_offers | reward_per_day, difficulty_per_day | Decimal Number |
| fact_offer_interactions | customer_id, offer_id | Text |
| fact_offer_interactions | event_date | Date |
| fact_offer_interactions | time_hours | Decimal Number |
| fact_kpi_metrics | metric_value | Decimal Number |
| fact_segment_performance | avg_completion_rate, avg_view_rate | Decimal Number |
| fact_segment_performance | total_revenue, revenue_per_customer | Decimal Number |
| dim_date | date | Date |
| dim_date | date_key | Text |
| dim_date | is_weekend, is_weekday, is_holiday_period | True/False |
| dim_date | day_since_start | Whole Number |
| transactions | transaction_date | Date |
| transactions | transaction_amount | Decimal Number |
| offer_events | event_date | Date |

### Star Schema Relationships
Create these in **Model View** (drag field between tables):

**Primary relationships (star schema):**
| From (Many) | To (One) | Cardinality | Cross-filter |
|---|---|---|---|
| fact_offer_interactions[customer_id] | dim_customers[customer_id] | Many-to-One | Single |
| fact_offer_interactions[offer_id] | dim_offers[offer_id] | Many-to-One | Single |
| fact_offer_interactions[event_date_key] | dim_date[date_key] | Many-to-One | Single |
| transactions[customer_id] | dim_customers[customer_id] | Many-to-One | Single |
| transactions[transaction_date] | dim_date[date] | Many-to-One | Single |

**Secondary relationships (supporting tables):**
| From (Many) | To (One) | Cardinality | Cross-filter |
|---|---|---|---|
| offer_events[customer_id] | dim_customers[customer_id] | Many-to-One | Single |
| offer_events[offer_id] | dim_offers[offer_id] | Many-to-One | Single |
| offer_events[event_date] | dim_date[date] | Many-to-One | Single |

**Important:**
- Mark `dim_date` as the **Date Table**: right-click to "Mark as Date Table" to select `date` column
- Set `date_key` in `dim_date` as the related column for `event_date_key` lookups
- Ensure cross-filter direction is **Single** (dimension to fact) to preserve star schema filter propagation

## Visualizations - 4 Dashboard Pages

### Page 1: Executive Summary (KPIs + Segments)

| Visual | Fields | Notes |
|---|---|---|
| **KPI Card** | Total Customers (from fact_kpi_metrics where metric_name = "Total Customers") | Large number format |
| **KPI Card** | Average Completion Rate (from fact_kpi_metrics) | % format, 1 decimal |
| **KPI Card** | Total Revenue (from fact_kpi_metrics where metric_name = "Total Revenue") | Currency $ |
| **KPI Card** | Recommendation Lift % (from fact_kpi_metrics) | Show green if > 0 |
| **KPI Card** | Model AUC-ROC (from fact_kpi_metrics) | Gauge-style, green if > 0.8 |
| **Donut Chart** | dim_customers[cluster_label] (Count) | Starbucks green/gold colors |
| **Clustered Bar Chart** | Axis: cluster_label, Values: avg_completion_rate, avg_view_rate, avg_transaction_amount | From fact_segment_performance |
| **Table** | fact_segment_performance (all columns) | Segment comparison with revenue |
| **Decomposition Tree** | Analyze: Avg Completion Rate, By: cluster_label to engagement_level to rfm_segment | Drill into segments |

**Slicers** (top ribbon): cluster_label, engagement_level, offer_type_category

---

### Page 2: Offer Funnel & Performance

| Visual | Fields | Notes |
|---|---|---|
| **Funnel Chart** | Values: COUNT(fact_offer_interactions[event_type]) | Stages: received to viewed to completed |
| **Funnel (per segment)** | Copy funnel, add dim_customers[cluster_label] filter | Shows funnel differs by segment |
| **Scatter Plot** | X: dim_offers[difficulty], Y: dim_offers[reward], Size: COUNT(completed interactions), Legend: offer_type | Sweet-spot analysis |
| **Matrix (Heatmap)** | Rows: dim_offers[offer_type], Columns: channel flags, Values: AVG(dim_customers[completion_rate]) | Channel×type combo |
| **100% Stacked Bar** | Axis: dim_offers[offer_type], Values: event_type count (%), Legend: event_type | Conversion per offer type |
| **Ribbon Chart** | Axis: dim_date[date], Values: COUNT(fact_offer_interactions), Legend: event_type | Volume trends over time |

---

### Page 3: Customer Segments Deep Dive

| Visual | Fields | Notes |
|---|---|---|
| **Scatter Plot** | X: dim_customers[income_imputed], Y: dim_customers[age_imputed], Legend: cluster_label | Demographic clusters |
| **Small Multiples (Line)** | Axis: dim_date[date], Values: SUM(transactions[transaction_amount]), Small multiples: cluster_label | Spend trends per segment |
| **Clustered Bar Chart** | Axis: cluster_label, Values: avg_income, avg_age, avg_transaction_amount | From fact_segment_performance |
| **Stacked Column** | Axis: cluster_label, Values: COUNT(customers), Legend: engagement_level | Engagement distribution |
| **Key Influencers** | Analyze: cluster_label, Explain by: income, age, completion_rate, rfm_segment | ML-driven segment drivers |
| **Scatter with CLV** | X: clv_proxy, Y: completion_rate, Size: total_spend, Legend: rfm_segment | Value vs engagement |

---

### Page 4: Model & Business Impact

| Visual | Fields | Notes |
|---|---|---|
| **Card** | Best model AUC-ROC (from fact_kpi_metrics) | Gauge-style |
| **Card** | Recommendation Lift % | Green if positive |
| **Bar Chart** | Axis: offer_type (from ate_results), Values: ate_dollars | Causal impact per type |
| **Clustered Bar** | Axis: offer_type, Values: control_mean, treatment_mean | Before/after comparison |
| **Table** | fact_segment_performance (cluster_id, cluster_label, segment_size, revenue_per_customer, best_offer_type) | Full segment KPIs |
| **Gauge** | ATE Any Offer value vs $0 baseline | From fact_kpi_metrics |
| **Multi-row card** | ATE Bogo, ATE Discount, ATE Informational (from fact_kpi_metrics filter category = "Causal Impact") | Side-by-side impact |

---

## DAX Measures

Copy these into the model (Home to New Measure):

```dax
-- ============================================
-- STAR SCHEMA MEASURES (using fact/dim tables)
-- ============================================

-- --------------------------------------------
-- CUSTOMER METRICS (from dim_customers)
-- --------------------------------------------

Total Customers = DISTINCTCOUNT(dim_customers[customer_id])

Avg Completion Rate = AVERAGE(dim_customers[completion_rate])

Avg View Rate = AVERAGE(dim_customers[view_rate])

Avg Transaction = AVERAGE(dim_customers[trans_avg])

High Engagement Customers =
    CALCULATE(
        DISTINCTCOUNT(dim_customers[customer_id]),
        dim_customers[engagement_level] = "High"
    )

-- --------------------------------------------
-- OFFER FUNNEL (from fact_offer_interactions)
-- --------------------------------------------

Offers Received = COUNTROWS(
    FILTER(fact_offer_interactions, fact_offer_interactions[event_type] = "received")
)

Offers Viewed = COUNTROWS(
    FILTER(fact_offer_interactions, fact_offer_interactions[event_type] = "viewed")
)

Offers Completed = COUNTROWS(
    FILTER(fact_offer_interactions, fact_offer_interactions[event_type] = "completed")
)

View Rate % = DIVIDE([Offers Viewed], [Offers Received])

Completion Rate % = DIVIDE([Offers Completed], [Offers Received])

-- Funnel by offer type (cross-filtered from dim_offers)
BOGO Offers Received = CALCULATE([Offers Received], dim_offers[offer_type] = "bogo")
Discount Offers Received = CALCULATE([Offers Received], dim_offers[offer_type] = "discount")
Info Offers Received = CALCULATE([Offers Received], dim_offers[offer_type] = "informational")

-- --------------------------------------------
-- FINANCIAL METRICS (from fact_kpi_metrics)
-- --------------------------------------------

KPI Total Customers = CALCULATE(
    SUM(fact_kpi_metrics[metric_value]),
    fact_kpi_metrics[metric_name] = "Total Customers"
)

KPI Total Revenue = CALCULATE(
    SUM(fact_kpi_metrics[metric_value]),
    fact_kpi_metrics[metric_name] = "Total Revenue"
)

KPI Avg Completion Rate = CALCULATE(
    SUM(fact_kpi_metrics[metric_value]),
    fact_kpi_metrics[metric_name] = "Average Completion Rate"
)

KPI Recommendation Lift = CALCULATE(
    SUM(fact_kpi_metrics[metric_value]),
    fact_kpi_metrics[metric_name] = "Recommendation Lift %"
)

KPI Model AUC = CALCULATE(
    SUM(fact_kpi_metrics[metric_value]),
    fact_kpi_metrics[metric_name] = "Model AUC-ROC"
)

-- --------------------------------------------
-- SEGMENT METRICS (from fact_segment_performance)
-- --------------------------------------------

Segment Revenue Rank = RANKX(
    ALL(fact_segment_performance),
    CALCULATE(SUM(fact_segment_performance[total_revenue])),
    ,DESC
)

Best Segment = MINX(
    TOPN(1, fact_segment_performance, fact_segment_performance[total_revenue], DESC),
    fact_segment_performance[cluster_label]
)

-- --------------------------------------------
-- TIME INTELLIGENCE (requires dim_date as date table)
-- --------------------------------------------

Running Total Spend =
    CALCULATE(
        SUM(transactions[transaction_amount]),
        DATESYTD(dim_date[date])
    )

Interactions Last N Days =
    CALCULATE(
        COUNTROWS(fact_offer_interactions),
        FILTER(
            ALL(dim_date),
            dim_date[day_since_start] <= MAX(dim_date[day_since_start])
                && dim_date[day_since_start] >= MAX(dim_date[day_since_start]) - 7
        )
    )

-- Interactions during holiday period
Holiday Period Interactions =
    CALCULATE(
        COUNTROWS(fact_offer_interactions),
        dim_date[is_holiday_period] = TRUE()
    )

-- --------------------------------------------
-- MODEL PERFORMANCE (legacy table)
-- --------------------------------------------

Best AUC-ROC = MAXX(
    FILTER(model_performance, model_performance[metric] = "AUC-ROC"),
    model_performance[value]
)

Best Model = MINX(
    TOPN(1,
        FILTER(model_performance, model_performance[metric] = "AUC-ROC"),
        model_performance[value], DESC
    ),
    model_performance[model_name]
)

-- --------------------------------------------
-- DYNAMIC MEASURES (disconnected table for switching)
-- --------------------------------------------

-- Create a disconnected table (Home > Enter Data):
--   Table: param_metrics, Column: Metric
--   Rows: "Completion Rate", "View Rate", "Avg Spend", "Transaction Count"

Selected KPI =
    SWITCH(
        SELECTEDVALUE(param_metrics[Metric]),
        "Completion Rate", [Avg Completion Rate],
        "View Rate", [Avg View Rate],
        "Avg Spend", [Avg Transaction],
        "Transaction Count", COUNTROWS(fact_offer_interactions),
        [Avg Completion Rate]
    )

-- Dynamic title responds to slicers
Report Title =
    "Starbucks Segments" &
        IF(
            HASONEVALUE(dim_customers[cluster_label]),
            " - " & VALUES(dim_customers[cluster_label]),
            ""
        )
```

## Starbucks Theme Setup

1. **View tab** to **Themes** to **Customize current theme**
2. Set colors:
   - **Data colors**: `#00704A` (green primary), `#6B8E23` (olive), `#FFC72C` (gold), `#1E3932` (dark green), `#D4E9E2` (light mint)
   - **Background**: White `#FFFFFF`
   - **Cards & tiles**: `#F5F5F5` (light gray)
   - **Font**: Segoe UI, 12pt base
3. Save as `starbucks-theme.json` for reuse

## Layout & Interactivity Tips

- **Slicer panel**: Add a consistent left-rail with slicers for cluster_label, engagement_level, offer_type_category. Sync across all 4 pages via "View to Sync Slicers"
- **Page navigation**: Insert shape buttons (rounded rectangle) at the top of each page, assign Page Navigation action
- **Drill-through**: Right-click page tab to "Drill-through" to add cluster_label as drill-through field. Any page can right-click to drill to details
- **Tooltip pages**: Create a mini page (200x300) with cluster profile metrics. Assign it as the tooltip for cluster visuals
- **Mobile layout**: View to Mobile Layout to arrange cards vertically for phone viewing
- **KPI cards**: Use the fact_kpi_metrics table with visual-level filters on metric_name for quick card visuals

## Next Steps

1. Run `python src/reporting/export_powerbi.py` to regenerate CSVs with latest data
2. Open Power BI Desktop to Get Data to Folder to select `{export_dir}`
3. Set data types per the Power Query table above
4. Create relationships per the star schema diagram above
5. Mark `dim_date` as date table (right-click to "Mark as Date Table" to date column)
6. Paste DAX measures from this guide
7. Build the 4 dashboard pages following the visual tables above
8. Apply the Starbucks theme
9. Publish to Power BI Service to Share on portfolio!

---

**For questions or issues, refer to:**
- Power BI Documentation: https://docs.microsoft.com/power-bi/
- Project Repository: [your-repo-url]
- Technical Appendix: reports/technical_appendix.md
"""

    with open(export_dir / 'POWERBI_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)

    print(f"\n Power BI instructions saved to: {export_dir / 'POWERBI_INSTRUCTIONS.md'}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - POWER BI EXPORT")
    print("="*60)

    export_dir, datasets = save_all_exports()

    create_powerbi_instructions(export_dir, datasets)

    print("\n" + "="*60)
    print("POWER BI EXPORT COMPLETE")
    print("="*60)
    print(f" All CSV files saved to: {export_dir}")
    print(f" Read instructions: {export_dir / 'POWERBI_INSTRUCTIONS.md'}")
    print(f" Ready to import into Power BI Desktop!")
    print("="*60)
    print("\n Next steps:")
    print("  1. Open Power BI Desktop")
    print("  2. Import CSV files from 'powerbi_data/' folder")
    print("  3. Create star schema relationships (see POWERBI_INSTRUCTIONS.md)")
    print("  4. Mark dim_date as date table")
    print("  5. Build interactive dashboards")
    print("  6. Share on your portfolio with screenshots!")
