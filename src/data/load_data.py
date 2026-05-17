"""
Data ingestion and validation module for Starbucks customer segmentation project.

This module handles loading the three JSON datasets (portfolio, profile, transcript),
validates their schemas, checks data quality, and provides summary statistics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any


def load_json_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a JSON or JSONL file into a pandas DataFrame.
    
    Handles both:
    - JSON array format: [{"key": "value"}, ...]
    - JSONL format (one JSON object per line): {"key": "value"}\n{"key": "value"}\n...
    
    Args:
        file_path: Path to the JSON/JSONL file
        
    Returns:
        DataFrame containing the JSON data
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON is malformed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(path, 'r') as f:
        # Try to load as a single JSON array first
        content = f.read().strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return pd.DataFrame(parsed)
        except json.JSONDecodeError:
            pass
        
        # If that fails, try JSONL format (one object per line)
        f.seek(0)
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")
    
    return pd.DataFrame(data)


def validate_portfolio_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the portfolio.json schema and return validation results.
    
    Expected schema:
    - id (string): offer id
    - offer_type (string): BOGO, discount, informational
    - difficulty (int): minimum required spend
    - reward (int): reward given
    - duration (int): time in days
    - channels (list of strings)
    
    Args:
        df: Portfolio DataFrame
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'offer_types': df['offer_type'].unique().tolist() if 'offer_type' in df.columns else None,
        'valid_schema': True,
        'issues': []
    }
    
    # Check required columns
    required_cols = ['id', 'offer_type', 'difficulty', 'reward', 'duration', 'channels']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results['valid_schema'] = False
        results['issues'].append(f"Missing columns: {missing_cols}")
    
    # Validate offer_type values (actual data uses lowercase)
    if 'offer_type' in df.columns:
        valid_types = ['bogo', 'discount', 'informational']
        invalid_types = [t for t in df['offer_type'].unique() if t not in valid_types]
        if invalid_types:
            results['issues'].append(f"Invalid offer types: {invalid_types}")
        # Add normalized offer_type for consistency
        df['offer_type_normalized'] = df['offer_type'].str.upper()
    
    # Check numeric columns
    numeric_cols = ['difficulty', 'reward', 'duration']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                results['issues'].append(f"{col} is not numeric")
            elif df[col].min() < 0:
                results['issues'].append(f"{col} has negative values")
    
    return results


def validate_profile_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the profile.json schema and return validation results.
    
    Expected schema:
    - age (int): customer age
    - became_member_on (int): date as integer (YYYYMMDD)
    - gender (str): M, F, O (other)
    - id (str): customer id
    - income (float): customer income
    
    Args:
        df: Profile DataFrame
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'gender_values': df['gender'].value_counts().to_dict() if 'gender' in df.columns else None,
        'valid_schema': True,
        'issues': []
    }
    
    # Check required columns
    required_cols = ['age', 'became_member_on', 'gender', 'id', 'income']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results['valid_schema'] = False
        results['issues'].append(f"Missing columns: {missing_cols}")
    
    # Check for missing gender (represented as NaN or empty)
    if 'gender' in df.columns:
        missing_gender = df['gender'].isnull().sum()
        results['missing_gender_count'] = int(missing_gender)
    
    # Check for missing income
    if 'income' in df.columns:
        missing_income = df['income'].isnull().sum()
        results['missing_income_count'] = int(missing_income)
    
    # Validate age (should be positive, check for 118 as potential sentinel value)
    if 'age' in df.columns:
        age_118_count = (df['age'] == 118).sum()
        if age_118_count > 0:
            results['issues'].append(f"Age 118 detected for {age_118_count} records - sentinel value for missing data")
            results['age_118_count'] = int(age_118_count)
        
        # Calculate stats excluding sentinel value
        valid_ages = df[df['age'] != 118]['age']
        if len(valid_ages) > 0:
            results['age_stats'] = {
                'min': int(valid_ages.min()),
                'max': int(valid_ages.max()),
                'mean': float(valid_ages.mean()),
                'median': float(valid_ages.median())
            }
        
        # Add a flag for missing age
        df['age_missing'] = (df['age'] == 118).astype(int)
    
    return results


def validate_transcript_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the transcript.json schema and return validation results.
    
    Expected schema:
    - event (str): transaction, offer received, offer viewed, offer completed
    - person (str): customer id
    - time (int): time in hours
    - value (dict): offer id or transaction amount
    
    Args:
        df: Transcript DataFrame
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'event_types': df['event'].value_counts().to_dict() if 'event' in df.columns else None,
        'valid_schema': True,
        'issues': []
    }
    
    # Check required columns
    required_cols = ['event', 'person', 'time', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results['valid_schema'] = False
        results['issues'].append(f"Missing columns: {missing_cols}")
    
    # Validate event types (actual data uses lowercase)
    if 'event' in df.columns:
        valid_events = ['transaction', 'offer received', 'offer viewed', 'offer completed']
        invalid_events = [e for e in df['event'].unique() if e not in valid_events]
        if invalid_events:
            results['issues'].append(f"Invalid event types: {invalid_events}")
    
    # Check time range
    if 'time' in df.columns:
        results['time_range'] = {
            'min': int(df['time'].min()),
            'max': int(df['time'].max()),
            'duration_days': int(df['time'].max() / 24) if df['time'].max() > 0 else 0
        }
    
    # Extract offer IDs from value column for validation
    if 'value' in df.columns:
        offer_ids = []
        transaction_amounts = []
        for val in df['value']:
            if isinstance(val, dict):
                # Handle both "offer id" (with space) and "offer_id" formats
                offer_id = val.get('offer id') or val.get('offer_id')
                if offer_id:
                    offer_ids.append(offer_id)
                # Check for transaction amount
                if 'amount' in val:
                    transaction_amounts.append(val['amount'])
        
        results['unique_offer_ids_in_transcript'] = len(set(offer_ids))
        results['transaction_count'] = len(transaction_amounts)
        if transaction_amounts:
            results['transaction_amount_stats'] = {
                'min': float(min(transaction_amounts)),
                'max': float(max(transaction_amounts)),
                'mean': float(np.mean(transaction_amounts))
            }
    
    return results


def load_all_datasets(base_path: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three Starbucks datasets.
    
    Args:
        base_path: Base directory containing the JSON files
        
    Returns:
        Tuple of (portfolio_df, profile_df, transcript_df)
    """
    portfolio = load_json_to_dataframe(f"{base_path}/portfolio.json")
    profile = load_json_to_dataframe(f"{base_path}/profile.json")
    transcript = load_json_to_dataframe(f"{base_path}/transcript.json")
    
    return portfolio, profile, transcript


def generate_data_quality_report(portfolio: pd.DataFrame, profile: pd.DataFrame, 
                                  transcript: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report for all datasets.
    
    Args:
        portfolio: Portfolio DataFrame
        profile: Profile DataFrame
        transcript: Transcript DataFrame
        output_path: Optional path to save the report as JSON
        
    Returns:
        Dictionary containing the full quality report
    """
    report = {
        'portfolio_validation': validate_portfolio_schema(portfolio),
        'profile_validation': validate_profile_schema(profile),
        'transcript_validation': validate_transcript_schema(transcript),
        'summary': {}
    }
    
    # Add summary statistics
    report['summary'] = {
        'total_offers': len(portfolio),
        'total_customers': len(profile),
        'total_events': len(transcript),
        'unique_customers_in_transcript': transcript['person'].nunique() if 'person' in transcript.columns else 0,
        'date_range_days': int(transcript['time'].max() / 24) if 'time' in transcript.columns else 0
    }
    
    # Check for data consistency issues
    report['summary']['customers_not_in_transcript'] = 0
    report['summary']['offers_not_in_transcript'] = 0
    
    if 'person' in transcript.columns and 'id' in profile.columns:
        customers_in_transcript = set(transcript['person'].unique())
        customers_in_profile = set(profile['id'].unique())
        report['summary']['customers_not_in_transcript'] = len(customers_in_profile - customers_in_transcript)
        report['summary']['customers_in_transcript_not_in_profile'] = len(customers_in_transcript - customers_in_profile)
    
    if 'value' in transcript.columns and 'id' in portfolio.columns:
        # Extract offer ids from transcript value column
        offer_ids_in_transcript = set()
        for val in transcript['value']:
            if isinstance(val, dict):
                # Handle both "offer id" (with space) and "offer_id" formats
                offer_id = val.get('offer id') or val.get('offer_id')
                if offer_id:
                    offer_ids_in_transcript.add(offer_id)
        
        offer_ids_in_portfolio = set(portfolio['id'].unique())
        report['summary']['offers_not_in_transcript'] = len(offer_ids_in_portfolio - offer_ids_in_transcript)
        report['summary']['offers_in_transcript_not_in_portfolio'] = len(offer_ids_in_transcript - offer_ids_in_portfolio)
    
    # Save report if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 60)
    print("STARBUCKS CUSTOMER SEGMENTATION - DATA INGESTION & VALIDATION")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    portfolio, profile, transcript = load_all_datasets()
    print(f" Portfolio: {portfolio.shape[0]} offers, {portfolio.shape[1]} columns")
    print(f" Profile: {profile.shape[0]} customers, {profile.shape[1]} columns")
    print(f" Transcript: {transcript.shape[0]} events, {transcript.shape[1]} columns")
    
    # Generate quality report
    print("\nGenerating data quality report...")
    report = generate_data_quality_report(portfolio, profile, transcript, 
                                          output_path='reports/data_quality_report.json')
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"Total offers: {report['summary']['total_offers']}")
    print(f"Total customers: {report['summary']['total_customers']}")
    print(f"Total events: {report['summary']['total_events']:,}")
    print(f"Unique customers in transcript: {report['summary']['unique_customers_in_transcript']}")
    print(f"Date range: {report['summary']['date_range_days']} days")
    
    # Check for issues
    all_issues = []
    for dataset in ['portfolio_validation', 'profile_validation', 'transcript_validation']:
        all_issues.extend(report[dataset].get('issues', []))
    
    if all_issues:
        print("\n  Data Quality Issues Found:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("\n No major data quality issues found")
    
    print("\n Full report saved to: reports/data_quality_report.json")
    print("=" * 60)
