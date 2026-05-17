"""
Exploratory Data Analysis (EDA) module for Starbucks customer segmentation project.

This module performs comprehensive EDA to answer Tier 2 research questions:
1. Offer characteristics and correlation with completion rates
2. Event funnel analysis (received → viewed → completed)
3. Demographic patterns and missing data biases
4. Transaction behavior analysis
5. Marketing channel effectiveness

Generates publication-quality visualizations saved to reports/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_figure_directory(base_path: str = '.') -> Path:
    """Create directory for saving figures."""
    fig_dir = Path(base_path) / 'reports' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def analyze_offer_characteristics(portfolio: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze offer characteristics and their correlation with completion rates.
    
    Answers Research Question: 
    "What is the distribution of offer characteristics (duration, difficulty, reward) 
    across offer types, and how do these correlate with completion rates?"
    
    Args:
        portfolio: Portfolio DataFrame
        transcript: Transcript DataFrame
        
    Returns:
        Dictionary with offer analysis results
    """
    print("\n" + "="*60)
    print("OFFER CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # Extract offer IDs from transcript
    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    
    # Calculate completion rates by offer
    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    
    # Merge with portfolio data
    offer_merged = offer_events.merge(portfolio, left_on='offer_id', right_on='id', how='left')
    
    # Calculate metrics by offer
    offer_stats = []
    for offer_id in offer_merged['offer_id'].unique():
        offer_data = offer_merged[offer_merged['offer_id'] == offer_id]
        stats = {
            'offer_id': offer_id,
            'offer_type': offer_data['offer_type'].iloc[0],
            'difficulty': offer_data['difficulty'].iloc[0],
            'reward': offer_data['reward'].iloc[0],
            'duration': offer_data['duration'].iloc[0],
            'n_received': (offer_data['event'] == 'offer received').sum(),
            'n_viewed': (offer_data['event'] == 'offer viewed').sum(),
            'n_completed': (offer_data['event'] == 'offer completed').sum(),
        }
        stats['view_rate'] = stats['n_viewed'] / stats['n_received'] if stats['n_received'] > 0 else 0
        stats['completion_rate'] = stats['n_completed'] / stats['n_received'] if stats['n_received'] > 0 else 0
        stats['view_to_completion_rate'] = stats['n_completed'] / stats['n_viewed'] if stats['n_viewed'] > 0 else 0
        offer_stats.append(stats)
    
    offer_stats_df = pd.DataFrame(offer_stats)
    
    # Print summary statistics
    print("\nOffer Completion Rates by Type:")
    print(offer_stats_df.groupby('offer_type')[['view_rate', 'completion_rate', 'view_to_completion_rate']].mean())
    
    print("\nCorrelation between offer characteristics and completion rate:")
    corr_cols = ['difficulty', 'reward', 'duration', 'completion_rate']
    corr_matrix = offer_stats_df[corr_cols].corr()
    print(corr_matrix['completion_rate'].sort_values(ascending=False))
    
    return {
        'offer_stats_df': offer_stats_df,
        'correlations': corr_matrix
    }


def create_offer_visualizations(offer_stats_df: pd.DataFrame, fig_dir: Path) -> None:
    """Create visualizations for offer characteristics analysis."""
    
    # Figure 1: Offer characteristics by type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Offer Characteristics by Type', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(['difficulty', 'reward', 'duration']):
        ax = axes[idx]
        sns.boxplot(data=offer_stats_df, x='offer_type', y=col, ax=ax)
        ax.set_title(f'{col.capitalize()} Distribution')
        ax.set_xlabel('Offer Type')
        ax.set_ylabel(col.capitalize())
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'offer_characteristics_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Completion rates by offer (using plotly for interactivity)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('View Rate by Offer', 'Completion Rate by Offer'),
                        specs=[[{'type': 'bar'}, {'type': 'bar'}]])
    
    # View rate
    fig.add_trace(
        go.Bar(
            x=offer_stats_df['offer_id'],
            y=offer_stats_df['view_rate'],
            marker_color='lightblue',
            name='View Rate',
            hovertemplate='Offer: %{x}<br>View Rate: %{y:.2%}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Completion rate
    fig.add_trace(
        go.Bar(
            x=offer_stats_df['offer_id'],
            y=offer_stats_df['completion_rate'],
            marker_color='lightgreen',
            name='Completion Rate',
            hovertemplate='Offer: %{x}<br>Completion Rate: %{y:.2%}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text='Offer Performance Metrics',
        showlegend=False,
        height=500,
        width=1200
    )
    fig.write_html(fig_dir / 'offer_performance_interactive.html')
    
    print(f" Offer visualizations saved to {fig_dir}")


def analyze_event_funnel(transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the event funnel from offer received to viewed to completed.
    
    Answers Research Question:
    "What proportion of offers are viewed after being received, and what is the 
    time-to-view distribution across offer types?"
    
    Args:
        transcript: Transcript DataFrame
        
    Returns:
        Dictionary with funnel analysis results
    """
    print("\n" + "="*60)
    print("EVENT FUNNEL ANALYSIS")
    print("="*60)
    
    # Extract offer IDs
    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    
    # Filter to offer events only
    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    
    # Calculate funnel metrics
    total_received = (offer_events['event'] == 'offer received').sum()
    total_viewed = (offer_events['event'] == 'offer viewed').sum()
    total_completed = (offer_events['event'] == 'offer completed').sum()
    
    print(f"\nOverall Funnel:")
    print(f"  Offers Received: {total_received:,}")
    print(f"  Offers Viewed:  {total_viewed:,} ({total_viewed/total_received:.2%})")
    print(f"  Offers Completed: {total_completed:,} ({total_completed/total_received:.2%})")
    print(f"  View → Complete: {total_completed/total_viewed:.2%}")
    
    # Time-to-view analysis
    # For each offer received, find the corresponding view time
    received = offer_events[offer_events['event'] == 'offer received'][['person', 'offer_id', 'time']].copy()
    received.rename(columns={'time': 'receive_time'}, inplace=True)
    
    viewed = offer_events[offer_events['event'] == 'offer viewed'][['person', 'offer_id', 'time']].copy()
    viewed.rename(columns={'time': 'view_time'}, inplace=True)
    
    # Merge to get time-to-view
    merged = received.merge(viewed, on=['person', 'offer_id'], how='left')
    merged['time_to_view'] = merged['view_time'] - merged['receive_time']
    
    # Statistics for viewed offers only
    viewed_offers = merged[merged['view_time'].notna()].copy()
    if len(viewed_offers) > 0:
        print(f"\nTime-to-View Statistics (for viewed offers):")
        print(f"  Mean: {viewed_offers['time_to_view'].mean():.1f} hours")
        print(f"  Median: {viewed_offers['time_to_view'].median():.1f} hours")
        print(f"  Min: {viewed_offers['time_to_view'].min():.1f} hours")
        print(f"  Max: {viewed_offers['time_to_view'].max():.1f} hours")
    
    return {
        'total_received': total_received,
        'total_viewed': total_viewed,
        'total_completed': total_completed,
        'view_rate': total_viewed / total_received,
        'completion_rate': total_completed / total_received,
        'time_to_view_stats': viewed_offers['time_to_view'].describe() if len(viewed_offers) > 0 else None
    }


def create_funnel_visualizations(funnel_results: Dict, fig_dir: Path, portfolio: pd.DataFrame) -> None:
    """Create visualizations for funnel analysis."""
    
    # Figure 1: Overall funnel (matplotlib)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ['Received', 'Viewed', 'Completed']
    values = [funnel_results['total_received'], 
              funnel_results['total_viewed'], 
              funnel_results['total_completed']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(stages, values, color=colors, edgecolor='black', linewidth=2)
    ax.set_title('Offer Event Funnel', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_xlabel('Event Stage', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value):,}\n({value/funnel_results["total_received"]:.1%})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'offer_funnel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Funnel by offer type (plotly)
    # Merge transcript with portfolio to get offer types
    transcript_expanded = pd.read_json('transcript.json', lines=True)
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    
    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    offer_events = offer_events.merge(portfolio, left_on='offer_id', right_on='id', how='left')
    
    # Calculate funnel by type
    funnel_by_type = []
    for offer_type in offer_events['offer_type'].unique():
        type_data = offer_events[offer_events['offer_type'] == offer_type]
        funnel_by_type.append({
            'offer_type': offer_type,
            'received': (type_data['event'] == 'offer received').sum(),
            'viewed': (type_data['event'] == 'offer viewed').sum(),
            'completed': (type_data['event'] == 'offer completed').sum()
        })
    
    funnel_df = pd.DataFrame(funnel_by_type)
    
    fig = go.Figure()
    colors = {'bogo': '#FF6B6B', 'discount': '#4ECDC4', 'informational': '#45B7D1'}
    
    for idx, stage in enumerate(['received', 'viewed', 'completed']):
        fig.add_trace(go.Bar(
            name=stage.capitalize(),
            x=funnel_df['offer_type'],
            y=funnel_df[stage],
            marker_color=list(colors.values())[idx],
            offsetgroup=idx
        ))
    
    fig.update_layout(
        title='Offer Funnel by Type',
        xaxis_title='Offer Type',
        yaxis_title='Number of Events',
        barmode='group',
        height=500,
        width=800
    )
    fig.write_html(fig_dir / 'funnel_by_type_interactive.html')
    
    print(f" Funnel visualizations saved to {fig_dir}")


def analyze_demographics(profile: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze customer demographics and missing data patterns.
    
    Answers Research Question:
    "What demographic patterns and missing data biases exist in the customer base?"
    
    Args:
        profile: Profile DataFrame
        
    Returns:
        Dictionary with demographic analysis results
    """
    print("\n" + "="*60)
    print("DEMOGRAPHIC ANALYSIS")
    print("="*60)
    
    # Handle missing data (age 118 = missing)
    profile_clean = profile.copy()
    profile_clean['age_missing'] = (profile_clean['age'] == 118).astype(int)
    profile_clean['age_clean'] = profile_clean['age'].replace(118, np.nan)
    
    # Gender distribution
    print("\nGender Distribution:")
    gender_counts = profile['gender'].value_counts()
    print(gender_counts)
    print(f"Missing gender: {gender_counts.get(np.nan, 0)} ({gender_counts.get(np.nan, 0)/len(profile):.2%})")
    
    # Age distribution (excluding missing)
    age_data = profile_clean[profile_clean['age_missing'] == 0]['age']
    print(f"\nAge Distribution (excluding missing):")
    print(f"  Mean: {age_data.mean():.1f}")
    print(f"  Median: {age_data.median():.1f}")
    print(f"  Min: {age_data.min()}")
    print(f"  Max: {age_data.max()}")
    
    # Income distribution (excluding missing)
    income_data = profile_clean[profile_clean['income'].notna()]['income']
    print(f"\nIncome Distribution (excluding missing):")
    print(f"  Mean: ${income_data.mean():,.0f}")
    print(f"  Median: ${income_data.median():,.0f}")
    print(f"  Min: ${income_data.min():,.0f}")
    print(f"  Max: ${income_data.max():,.0f}")
    
    # Membership tenure
    profile_clean['became_member_on'] = pd.to_datetime(profile_clean['became_member_on'], format='%Y%m%d')
    profile_clean['tenure_days'] = (pd.Timestamp('2018-07-26') - profile_clean['became_member_on']).dt.days
    print(f"\nTenure Distribution:")
    print(f"  Mean: {profile_clean['tenure_days'].mean():.0f} days")
    print(f"  Median: {profile_clean['tenure_days'].median():.0f} days")
    
    # Missing data patterns
    print(f"\nMissing Data Summary:")
    print(f"  Records with missing gender: {(profile['gender'].isna()).sum()} ({profile['gender'].isna().mean():.2%})")
    print(f"  Records with missing income: {(profile['income'].isna()).sum()} ({profile['income'].isna().mean():.2%})")
    print(f"  Records with age=118 (missing): {(profile['age'] == 118).sum()} ({(profile['age'] == 118).mean():.2%})")
    
    return {
        'profile_clean': profile_clean,
        'gender_counts': gender_counts,
        'age_stats': age_data.describe(),
        'income_stats': income_data.describe(),
        'tenure_stats': profile_clean['tenure_days'].describe()
    }


def create_demographic_visualizations(demo_results: Dict, fig_dir: Path) -> None:
    """Create visualizations for demographic analysis."""
    
    profile_clean = demo_results['profile_clean']
    
    # Figure 1: Demographic distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Customer Demographics Analysis', fontsize=16, fontweight='bold')
    
    # Age distribution
    age_data = profile_clean[profile_clean['age_missing'] == 0]['age']
    axes[0, 0].hist(age_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Age Distribution (Excluding Missing)')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(age_data.mean(), color='red', linestyle='--', label=f'Mean: {age_data.mean():.1f}')
    axes[0, 0].legend()
    
    # Income distribution
    income_data = profile_clean[profile_clean['income'].notna()]['income']
    axes[0, 1].hist(income_data, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Income Distribution (Excluding Missing)')
    axes[0, 1].set_xlabel('Income ($)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(income_data.mean(), color='red', linestyle='--', label=f'Mean: ${income_data.mean():,.0f}')
    axes[0, 1].legend()
    
    # Gender distribution
    gender_counts = profile_clean['gender'].value_counts()
    axes[0, 2].bar(gender_counts.index.astype(str), gender_counts.values, 
                   color=['blue', 'pink', 'gray', 'lightgray'])
    axes[0, 2].set_title('Gender Distribution')
    axes[0, 2].set_xlabel('Gender')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Tenure distribution
    axes[1, 0].hist(profile_clean['tenure_days'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title('Membership Tenure (Days)')
    axes[1, 0].set_xlabel('Tenure (days)')
    axes[1, 0].set_ylabel('Count')
    
    # Missing data patterns
    missing_data = pd.DataFrame({
        'Missing': ['Gender', 'Income', 'Age'],
        'Count': [
            profile_clean['gender'].isna().sum(),
            profile_clean['income'].isna().sum(),
            (profile_clean['age'] == 118).sum()
        ]
    })
    axes[1, 1].bar(missing_data['Missing'], missing_data['Count'], color='red', alpha=0.6)
    axes[1, 1].set_title('Missing Data Counts')
    axes[1, 1].set_ylabel('Number of Records')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Age vs Income scatter (for non-missing)
    valid_data = profile_clean[(profile_clean['age_missing'] == 0) & (profile_clean['income'].notna())].copy()
    scatter = axes[1, 2].scatter(valid_data['age'], valid_data['income'], alpha=0.3, c='purple', s=10)
    axes[1, 2].set_title('Age vs Income (Non-Missing)')
    axes[1, 2].set_xlabel('Age')
    axes[1, 2].set_ylabel('Income ($)')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'demographic_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Demographic visualizations saved to {fig_dir}")


def analyze_transaction_behavior(profile: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze transaction behaviors of customers who respond to offers vs non-responders.
    
    Answers Research Question:
    "How do transaction behaviors (frequency, average amount) differ between 
    customers who respond to offers vs non-responders?"
    
    Args:
        profile: Profile DataFrame
        transcript: Transcript DataFrame
        
    Returns:
        Dictionary with transaction behavior analysis
    """
    print("\n" + "="*60)
    print("TRANSACTION BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Extract transaction amounts
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transactions = transactions[transactions['amount'].notna()].copy()
    
    # Calculate transaction stats per customer
    customer_transactions = transactions.groupby('person').agg({
        'amount': ['count', 'sum', 'mean', 'std']
    }).reset_index()
    customer_transactions.columns = ['customer_id', 'trans_count', 'trans_total', 'trans_avg', 'trans_std']
    
    # Identify responders (customers who completed at least one offer)
    responders = set(transcript[transcript['event'] == 'offer completed']['person'].unique())
    customer_transactions['is_responder'] = customer_transactions['customer_id'].isin(responders).astype(int)
    
    # Compare responders vs non-responders
    print("\nTransaction Behavior: Responders vs Non-Responders")
    print("\nResponders (completed ≥1 offer):")
    responder_data = customer_transactions[customer_transactions['is_responder'] == 1]
    print(f"  Count: {len(responder_data):,}")
    print(f"  Avg transactions per customer: {responder_data['trans_count'].mean():.1f}")
    print(f"  Avg transaction amount: ${responder_data['trans_avg'].mean():.2f}")
    print(f"  Avg total spend: ${responder_data['trans_total'].mean():.2f}")
    
    print("\nNon-Responders (completed 0 offers):")
    non_responder_data = customer_transactions[customer_transactions['is_responder'] == 0]
    print(f"  Count: {len(non_responder_data):,}")
    print(f"  Avg transactions per customer: {non_responder_data['trans_count'].mean():.1f}")
    print(f"  Avg transaction amount: ${non_responder_data['trans_avg'].mean():.2f}")
    print(f"  Avg total spend: ${non_responder_data['trans_total'].mean():.2f}")
    
    return {
        'customer_transactions': customer_transactions,
        'responders': responders,
        'responder_stats': responder_data.describe(),
        'non_responder_stats': non_responder_data.describe()
    }


def create_transaction_visualizations(trans_results: Dict, fig_dir: Path) -> None:
    """Create visualizations for transaction behavior analysis."""
    
    customer_transactions = trans_results['customer_transactions']
    
    # Figure 1: Transaction behavior comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Transaction Behavior: Responders vs Non-Responders', fontsize=16, fontweight='bold')
    
    # Transaction count
    sns.boxplot(data=customer_transactions, x='is_responder', y='trans_count', ax=axes[0])
    axes[0].set_title('Transaction Count')
    axes[0].set_xlabel('Responder (1=Yes, 0=No)')
    axes[0].set_ylabel('Number of Transactions')
    
    # Average transaction amount
    sns.boxplot(data=customer_transactions, x='is_responder', y='trans_avg', ax=axes[1])
    axes[1].set_title('Average Transaction Amount')
    axes[1].set_xlabel('Responder (1=Yes, 0=No)')
    axes[1].set_ylabel('Average Amount ($)')
    
    # Total spend
    sns.boxplot(data=customer_transactions, x='is_responder', y='trans_total', ax=axes[2])
    axes[2].set_title('Total Spend')
    axes[2].set_xlabel('Responder (1=Yes, 0=No)')
    axes[2].set_ylabel('Total Spend ($)')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'transaction_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Transaction visualizations saved to {fig_dir}")


def generate_eda_report(results: Dict[str, Any], output_path: str = 'reports/eda_summary.json') -> None:
    """Generate a comprehensive EDA summary report."""
    
    report = {
        'offer_analysis': {
            'summary': results.get('offer_analysis', {}).get('offer_stats_df', pd.DataFrame()).to_dict('records') if 'offer_analysis' in results else []
        },
        'funnel_analysis': results.get('funnel_analysis', {}),
        'demographic_analysis': {
            'gender_counts': results.get('demographic_analysis', {}).get('gender_counts', {}).to_dict() if 'demographic_analysis' in results else {}
        },
        'transaction_analysis': {
            'responder_count': len(results.get('transaction_analysis', {}).get('responders', [])),
            'total_customers_with_transactions': len(results.get('transaction_analysis', {}).get('customer_transactions', []))
        }
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    report = convert_numpy(report)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n EDA summary report saved to: {output_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Setup
    fig_dir = setup_figure_directory()
    
    # Load data
    print("\nLoading datasets...")
    from load_data import load_all_datasets
    portfolio, profile, transcript = load_all_datasets()
    print(f" Loaded {len(portfolio)} offers, {len(profile)} customers, {len(transcript)} events")
    
    # Run analyses
    results = {}
    
    # 1. Offer characteristics
    results['offer_analysis'] = analyze_offer_characteristics(portfolio, transcript)
    create_offer_visualizations(results['offer_analysis']['offer_stats_df'], fig_dir)
    
    # 2. Event funnel
    results['funnel_analysis'] = analyze_event_funnel(transcript)
    create_funnel_visualizations(results['funnel_analysis'], fig_dir, portfolio)
    
    # 3. Demographics
    results['demographic_analysis'] = analyze_demographics(profile)
    create_demographic_visualizations(results['demographic_analysis'], fig_dir)
    
    # 4. Transaction behavior
    results['transaction_analysis'] = analyze_transaction_behavior(profile, transcript)
    create_transaction_visualizations(results['transaction_analysis'], fig_dir)
    
    # Generate summary report
    generate_eda_report(results)
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)
    print(f" All visualizations saved to: {fig_dir}")
    print(f" Summary report saved to: reports/eda_summary.json")
    print("="*60)
