"""
Exploratory Data Analysis (EDA) module for Starbucks customer segmentation project.

This module performs comprehensive EDA to answer Tier 2 research questions:
1. Offer characteristics and correlation with completion rates
2. Event funnel analysis (received to viewed to completed)
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
from scipy import stats
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
    print(f"  View to Complete: {total_completed/total_viewed:.2%}")
    
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


def _cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float(np.mean(group1) - np.mean(group2)) / pooled_std


def _bootstrap_ci(data: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                  stat_func=np.mean) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return lower, upper


def _two_proportion_z_test(success1: int, n1: int, success2: int, n2: int) -> Tuple[float, float]:
    """Two-proportion z-test returning (z_stat, p_value)."""
    p1 = success1 / n1
    p2 = success2 / n2
    p_pooled = (success1 + success2) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def hypothesis_testing(profile: pd.DataFrame, transcript: pd.DataFrame,
                       portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical hypothesis tests for key business questions.

    Tests:
    - Mann-Whitney U: responders vs non-responders transaction amounts
    - Chi-squared: gender vs offer completion association
    - Kruskal-Wallis: income differences across offer types
    - Two-proportion z-test: completion rate BOGO vs discount
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)

    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )

    responders = set(transcript[transcript['event'] == 'offer completed']['person'].unique())

    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transactions = transactions[transactions['amount'].notna()]

    resp_amounts = transactions[transactions['person'].isin(responders)]['amount'].values
    non_resp_amounts = transactions[transactions['person'].isin(responders) == False]['amount'].values

    u_stat, u_pvalue = stats.mannwhitneyu(resp_amounts, non_resp_amounts, alternative='two-sided')
    cohen_d_amount = _cohen_d(resp_amounts, non_resp_amounts)

    print("\n--- Mann-Whitney U Test: Transaction Amounts (Responders vs Non-Responders) ---")
    print(f"  U-statistic: {u_stat:.2f}")
    print(f"  p-value: {u_pvalue:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if u_pvalue < 0.05 else 'No'}")
    print(f"  Cohen's d: {cohen_d_amount:.4f}")
    print(f"  Responder mean: ${np.mean(resp_amounts):.2f}, Non-responder mean: ${np.mean(non_resp_amounts):.2f}")

    completed_persons = set(transcript[transcript['event'] == 'offer completed']['person'].unique())
    not_completed_persons = set(transcript['person'].unique()) - completed_persons

    profile_for_chi = profile[profile['gender'].notna()].copy()
    profile_for_chi['completed_offer'] = profile_for_chi['id'].isin(completed_persons).astype(int)

    contingency = pd.crosstab(profile_for_chi['gender'], profile_for_chi['completed_offer'])
    chi2, chi2_p, chi2_dof, chi2_expected = stats.chi2_contingency(contingency)
    n_chi = contingency.values.sum()
    cramers_v = np.sqrt(chi2 / (n_chi * (min(contingency.shape) - 1))) if n_chi > 0 and min(contingency.shape) > 1 else 0.0

    print("\n--- Chi-Squared Test: Gender vs Offer Completion ---")
    print(f"  Chi2-statistic: {chi2:.2f}")
    print(f"  p-value: {chi2_p:.6f}")
    print(f"  Degrees of freedom: {chi2_dof}")
    print(f"  Significant (α=0.05): {'Yes' if chi2_p < 0.05 else 'No'}")
    print(f"  Cramér's V: {cramers_v:.4f}")
    print(f"  Contingency table:\n{contingency}")

    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    offer_merged = offer_events.merge(portfolio, left_on='offer_id', right_on='id', how='left')

    completed_merged = offer_merged[offer_merged['event'] == 'offer completed'].copy()
    completed_merged = completed_merged.merge(
        profile[['id', 'income']], left_on='person', right_on='id', how='left', suffixes=('', '_profile')
    )
    completed_merged = completed_merged[completed_merged['income'].notna()]

    income_groups = []
    offer_type_labels = []
    for ot in completed_merged['offer_type'].dropna().unique():
        grp = completed_merged[completed_merged['offer_type'] == ot]['income'].values
        income_groups.append(grp)
        offer_type_labels.append(ot)
        print(f"  {ot}: n={len(grp)}, median=${np.median(grp):,.0f}")

    if len(income_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*income_groups)
    else:
        kw_stat, kw_p = 0.0, 1.0

    print("\n--- Kruskal-Wallis Test: Income Across Offer Types ---")
    print(f"  H-statistic: {kw_stat:.2f}")
    print(f"  p-value: {kw_p:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if kw_p < 0.05 else 'No'}")

    bogo_received = offer_merged[(offer_merged['offer_type'] == 'bogo') & (offer_merged['event'] == 'offer received')]
    discount_received = offer_merged[(offer_merged['offer_type'] == 'discount') & (offer_merged['event'] == 'offer received')]

    bogo_ids = set(bogo_received['offer_id'].unique())
    discount_ids = set(discount_received['offer_id'].unique())

    bogo_completed = offer_merged[(offer_merged['offer_id'].isin(bogo_ids)) & (offer_merged['event'] == 'offer completed')]
    discount_completed = offer_merged[(offer_merged['offer_id'].isin(discount_ids)) & (offer_merged['event'] == 'offer completed')]

    n_bogo_received = len(bogo_received)
    n_discount_received = len(discount_received)
    n_bogo_completed = len(bogo_completed)
    n_discount_completed = len(discount_completed)

    bogo_rate = n_bogo_completed / n_bogo_received if n_bogo_received > 0 else 0
    discount_rate = n_discount_completed / n_discount_received if n_discount_received > 0 else 0

    z_stat, z_p = _two_proportion_z_test(n_bogo_completed, n_bogo_received,
                                          n_discount_completed, n_discount_received)

    print("\n--- Two-Proportion Z-Test: BOGO vs Discount Completion Rate ---")
    print(f"  BOGO completion rate: {bogo_rate:.4f} ({n_bogo_completed}/{n_bogo_received})")
    print(f"  Discount completion rate: {discount_rate:.4f} ({n_discount_completed}/{n_discount_received})")
    print(f"  Z-statistic: {z_stat:.4f}")
    print(f"  p-value: {z_p:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if z_p < 0.05 else 'No'}")

    return {
        'mann_whitney': {
            'u_statistic': u_stat, 'p_value': u_pvalue,
            'cohens_d': cohen_d_amount, 'significant': u_pvalue < 0.05,
            'responder_mean': float(np.mean(resp_amounts)),
            'non_responder_mean': float(np.mean(non_resp_amounts))
        },
        'chi_squared': {
            'chi2_statistic': chi2, 'p_value': chi2_p,
            'dof': chi2_dof, 'cramers_v': cramers_v, 'significant': chi2_p < 0.05
        },
        'kruskal_wallis': {
            'h_statistic': kw_stat, 'p_value': kw_p, 'significant': kw_p < 0.05
        },
        'two_proportion_z': {
            'z_statistic': z_stat, 'p_value': z_p,
            'bogo_rate': bogo_rate, 'discount_rate': discount_rate,
            'significant': z_p < 0.05
        }
    }


def effect_size_analysis(profile: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Cohen's d effect sizes for responder vs non-responder comparisons
    across transaction frequency, average amount, total spend, and customer demographics.
    """
    print("\n" + "=" * 60)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 60)

    responders = set(transcript[transcript['event'] == 'offer completed']['person'].unique())

    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transactions = transactions[transactions['amount'].notna()]

    cust_txn = transactions.groupby('person').agg(
        trans_count=('amount', 'count'),
        trans_total=('amount', 'sum'),
        trans_avg=('amount', 'mean')
    ).reset_index()
    cust_txn['is_responder'] = cust_txn['person'].isin(responders).astype(int)

    profile_clean = profile.copy()
    profile_clean['age_clean'] = profile_clean['age'].replace(118, np.nan)
    cust_txn = cust_txn.merge(
        profile_clean[['id', 'age_clean', 'income']],
        left_on='person', right_on='id', how='left'
    )

    resp = cust_txn[cust_txn['is_responder'] == 1]
    non_resp = cust_txn[cust_txn['is_responder'] == 0]

    metrics = ['trans_count', 'trans_avg', 'trans_total', 'age_clean', 'income']
    labels = ['Transaction Count', 'Avg Transaction Amount', 'Total Spend', 'Age', 'Income']

    results = []
    print("\nCohen's d: Responders vs Non-Responders")
    print("-" * 55)
    for metric, label in zip(metrics, labels):
        r_vals = resp[metric].dropna().values
        nr_vals = non_resp[metric].dropna().values
        if len(r_vals) > 1 and len(nr_vals) > 1:
            d = _cohen_d(r_vals, nr_vals)
        else:
            d = 0.0
        magnitude = 'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
        print(f"  {label:30s}  d = {d:+.4f}  ({magnitude})")
        results.append({'metric': metric, 'label': label, 'cohens_d': d, 'magnitude': magnitude})

    return {'effect_sizes': results}


def confidence_interval_analysis(profile: pd.DataFrame, transcript: pd.DataFrame,
                                  portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute 95% bootstrap confidence intervals for key metrics:
    completion rates by offer type, mean transaction amounts by responder status.
    """
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVAL ANALYSIS (Bootstrap, 95% CI)")
    print("=" * 60)

    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    offer_merged = offer_events.merge(portfolio, left_on='offer_id', right_on='id', how='left')

    ci_results = {}

    print("\nCompletion Rate 95% CI by Offer Type:")
    print("-" * 55)
    for offer_type in offer_merged['offer_type'].dropna().unique():
        ot_data = offer_merged[offer_merged['offer_type'] == offer_type]
        n_received = (ot_data['event'] == 'offer received').sum()
        n_completed = (ot_data['event'] == 'offer completed').sum()

        per_person = ot_data.groupby('person').apply(
            lambda g: int((g['event'] == 'offer completed').any())
        ).values

        if len(per_person) > 1:
            ci_lo, ci_hi = _bootstrap_ci(per_person, stat_func=np.mean)
        else:
            ci_lo, ci_hi = float(np.mean(per_person)), float(np.mean(per_person))

        rate = n_completed / n_received if n_received > 0 else 0
        print(f"  {offer_type:15s}  rate={rate:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        ci_results[f'completion_rate_{offer_type}'] = {
            'rate': rate, 'ci_lower': ci_lo, 'ci_upper': ci_hi
        }

    responders = set(transcript[transcript['event'] == 'offer completed']['person'].unique())
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transactions = transactions[transactions['amount'].notna()]

    resp_amt = transactions[transactions['person'].isin(responders)]['amount'].values
    non_resp_amt = transactions[transactions['person'].isin(responders) == False]['amount'].values

    print("\nMean Transaction Amount 95% CI:")
    print("-" * 55)
    if len(resp_amt) > 1:
        r_lo, r_hi = _bootstrap_ci(resp_amt)
        print(f"  Responders     mean=${np.mean(resp_amt):.2f}  CI=[${r_lo:.2f}, ${r_hi:.2f}]")
        ci_results['responder_mean_amount'] = {
            'mean': float(np.mean(resp_amt)), 'ci_lower': r_lo, 'ci_upper': r_hi
        }
    if len(non_resp_amt) > 1:
        nr_lo, nr_hi = _bootstrap_ci(non_resp_amt)
        print(f"  Non-Responders mean=${np.mean(non_resp_amt):.2f}  CI=[${nr_lo:.2f}, ${nr_hi:.2f}]")
        ci_results['non_responder_mean_amount'] = {
            'mean': float(np.mean(non_resp_amt)), 'ci_lower': nr_lo, 'ci_upper': nr_hi
        }

    return ci_results


def correlation_analysis(portfolio: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlation matrices with p-values
    for offer characteristics vs completion rates.
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS (with p-values)")
    print("=" * 60)

    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    offer_merged = offer_events.merge(portfolio, left_on='offer_id', right_on='id', how='left')

    offer_stats = []
    for offer_id in offer_merged['offer_id'].unique():
        od = offer_merged[offer_merged['offer_id'] == offer_id]
        n_received = (od['event'] == 'offer received').sum()
        n_completed = (od['event'] == 'offer completed').sum()
        offer_stats.append({
            'offer_id': offer_id,
            'offer_type': od['offer_type'].iloc[0],
            'difficulty': od['difficulty'].iloc[0],
            'reward': od['reward'].iloc[0],
            'duration': od['duration'].iloc[0],
            'completion_rate': n_completed / n_received if n_received > 0 else 0,
            'view_rate': (od['event'] == 'offer viewed').sum() / n_received if n_received > 0 else 0,
        })
    offer_stats_df = pd.DataFrame(offer_stats)

    corr_cols = ['difficulty', 'reward', 'duration', 'completion_rate', 'view_rate']

    print("\nPearson Correlations with p-values:")
    print("-" * 55)
    pearson_r = pd.DataFrame(index=corr_cols, columns=corr_cols, dtype=float)
    pearson_p = pd.DataFrame(index=corr_cols, columns=corr_cols, dtype=float)

    for c1 in corr_cols:
        for c2 in corr_cols:
            r, p = stats.pearsonr(offer_stats_df[c1], offer_stats_df[c2])
            pearson_r.loc[c1, c2] = r
            pearson_p.loc[c1, c2] = p

    print(pearson_r.round(4).to_string())
    print("\nP-values:")
    print(pearson_p.round(6).to_string())

    print("\nSpearman Correlations with p-values:")
    print("-" * 55)
    spearman_r = pd.DataFrame(index=corr_cols, columns=corr_cols, dtype=float)
    spearman_p = pd.DataFrame(index=corr_cols, columns=corr_cols, dtype=float)

    for c1 in corr_cols:
        for c2 in corr_cols:
            r, p = stats.spearmanr(offer_stats_df[c1], offer_stats_df[c2])
            spearman_r.loc[c1, c2] = r
            spearman_p.loc[c1, c2] = p

    print(spearman_r.round(4).to_string())
    print("\nP-values:")
    print(spearman_p.round(6).to_string())

    target_col = 'completion_rate'
    feature_cols = [c for c in corr_cols if c != target_col]
    print(f"\nKey correlations with {target_col}:")
    for c in feature_cols:
        pr = pearson_r.loc[c, target_col]
        pp = pearson_p.loc[c, target_col]
        sr = spearman_r.loc[c, target_col]
        sp = spearman_p.loc[c, target_col]
        print(f"  {c:15s}  Pearson r={pr:+.4f} (p={pp:.4f})  Spearman ρ={sr:+.4f} (p={sp:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Correlation Matrices: Offer Characteristics vs Completion/View Rates', fontsize=14, fontweight='bold')

    mask = np.triu(np.ones_like(pearson_r, dtype=bool), k=1)
    sns.heatmap(pearson_r.astype(float), annot=True, fmt='.3f', ax=axes[0],
                cmap='coolwarm', center=0, mask=mask, vmin=-1, vmax=1)
    axes[0].set_title('Pearson Correlation')

    sns.heatmap(spearman_r.astype(float), annot=True, fmt='.3f', ax=axes[1],
                cmap='coolwarm', center=0, mask=mask, vmin=-1, vmax=1)
    axes[1].set_title('Spearman Correlation')

    plt.tight_layout()
    fig_dir = setup_figure_directory()
    plt.savefig(fig_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'offer_stats_df': offer_stats_df
    }


def cohort_analysis(profile: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Membership cohort analysis: how offer response varies by join-year cohort.
    """
    print("\n" + "=" * 60)
    print("MEMBERSHIP COHORT ANALYSIS")
    print("=" * 60)

    profile_clean = profile.copy()
    profile_clean['became_member_on'] = pd.to_datetime(profile_clean['became_member_on'], format='%Y%m%d')
    profile_clean['join_year'] = profile_clean['became_member_on'].dt.year

    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )

    completed = transcript_expanded[transcript_expanded['event'] == 'offer completed'][['person', 'offer_id']].copy()
    received = transcript_expanded[transcript_expanded['event'] == 'offer received'][['person', 'offer_id']].copy()
    viewed = transcript_expanded[transcript_expanded['event'] == 'offer viewed'][['person', 'offer_id']].copy()

    completed_per_person = completed.groupby('person').size().reset_index(name='n_completed')
    received_per_person = received.groupby('person').size().reset_index(name='n_received')
    viewed_per_person = viewed.groupby('person').size().reset_index(name='n_viewed')

    cohort_df = profile_clean[['id', 'join_year']].merge(
        received_per_person, left_on='id', right_on='person', how='left'
    ).merge(
        completed_per_person, left_on='id', right_on='person', how='left', suffixes=('', '_comp')
    ).merge(
        viewed_per_person, left_on='id', right_on='person', how='left', suffixes=('', '_view')
    )

    cohort_df['n_completed'] = cohort_df['n_completed'].fillna(0)
    cohort_df['n_viewed'] = cohort_df['n_viewed'].fillna(0)
    cohort_df['n_received'] = cohort_df['n_received'].fillna(0)

    cohort_agg = cohort_df.groupby('join_year').agg(
        n_customers=('id', 'count'),
        total_received=('n_received', 'sum'),
        total_completed=('n_completed', 'sum'),
        total_viewed=('n_viewed', 'sum')
    ).reset_index()

    cohort_agg['completion_rate'] = cohort_agg['total_completed'] / cohort_agg['total_received'].replace(0, np.nan)
    cohort_agg['view_rate'] = cohort_agg['total_viewed'] / cohort_agg['total_received'].replace(0, np.nan)
    cohort_agg['avg_completed_per_person'] = cohort_agg['total_completed'] / cohort_agg['n_customers']

    print("\nCohort Performance by Join Year:")
    print("-" * 80)
    print(f"{'Year':>6s}  {'Customers':>10s}  {'Received':>10s}  {'Completed':>10s}  "
          f"{'View Rate':>10s}  {'Compl Rate':>11s}  {'Avg Compl/P':>12s}")
    for _, row in cohort_agg.iterrows():
        print(f"{int(row['join_year']):>6d}  {int(row['n_customers']):>10d}  "
              f"{int(row['total_received']):>10d}  {int(row['total_completed']):>10d}  "
              f"{row['view_rate']:>10.4f}  {row['completion_rate']:>11.4f}  "
              f"{row['avg_completed_per_person']:>12.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Membership Cohort Analysis: Offer Response by Join Year', fontsize=14, fontweight='bold')

    axes[0].bar(cohort_agg['join_year'].astype(int), cohort_agg['n_customers'], color='steelblue')
    axes[0].set_title('Customers per Cohort')
    axes[0].set_xlabel('Join Year')
    axes[0].set_ylabel('Number of Customers')

    axes[1].bar(cohort_agg['join_year'].astype(int), cohort_agg['completion_rate'], color='seagreen')
    axes[1].set_title('Completion Rate by Cohort')
    axes[1].set_xlabel('Join Year')
    axes[1].set_ylabel('Completion Rate')
    axes[1].set_ylim(0, 1)

    axes[2].bar(cohort_agg['join_year'].astype(int), cohort_agg['avg_completed_per_person'], color='coral')
    axes[2].set_title('Avg Completed Offers per Person')
    axes[2].set_xlabel('Join Year')
    axes[2].set_ylabel('Avg Completed')

    plt.tight_layout()
    fig_dir = setup_figure_directory()
    plt.savefig(fig_dir / 'cohort_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {'cohort_agg': cohort_agg}


def channel_effectiveness_analysis(portfolio: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze which marketing channels (email, mobile, social, web) are most
    effective for each offer type based on view and completion rates.
    """
    print("\n" + "=" * 60)
    print("CHANNEL EFFECTIVENESS ANALYSIS")
    print("=" * 60)

    all_channels = ['email', 'mobile', 'social', 'web']

    channel_flags = portfolio.copy()
    for ch in all_channels:
        channel_flags[f'channel_{ch}'] = channel_flags['channels'].apply(
            lambda lst: 1 if isinstance(lst, list) and ch in lst else 0
        )

    transcript_expanded = transcript.copy()
    transcript_expanded['offer_id'] = transcript_expanded['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    offer_events = transcript_expanded[transcript_expanded['offer_id'].notna()].copy()
    offer_merged = offer_events.merge(channel_flags, left_on='offer_id', right_on='id', how='left')

    channel_stats = []
    for ch in all_channels:
        with_channel = offer_merged[offer_merged[f'channel_{ch}'] == 1]
        without_channel = offer_merged[offer_merged[f'channel_{ch}'] == 0]

        n_recv_w = (with_channel['event'] == 'offer received').sum()
        n_recv_wo = (without_channel['event'] == 'offer received').sum()
        n_view_w = (with_channel['event'] == 'offer viewed').sum()
        n_view_wo = (without_channel['event'] == 'offer viewed').sum()
        n_comp_w = (with_channel['event'] == 'offer completed').sum()
        n_comp_wo = (without_channel['event'] == 'offer completed').sum()

        view_rate_w = n_view_w / n_recv_w if n_recv_w > 0 else np.nan
        view_rate_wo = n_view_wo / n_recv_wo if n_recv_wo > 0 else np.nan
        comp_rate_w = n_comp_w / n_recv_w if n_recv_w > 0 else np.nan
        comp_rate_wo = n_comp_wo / n_recv_wo if n_recv_wo > 0 else np.nan

        channel_stats.append({
            'channel': ch,
            'view_rate_with': view_rate_w,
            'view_rate_without': view_rate_wo,
            'completion_rate_with': comp_rate_w,
            'completion_rate_without': comp_rate_wo,
            'n_received_with': n_recv_w,
            'n_received_without': n_recv_wo
        })

    channel_stats_df = pd.DataFrame(channel_stats)

    print("\nOverall Channel Effectiveness:")
    print("-" * 70)
    print(f"{'Channel':>10s}  {'View Rate (w/)':>15s}  {'View Rate (w/o)':>16s}  "
          f"{'Compl Rate (w/)':>16s}  {'Compl Rate (w/o)':>17s}")
    for _, row in channel_stats_df.iterrows():
        print(f"{row['channel']:>10s}  {row['view_rate_with']:>15.4f}  "
              f"{row['view_rate_without']:>16.4f}  "
              f"{row['completion_rate_with']:>16.4f}  "
              f"{row['completion_rate_without']:>17.4f}")

    by_type_stats = []
    for offer_type in offer_merged['offer_type'].dropna().unique():
        ot_data = offer_merged[offer_merged['offer_type'] == offer_type]
        for ch in all_channels:
            with_ch = ot_data[ot_data[f'channel_{ch}'] == 1]
            n_recv = (with_ch['event'] == 'offer received').sum()
            n_view = (with_ch['event'] == 'offer viewed').sum()
            n_comp = (with_ch['event'] == 'offer completed').sum()
            by_type_stats.append({
                'offer_type': offer_type,
                'channel': ch,
                'view_rate': n_view / n_recv if n_recv > 0 else np.nan,
                'completion_rate': n_comp / n_recv if n_recv > 0 else np.nan,
                'n_received': n_recv
            })

    by_type_df = pd.DataFrame(by_type_stats)

    print("\nChannel Effectiveness by Offer Type:")
    print("-" * 70)
    for offer_type in by_type_df['offer_type'].unique():
        print(f"\n  {offer_type.upper()}:")
        ot_rows = by_type_df[by_type_df['offer_type'] == offer_type]
        print(f"  {'Channel':>10s}  {'View Rate':>10s}  {'Compl Rate':>11s}  {'N Received':>11s}")
        for _, row in ot_rows.iterrows():
            vr = f"{row['view_rate']:.4f}" if not np.isnan(row['view_rate']) else "N/A"
            cr = f"{row['completion_rate']:.4f}" if not np.isnan(row['completion_rate']) else "N/A"
            print(f"  {row['channel']:>10s}  {vr:>10s}  {cr:>11s}  {int(row['n_received']):>11d}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Channel Effectiveness by Offer Type', fontsize=14, fontweight='bold')

    pivot_view = by_type_df.pivot(index='channel', columns='offer_type', values='view_rate')
    pivot_view.plot(kind='bar', ax=axes[0])
    axes[0].set_title('View Rate by Channel & Offer Type')
    axes[0].set_ylabel('View Rate')
    axes[0].set_xlabel('Channel')
    axes[0].tick_params(axis='x', rotation=45)

    pivot_comp = by_type_df.pivot(index='channel', columns='offer_type', values='completion_rate')
    pivot_comp.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Completion Rate by Channel & Offer Type')
    axes[1].set_ylabel('Completion Rate')
    axes[1].set_xlabel('Channel')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig_dir = setup_figure_directory()
    plt.savefig(fig_dir / 'channel_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'channel_stats': channel_stats_df,
        'channel_by_type': by_type_df
    }


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
        },
        'hypothesis_testing': results.get('hypothesis_testing', {}),
        'effect_sizes': results.get('effect_sizes', {}),
        'confidence_intervals': results.get('confidence_intervals', {}),
        'correlation_analysis': {
            'pearson_completion_rate': results.get('correlation_analysis', {}).get('pearson_r', pd.DataFrame()).get('completion_rate', {}).to_dict() if 'correlation_analysis' in results else {},
            'spearman_completion_rate': results.get('correlation_analysis', {}).get('spearman_r', pd.DataFrame()).get('completion_rate', {}).to_dict() if 'correlation_analysis' in results else {}
        },
        'cohort_analysis': results.get('cohort_analysis', {}).get('cohort_agg', pd.DataFrame()).to_dict('records') if 'cohort_analysis' in results else [],
        'channel_effectiveness': results.get('channel_effectiveness', {}).get('channel_stats', pd.DataFrame()).to_dict('records') if 'channel_effectiveness' in results else []
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
    
    # 5. Hypothesis testing
    results['hypothesis_testing'] = hypothesis_testing(profile, transcript, portfolio)
    
    # 6. Effect sizes
    results['effect_sizes'] = effect_size_analysis(profile, transcript)
    
    # 7. Confidence intervals
    results['confidence_intervals'] = confidence_interval_analysis(profile, transcript, portfolio)
    
    # 8. Correlation analysis
    results['correlation_analysis'] = correlation_analysis(portfolio, transcript)
    
    # 9. Cohort analysis
    results['cohort_analysis'] = cohort_analysis(profile, transcript)
    
    # 10. Channel effectiveness
    results['channel_effectiveness'] = channel_effectiveness_analysis(portfolio, transcript)
    
    # Generate summary report
    generate_eda_report(results)
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)
    print(f" All visualizations saved to: {fig_dir}")
    print(f" Summary report saved to: reports/eda_summary.json")
    print("="*60)
