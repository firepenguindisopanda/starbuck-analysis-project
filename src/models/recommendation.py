"""
Causal Inference & Recommendation System module for Starbucks project.

This module:
1. Calculates Average Treatment Effect (ATE) of offers on transaction spend
2. Builds a rule-based offer recommendation system
3. Simulates lift from personalized vs. random targeting

Answers research questions:
- "What is the ATE of sending BOGO vs discount offers on transaction spend?"
- "Can we design a recommendation system that improves campaign performance by >10%?"

Follows Data Scientist principles: causal thinking, measurable business impact, explainable rules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Tuple
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def load_data_for_causal_analysis(base_path: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data needed for causal inference analysis.
    
    Args:
        base_path: Base directory
        
    Returns:
        Tuple of (portfolio, profile, transcript)
    """
    portfolio = pd.read_json(f"{base_path}/portfolio.json", lines=True)
    profile = pd.read_json(f"{base_path}/profile.json", lines=True)
    transcript = pd.read_json(f"{base_path}/transcript.json", lines=True)
    
    # Extract offer_id and transaction amounts
    transcript['offer_id'] = transcript['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    transcript['transaction_amount'] = transcript['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) and 'amount' in x else None
    )
    
    return portfolio, profile, transcript


def calculate_ate_by_offer_type(portfolio: pd.DataFrame, transcript: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Average Treatment Effect (ATE) of offers on transaction spend.
    
      LIMITATION: This is a simplified ATE calculation comparing mean transaction
    amounts between customers who received offers vs. those who did not. It does
    NOT adjust for confounding variables (e.g., income, engagement level) that
    influence both offer assignment and spending. Results should be interpreted as
    correlational, not strictly causal.
    
    For a more rigorous causal analysis, see the "Future Work" section notes on
    propensity score matching (PSM) and doubly-robust estimation.
    
    Args:
        portfolio: Portfolio DataFrame
        transcript: Transcript DataFrame
        
    Returns:
        Dictionary with ATE results by offer type
    """
    print("\n" + "="*60)
    print("CAUSAL INFERENCE: AVERAGE TREATMENT EFFECT (ATE)")
    print("="*60)
    
    # Get transactions with and without offer exposure
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['transaction_amount']
    
    # Get offer events
    offer_events = transcript[transcript['offer_id'].notna()].copy()
    
    # Merge to get offer types
    offer_events = offer_events.merge(
        portfolio[['id', 'offer_type']], 
        left_on='offer_id', 
        right_on='id', 
        how='left'
    )
    
    # For each transaction, determine which offer type (if any) the customer was exposed to
    # We'll use a simplified approach: compare transaction amounts for customers
    # who received offers vs those who didn't during the same time period
    
    # Get unique customers
    all_customers = set(transcript['person'].unique())
    customers_with_offers = set(offer_events['person'].unique())
    customers_without_offers = all_customers - customers_with_offers
    
    print(f"\nCustomer groups:")
    print(f"  Customers with offers: {len(customers_with_offers):,}")
    print(f"  Customers without offers: {len(customers_without_offers):,}")
    
    # Calculate average transaction amount by group
    trans_with_offers = transactions[transactions['person'].isin(customers_with_offers)]
    trans_without_offers = transactions[transactions['person'].isin(customers_without_offers)]
    
    ate_results = {}
    
    # Overall ATE (any offer vs no offer)
    ate_results['any_offer'] = {
        'control_mean': trans_without_offers['amount'].mean() if len(trans_without_offers) > 0 else 0,
        'treatment_mean': trans_with_offers['amount'].mean() if len(trans_with_offers) > 0 else 0,
        'control_count': len(trans_without_offers),
        'treatment_count': len(trans_with_offers)
    }
    ate_results['any_offer']['ate'] = (
        ate_results['any_offer']['treatment_mean'] - ate_results['any_offer']['control_mean']
    )
    ate_results['any_offer']['ate_percent'] = (
        ate_results['any_offer']['ate'] / ate_results['any_offer']['control_mean'] * 100
        if ate_results['any_offer']['control_mean'] > 0 else 0
    )
    
    print(f"\nATE: Any Offer vs No Offer")
    print(f"  Control (no offer) avg transaction: ${ate_results['any_offer']['control_mean']:.2f}")
    print(f"  Treatment (any offer) avg transaction: ${ate_results['any_offer']['treatment_mean']:.2f}")
    print(f"  ATE: ${ate_results['any_offer']['ate']:.2f} ({ate_results['any_offer']['ate_percent']:.1f}%)")
    
    # ATE by offer type
    for offer_type in ['bogo', 'discount', 'informational']:
        # Customers who received this offer type
        type_offers = offer_events[offer_events['offer_type'] == offer_type]
        type_customers = set(type_offers['person'].unique())
        
        # Customers who did NOT receive this offer type
        non_type_customers = all_customers - type_customers
        
        # Transaction amounts
        type_trans = transactions[transactions['person'].isin(type_customers)]
        non_type_trans = transactions[transactions['person'].isin(non_type_customers)]
        
        ate_results[offer_type] = {
            'control_mean': non_type_trans['amount'].mean() if len(non_type_trans) > 0 else 0,
            'treatment_mean': type_trans['amount'].mean() if len(type_trans) > 0 else 0,
            'control_count': len(non_type_trans),
            'treatment_count': len(type_trans)
        }
        ate_results[offer_type]['ate'] = (
            ate_results[offer_type]['treatment_mean'] - ate_results[offer_type]['control_mean']
        )
        ate_results[offer_type]['ate_percent'] = (
            ate_results[offer_type]['ate'] / ate_results[offer_type]['control_mean'] * 100
            if ate_results[offer_type]['control_mean'] > 0 else 0
        )
        
        print(f"\nATE: {offer_type.upper()} vs No {offer_type.upper()}")
        print(f"  Control (no {offer_type}) avg transaction: ${ate_results[offer_type]['control_mean']:.2f}")
        print(f"  Treatment ({offer_type}) avg transaction: ${ate_results[offer_type]['treatment_mean']:.2f}")
        print(f"  ATE: ${ate_results[offer_type]['ate']:.2f} ({ate_results[offer_type]['ate_percent']:.1f}%)")
    
    return ate_results


def build_recommendation_rules(customer_clusters_path: str = 'data/processed/customer_clusters.csv',
                                cluster_profiles_path: str = 'data/processed/cluster_profiles.csv',
                                portfolio_path: str = 'portfolio.json') -> Dict[str, Any]:
    """
    Build rule-based offer recommendation system based on cluster analysis.
    
    Args:
        customer_clusters_path: Path to customer cluster assignments
        cluster_profiles_path: Path to cluster profiles
        portfolio_path: Path to portfolio JSON
        
    Returns:
        Dictionary with recommendation rules and expected performance
    """
    print("\n" + "="*60)
    print("BUILDING RULE-BASED RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Load data
    customer_clusters = pd.read_csv(customer_clusters_path)
    cluster_profiles = pd.read_csv(cluster_profiles_path)
    portfolio = pd.read_json(portfolio_path, lines=True)
    
    # Load customer features for behavioral data
    customer_features = pd.read_csv('data/processed/customer_features.csv')
    
    # Merge cluster assignments with features
    customers_with_clusters = customer_features.merge(customer_clusters, on='id', how='left')
    
    # Define recommendation rules based on cluster profiles
    # From Phase 4 clustering results:
    # Cluster 0: Unknown gender, low completion (11.4%) - avoid sending offers
    # Cluster 1: High discount completion (69.7%) - send discount offers
    # Cluster 2: High BOGO completion (70.9%) - send BOGO offers
    # Cluster 3: Low completion (14.8%) - send informational or no offers
    
    rules = {
        0: {'primary': 'informational', 'secondary': None, 'rationale': 'Unknown demographics, low engagement'},
        1: {'primary': 'discount', 'secondary': 'bogo', 'rationale': 'High discount completion rate (69.7%)'},
        2: {'primary': 'bogo', 'secondary': 'discount', 'rationale': 'High BOGO completion rate (70.9%)'},
        3: {'primary': 'informational', 'secondary': None, 'rationale': 'Low completion rates across all types'}
    }
    
    print("\nRecommendation Rules by Cluster:")
    for cluster_id, rule in rules.items():
        cluster_data = cluster_profiles[cluster_profiles['cluster_id'] == cluster_id].iloc[0]
        print(f"\n  Cluster {cluster_id} ({int(cluster_data['size'])} customers):")
        print(f"    Primary offer: {rule['primary']}")
        print(f"    Secondary offer: {rule['secondary']}")
        print(f"    Rationale: {rule['rationale']}")
    
    # Calculate expected performance improvement
    # Simulate: for each customer, recommend primary offer and calculate expected completion rate
    
    # Get offer completion rates by type from historical data
    transcript = pd.read_json('transcript.json', lines=True)
    transcript['offer_id'] = transcript['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    
    # Calculate historical completion rates by offer type
    completed = transcript[transcript['event'] == 'offer completed']
    received = transcript[transcript['event'] == 'offer received']
    
    # Merge with portfolio to get offer types
    portfolio_clean = portfolio.copy()
    completed_merged = completed.merge(
        portfolio_clean[['id', 'offer_type']], 
        left_on='offer_id', 
        right_on='id', 
        how='left'
    )
    received_merged = received.merge(
        portfolio_clean[['id', 'offer_type']], 
        left_on='offer_id', 
        right_on='id', 
        how='left'
    )
    
    # Completion rate by offer type
    completion_by_type = {}
    for offer_type in ['bogo', 'discount', 'informational']:
        type_received = received_merged[received_merged['offer_type'] == offer_type]['person'].nunique()
        type_completed = completed_merged[completed_merged['offer_type'] == offer_type]['person'].nunique()
        completion_by_type[offer_type] = type_completed / type_received if type_received > 0 else 0
    
    print(f"\nHistorical Completion Rates by Offer Type:")
    for offer_type, rate in completion_by_type.items():
        print(f"  {offer_type}: {rate:.1%}")
    
    # Simulate random targeting (baseline)
    # Assume random assignment proportional to historical distribution
    total_offers = received_merged['person'].count()
    random_completion_rate = sum(completion_by_type.values()) / len(completion_by_type)
    
    # Simulate rule-based targeting
    rule_based_completions = 0
    total_customers = len(customers_with_clusters)
    
    for cluster_id, rule in rules.items():
        cluster_customers = customers_with_clusters[customers_with_clusters['cluster'] == cluster_id]
        n_customers = len(cluster_customers)
        
        # Primary offer completion rate for this cluster
        primary_type = rule['primary']
        # Use cluster-specific completion rate if available, otherwise use historical
        cluster_col = f"{primary_type}_completion_rate"
        if cluster_col in cluster_customers.columns:
            cluster_rate = cluster_customers[cluster_col].mean()
        else:
            cluster_rate = completion_by_type.get(primary_type, 0)
        
        rule_based_completions += n_customers * cluster_rate
    
    rule_based_completion_rate = rule_based_completions / total_customers
    
    print(f"\nPerformance Simulation:")
    print(f"  Random targeting (baseline): {random_completion_rate:.1%}")
    print(f"  Rule-based targeting: {rule_based_completion_rate:.1%}")
    print(f"  Lift: {((rule_based_completion_rate / random_completion_rate) - 1):.1%}")
    
    return {
        'rules': rules,
        'completion_by_type': completion_by_type,
        'random_completion_rate': random_completion_rate,
        'rule_based_completion_rate': rule_based_completion_rate,
        'lift_percent': ((rule_based_completion_rate / random_completion_rate) - 1) * 100
    }


def visualize_causal_results(ate_results: Dict, recommendation_results: Dict, fig_dir: Path) -> None:
    """
    Visualize causal inference and recommendation system results.
    
    Args:
        ate_results: ATE calculation results
        recommendation_results: Recommendation system results
        fig_dir: Directory to save figures
    """
    print("\nCreating causal inference visualizations...")
    
    # Figure 1: ATE by offer type
    offer_types = ['any_offer', 'bogo', 'discount', 'informational']
    ate_values = [ate_results[t]['ate'] for t in offer_types]
    ate_percent = [ate_results[t]['ate_percent'] for t in offer_types]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Average Treatment Effect (ATE) by Offer Type', fontsize=16, fontweight='bold')
    
    # ATE in dollars
    bars1 = axes[0].bar(offer_types, ate_values, color='skyblue', edgecolor='black')
    axes[0].set_ylabel('ATE ($)')
    axes[0].set_title('ATE: Treatment - Control ($)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars1, ate_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'${val:.2f}', ha='center', va='bottom' if val >= 0 else 'top')
    
    # ATE in percent
    bars2 = axes[1].bar(offer_types, ate_percent, color='lightgreen', edgecolor='black')
    axes[1].set_ylabel('ATE (%)')
    axes[1].set_title('ATE: Percent Lift (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars2, ate_percent):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'ate_by_offer_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Recommendation system performance comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Random Targeting', 'Rule-Based Targeting']
    completion_rates = [
        recommendation_results['random_completion_rate'],
        recommendation_results['rule_based_completion_rate']
    ]
    
    bars = ax.bar(methods, completion_rates, color=['gray', 'green'], edgecolor='black', alpha=0.7)
    ax.set_ylabel('Completion Rate')
    ax.set_title('Recommendation System Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(completion_rates) * 1.2])
    
    # Add value labels
    for bar, rate in zip(bars, completion_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add lift annotation
    lift = recommendation_results['lift_percent']
    ax.text(1, max(completion_rates) * 1.1, f'Lift: +{lift:.1f}%', 
             ha='center', fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'recommendation_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Plotly interactive ATE visualization
    ate_df = pd.DataFrame({
        'Offer Type': offer_types,
        'ATE ($)': ate_values,
        'ATE (%)': ate_percent
    })
    
    fig = px.bar(
        ate_df, x='Offer Type', y='ATE (%)',
        title='Average Treatment Effect (ATE) by Offer Type',
        color='ATE (%)',
        color_continuous_scale='RdYlGn',
        text='ATE (%)'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(width=800, height=500)
    fig.write_html(fig_dir / 'ate_interactive.html')
    
    print(f" Causal inference visualizations saved to {fig_dir}")


def save_recommendation_system(rules: Dict, recommendation_results: Dict, 
                               base_path: str = '.') -> None:
    """
    Save recommendation system rules and results.
    
    Args:
        rules: Recommendation rules by cluster
        recommendation_results: Performance results
        base_path: Base directory
    """
    output_dir = Path(base_path) / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save rules
    rules_df = pd.DataFrame([
        {'cluster_id': k, 'primary_offer': v['primary'], 'secondary_offer': v['secondary'], 'rationale': v['rationale']}
        for k, v in rules.items()
    ])
    rules_df.to_csv(output_dir / 'recommendation_rules.csv', index=False)
    
    # Save results
    with open(output_dir / 'recommendation_results.json', 'w') as f:
        json.dump(recommendation_results, f, indent=2, default=str)
    
    print(f"\n Recommendation system saved to {output_dir}")


def generate_causal_report(ate_results: Dict, recommendation_results: Dict,
                           output_path: str = 'reports/causal_report.json') -> None:
    """
    Generate a comprehensive causal inference report.
    
    Args:
        ate_results: ATE calculation results
        recommendation_results: Recommendation system results
        output_path: Path to save the report
    """
    report = {
        'ate_results': ate_results,
        'recommendation_results': {
            'rules': recommendation_results['rules'],
            'random_completion_rate': float(recommendation_results['random_completion_rate']),
            'rule_based_completion_rate': float(recommendation_results['rule_based_completion_rate']),
            'lift_percent': float(recommendation_results['lift_percent'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n Causal inference report saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - CAUSAL INFERENCE & RECOMMENDATION")
    print("="*60)
    
    # Setup
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    portfolio, profile, transcript = load_data_for_causal_analysis()
    print(f" Loaded {len(portfolio)} offers, {len(profile)} customers, {len(transcript)} events")
    
    # Calculate ATE
    ate_results = calculate_ate_by_offer_type(portfolio, transcript)
    
    # Build recommendation system
    recommendation_results = build_recommendation_rules()
    rules = recommendation_results['rules']
    
    # Visualize results
    visualize_causal_results(ate_results, recommendation_results, fig_dir)
    
    # Save results
    save_recommendation_system(rules, recommendation_results)
    
    # Generate report
    generate_causal_report(ate_results, recommendation_results)
    
    print("\n" + "="*60)
    print("CAUSAL INFERENCE & RECOMMENDATION COMPLETE")
    print("="*60)
    print(f" ATE calculated for all offer types")
    print(f" Recommendation system built with {len(rules)} rules")
    print(f" Expected lift: +{recommendation_results['lift_percent']:.1f}%")
    print(f" Visualizations saved to: {fig_dir}")
    print("="*60)
