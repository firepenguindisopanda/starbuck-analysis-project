"""
Causal Inference & Recommendation System module for Starbucks project.

This module:
1. Calculates Average Treatment Effect (ATE) of offers on transaction spend
2. Builds a rule-based offer recommendation system
3. Simulates lift from personalized vs. random targeting
4. Provides bootstrap confidence intervals for ATE
5. Performs propensity score matching for causal adjustment
6. Conducts statistical significance tests (Welch's t-test, Cohen's d)
7. Analyzes heterogeneous treatment effects by customer segment
8. Simulates A/B test framework for targeting validation

Answers research questions:
- "What is the ATE of sending BOGO vs discount offers on transaction spend?"
- "Can we design a recommendation system that improves campaign performance by >10%?"
- "Are observed treatment effects statistically significant?"
- "Which customer segments benefit most from each offer type?"

Follows Data Scientist principles: causal thinking, measurable business impact, explainable rules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Tuple, Optional, List
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    
    For a more rigorous causal analysis, see propensity_score_matching() and
    bootstrap_ate_ci() below.
    
    Args:
        portfolio: Portfolio DataFrame
        transcript: Transcript DataFrame
        
    Returns:
        Dictionary with ATE results by offer type
    """
    print("\n" + "="*60)
    print("CAUSAL INFERENCE: AVERAGE TREATMENT EFFECT (ATE)")
    print("="*60)
    
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['transaction_amount']
    
    offer_events = transcript[transcript['offer_id'].notna()].copy()
    
    offer_events = offer_events.merge(
        portfolio[['id', 'offer_type']], 
        left_on='offer_id', 
        right_on='id', 
        how='left'
    )
    
    all_customers = set(transcript['person'].unique())
    customers_with_offers = set(offer_events['person'].unique())
    customers_without_offers = all_customers - customers_with_offers
    
    print(f"\nCustomer groups:")
    print(f"  Customers with offers: {len(customers_with_offers):,}")
    print(f"  Customers without offers: {len(customers_without_offers):,}")
    
    trans_with_offers = transactions[transactions['person'].isin(customers_with_offers)]
    trans_without_offers = transactions[transactions['person'].isin(customers_without_offers)]
    
    ate_results = {}
    
    ate_results['any_offer'] = {
        'control_mean': trans_without_offers['amount'].mean() if len(trans_without_offers) > 0 else 0,
        'treatment_mean': trans_with_offers['amount'].mean() if len(trans_with_offers) > 0 else 0,
        'control_count': len(trans_without_offers),
        'treatment_count': len(trans_with_offers),
        'control_amounts': trans_without_offers['amount'].values,
        'treatment_amounts': trans_with_offers['amount'].values
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
    
    for offer_type in ['bogo', 'discount', 'informational']:
        type_offers = offer_events[offer_events['offer_type'] == offer_type]
        type_customers = set(type_offers['person'].unique())
        
        non_type_customers = all_customers - type_customers
        
        type_trans = transactions[transactions['person'].isin(type_customers)]
        non_type_trans = transactions[transactions['person'].isin(non_type_customers)]
        
        ate_results[offer_type] = {
            'control_mean': non_type_trans['amount'].mean() if len(non_type_trans) > 0 else 0,
            'treatment_mean': type_trans['amount'].mean() if len(type_trans) > 0 else 0,
            'control_count': len(non_type_trans),
            'treatment_count': len(type_trans),
            'control_amounts': non_type_trans['amount'].values,
            'treatment_amounts': type_trans['amount'].values
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


def bootstrap_ate_ci(ate_results: Dict[str, Any], n_bootstrap: int = 1000, 
                     confidence: float = 0.95, random_state: int = 42) -> Dict[str, Any]:
    """
    Compute bootstrap confidence intervals for ATE estimates.
    
    Resamples with replacement n_bootstrap times, computes ATE for each
    bootstrap sample, and returns the confidence interval at the specified level.
    
    Args:
        ate_results: Output from calculate_ate_by_offer_type()
        n_bootstrap: Number of bootstrap iterations (default 1000)
        confidence: Confidence level for interval (default 0.95)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with bootstrap CI results per offer type, including
        whether the CI includes 0 (statistical significance indicator)
    """
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS FOR ATE")
    print("="*60)
    
    rng = np.random.RandomState(random_state)
    alpha = 1 - confidence
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100
    
    bootstrap_results = {}
    
    for offer_type in ['any_offer', 'bogo', 'discount', 'informational']:
        control_data = ate_results[offer_type]['control_amounts']
        treatment_data = ate_results[offer_type]['treatment_amounts']
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            bootstrap_results[offer_type] = {
                'ci_lower': 0, 'ci_upper': 0,
                'ci_includes_zero': True, 'significant': False,
                'bootstrap_ates': []
            }
            print(f"\n  {offer_type}: Insufficient data for bootstrap")
            continue
        
        bootstrap_ates = []
        for _ in range(n_bootstrap):
            ctrl_sample = rng.choice(control_data, size=len(control_data), replace=True)
            treat_sample = rng.choice(treatment_data, size=len(treatment_data), replace=True)
            bootstrap_ates.append(treat_sample.mean() - ctrl_sample.mean())
        
        bootstrap_ates = np.array(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, lower_pct)
        ci_upper = np.percentile(bootstrap_ates, upper_pct)
        ci_includes_zero = ci_lower <= 0 <= ci_upper
        
        bootstrap_results[offer_type] = {
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_includes_zero': bool(ci_includes_zero),
            'significant': not ci_includes_zero,
            'bootstrap_mean': float(np.mean(bootstrap_ates)),
            'bootstrap_std': float(np.std(bootstrap_ates)),
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence
        }
        
        sig_marker = "***" if not ci_includes_zero else ""
        print(f"\n  {offer_type.upper()}:")
        print(f"    ATE: ${ate_results[offer_type]['ate']:.2f}")
        print(f"    {confidence*100:.0f}% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")
        print(f"    CI includes 0: {ci_includes_zero}")
        print(f"    Statistically significant: {not ci_includes_zero} {sig_marker}")
    
    n_significant = sum(1 for r in bootstrap_results.values() if r.get('significant', False))
    print(f"\n  {n_significant}/{len(bootstrap_results)} offer types have statistically significant ATE")
    
    return bootstrap_results


def propensity_score_matching(portfolio: pd.DataFrame, profile: pd.DataFrame,
                               transcript: pd.DataFrame,
                               customer_features_path: str = 'data/processed/customer_features.csv'
                               ) -> Dict[str, Any]:
    """
    Perform propensity score matching to adjust for confounding in ATE estimation.
    
    Fits a logistic regression to estimate P(treatment | covariates) using
    age, income, gender, and tenure. Matches treated and control units using
    nearest-neighbor matching on propensity scores, then computes ATE on
    the matched sample. Reports balance diagnostics (standardized mean
    differences before and after matching).
    
    Args:
        portfolio: Portfolio DataFrame
        profile: Profile DataFrame
        transcript: Transcript DataFrame
        customer_features_path: Path to customer features CSV
        
    Returns:
        Dictionary with PSM results, including matched ATE, propensity model
        metrics, and balance diagnostics
    """
    print("\n" + "="*60)
    print("PROPENSITY SCORE MATCHING (PSM)")
    print("="*60)
    
    customer_features = pd.read_csv(customer_features_path)
    
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['transaction_amount']
    
    offer_events = transcript[transcript['offer_id'].notna()].copy()
    offer_events = offer_events.merge(
        portfolio[['id', 'offer_type']], left_on='offer_id', right_on='id', how='left'
    )
    
    all_customers = set(transcript['person'].unique())
    customers_with_offers = set(offer_events['person'].unique())
    customers_without_offers = all_customers - customers_with_offers
    
    profile_clean = profile.copy()
    profile_clean['gender'] = profile_clean['gender'].fillna('Unknown')
    profile_clean['age'] = profile_clean['age'].fillna(profile_clean['age'].median())
    profile_clean['income'] = profile_clean['income'].fillna(profile_clean['income'].median())
    
    profile_clean['tenure_days'] = (
        pd.Timestamp('2018-07-26') - pd.to_datetime(profile_clean['became_member_on'], format='%Y%m%d')
    ).dt.days
    profile_clean['tenure_months'] = profile_clean['tenure_days'] / 30.44
    profile_clean['tenure_months'] = profile_clean['tenure_months'].fillna(profile_clean['tenure_months'].median())
    
    gender_dummies = pd.get_dummies(profile_clean['gender'], prefix='gender')
    profile_features = pd.concat([
        profile_clean[['id', 'age', 'income', 'tenure_months']],
        gender_dummies
    ], axis=1)
    
    psm_results = {}
    
    for offer_type in ['bogo', 'discount', 'informational']:
        print(f"\n--- Propensity Score Matching for {offer_type.upper()} ---")
        
        type_offers = offer_events[offer_events['offer_type'] == offer_type]
        type_customers = set(type_offers['person'].unique())
        
        treated_ids = type_customers & set(profile_features['id'].values)
        control_ids = customers_without_offers & set(profile_features['id'].values)
        
        if not treated_ids or not control_ids:
            print(f"  Skipping {offer_type}: insufficient treated/control units")
            continue
        
        treated_df = profile_features[profile_features['id'].isin(treated_ids)].copy()
        control_df = profile_features[profile_features['id'].isin(control_ids)].copy()
        
        treated_df['treatment'] = 1
        control_df['treatment'] = 0
        
        combined = pd.concat([treated_df, control_df], ignore_index=True)
        
        covariate_cols = [c for c in combined.columns if c not in ['id', 'treatment']]
        covariate_cols = [c for c in covariate_cols if combined[c].dtype in ['float64', 'int64', 'bool', 'uint8']]
        
        X = combined[covariate_cols].fillna(0).values
        y = combined['treatment'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, y)
        
        combined['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
        
        try:
            auc = roc_auc_score(y, combined['propensity_score'])
        except ValueError:
            auc = 0.5
        print(f"  Propensity model AUC: {auc:.3f}")
        
        treated_ps = combined[combined['treatment'] == 1][['id', 'propensity_score']].copy()
        control_ps = combined[combined['treatment'] == 0][['id', 'propensity_score']].copy()
        
        matched_treated = []
        matched_control = []
        used_control_indices = set()
        
        control_ps_sorted = control_ps.reset_index(drop=True)
        
        for _, t_row in treated_ps.iterrows():
            t_ps = t_row['propensity_score']
            distances = np.abs(control_ps_sorted['propensity_score'].values - t_ps)
            
            available_mask = ~control_ps_sorted.index.isin(used_control_indices)
            if not available_mask.any():
                break
            
            available_indices_arr = np.where(available_mask)[0]
            if len(available_indices_arr) == 0:
                break
            available_distances = distances[available_indices_arr]
            best_local_pos = np.argmin(available_distances)
            best_local_idx = available_indices_arr[best_local_pos]
            
            if available_distances[best_local_pos] < 0.2:
                matched_treated.append(t_row['id'])
                matched_control.append(control_ps_sorted.loc[best_local_idx, 'id'])
                used_control_indices.add(best_local_idx)
        
        n_matched = len(matched_treated)
        print(f"  Matched pairs: {n_matched}/{len(treated_ps)}")
        
        if n_matched < 10:
            print(f"  Skipping {offer_type}: too few matched pairs ({n_matched})")
            continue
        
        trans_by_person = transactions.groupby('person')['amount'].mean()
        
        treated_means = [trans_by_person.get(pid, 0) for pid in matched_treated]
        control_means = [trans_by_person.get(pid, 0) for pid in matched_control]
        
        matched_ate = np.mean(treated_means) - np.mean(control_means)
        
        balance_before = {}
        balance_after = {}
        for col in covariate_cols[:6]:
            t_vals_all = combined[combined['treatment'] == 1][col].values
            c_vals_all = combined[combined['treatment'] == 0][col].values
            smd_before = (t_vals_all.mean() - c_vals_all.mean()) / np.sqrt(
                (t_vals_all.std()**2 + c_vals_all.std()**2) / 2
            ) if (t_vals_all.std() > 0 or c_vals_all.std() > 0) else 0
            balance_before[col] = float(smd_before)
            
            treated_matched_df = combined[combined['id'].isin(matched_treated)]
            control_matched_df = combined[combined['id'].isin(matched_control)]
            t_vals_m = treated_matched_df[col].values
            c_vals_m = control_matched_df[col].values
            if len(t_vals_m) > 0 and len(c_vals_m) > 0:
                smd_after = (t_vals_m.mean() - c_vals_m.mean()) / np.sqrt(
                    (t_vals_m.std()**2 + c_vals_m.std()**2) / 2
                ) if (t_vals_m.std() > 0 or c_vals_m.std() > 0) else 0
            else:
                smd_after = 0
            balance_after[col] = float(smd_after)
        
        psm_results[offer_type] = {
            'ate_naive': ate_results_by_type if 'ate_results_by_type' in dir() else None,
            'ate_matched': float(matched_ate),
            'n_matched_pairs': n_matched,
            'n_treated_total': len(treated_ps),
            'n_control_total': len(control_ps),
            'propensity_auc': float(auc),
            'balance_before': balance_before,
            'balance_after': balance_after,
            'matched_treated_ids': matched_treated,
            'matched_control_ids': matched_control
        }
        
        naive_ate_for_type = psm_results[offer_type].pop('ate_naive', None)
        
        print(f"  Matched ATE: ${matched_ate:.2f}")
        print(f"  Balance diagnostics (top 6 covariates):")
        for col in list(balance_before.keys())[:6]:
            print(f"    {col}: SMD before={balance_before[col]:.3f}, after={balance_after[col]:.3f}")
    
    return psm_results


def statistical_significance_tests(ate_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Welch's t-test and compute Cohen's d effect size for each ATE comparison.
    
    Args:
        ate_results: Output from calculate_ate_by_offer_type()
        
    Returns:
        Dictionary with test results per offer type: t-statistic, p-value,
        degrees of freedom, Cohen's d, and significance flag
    """
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    
    test_results = {}
    
    for offer_type in ['any_offer', 'bogo', 'discount', 'informational']:
        control = ate_results[offer_type]['control_amounts']
        treatment = ate_results[offer_type]['treatment_amounts']
        
        if len(control) < 2 or len(treatment) < 2:
            print(f"\n  {offer_type.upper()}: Insufficient data for testing")
            test_results[offer_type] = {
                't_statistic': None, 'p_value': None,
                'cohens_d': None, 'significant': None
            }
            continue
        
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        pooled_std = np.sqrt(
            ((len(treatment) - 1) * np.var(treatment, ddof=1) +
             (len(control) - 1) * np.var(control, ddof=1)) /
            (len(treatment) + len(control) - 2)
        )
        cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std if pooled_std > 0 else 0
        
        if abs(cohens_d) < 0.2:
            effect_label = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_label = "small"
        elif abs(cohens_d) < 0.8:
            effect_label = "medium"
        else:
            effect_label = "large"
        
        test_results[offer_type] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'degrees_freedom': int(len(treatment) + len(control) - 2),
            'cohens_d': float(cohens_d),
            'effect_size_label': effect_label,
            'significant': p_value < 0.05,
            'control_n': int(len(control)),
            'treatment_n': int(len(treatment)),
            'control_mean': float(np.mean(control)),
            'treatment_mean': float(np.mean(treatment))
        }
        
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"\n  {offer_type.upper()}:")
        print(f"    Welch's t-test: t={t_stat:.2f}, p={p_value:.2e} {sig_marker}")
        print(f"    Cohen's d: {cohens_d:.4f} ({effect_label} effect)")
        print(f"    Significant at α=0.05: {p_value < 0.05}")
        print(f"    Treatment n={len(treatment):,}, Control n={len(control):,}")
    
    return test_results


def analyze_heterogeneous_effects(portfolio: pd.DataFrame, transcript: pd.DataFrame,
                                   customer_clusters_path: str = 'data/processed/customer_clusters.csv',
                                   fig_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Compute ATE by customer segment (cluster) and identify which segments
    benefit most from each offer type. Creates a heatmap visualization.
    
    Args:
        portfolio: Portfolio DataFrame
        transcript: Transcript DataFrame
        customer_clusters_path: Path to customer cluster assignments CSV
        fig_dir: Directory to save heatmap figure (optional)
        
    Returns:
        Dictionary with segment-level ATE results and best offer per segment
    """
    print("\n" + "="*60)
    print("HETEROGENEOUS TREATMENT EFFECTS BY SEGMENT")
    print("="*60)
    
    customer_clusters = pd.read_csv(customer_clusters_path)
    
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    transactions['amount'] = transactions['transaction_amount']
    
    offer_events = transcript[transcript['offer_id'].notna()].copy()
    offer_events = offer_events.merge(
        portfolio[['id', 'offer_type']], left_on='offer_id', right_on='id', how='left'
    )
    
    all_customers = set(transcript['person'].unique())
    customers_with_offers = set(offer_events['person'].unique())
    customers_without_offers = all_customers - customers_with_offers
    
    trans_by_person = transactions.groupby('person')['amount'].mean().reset_index()
    trans_by_person.columns = ['person', 'avg_amount']
    
    cluster_mapping = dict(zip(customer_clusters['id'], customer_clusters['cluster']))
    
    unique_clusters = sorted(customer_clusters['cluster'].unique())
    offer_types = ['bogo', 'discount', 'informational']
    
    het_results = {}
    heatmap_data = []
    
    for cluster_id in unique_clusters:
        cluster_customer_ids = set(
            customer_clusters[customer_clusters['cluster'] == cluster_id]['id'].values
        )
        cluster_customers = cluster_customer_ids & all_customers
        
        if len(cluster_customers) < 5:
            continue
        
        cluster_trans = trans_by_person[trans_by_person['person'].isin(cluster_customers)]
        overall_ate_cluster = None
        
        het_results[int(cluster_id)] = {
            'n_customers': len(cluster_customers),
            'offer_ates': {}
        }
        
        for offer_type in offer_types:
            type_offers = offer_events[offer_events['offer_type'] == offer_type]
            type_customers = (set(type_offers['person'].unique()) & cluster_customers)
            
            non_type_customers = all_customers - set(type_offers['person'].unique())
            non_type_in_cluster = non_type_customers & cluster_customers
            
            if len(type_customers) < 3 or len(non_type_in_cluster) < 3:
                het_results[int(cluster_id)]['offer_ates'][offer_type] = {
                    'ate': None, 'ate_percent': None,
                    'treatment_n': len(type_customers), 'control_n': len(non_type_in_cluster)
                }
                heatmap_data.append({
                    'cluster': int(cluster_id),
                    'offer_type': offer_type,
                    'ate': np.nan
                })
                continue
            
            type_trans = cluster_trans[cluster_trans['person'].isin(type_customers)]
            ctrl_trans = cluster_trans[cluster_trans['person'].isin(non_type_in_cluster)]
            
            treatment_mean = type_trans['avg_amount'].mean() if len(type_trans) > 0 else 0
            control_mean = ctrl_trans['avg_amount'].mean() if len(ctrl_trans) > 0 else 0
            ate = treatment_mean - control_mean
            ate_pct = (ate / control_mean * 100) if control_mean > 0 else 0
            
            het_results[int(cluster_id)]['offer_ates'][offer_type] = {
                'ate': float(ate),
                'ate_percent': float(ate_pct),
                'treatment_mean': float(treatment_mean),
                'control_mean': float(control_mean),
                'treatment_n': int(len(type_customers)),
                'control_n': int(len(non_type_in_cluster))
            }
            
            heatmap_data.append({
                'cluster': int(cluster_id),
                'offer_type': offer_type,
                'ate': float(ate)
            })
        
        best_offer = None
        best_ate = -np.inf
        for ot, data in het_results[int(cluster_id)]['offer_ates'].items():
            if data['ate'] is not None and data['ate'] > best_ate:
                best_ate = data['ate']
                best_offer = ot
        
        het_results[int(cluster_id)]['best_offer'] = best_offer
        het_results[int(cluster_id)]['best_ate'] = float(best_ate) if best_ate is not None else None
        
        print(f"\n  Cluster {int(cluster_id)} ({len(cluster_customers)} customers):")
        print(f"    Best offer type: {best_offer} (ATE: ${best_ate:.2f})" if best_offer else "    Best offer type: N/A")
        for ot in offer_types:
            d = het_results[int(cluster_id)]['offer_ates'][ot]
            if d['ate'] is not None:
                print(f"    {ot}: ATE=${d['ate']:.2f} ({d['ate_percent']:.1f}%), n_treat={d['treatment_n']}, n_ctrl={d['control_n']}")
            else:
                print(f"    {ot}: Insufficient data")
    
    if fig_dir is not None:
        heatmap_df = pd.DataFrame(heatmap_data)
        if len(heatmap_df) > 0 and heatmap_df['ate'].notna().any():
            pivot = heatmap_df.pivot(index='cluster', columns='offer_type', values='ate')
            
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, max(4, len(unique_clusters) * 1.2)))
            sns.heatmap(
                pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=ax_heatmap,
                cbar_kws={'label': 'ATE ($)'})
            ax_heatmap.set_title('Heterogeneous Treatment Effects: Segment x Offer Type ATE',
                                 fontsize=14, fontweight='bold')
            ax_heatmap.set_ylabel('Customer Segment (Cluster)')
            ax_heatmap.set_xlabel('Offer Type')
            plt.tight_layout()
            plt.savefig(fig_dir / 'heterogeneous_ate_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            fig_interactive = px.imshow(
                pivot,
                color_continuous_scale='RdYlGn',
                title='Heterogeneous Treatment Effects: Segment × Offer Type ATE ($)',
                labels=dict(x='Offer Type', y='Customer Segment', color='ATE ($)')
            )
            fig_interactive.write_html(fig_dir / 'heterogeneous_ate_interactive.html')
            
            print(f"\n  Heatmap saved to {fig_dir}")
    
    return het_results


def simulate_ab_test(recommendation_results: Dict[str, Any],
                     ate_results: Dict[str, Any],
                     customer_clusters_path: str = 'data/processed/customer_clusters.csv',
                     n_simulations: int = 1000, random_state: int = 42) -> Dict[str, Any]:
    """
    Simulate an A/B test comparing rule-based vs random targeting, compute
    required sample size, estimate expected effect size, and calculate power.
    
    Args:
        recommendation_results: Output from build_recommendation_rules()
        ate_results: Output from calculate_ate_by_offer_type()
        customer_clusters_path: Path to customer clusters CSV
        n_simulations: Number of Monte Carlo simulations
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with A/B test simulation results, including required
        sample size, estimated power, and simulation details
    """
    print("\n" + "="*60)
    print("A/B TEST SIMULATION FRAMEWORK")
    print("="*60)
    
    rng = np.random.RandomState(random_state)
    
    customer_clusters = pd.read_csv(customer_clusters_path)
    total_customers = len(customer_clusters)
    
    rule_rate = recommendation_results['rule_based_completion_rate']
    random_rate = recommendation_results['random_completion_rate']
    
    lift = (rule_rate - random_rate)
    lift_pct = recommendation_results['lift_percent']
    
    print(f"\n  Baseline completion rate (random): {random_rate:.4f}")
    print(f"  Treatment completion rate (rule-based): {rule_rate:.4f}")
    print(f"  Absolute lift: {lift:.4f} ({lift_pct:.1f}%)")
    
    p1 = random_rate
    p2 = rule_rate
    p_avg = (p1 + p2) / 2
    effect_size = abs(p2 - p1) / np.sqrt(p_avg * (1 - p_avg)) if p_avg > 0 else 0
    
    print(f"\n  Effect size (Cohen's h): {effect_size:.4f}")
    
    alpha = 0.05
    power_target = 0.80
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power_target)
    
    if abs(p1 - p2) > 1e-10:
        n_per_group = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
                        z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    else:
        n_per_group = float('inf')
    
    n_per_group = int(np.ceil(n_per_group))
    total_sample = 2 * n_per_group
    
    print(f"\n  Required sample size per group: {n_per_group:,}")
    print(f"  Total sample required: {total_sample:,}")
    print(f"  Available customers: {total_customers:,}")
    print(f"  Feasibility: {'Feasible' if total_sample <= total_customers else 'Need larger population'}")
    
    power_results = []
    sample_sizes = [500, 1000, 2000, 5000, 10000, n_per_group if n_per_group < total_customers else total_customers]
    sample_sizes = sorted(set(s for s in sample_sizes if s <= total_customers))
    
    for n in sample_sizes:
        simulated_powers = []
        for _ in range(n_simulations):
            control = rng.binomial(1, p1, size=n)
            treatment = rng.binomial(1, p2, size=n)
            
            control_rate = control.mean()
            treatment_rate = treatment.mean()
            
            if control_rate == 0 and treatment_rate == 0:
                continue
            
            se = np.sqrt(control_rate * (1 - control_rate) / n + treatment_rate * (1 - treatment_rate) / n)
            if se == 0:
                continue
            
            z = (treatment_rate - control_rate) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            simulated_powers.append(p_val < alpha)
        
        power = np.mean(simulated_powers) if simulated_powers else 0
        power_results.append({
            'n_per_group': n,
            'total_n': 2 * n,
            'power': float(power)
        })
    
    print(f"\n  Power analysis (by sample size per group):")
    for pr in power_results:
        bar = '#' * int(pr['power'] * 20)
        print(f"    n={pr['n_per_group']:>6,}: power={pr['power']:.3f} |{bar}")
    
    print(f"\n  Monte Carlo A/B Test Simulation ({n_simulations} runs):")
    sim_lifts = []
    sim_significant = []
    sim_n = min(n_per_group, total_customers // 2)
    
    for _ in range(n_simulations):
        control_sim = rng.binomial(1, p1, size=sim_n)
        treatment_sim = rng.binomial(1, p2, size=sim_n)
        
        sim_ctrl_rate = control_sim.mean()
        sim_treat_rate = treatment_sim.mean()
        sim_lift = sim_treat_rate - sim_ctrl_rate
        sim_lifts.append(sim_lift)
        
        se = np.sqrt(sim_ctrl_rate * (1 - sim_ctrl_rate) / sim_n +
                      sim_treat_rate * (1 - sim_treat_rate) / sim_n)
        if se > 0:
            z = sim_lift / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            sim_significant.append(p_val < alpha)
    
    empirical_power = np.mean(sim_significant) if sim_significant else 0
    mean_lift = np.mean(sim_lifts)
    lift_ci_lower = np.percentile(sim_lifts, 2.5)
    lift_ci_upper = np.percentile(sim_lifts, 97.5)
    
    print(f"    Mean simulated lift: {mean_lift:.4f}")
    print(f"    95% CI of lift: [{lift_ci_lower:.4f}, {lift_ci_upper:.4f}]")
    print(f"    Empirical power at n={sim_n:,}: {empirical_power:.3f}")
    
    ab_test_results = {
        'baseline_rate': float(random_rate),
        'treatment_rate': float(rule_rate),
        'absolute_lift': float(lift),
        'lift_percent': float(lift_pct),
        'effect_size_cohens_h': float(effect_size),
        'required_sample_per_group': n_per_group,
        'required_sample_total': total_sample,
        'feasible': total_sample <= total_customers,
        'power_analysis': power_results,
        'simulation': {
            'n_simulations': n_simulations,
            'sample_per_group': sim_n,
            'mean_lift': float(mean_lift),
            'lift_ci_lower': float(lift_ci_lower),
            'lift_ci_upper': float(lift_ci_upper),
            'empirical_power': float(empirical_power),
            'alpha': alpha,
            'power_target': power_target
        }
    }
    
    return ab_test_results


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
    
    customer_clusters = pd.read_csv(customer_clusters_path)
    cluster_profiles = pd.read_csv(cluster_profiles_path)
    portfolio = pd.read_json(portfolio_path, lines=True)
    
    customer_features = pd.read_csv('data/processed/customer_features.csv')
    
    customers_with_clusters = customer_features.merge(customer_clusters, on='id', how='left')
    
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
    
    transcript = pd.read_json('transcript.json', lines=True)
    transcript['offer_id'] = transcript['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    
    completed = transcript[transcript['event'] == 'offer completed']
    received = transcript[transcript['event'] == 'offer received']
    
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
    
    completion_by_type = {}
    for offer_type in ['bogo', 'discount', 'informational']:
        type_received = received_merged[received_merged['offer_type'] == offer_type]['person'].nunique()
        type_completed = completed_merged[completed_merged['offer_type'] == offer_type]['person'].nunique()
        completion_by_type[offer_type] = type_completed / type_received if type_received > 0 else 0
    
    print(f"\nHistorical Completion Rates by Offer Type:")
    for offer_type, rate in completion_by_type.items():
        print(f"  {offer_type}: {rate:.1%}")
    
    random_completion_rate = sum(completion_by_type.values()) / len(completion_by_type)
    
    rule_based_completions = 0
    total_customers = len(customers_with_clusters)
    
    for cluster_id, rule in rules.items():
        cluster_customers = customers_with_clusters[customers_with_clusters['cluster'] == cluster_id]
        n_customers = len(cluster_customers)
        
        primary_type = rule['primary']
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


def visualize_causal_results(ate_results: Dict, recommendation_results: Dict, fig_dir: Path,
                               bootstrap_results: Optional[Dict] = None,
                               test_results: Optional[Dict] = None,
                               het_results: Optional[Dict] = None,
                               ab_test_results: Optional[Dict] = None) -> None:
    """
    Visualize causal inference and recommendation system results.
    
    Args:
        ate_results: ATE calculation results
        recommendation_results: Recommendation system results
        fig_dir: Directory to save figures
        bootstrap_results: Bootstrap CI results (optional)
        test_results: Statistical significance test results (optional)
        het_results: Heterogeneous treatment effect results (optional)
        ab_test_results: A/B test simulation results (optional)
    """
    print("\nCreating causal inference visualizations...")
    
    offer_types = ['any_offer', 'bogo', 'discount', 'informational']
    ate_values = [ate_results[t]['ate'] for t in offer_types]
    ate_percent = [ate_results[t]['ate_percent'] for t in offer_types]
    
    n_figs = 4
    fig_num = 0
    
    fig_num += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Average Treatment Effect (ATE) by Offer Type', fontsize=16, fontweight='bold')
    
    bars1 = axes[0].bar(offer_types, ate_values, color='skyblue', edgecolor='black')
    axes[0].set_ylabel('ATE ($)')
    axes[0].set_title('ATE: Treatment - Control ($)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars1, ate_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'${val:.2f}', ha='center', va='bottom' if val >= 0 else 'top')
    
    bars2 = axes[1].bar(offer_types, ate_percent, color='lightgreen', edgecolor='black')
    axes[1].set_ylabel('ATE (%)')
    axes[1].set_title('ATE: Percent Lift (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars2, ate_percent):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'ate_by_offer_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig_num += 1
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
    
    for bar, rate in zip(bars, completion_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    lift = recommendation_results['lift_percent']
    ax.text(1, max(completion_rates) * 1.1, f'Lift: +{lift:.1f}%', 
             ha='center', fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'recommendation_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig_num += 1
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
    
    if bootstrap_results is not None:
        fig_num += 1
        fig_boot, ax_boot = plt.subplots(figsize=(10, 6))
        
        ci_data = {}
        for ot in offer_types:
            if ot in bootstrap_results and 'ci_lower' in bootstrap_results[ot]:
                ci_data[ot] = bootstrap_results[ot]
        
        if ci_data:
            labels = list(ci_data.keys())
            ates = [ate_results[ot]['ate'] for ot in labels]
            lowers = [ci_data[ot]['ci_lower'] for ot in labels]
            uppers = [ci_data[ot]['ci_upper'] for ot in labels]
            errors = [[ates[i] - lowers[i] for i in range(len(labels))],
                       [uppers[i] - ates[i] for i in range(len(labels))]]
            
            colors = ['green' if not ci_data[ot]['ci_includes_zero'] else 'orange' for ot in labels]
            bars = ax_boot.bar(labels, ates, yerr=errors, color=colors, edgecolor='black',
                               capsize=5, alpha=0.7)
            ax_boot.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='No effect (0)')
            ax_boot.set_ylabel('ATE ($)')
            ax_boot.set_title('ATE with 95% Bootstrap Confidence Intervals\n(Green = significant, Orange = not significant)',
                               fontsize=13, fontweight='bold')
            ax_boot.legend()
            
            for ot in labels:
                if ot in ci_data:
                    sig = "Sig." if not ci_data[ot]['ci_includes_zero'] else "Not sig."
                    print(f"    CI annotation for {ot}: {sig}")
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'bootstrap_ci.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            fig_boot_interactive = go.Figure()
            for i, ot in enumerate(labels):
                fig_boot_interactive.add_trace(go.Bar(
                    name=ot,
                    x=[ot], y=[ates[i]],
                    error_y=dict(type='data', array=[uppers[i] - ates[i]],
                                  arrayminus=[ates[i] - lowers[i]]),
                    marker_color=colors[i]
                ))
            fig_boot_interactive.add_hline(y=0, line_dash='dash', line_color='red',
                                            annotation_text='No effect')
            fig_boot_interactive.update_layout(
                title='ATE with 95% Bootstrap Confidence Intervals',
                yaxis_title='ATE ($)'
            )
            fig_boot_interactive.write_html(fig_dir / 'bootstrap_ci_interactive.html')
    
    if test_results is not None:
        fig_num += 1
        fig_test, axes_test = plt.subplots(1, 2, figsize=(14, 6))
        fig_test.suptitle('Statistical Significance Tests', fontsize=16, fontweight='bold')
        
        test_labels = [ot for ot in offer_types if ot in test_results and test_results[ot].get('p_value') is not None]
        p_values = [-np.log10(test_results[ot]['p_value']) for ot in test_labels]
        cohens_d_vals = [test_results[ot]['cohens_d'] for ot in test_labels]
        
        if test_labels:
            colors_pval = ['green' if test_results[ot]['significant'] else 'red' for ot in test_labels]
            axes_test[0].bar(test_labels, p_values, color=colors_pval, edgecolor='black', alpha=0.7)
            axes_test[0].axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                                label='α=0.05 threshold')
            axes_test[0].axhline(y=-np.log10(0.01), color='darkred', linestyle=':', 
                                label='α=0.01 threshold')
            axes_test[0].set_ylabel('-log10(p-value)')
            axes_test[0].set_title('Statistical Significance (p-values)')
            axes_test[0].legend()
            axes_test[0].tick_params(axis='x', rotation=45)
            
            colors_d = ['green' if abs(d) >= 0.5 else 'orange' if abs(d) >= 0.2 else 'red' 
                        for d in cohens_d_vals]
            bars_d = axes_test[1].bar(test_labels, cohens_d_vals, color=colors_d, edgecolor='black', alpha=0.7)
            axes_test[1].axhline(y=0, color='black', linewidth=0.5)
            axes_test[1].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
            axes_test[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
            axes_test[1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large (0.8)')
            axes_test[1].set_ylabel("Cohen's d")
            axes_test[1].set_title("Effect Size (Cohen's d)")
            axes_test[1].legend()
            axes_test[1].tick_params(axis='x', rotation=45)
            
            for bar, d in zip(bars_d, cohens_d_vals):
                axes_test[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{d:.3f}', ha='center', va='bottom' if d >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'statistical_tests.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    if ab_test_results is not None:
        fig_num += 1
        fig_ab, axes_ab = plt.subplots(1, 2, figsize=(14, 6))
        fig_ab.suptitle('A/B Test Simulation Results', fontsize=16, fontweight='bold')
        
        power_df = pd.DataFrame(ab_test_results['power_analysis'])
        axes_ab[0].plot(power_df['n_per_group'], power_df['power'], 'bo-', linewidth=2, markersize=8)
        axes_ab[0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% power target')
        axes_ab[0].set_xlabel('Sample Size per Group')
        axes_ab[0].set_ylabel('Statistical Power')
        axes_ab[0].set_title('Power Curve')
        axes_ab[0].legend()
        axes_ab[0].set_ylim([0, 1.05])
        axes_ab[0].grid(True, alpha=0.3)
        
        sim = ab_test_results['simulation']
        axes_ab[1].hist(sim['mean_lift'], bins=30 if isinstance(sim.get('mean_lift'), (list, np.ndarray)) else 30,
                        alpha=0.7, color='skyblue', edgecolor='black')
        axes_ab[1].axvline(x=0, color='red', linestyle='--', label='No effect')
        axes_ab[1].set_xlabel('Simulated Lift (completion rate difference)')
        axes_ab[1].set_ylabel('Frequency')
        axes_ab[1].set_title(f'Distribution of Simulated Lift\n(n={sim["sample_per_group"]:,} per group)')
        axes_ab[1].legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'ab_test_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
    
    rules_df = pd.DataFrame([
        {'cluster_id': k, 'primary_offer': v['primary'], 'secondary_offer': v['secondary'], 'rationale': v['rationale']}
        for k, v in rules.items()
    ])
    rules_df.to_csv(output_dir / 'recommendation_rules.csv', index=False)
    
    with open(output_dir / 'recommendation_results.json', 'w') as f:
        json.dump(recommendation_results, f, indent=2, default=str)
    
    print(f"\n Recommendation system saved to {output_dir}")


def generate_causal_report(ate_results: Dict, recommendation_results: Dict,
                           output_path: str = 'reports/causal_report.json',
                           bootstrap_results: Optional[Dict] = None,
                           test_results: Optional[Dict] = None,
                           psm_results: Optional[Dict] = None,
                           het_results: Optional[Dict] = None,
                           ab_test_results: Optional[Dict] = None) -> None:
    """
    Generate a comprehensive causal inference report.
    
    Args:
        ate_results: ATE calculation results
        recommendation_results: Recommendation system results
        output_path: Path to save the report
        bootstrap_results: Bootstrap CI results (optional)
        test_results: Statistical test results (optional)
        psm_results: Propensity score matching results (optional)
        het_results: Heterogeneous treatment effect results (optional)
        ab_test_results: A/B test simulation results (optional)
    """
    
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(i) for i in obj]
        return obj
    
    ate_clean = {}
    for k, v in ate_results.items():
        ate_clean[k] = {key: val for key, val in v.items() if key != 'control_amounts' and key != 'treatment_amounts'}
    
    report = {
        'ate_results': _make_serializable(ate_clean),
        'recommendation_results': {
            'rules': recommendation_results['rules'],
            'random_completion_rate': float(recommendation_results['random_completion_rate']),
            'rule_based_completion_rate': float(recommendation_results['rule_based_completion_rate']),
            'lift_percent': float(recommendation_results['lift_percent'])
        }
    }
    
    if bootstrap_results is not None:
        report['bootstrap_confidence_intervals'] = _make_serializable(bootstrap_results)
    
    if test_results is not None:
        report['statistical_significance'] = _make_serializable(test_results)
    
    if psm_results is not None:
        psm_clean = {}
        for ot, data in psm_results.items():
            psm_clean[ot] = {k: v for k, v in data.items() 
                             if k not in ['matched_treated_ids', 'matched_control_ids']}
        report['propensity_score_matching'] = _make_serializable(psm_clean)
    
    if het_results is not None:
        report['heterogeneous_effects'] = _make_serializable(het_results)
    
    if ab_test_results is not None:
        report['ab_test_simulation'] = _make_serializable(ab_test_results)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n Causal inference report saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - CAUSAL INFERENCE & RECOMMENDATION")
    print("="*60)
    
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading data...")
    portfolio, profile, transcript = load_data_for_causal_analysis()
    print(f" Loaded {len(portfolio)} offers, {len(profile)} customers, {len(transcript)} events")
    
    ate_results = calculate_ate_by_offer_type(portfolio, transcript)
    
    bootstrap_results = bootstrap_ate_ci(ate_results, n_bootstrap=1000, confidence=0.95)
    
    psm_results = propensity_score_matching(portfolio, profile, transcript)
    
    test_results = statistical_significance_tests(ate_results)
    
    het_results = analyze_heterogeneous_effects(portfolio, transcript, fig_dir=fig_dir)
    
    recommendation_results = build_recommendation_rules()
    rules = recommendation_results['rules']
    
    ab_test_results = simulate_ab_test(recommendation_results, ate_results)
    
    visualize_causal_results(
        ate_results, recommendation_results, fig_dir,
        bootstrap_results=bootstrap_results,
        test_results=test_results,
        het_results=het_results,
        ab_test_results=ab_test_results
    )
    
    save_recommendation_system(rules, recommendation_results)
    
    generate_causal_report(
        ate_results, recommendation_results,
        bootstrap_results=bootstrap_results,
        test_results=test_results,
        psm_results=psm_results,
        het_results=het_results,
        ab_test_results=ab_test_results
    )
    
    print("\n" + "="*60)
    print("CAUSAL INFERENCE & RECOMMENDATION COMPLETE")
    print("="*60)
    print(f" ATE calculated for all offer types")
    print(f" Bootstrap 95% CIs computed (1000 iterations)")
    print(f" Propensity score matching completed")
    print(f" Statistical significance tests (Welch's t-test, Cohen's d)")
    print(f" Heterogeneous treatment effects by segment")
    print(f" A/B test simulation framework")
    print(f" Recommendation system built with {len(rules)} rules")
    print(f" Expected lift: +{recommendation_results['lift_percent']:.1f}%")
    print(f" Visualizations saved to: {fig_dir}")
    print("="*60)