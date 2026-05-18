"""
Customer Segmentation (Clustering) module for Starbucks project.

This module performs customer segmentation using unsupervised learning:
1. Determines optimal number of clusters using elbow method and silhouette scores
2. Applies K-Means clustering (primary) and DBSCAN (secondary)
3. Interprets and visualizes clusters for business insights
4. Answers research question: "Can we identify 3-5 distinct customer segments 
   with homogeneous offer response behaviors?"

Follows Data Scientist principles: reproducible, explainable, validated.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_customer_features(base_path: str = '.') -> pd.DataFrame:
    """
    Load engineered customer features from Phase 3.
    
    Args:
        base_path: Base directory
        
    Returns:
        Customer features DataFrame
    """
    features_path = Path(base_path) / 'data' / 'processed' / 'customer_features.csv'
    return pd.read_csv(features_path)


def preprocess_for_clustering(customer_features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Preprocess customer features for clustering.
    
    Args:
        customer_features: Raw customer features
        
    Returns:
        Tuple of (preprocessed DataFrame, fitted scaler, feature names used)
    """
    print("\n" + "="*60)
    print("PREPROCESSING FOR CLUSTERING")
    print("="*60)
    
    # Exclude ID and non-feature columns
    exclude_cols = ['id', 'customer_id', 'gender', 'age_bin', 'income_bin', 'tenure_bin']
    feature_cols = [col for col in customer_features.columns if col not in exclude_cols]
    
    # Select features for clustering
    # Focus on behavioral and demographic features that drive segmentation
    clustering_features = [
        # Demographics
        'age_imputed', 'income_imputed', 'tenure_months',
        'age_missing', 'income_missing',
        'gender_M', 'gender_F', 'gender_O', 'gender_Unknown',
        
        # Transaction behavior
        'trans_count', 'trans_total', 'trans_avg', 'trans_std',
        
        # Offer response behavior
        'offers_received', 'offers_viewed', 'offers_completed',
        'view_rate', 'completion_rate', 'view_to_completion_rate',
        
        # Offer-type specific behavior
        'bogo_received', 'bogo_viewed', 'bogo_completed',
        'bogo_view_rate', 'bogo_completion_rate',
        'discount_received', 'discount_viewed', 'discount_completed',
        'discount_view_rate', 'discount_completion_rate',
        'info_received', 'info_viewed', 'info_view_rate'
    ]
    
    # Keep only columns that exist
    clustering_features = [col for col in clustering_features if col in customer_features.columns]
    
    print(f"Using {len(clustering_features)} features for clustering:")
    for f in clustering_features:
        print(f"  - {f}")
    
    X = customer_features[clustering_features].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features)
    
    print(f" Preprocessed data shape: {X_scaled_df.shape}")
    print(f" Feature means (should be ~0): {X_scaled_df.mean().mean():.3f}")
    print(f" Feature stds (should be ~1): {X_scaled_df.std().mean():.3f}")
    
    return X_scaled_df, scaler, clustering_features


def compute_gap_statistic(X: pd.DataFrame, max_k: int = 15, n_refs: int = 10) -> Tuple[List[float], List[float], int]:
    """
    Compute gap statistic for determining optimal number of clusters.
    
    Generates B reference datasets from a uniform distribution along each
    feature axis and compares within-cluster dispersion to the reference.
    
    Args:
        X: Preprocessed feature matrix
        max_k: Maximum number of clusters to test
        n_refs: Number of reference datasets (B in the Tibshirani paper)
        
    Returns:
        Tuple of (gap_values, gap_std_errors, optimal_k_gap)
    """
    print("\n" + "-"*40)
    print("Computing Gap Statistic...")
    print("-"*40)
    
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    n_samples, n_features = X_arr.shape
    K_range = range(2, max_k + 1)
    
    gap_values = []
    gap_std_errors = []
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_arr)
        wcss_obs = np.log(km.inertia_)
        
        ref_wcss = []
        for _ in range(n_refs):
            reference = np.random.uniform(
                X_arr.min(axis=0),
                X_arr.max(axis=0),
                size=(n_samples, n_features)
            )
            km_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_ref.fit(reference)
            ref_wcss.append(np.log(km_ref.inertia_))
        
        ref_wcss = np.array(ref_wcss)
        gap = ref_wcss.mean() - wcss_obs
        gap_values.append(gap)
        gap_std_errors.append(np.sqrt(1 + 1 / n_refs) * ref_wcss.std())
        
        print(f"  k={k}: Gap={gap:.3f}, SE={gap_std_errors[-1]:.3f}")
    
    optimal_k_gap = 2
    for i in range(len(gap_values) - 1):
        if gap_values[i] >= gap_values[i + 1] - gap_std_errors[i + 1]:
            optimal_k_gap = list(K_range)[i]
            break
    
    print(f" Optimal k (Gap Statistic): {optimal_k_gap}")
    return gap_values, gap_std_errors, optimal_k_gap


def find_optimal_clusters(X: pd.DataFrame, max_k: int = 15) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method, silhouette scores,
    Calinski-Harabasz index, Davies-Bouldin index, and gap statistic.
    
    Args:
        X: Preprocessed feature matrix
        max_k: Maximum number of clusters to test
        
    Returns:
        Dictionary with optimization results
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    wcss = []
    silhouette_scores = []
    calinski_scores = []
    db_scores = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X, kmeans.labels_))
        
        print(f"  k={k}: WCSS={wcss[-1]:.0f}, Silhouette={silhouette_scores[-1]:.3f}, "
              f"Calinski={calinski_scores[-1]:.1f}, DB={db_scores[-1]:.3f}")
    
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    optimal_k_db = K_range[np.argmin(db_scores)]
    
    wcss_diff = np.diff(wcss)
    wcss_diff2 = np.diff(wcss_diff)
    optimal_k_elbow = K_range[np.argmax(wcss_diff2) + 1] if len(wcss_diff2) > 0 else 3
    
    gap_values, gap_std_errors, optimal_k_gap = compute_gap_statistic(X, max_k=max_k)
    
    print(f"\n Optimal k (Silhouette): {optimal_k_silhouette}")
    print(f" Optimal k (Elbow): {optimal_k_elbow}")
    print(f" Optimal k (Davies-Bouldin): {optimal_k_db}")
    print(f" Optimal k (Gap Statistic): {optimal_k_gap}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Cluster Optimization Metrics', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(K_range, wcss, 'bo-')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('WCSS (Inertia)')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].axvline(optimal_k_elbow, color='r', linestyle='--', alpha=0.5)
    
    axes[0, 1].plot(K_range, silhouette_scores, 'go-')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Scores')
    axes[0, 1].axvline(optimal_k_silhouette, color='r', linestyle='--', alpha=0.5)
    
    axes[0, 2].plot(K_range, calinski_scores, 'mo-')
    axes[0, 2].set_xlabel('Number of Clusters (k)')
    axes[0, 2].set_ylabel('Calinski-Harabasz Index')
    axes[0, 2].set_title('Calinski-Harabasz (Higher is better)')
    
    axes[1, 0].plot(K_range, db_scores, 'ro-')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin (Lower is better)')
    axes[1, 0].axvline(optimal_k_db, color='g', linestyle='--', alpha=0.5)
    
    K_list = list(K_range)
    axes[1, 1].plot(K_list, gap_values, 'co-')
    axes[1, 1].fill_between(
        K_list,
        np.array(gap_values) - np.array(gap_std_errors),
        np.array(gap_values) + np.array(gap_std_errors),
        alpha=0.2, color='c'
    )
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Gap Statistic')
    axes[1, 1].set_title('Gap Statistic')
    axes[1, 1].axvline(optimal_k_gap, color='r', linestyle='--', alpha=0.5)
    
    axes[1, 2].axis('off')
    summary_text = (
        f"Optimal k Summary\n"
        f"Elbow: {optimal_k_elbow}\n"
        f"Silhouette: {optimal_k_silhouette}\n"
        f"Davies-Bouldin: {optimal_k_db}\n"
        f"Gap Statistic: {optimal_k_gap}"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, transform=axes[1, 2].transAxes,
                     fontsize=14, verticalalignment='center', horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / 'cluster_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'K_range': list(K_range),
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': db_scores,
        'gap_values': gap_values,
        'gap_std_errors': gap_std_errors,
        'optimal_k_silhouette': optimal_k_silhouette,
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_db': optimal_k_db,
        'optimal_k_gap': optimal_k_gap
    }


def apply_kmeans_clustering(X: pd.DataFrame, k: int, feature_names: List[str]) -> Tuple[pd.DataFrame, KMeans]:
    """
    Apply K-Means clustering with specified k.
    
    Args:
        X: Preprocessed feature matrix
        k: Number of clusters
        feature_names: List of feature names
        
    Returns:
        Tuple of (DataFrame with cluster labels, fitted KMeans model)
    """
    print("\n" + "="*60)
    print(f"APPLYING K-MEANS CLUSTERING (k={k})")
    print("="*60)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Add labels to DataFrame
    result = X.copy()
    result['cluster'] = labels
    
    # Calculate cluster statistics
    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} customers ({count/len(labels):.1%})")
    
    # Feature importance per cluster (distance from global mean)
    print(f"\nCluster centers (top distinguishing features):")
    centers = kmeans.cluster_centers_
    
    for i in range(k):
        cluster_center = centers[i]
        # Calculate which features are most distinctive for this cluster
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'cluster_center': cluster_center,
            'global_mean': X.mean(),
            'difference': cluster_center - X.mean()
        })
        feature_importance['abs_diff'] = feature_importance['difference'].abs()
        top_features = feature_importance.nlargest(5, 'abs_diff')
        
        print(f"\n  Cluster {i} top features:")
        for _, row in top_features.iterrows():
            direction = "+" if row['difference'] > 0 else "-"
            print(f"    {direction} {row['feature']}: {row['cluster_center']:.2f} (diff: {row['difference']:+.2f})")
    
    return result, kmeans


def visualize_clusters_pca(X: pd.DataFrame, labels: np.ndarray, k: int, fig_dir: Path) -> None:
    """
    Visualize clusters using PCA dimensionality reduction.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        k: Number of clusters
        fig_dir: Directory to save figures
    """
    print("\nCreating PCA visualization...")
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': labels.astype(str)
    })
    
    # Matplotlib static plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=labels, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(f'Customer Segments - PCA Visualization (k={k})', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly interactive plot
    fig = px.scatter(
        pca_df, x='PC1', y='PC2', color='Cluster',
        title=f'Customer Segments - Interactive PCA (k={k})',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
        },
        width=800, height=600
    )
    fig.write_html(fig_dir / 'cluster_pca_interactive.html')
    
    print(f" PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    print(f" PCA plots saved to {fig_dir}")


def interpret_clusters(customer_features: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    """
    Interpret clusters by analyzing cluster characteristics.
    
    Args:
        customer_features: Original customer features with ID
        labels: Cluster labels
        k: Number of clusters
        
    Returns:
        DataFrame with cluster profiles
    """
    print("\n" + "="*60)
    print("CLUSTER INTERPRETATION")
    print("="*60)
    
    # Add cluster labels to customer features
    df = customer_features.copy()
    df['cluster'] = labels
    
    # Define key metrics for interpretation
    metrics = [
        'age_imputed', 'income_imputed', 'tenure_months',
        'trans_count', 'trans_avg', 'trans_total',
        'offers_received', 'offers_viewed', 'offers_completed',
        'view_rate', 'completion_rate',
        'bogo_completion_rate', 'discount_completion_rate'
    ]
    
    # Calculate cluster profiles
    cluster_profiles = []
    
    for cluster_id in range(k):
        cluster_data = df[df['cluster'] == cluster_id]
        
        profile = {'cluster_id': cluster_id, 'size': len(cluster_data)}
        
        for metric in metrics:
            if metric in cluster_data.columns:
                profile[f'{metric}_mean'] = cluster_data[metric].mean()
                profile[f'{metric}_median'] = cluster_data[metric].median()
        
        # Add gender distribution
        if 'gender' in cluster_data.columns:
            gender_counts = cluster_data['gender'].value_counts(normalize=True)
            for gender in ['M', 'F', 'O', 'Unknown']:
                profile[f'gender_{gender}_pct'] = gender_counts.get(gender, 0)
        
        cluster_profiles.append(profile)
    
    profiles_df = pd.DataFrame(cluster_profiles)
    
    # Print interpretation
    print("\nCluster Profiles:")
    for _, profile in profiles_df.iterrows():
        print(f"\n{'='*40}")
        print(f"CLUSTER {int(profile['cluster_id'])} ({int(profile['size'])} customers, "
              f"{profile['size']/len(df):.1%})")
        print(f"{'='*40}")
        
        # Demographics
        print(f"  Demographics:")
        print(f"    Avg Age: {profile.get('age_imputed_mean', 0):.1f}")
        print(f"    Avg Income: ${profile.get('income_imputed_mean', 0):,.0f}")
        print(f"    Avg Tenure: {profile.get('tenure_months_mean', 0):.1f} months")
        
        # Transaction behavior
        print(f"  Transaction Behavior:")
        print(f"    Avg Transactions: {profile.get('trans_count_mean', 0):.1f}")
        print(f"    Avg Transaction Amount: ${profile.get('trans_avg_mean', 0):.2f}")
        print(f"    Avg Total Spend: ${profile.get('trans_total_mean', 0):.2f}")
        
        # Offer response
        print(f"  Offer Response:")
        print(f"    Offers Received: {profile.get('offers_received_mean', 0):.1f}")
        print(f"    View Rate: {profile.get('view_rate_mean', 0):.1%}")
        print(f"    Completion Rate: {profile.get('completion_rate_mean', 0):.1%}")
        
        # Gender distribution
        print(f"  Gender Distribution:")
        for gender in ['M', 'F', 'O', 'Unknown']:
            pct = profile.get(f'gender_{gender}_pct', 0)
            if pct > 0.01:
                print(f"    {gender}: {pct:.1%}")
    
    return profiles_df


def create_cluster_visualizations(profiles_df: pd.DataFrame, df_with_clusters: pd.DataFrame, fig_dir: Path) -> None:
    """
    Create visualizations for cluster interpretation.
    
    Args:
        profiles_df: Cluster profiles DataFrame
        df_with_clusters: Customer data with cluster labels
        fig_dir: Directory to save figures
    """
    print("\nCreating cluster visualization plots...")
    
    # Figure 1: Cluster size comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(profiles_df['cluster_id'].astype(str), profiles_df['size'], 
                  color='skyblue', edgecolor='black')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, size in zip(bars, profiles_df['size']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(size)}\n({size/len(df_with_clusters):.1%})',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Radar chart comparison (using plotly)
    # Select key metrics for radar chart
    radar_metrics = ['completion_rate_mean', 'view_rate_mean', 'trans_avg_mean', 
                     'income_imputed_mean', 'tenure_months_mean']
    
    # Normalize metrics to 0-1 scale for radar chart
    radar_data = profiles_df[radar_metrics].copy()
    for col in radar_metrics:
        radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
    
    # Create radar chart
    fig = go.Figure()
    
    categories = ['Completion Rate', 'View Rate', 'Avg Transaction', 'Income', 'Tenure']
    
    for _, row in radar_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Cluster {int(row.name)}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Cluster Comparison - Normalized Metrics',
        width=800,
        height=600
    )
    
    fig.write_html(fig_dir / 'cluster_radar_interactive.html')
    
    # Figure 3: Boxplots comparing clusters on key metrics
    key_metrics = ['age_imputed', 'income_imputed', 'completion_rate', 'trans_total']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cluster Comparison - Key Metrics Distribution', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df_with_clusters, x='cluster', y=metric, ax=ax)
        ax.set_title(f'{metric}')
        ax.set_xlabel('Cluster')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Cluster visualizations saved to {fig_dir}")


def analyze_cluster_stability(X: pd.DataFrame, k: int, seeds: List[int] = None) -> Dict[str, Any]:
    """
    Analyze cluster stability by running K-Means with multiple random seeds
    and computing pairwise Adjusted Rand Index.
    
    Args:
        X: Preprocessed feature matrix
        k: Number of clusters
        seeds: List of random seeds to use (default: [42, 123, 456, 789, 2024])
        
    Returns:
        Dictionary with stability analysis results
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024]
    
    print("\n" + "="*60)
    print("CLUSTER STABILITY ANALYSIS")
    print("="*60)
    print(f"Running K-Means k={k} with {len(seeds)} random seeds: {seeds}")
    
    label_sets = []
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(X)
        label_sets.append(km.labels_)
        print(f"  Seed {seed}: inertia={km.inertia_:.1f}, "
              f"sils={silhouette_score(X, km.labels_):.3f}")
    
    pairwise_ari = []
    ari_matrix = np.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari = adjusted_rand_score(label_sets[i], label_sets[j])
            pairwise_ari.append(ari)
            ari_matrix[i][j] = ari
            ari_matrix[j][i] = ari
    np.fill_diagonal(ari_matrix, 1.0)
    
    mean_ari = np.mean(pairwise_ari)
    std_ari = np.std(pairwise_ari)
    min_ari = np.min(pairwise_ari)
    
    if mean_ari >= 0.90:
        interpretation = "Highly stable - clusters are robust across initializations"
    elif mean_ari >= 0.75:
        interpretation = "Moderately stable - most clusters are consistent"
    elif mean_ari >= 0.50:
        interpretation = "Marginal stability - some clusters may be unreliable"
    else:
        interpretation = "Low stability - cluster structure is sensitive to initialization"
    
    print(f"\n Pairwise ARI Statistics:")
    print(f"  Mean ARI:  {mean_ari:.3f}")
    print(f"  Std ARI:   {std_ari:.3f}")
    print(f"  Min ARI:   {min_ari:.3f}")
    print(f"  Interpretation: {interpretation}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ari_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(seeds)))
    ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_yticklabels([str(s) for s in seeds])
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Random Seed')
    ax.set_title(f'Cluster Stability - Pairwise ARI Matrix (Mean={mean_ari:.3f})',
                 fontsize=13, fontweight='bold')
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            ax.text(j, i, f'{ari_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if ari_matrix[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Adjusted Rand Index')
    plt.tight_layout()
    
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / 'cluster_stability_ari.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'seeds': seeds,
        'pairwise_ari': pairwise_ari,
        'ari_matrix': ari_matrix.tolist(),
        'mean_ari': float(mean_ari),
        'std_ari': float(std_ari),
        'min_ari': float(min_ari),
        'interpretation': interpretation
    }


def validate_clusters_business(customer_features: pd.DataFrame, labels: np.ndarray, k: int,
                                fig_dir: Path = None) -> pd.DataFrame:
    """
    Validate clusters using business-relevant metrics.
    
    Computes revenue per customer, offer ROI, and churn risk proxy per cluster,
    creates a comparison table, and generates a visualization.
    
    Args:
        customer_features: Original customer features DataFrame
        labels: Cluster labels
        k: Number of clusters
        fig_dir: Directory to save figures
        
    Returns:
        DataFrame with business metrics per cluster
    """
    print("\n" + "="*60)
    print("BUSINESS METRIC VALIDATION")
    print("="*60)
    
    if fig_dir is None:
        fig_dir = Path('reports/figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
    
    df = customer_features.copy()
    df['cluster'] = labels
    
    reward_cols = [col for col in df.columns if 'reward' in col.lower()]
    if reward_cols:
        df['_total_reward_cost'] = df[reward_cols].sum(axis=1)
    else:
        df['_total_reward_cost'] = 0
    
    business_metrics = []
    for cluster_id in range(k):
        cluster_data = df[df['cluster'] == cluster_id]
        n = len(cluster_data)
        
        total_spend = cluster_data['trans_total'].mean() if 'trans_total' in cluster_data.columns else 0
        total_reward_cost = cluster_data['_total_reward_cost'].mean()
        
        if total_reward_cost > 0:
            offer_roi = total_spend / total_reward_cost
        else:
            offer_roi = float('inf')
        
        engagement_metrics = []
        for col_name, default in [('view_rate', 0), ('completion_rate', 0), ('trans_count', 0)]:
            if col_name in cluster_data.columns:
                engagement_metrics.append(cluster_data[col_name].mean())
            else:
                engagement_metrics.append(default)
        churn_risk = max(0, 1 - np.mean(engagement_metrics))
        
        metric_row = {
            'cluster_id': cluster_id,
            'size': n,
            'pct_of_total': n / len(df),
            'revenue_per_customer': total_spend,
            'total_reward_cost_per_customer': total_reward_cost,
            'offer_roi': offer_roi,
            'avg_view_rate': engagement_metrics[0],
            'avg_completion_rate': engagement_metrics[1],
            'avg_trans_count': engagement_metrics[2],
            'churn_risk_proxy': churn_risk
        }
        business_metrics.append(metric_row)
    
    business_df = pd.DataFrame(business_metrics)
    
    business_df['offer_roi'] = business_df['offer_roi'].replace([float('inf')], 999.0)
    
    print("\nBusiness Metrics by Cluster:")
    print(business_df.to_string(index=False))
    
    print("\nBusiness Validation Interpretation:")
    best_revenue_cluster = business_df.loc[business_df['revenue_per_customer'].idxmax(), 'cluster_id']
    best_roi_cluster = business_df.loc[business_df['offer_roi'].idxmax(), 'cluster_id']
    highest_churn_cluster = business_df.loc[business_df['churn_risk_proxy'].idxmax(), 'cluster_id']
    
    print(f"  Highest revenue per customer: Cluster {best_revenue_cluster} "
          f"(${business_df.loc[business_df['revenue_per_customer'].idxmax(), 'revenue_per_customer']:,.2f})")
    print(f"  Best offer ROI: Cluster {best_roi_cluster} "
          f"({business_df.loc[business_df['offer_roi'].idxmax(), 'offer_roi']:.2f}x)")
    print(f"  Highest churn risk: Cluster {highest_churn_cluster} "
          f"(proxy={business_df.loc[business_df['churn_risk_proxy'].idxmax(), 'churn_risk_proxy']:.3f})")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Business Metrics by Cluster', fontsize=16, fontweight='bold')
    
    cluster_labels = [f'C{i}' for i in range(k)]
    colors = sns.color_palette("husl", k)
    
    axes[0, 0].bar(cluster_labels, business_df['revenue_per_customer'], color=colors, edgecolor='black')
    axes[0, 0].set_title('Revenue per Customer')
    axes[0, 0].set_ylabel('Average Total Spend ($)')
    for i, v in enumerate(business_df['revenue_per_customer']):
        axes[0, 0].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    axes[0, 1].bar(cluster_labels, business_df['offer_roi'], color=colors, edgecolor='black')
    axes[0, 1].set_title('Offer ROI (Spend / Reward Cost)')
    axes[0, 1].set_ylabel('ROI Multiplier')
    for i, v in enumerate(business_df['offer_roi']):
        axes[0, 1].text(i, v, f'{v:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 0].bar(cluster_labels, business_df['churn_risk_proxy'], color=colors, edgecolor='black')
    axes[1, 0].set_title('Churn Risk Proxy')
    axes[1, 0].set_ylabel('Risk Score')
    for i, v in enumerate(business_df['churn_risk_proxy']):
        axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    size_data = business_df['size'].values
    axes[1, 1].pie(size_data, labels=cluster_labels, colors=colors,
                    autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_business_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Business metrics visualization saved to {fig_dir / 'cluster_business_metrics.png'}")
    
    return business_df


def generate_cluster_narratives(profiles_df: pd.DataFrame, business_df: pd.DataFrame = None,
                                 k: int = 4) -> Dict[str, Any]:
    """
    Generate human-readable cluster narratives with business-friendly names
    and actionable recommendations.
    
    Args:
        profiles_df: Cluster profiles DataFrame from interpret_clusters()
        business_df: Business metrics DataFrame from validate_clusters_business() (optional)
        k: Number of clusters
        
    Returns:
        Dictionary with narrative descriptions, segment names, and recommendations
    """
    print("\n" + "="*60)
    print("CLUSTER NARRATIVES & SEGMENT NAMING")
    print("="*60)
    
    NAMING_TEMPLATES = {
        'high_value_loyalist': {
            'name': 'High-Value Loyalists',
            'description': 'Premium customers with high spending, strong offer engagement, '
                           'and long tenure. These are your brand advocates.',
            'recommendations': [
                'Offer exclusive premium rewards to maintain loyalty',
                'Introduce referral programs leveraging their advocacy',
                'Test high-value personalized offers (e.g., limited-edition items)',
                'Minimize discounting; they respond to quality, not price'
            ]
        },
        'discount_seeker': {
            'name': 'Discount Seekers',
            'description': 'Price-sensitive customers who actively pursue and complete discount '
                           'offers. High offer engagement but moderate spending.',
            'recommendations': [
                'Continue targeted discount offers to maintain engagement',
                'Gradually introduce BOGO to shift spending upward',
                'Monitor for offer fatigue and adjust frequency',
                'Use bundle offers to increase average transaction value'
            ]
        },
        'bogo_responder': {
            'name': 'BOGO Responders',
            'description': 'Customers who respond strongly to buy-one-get-one offers. '
                           'Good engagement but may be promotion-dependent.',
            'recommendations': [
                'Balance BOGO frequency to avoid training only promotional purchases',
                'Cross-sell complementary products during BOGO redemption',
                'Introduce loyalty tier rewards alongside BOGO',
                'Track attribution to ensure incremental spend'
            ]
        },
        'low_engagement': {
            'name': 'Low-Engagement Customers',
            'description': 'Customers with low view and completion rates, low spending, '
                           'and minimal offer interaction. Highest churn risk.',
            'recommendations': [
                'Deploy re-engagement campaigns with simpler offers',
                'Send push notifications to increase view rates',
                'Reduce offer complexity - focus on single-step completion',
                'Consider win-back offers with expiration urgency'
            ]
        },
        'moderate_spender': {
            'name': 'Moderate Spenders',
            'description': 'Average customers with moderate spending and engagement. '
                           'No extreme behaviors; they form your dependable base.',
            'recommendations': [
                'Maintain baseline offer cadence to sustain engagement',
                'Test offer types to identify pathways to higher-value behavior',
                'Use A/B testing to find optimal offer frequency',
                'Target upsell opportunities during regular purchasing patterns'
            ]
        },
        'tenured_conservative': {
            'name': 'Tenured Conservatives',
            'description': 'Long-tenured customers with conservative spending patterns. '
                           'Stable but not optimizing their value.',
            'recommendations': [
                'Introduce new product lines to reinvigorate interest',
                'Leverage tenure for feedback and co-creation opportunities',
                'Offer loyalty anniversary rewards',
                'Test premium product trials to broaden their range'
            ]
        },
        'high_income_casual': {
            'name': 'High-Income Casuals',
            'description': 'High-income customers with surprisingly low engagement or spending. '
                           'Opportunity for significant value capture.',
            'recommendations': [
                'Offer premium experiences (reserve roastery events, etc.)',
                'Use aspirational messaging tied to quality and craft',
                'Reduce friction in digital ordering'    ,
                'Personalize offers based on infrequent high-value purchases'
            ]
        }
    }
    
    def classify_cluster(profile_row) -> str:
        spending = profile_row.get('trans_total_mean', 0)
        completion = profile_row.get('completion_rate_mean', 0)
        view_rate = profile_row.get('view_rate_mean', 0)
        bogo_comp = profile_row.get('bogo_completion_rate_mean', 0)
        discount_comp = profile_row.get('discount_completion_rate_mean', 0)
        tenure = profile_row.get('tenure_months_mean', 0)
        income = profile_row.get('income_imputed_mean', 0)
        
        median_spend = profiles_df['trans_total_mean'].median() if 'trans_total_mean' in profiles_df.columns else spending
        median_completion = profiles_df['completion_rate_mean'].median() if 'completion_rate_mean' in profiles_df.columns else completion
        median_income = profiles_df['income_imputed_mean'].median() if 'income_imputed_mean' in profiles_df.columns else income
        median_tenure = profiles_df['tenure_months_mean'].median() if 'tenure_months_mean' in profiles_df.columns else tenure
        
        if spending > median_spend and completion > median_completion:
            return 'high_value_loyalist'
        elif completion < 0.30 or view_rate < 0.40:
            return 'low_engagement'
        elif bogo_comp > discount_comp and bogo_comp > 0.50:
            return 'bogo_responder'
        elif discount_comp > bogo_comp and discount_comp > 0.50:
            return 'discount_seeker'
        elif income > median_income and spending <= median_spend:
            return 'high_income_casual'
        elif tenure > median_tenure and spending <= median_spend:
            return 'tenured_conservative'
        else:
            return 'moderate_spender'
    
    narratives = {}
    segment_assignments = {}
    
    assigned_types = []
    for _, row in profiles_df.iterrows():
        cluster_type = classify_cluster(row)
        if cluster_type in assigned_types:
            cluster_type = 'moderate_spender'
        assigned_types.append(cluster_type)
    
    for i, (_, row) in enumerate(profiles_df.iterrows()):
        cluster_id = int(row['cluster_id'])
        cluster_type = assigned_types[i]
        template = NAMING_TEMPLATES[cluster_type]
        
        narrative = {
            'cluster_id': cluster_id,
            'segment_name': template['name'],
            'segment_type': cluster_type,
            'size': int(row['size']),
            'pct_of_total': float(row['size'] / profiles_df['size'].sum()),
            'description': template['description'],
            'key_characteristics': {}
        }
        
        char_metrics = {
            'age': row.get('age_imputed_mean'),
            'income': row.get('income_imputed_mean'),
            'tenure_months': row.get('tenure_months_mean'),
            'avg_spend': row.get('trans_total_mean'),
            'avg_transaction': row.get('trans_avg_mean'),
            'view_rate': row.get('view_rate_mean'),
            'completion_rate': row.get('completion_rate_mean'),
            'bogo_completion_rate': row.get('bogo_completion_rate_mean'),
            'discount_completion_rate': row.get('discount_completion_rate_mean')
        }
        narrative['key_characteristics'] = {mk: mv for mk, mv in char_metrics.items() if mv is not None and not (isinstance(mv, float) and np.isnan(mv))}
        
        if business_df is not None:
            biz_row = business_df[business_df['cluster_id'] == cluster_id]
            if len(biz_row) > 0:
                biz_row = biz_row.iloc[0]
                narrative['business_metrics'] = {
                    'revenue_per_customer': float(biz_row['revenue_per_customer']),
                    'offer_roi': float(biz_row['offer_roi']),
                    'churn_risk_proxy': float(biz_row['churn_risk_proxy'])
                }
        
        narrative['recommendations'] = template['recommendations']
        narratives[f'cluster_{cluster_id}'] = narrative
        segment_assignments[cluster_id] = template['name']
    
    for key, narrative in narratives.items():
        print(f"\n{'─'*50}")
        print(f"  CLUSTER {narrative['cluster_id']}: {narrative['segment_name']}")
        print(f"{'─'*50}")
        print(f"  Size: {narrative['size']} customers ({narrative['pct_of_total']:.1%})")
        print(f"  {narrative['description']}")
        print(f"  Key Characteristics:")
        for metric, value in narrative['key_characteristics'].items():
            if metric in ['age', 'income', 'tenure_months', 'avg_spend', 'avg_transaction']:
                print(f"    {metric}: {value:,.2f}")
            else:
                print(f"    {metric}: {value:.2%}" if value and value < 1 else f"    {metric}: {value}")
        if 'business_metrics' in narrative:
            bm = narrative['business_metrics']
            print(f"  Business Metrics:")
            print(f"    Revenue/Customer: ${bm['revenue_per_customer']:,.2f}")
            print(f"    Offer ROI: {bm['offer_roi']:.1f}x")
            print(f"    Churn Risk: {bm['churn_risk_proxy']:.3f}")
        print(f"  Recommendations:")
        for rec in narrative['recommendations']:
            print(f"    to {rec}")
    
    segment_df = pd.DataFrame([
        {'cluster_id': n['cluster_id'], 'segment_name': n['segment_name'],
         'segment_type': n['segment_type'], 'size': n['size']}
        for n in narratives.values()
    ])
    segment_df.to_csv(Path('data/processed') / 'segment_names.csv', index=False)
    
    print(f"\n Segment names saved to data/processed/segment_names.csv")
    
    return {
        'narratives': narratives,
        'segment_assignments': segment_assignments
    }


def save_clustering_results(df_with_clusters: pd.DataFrame, profiles_df: pd.DataFrame, 
                             optimization_results: Dict, k: int, base_path: str = '.') -> None:
    """
    Save clustering results to disk.
    
    Args:
        df_with_clusters: Customer data with cluster labels
        profiles_df: Cluster profiles
        optimization_results: Results from cluster optimization
        k: Number of clusters used
        base_path: Base directory
    """
    output_dir = Path(base_path) / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save customer clusters
    df_with_clusters[['id', 'cluster']].to_csv(output_dir / 'customer_clusters.csv', index=False)
    
    # Save cluster profiles
    profiles_df.to_csv(output_dir / 'cluster_profiles.csv', index=False)
    
    # Save optimization results
    with open(output_dir / 'cluster_optimization.json', 'w') as f:
        json.dump(optimization_results, f, indent=2, default=str)
    
    print(f"\n Clustering results saved to {output_dir}")


def generate_clustering_report(profiles_df: pd.DataFrame, optimization_results: Dict, 
                                k: int, stability_results: Dict = None,
                                business_df: pd.DataFrame = None,
                                narrative_results: Dict = None,
                                output_path: str = 'reports/clustering_report.json') -> None:
    """
    Generate a comprehensive clustering report.
    
    Args:
        profiles_df: Cluster profiles DataFrame
        optimization_results: Optimization results
        k: Number of clusters
        stability_results: Results from analyze_cluster_stability() (optional)
        business_df: Business metrics DataFrame (optional)
        narrative_results: Results from generate_cluster_narratives() (optional)
        output_path: Path to save the report
    """
    opt_idx = k - 2
    report = {
        'method': 'K-Means',
        'optimal_k': k,
        'optimization': {
            'silhouette_score': optimization_results['silhouette_scores'][opt_idx],
            'calinski_score': optimization_results['calinski_scores'][opt_idx],
            'davies_bouldin_score': optimization_results['davies_bouldin_scores'][opt_idx],
            'wcss': optimization_results['wcss'][opt_idx],
            'optimal_k_silhouette': optimization_results['optimal_k_silhouette'],
            'optimal_k_elbow': optimization_results['optimal_k_elbow'],
            'optimal_k_db': optimization_results['optimal_k_db'],
            'optimal_k_gap': optimization_results['optimal_k_gap']
        },
        'cluster_profiles': profiles_df.to_dict('records')
    }
    
    if stability_results is not None:
        report['stability'] = {
            'mean_ari': stability_results['mean_ari'],
            'std_ari': stability_results['std_ari'],
            'min_ari': stability_results['min_ari'],
            'interpretation': stability_results['interpretation']
        }
    
    if business_df is not None:
        report['business_metrics'] = business_df.to_dict('records')
    
    if narrative_results is not None:
        report['segment_narratives'] = narrative_results['narratives']
        report['segment_assignments'] = narrative_results['segment_assignments']
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n Clustering report saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - CLUSTERING")
    print("="*60)
    
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading customer features...")
    customer_features = load_customer_features()
    print(f" Loaded {len(customer_features)} customers with {len(customer_features.columns)-1} features")
    
    X_scaled, scaler, feature_names = preprocess_for_clustering(customer_features)
    
    optimization_results = find_optimal_clusters(X_scaled, max_k=10)
    optimal_k = optimization_results['optimal_k_silhouette']
    
    k = 4
    print(f"\nUsing k={k} clusters for business interpretability")
    
    result_df, kmeans_model = apply_kmeans_clustering(X_scaled, k, feature_names)
    
    labels = result_df['cluster'].values
    
    customer_with_clusters = customer_features.copy()
    customer_with_clusters['cluster'] = labels
    
    stability_results = analyze_cluster_stability(X_scaled, k)
    
    visualize_clusters_pca(X_scaled, labels, k, fig_dir)
    
    profiles_df = interpret_clusters(customer_features, labels, k)
    
    create_cluster_visualizations(profiles_df, customer_with_clusters, fig_dir)
    
    business_df = validate_clusters_business(customer_features, labels, k, fig_dir)
    
    narrative_results = generate_cluster_narratives(profiles_df, business_df, k)
    
    save_clustering_results(customer_with_clusters, profiles_df, optimization_results, k)
    
    generate_clustering_report(profiles_df, optimization_results, k,
                               stability_results=stability_results,
                               business_df=business_df,
                               narrative_results=narrative_results)
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f" Identified {k} customer segments")
    print(f" Stability: Mean ARI = {stability_results['mean_ari']:.3f} ({stability_results['interpretation']})")
    for cid, name in narrative_results['segment_assignments'].items():
        size = int(profiles_df.loc[profiles_df['cluster_id'] == cid, 'size'].values[0])
        print(f"  Cluster {cid}: {name} ({size} customers)")
    print(f" Visualizations saved to: {fig_dir}")
    print(f" Results saved to: data/processed/")
    print("="*60)
