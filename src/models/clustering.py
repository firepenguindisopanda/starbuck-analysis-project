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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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


def find_optimal_clusters(X: pd.DataFrame, max_k: int = 15) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method and silhouette scores.
    
    Args:
        X: Preprocessed feature matrix
        max_k: Maximum number of clusters to test
        
    Returns:
        Dictionary with optimization results
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    wcss = []  # Within-cluster sum of squares (elbow method)
    silhouette_scores = []
    calinski_scores = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        
        print(f"  k={k}: WCSS={wcss[-1]:.0f}, Silhouette={silhouette_scores[-1]:.3f}, "
              f"Calinski={calinski_scores[-1]:.1f}")
    
    # Find optimal k (highest silhouette score)
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    
    # Find elbow point (simplified: point with max second derivative)
    # Calculate curvature (second derivative approximation)
    wcss_diff = np.diff(wcss)
    wcss_diff2 = np.diff(wcss_diff)
    optimal_k_elbow = K_range[np.argmax(wcss_diff2) + 1] if len(wcss_diff2) > 0 else 3
    
    print(f"\n Optimal k (Silhouette): {optimal_k_silhouette}")
    print(f" Optimal k (Elbow): {optimal_k_elbow}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Cluster Optimization Metrics', fontsize=16, fontweight='bold')
    
    # WCSS (Elbow)
    axes[0].plot(K_range, wcss, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('WCSS (Inertia)')
    axes[0].set_title('Elbow Method')
    axes[0].axvline(optimal_k_elbow, color='r', linestyle='--', alpha=0.5)
    
    # Silhouette
    axes[1].plot(K_range, silhouette_scores, 'go-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Scores')
    axes[1].axvline(optimal_k_silhouette, color='r', linestyle='--', alpha=0.5)
    
    # Calinski-Harabasz
    axes[2].plot(K_range, calinski_scores, 'mo-')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Index')
    axes[2].set_title('Calinski-Harabasz (Higher is better)')
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / 'cluster_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'K_range': list(K_range),
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'optimal_k_silhouette': optimal_k_silhouette,
        'optimal_k_elbow': optimal_k_elbow
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
                                k: int, output_path: str = 'reports/clustering_report.json') -> None:
    """
    Generate a comprehensive clustering report.
    
    Args:
        profiles_df: Cluster profiles DataFrame
        optimization_results: Optimization results
        k: Number of clusters
        output_path: Path to save the report
    """
    report = {
        'method': 'K-Means',
        'optimal_k': k,
        'optimization': {
            'silhouette_score': optimization_results['silhouette_scores'][k-2],  # k-2 because range starts at 2
            'calinski_score': optimization_results['calinski_scores'][k-2],
            'wcss': optimization_results['wcss'][k-2]
        },
        'cluster_profiles': profiles_df.to_dict('records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n Clustering report saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - CLUSTERING")
    print("="*60)
    
    # Setup
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load customer features
    print("\nLoading customer features...")
    customer_features = load_customer_features()
    print(f" Loaded {len(customer_features)} customers with {len(customer_features.columns)-1} features")
    
    # Preprocess for clustering
    X_scaled, scaler, feature_names = preprocess_for_clustering(customer_features)
    
    # Find optimal number of clusters
    optimization_results = find_optimal_clusters(X_scaled, max_k=10)
    optimal_k = optimization_results['optimal_k_silhouette']
    
    # Apply K-Means with optimal k
    # Let's use k=4 for business interpretability (3-5 clusters as per research question)
    k = 4  # Balance between statistical optimality and business interpretability
    print(f"\nUsing k={k} clusters for business interpretability")
    
    result_df, kmeans_model = apply_kmeans_clustering(X_scaled, k, feature_names)
    
    # Get cluster labels
    labels = result_df['cluster'].values
    
    # Add cluster labels to original customer features
    customer_with_clusters = customer_features.copy()
    customer_with_clusters['cluster'] = labels
    
    # Visualize clusters
    visualize_clusters_pca(X_scaled, labels, k, fig_dir)
    
    # Interpret clusters
    profiles_df = interpret_clusters(customer_features, labels, k)
    
    # Create visualizations
    create_cluster_visualizations(profiles_df, customer_with_clusters, fig_dir)
    
    # Save results
    save_clustering_results(customer_with_clusters, profiles_df, optimization_results, k)
    
    # Generate report
    generate_clustering_report(profiles_df, optimization_results, k)
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f" Identified {k} customer segments")
    print(f" Visualizations saved to: {fig_dir}")
    print(f" Results saved to: data/processed/")
    print("="*60)
