"""
SHAP (SHapley Additive exPlanations) analysis for Starbucks offer completion model.

This module:
1. Loads the trained XGBoost model from Phase 5
2. Calculates SHAP values for model interpretability
3. Generates publication-quality explainability visualizations
4. Answers: "Why does the model make specific predictions?"

SHAP values show exactly how each feature contributes to individual predictions,
making the "black box" model transparent and trustworthy for business stakeholders.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from typing import Tuple, List, Dict
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Initialize SHAP (needed for some versions)
shap.initjs()


def load_model_and_data(base_path: str = '.') -> Tuple:
    """
    Load the trained model, scaler, and sample data for SHAP analysis.
    
    Args:
        base_path: Base directory
        
    Returns:
        Tuple of (model, scaler, X_sample, feature_names)
    """
    print("\n" + "="*60)
    print("LOADING MODEL AND DATA FOR SHAP ANALYSIS")
    print("="*60)
    
    # Load model
    model_path = Path(base_path) / 'data' / 'processed' / 'best_model.pkl'
    model = joblib.load(model_path)
    print(f" Loaded model: {type(model).__name__}")
    
    # Load scaler (need to recreate or save/load it)
    # For now, we'll reload and preprocess the data
    interactions = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'interaction_features.csv')
    
    # Preprocess (same as predictive_modeling.py)
    exclude_cols = ['customer_id', 'offer_id', 'completed']
    exclude_patterns = ['_bin']
    
    feature_cols = []
    for col in interactions.columns:
        if col in exclude_cols:
            continue
        if any(pattern in col for pattern in exclude_patterns):
            continue
        if pd.api.types.is_numeric_dtype(interactions[col]):
            feature_cols.append(col)
    
    X = interactions[feature_cols].fillna(0)
    y = interactions['completed'].values
    
    # Sample data for SHAP (use 1000 examples for computational efficiency)
    from sklearn.model_selection import train_test_split
    _, X_sample, _, _ = train_test_split(X, y, test_size=1000, random_state=42, stratify=y)
    
    print(f" Loaded {len(X_sample)} samples for SHAP analysis")
    print(f" Number of features: {len(feature_cols)}")
    
    return model, X_sample.values, feature_cols


def calculate_shap_values(model, X_sample: np.ndarray) -> Tuple:
    """
    Calculate SHAP values for the model.
    
    Args:
        model: Trained model (XGBoost)
        X_sample: Sample feature matrix
        
    Returns:
        Tuple of (shap_values, X_sample)
    """
    print("\n" + "="*60)
    print("CALCULATING SHAP VALUES")
    print("="*60)
    
    # Use TreeExplainer for tree-based models (fast)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print(f" SHAP values calculated")
    print(f"  Shape: {shap_values.shape}")
    print(f"  Base value (expected model output): {explainer.expected_value:.4f}")
    
    return shap_values, X_sample, explainer.expected_value


def create_shap_visualizations(shap_values: np.ndarray, X_sample: np.ndarray, 
                               feature_names: List[str], expected_value: float,
                               fig_dir: Path) -> None:
    """
    Create comprehensive SHAP visualizations.
    
    Args:
        shap_values: Calculated SHAP values
        X_sample: Sample feature matrix
        feature_names: List of feature names
        expected_value: Base value from explainer
        fig_dir: Directory to save figures
    """
    print("\nCreating SHAP visualizations...")
    
    # Create a SHAP Explanation object for newer SHAP versions
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_sample,
        feature_names=feature_names
    )
    
    # Figure 1: Summary plot (beeswarm) - overall feature importance
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_exp, show=False, max_display=20)
    plt.title('SHAP Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: shap_summary_beeswarm.png")
    
    # Figure 2: Bar plot - mean absolute SHAP values
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_exp, show=False, max_display=20)
    plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: shap_summary_bar.png")
    
    # Figure 3: Waterfall plot for a single prediction (example)
    # Choose an example with high prediction probability
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_exp[0], show=False, max_display=10)
    plt.title(f'SHAP Waterfall Plot (Example: Customer-Offer Interaction)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_waterfall_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: shap_waterfall_example.png")
    
    # Figure 4: Dependence plot for top feature
    top_feature_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-1]
    top_feature_name = feature_names[top_feature_idx]
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values, X_sample, 
                         feature_names=feature_names, show=False)
    plt.title(f'SHAP Dependence Plot: {top_feature_name}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_dependence_top_feature.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: shap_dependence_top_feature.png")
    
    # Figure 5: Interaction plot (top 2 features)
    top_2_features = np.argsort(np.abs(shap_values).mean(axis=0))[-2:]
    
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(
        top_2_features[0], shap_values, X_sample,
        interaction_index=top_2_features[1],
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Interaction: {feature_names[top_2_features[0]]} × {feature_names[top_2_features[1]]}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_interaction_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: shap_interaction_plot.png")
    
    print(f"\n All SHAP visualizations saved to {fig_dir}")


def generate_shap_report(shap_values: np.ndarray, feature_names: List[str],
                         output_path: str = 'reports/shap_report.json') -> Dict:
    """
    Generate a detailed SHAP analysis report.
    
    Args:
        shap_values: Calculated SHAP values
        feature_names: List of feature names
        output_path: Path to save the report
        
    Returns:
        Dictionary with SHAP analysis summary
    """
    print("\n" + "="*60)
    print("GENERATING SHAP REPORT")
    print("="*60)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create feature importance ranking
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\nTop 15 Features by Mean |SHAP|:")
    print(shap_importance.head(15).to_string(index=False))
    
    # Calculate SHAP value statistics
    report = {
        'top_15_features': shap_importance.head(15).to_dict('records'),
        'shap_statistics': {
            'mean_abs_shap_global': float(mean_abs_shap.mean()),
            'std_shap_global': float(np.abs(shap_values).std()),
            'max_abs_shap': float(np.abs(shap_values).max()),
            'min_abs_shap': float(np.abs(shap_values).min())
        },
        'interpretation': {
            'top_feature': shap_importance.iloc[0]['feature'],
            'top_feature_shap': float(shap_importance.iloc[0]['mean_abs_shap']),
            'explanation': "Higher SHAP values indicate stronger push toward offer completion prediction"
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n SHAP report saved to: {output_path}")
    
    return report


def create_interactive_shap_dashboard(shap_values: np.ndarray, X_sample: np.ndarray,
                                       feature_names: List[str], expected_value: float,
                                       fig_dir: Path) -> None:
    """
    Create interactive SHAP visualizations using Plotly.
    
    Args:
        shap_values: Calculated SHAP values
        X_sample: Sample feature matrix
        feature_names: List of feature names
        expected_value: Base value from explainer
        fig_dir: Directory to save figures
    """
    print("\nCreating interactive SHAP dashboard...")
    
    # Calculate mean absolute SHAP values for top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_n = 20
    top_indices = np.argsort(mean_abs_shap)[-top_n:]
    
    # Create DataFrame for plotting
    shap_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_indices],
        'Mean |SHAP|': mean_abs_shap[top_indices]
    }).sort_values('Mean |SHAP|', ascending=True)
    
    # Interactive bar chart
    import plotly.express as px
    fig = px.bar(
        shap_df, x='Mean |SHAP|', y='Feature', orientation='h',
        title='SHAP Feature Importance (Interactive)',
        color='Mean |SHAP|', color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600, width=800)
    fig.write_html(fig_dir / 'shap_interactive_bar.html')
    
    print(f"   Saved: shap_interactive_bar.html")
    print(f"\n Interactive SHAP dashboard saved to {fig_dir}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - SHAP ANALYSIS")
    print("="*60)
    
    # Setup
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, X_sample, feature_names = load_model_and_data()
    
    # Calculate SHAP values
    shap_values, X_sample, expected_value = calculate_shap_values(model, X_sample)
    
    # Create visualizations
    create_shap_visualizations(shap_values, X_sample, feature_names, expected_value, fig_dir)
    
    # Generate report
    shap_report = generate_shap_report(shap_values, feature_names)
    
    # Create interactive dashboard
    create_interactive_shap_dashboard(shap_values, X_sample, feature_names, expected_value, fig_dir)
    
    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETE")
    print("="*60)
    print(f" Model interpretability achieved with SHAP")
    print(f" Top feature: {shap_report['interpretation']['top_feature']}")
    print(f" Visualizations saved to: {fig_dir}")
    print(f" Report saved to: reports/shap_report.json")
    print("="*60)
    print("\n Key Insight: SHAP values show exactly how each feature contributes")
    print("   to individual predictions, making the XGBoost model transparent!")
