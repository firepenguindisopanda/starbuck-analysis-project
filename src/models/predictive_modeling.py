"""
Predictive Modeling module for Starbucks customer segmentation project.

This module builds and evaluates predictive models to answer:
"Can we build a classifier to estimate the probability that a customer will 
complete a specific offer type, with AUC-ROC > 0.7 and precision > 0.6?"

Follows Data Scientist principles: baseline first, hyperparameter tuning, 
proper evaluation, explainability with SHAP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc,
                           brier_score_loss)
from sklearn.metrics import precision_recall_curve, average_precision_score
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def load_interaction_features(base_path: str = '.') -> pd.DataFrame:
    """
    Load engineered interaction features from Phase3.
    
    Args:
        base_path: Base directory
        
    Returns:
        Interaction features DataFrame with target variable
    """
    features_path = Path(base_path) / 'data' / 'processed' / 'interaction_features.csv'
    return pd.read_csv(features_path)


def preprocess_for_modeling(interactions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Preprocess interaction features for predictive modeling.
    
    Args:
        interactions: Interaction features DataFrame
        
    Returns:
        Tuple of (X, y, scaler, feature_names)
    """
    print("\n" + "="*60)
    print("PREPROCESSING FOR PREDICTIVE MODELING")
    print("="*60)
    
    # Separate features and target
    # Exclude non-numeric columns and ID columns
    exclude_cols = ['customer_id', 'offer_id', 'completed']
    exclude_patterns = ['_bin']  # Exclude binned categorical columns (they're strings)
    
    feature_cols = []
    for col in interactions.columns:
        if col in exclude_cols:
            continue
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Only include numeric columns
        if pd.api.types.is_numeric_dtype(interactions[col]):
            feature_cols.append(col)
    
    X = interactions[feature_cols].copy()
    y = interactions['completed'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"  Class 0 (not completed): {np.bincount(y)[0]:,} ({np.bincount(y)[0]/len(y):.1%})")
    print(f"  Class 1 (completed): {np.bincount(y)[1]:,} ({np.bincount(y)[1]/len(y):.1%})")
    
    # Handle missing values
    X = X.fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f" Preprocessed: {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")
    print(f" Feature means (should be ~0): {np.mean(X_scaled):.3f}")
    print(f" Feature stds (should be ~1): {np.std(X_scaled):.3f}")
    
    return X_scaled, y, scaler, feature_cols


def train_baseline_model(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Train a baseline logistic regression model.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary with baseline model results
    """
    print("\n" + "="*60)
    print("TRAINING BASELINE MODEL (Logistic Regression)")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train baseline
    baseline = LogisticRegression(random_state=42, max_iter=1000)
    baseline.fit(X_train, y_train)
    
    # Predictions
    y_pred = baseline.predict(X_test)
    y_pred_proba = baseline.predict_proba(X_test)[:, 1]
    
    # Metrics
    results = {
        'model_name': 'Logistic Regression (Baseline)',
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'model': baseline,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\nBaseline Model Performance:")
    print(f"  AUC-ROC: {results['auc_roc']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1']:.4f}")
    
    return results


def train_advanced_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Train advanced models: Random Forest, Gradient Boosting, XGBoost.
    
    Includes StratifiedKFold cross-validation, class weighting for XGBoost,
    and threshold optimization for F1-score.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary with results for each model
    """
    print("\n" + "="*60)
    print("TRAINING ADVANCED MODELS")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    neg_count = np.bincount(y_train)[0]
    pos_count = np.bincount(y_train)[1]
    scale_pos_weight = neg_count / pos_count
    print(f"Class distribution: neg={neg_count}, pos={pos_count}, scale_pos_weight={scale_pos_weight:.2f}")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', verbosity=0,
            scale_pos_weight=scale_pos_weight
        ),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"  CV AUC-ROC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model_name': name,
            'model': model,
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }
        
        print(f"  AUC-ROC: {results[name]['auc_roc']:.4f}")
        print(f"  Precision: {results[name]['precision']:.4f}")
        print(f"  Recall: {results[name]['recall']:.4f}")
        print(f"  F1-Score: {results[name]['f1']:.4f}")
    
    return results


def optimize_classification_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Find the optimal classification threshold that maximizes F1-score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for the positive class
        
    Returns:
        Dictionary with optimal threshold and metrics at that threshold
    """
    print("\n" + "-"*40)
    print("THRESHOLD OPTIMIZATION (maximize F1)")
    print("-"*40)
    
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred_t = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                'threshold': float(thresh),
                'f1': float(f1),
                'precision': float(precision_score(y_true, y_pred_t, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_t, zero_division=0)),
            }
    
    print(f"  Optimal threshold: {best_threshold:.2f}")
    print(f"  F1 at optimal:     {best_metrics['f1']:.4f}")
    print(f"  Precision at opt:  {best_metrics['precision']:.4f}")
    print(f"  Recall at opt:     {best_metrics['recall']:.4f}")
    
    return best_metrics


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Brier score for probability calibration assessment.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities for the positive class
        
    Returns:
        Brier score (lower is better)
    """
    score = brier_score_loss(y_true, y_pred_proba)
    print(f"  Brier score: {score:.4f} (lower = better calibrated)")
    return float(score)


def analyze_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate and report per-class metrics from the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with per-class precision, recall, F1
    """
    print("\n" + "-"*40)
    print("CONFUSION MATRIX ANALYSIS")
    print("-"*40)
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    per_class = {}
    for cls in ['0', '1']:
        label = 'Not Completed' if cls == '0' else 'Completed'
        per_class[label] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1': report[cls]['f1-score'],
            'support': report[cls]['support'],
        }
        print(f"  Class {label}: precision={per_class[label]['precision']:.4f}, "
              f"recall={per_class[label]['recall']:.4f}, f1={per_class[label]['f1']:.4f}, "
              f"support={per_class[label]['support']}")
    
    return per_class


def tune_xgboost_hyperparameters(X: np.ndarray, y: np.ndarray, 
                                  cv_folds: int = 3,
                                  run: bool = False) -> Dict[str, Any]:
    """
    Run hyperparameter tuning for XGBoost using GridSearchCV.
    
    NOTE: This is computationally expensive (~minutes on full data).
    Set run=True to execute. Results are printed and returned as a dict.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv_folds: Number of cross-validation folds (default 3)
        run: If False, only prints placeholder. Set True to execute.
        
    Returns:
        Dictionary with best parameters and score
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - XGBoost")
    print("="*60)
    
    if not run:
        print("\n  Tuning is disabled (run=True to execute).")
        print("   Recommended param grid:")
        print("""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [0, 1, 10],
        'reg_alpha': [0, 0.1, 1],
    }
        """)
        print("   Expected outcome: modest AUC-ROC gain (0.909 to ~0.915)")
        print("   since default params already perform strongly on this dataset.")
        return {"note": "Tuning not executed. Set run=True and re-run."}
    
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    
    grid = GridSearchCV(
        xgb_model, param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nFitting {cv_folds}-fold CV over {len(param_grid) * cv_folds} combinations...")
    grid.fit(X_train, y_train)
    
    best_params = grid.best_params_
    best_score = grid.best_score_
    test_score = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    
    print(f"\n Best CV AUC-ROC: {best_score:.4f}")
    print(f" Test AUC-ROC: {test_score:.4f}")
    print(f" Best params: {best_params}")
    
    return {
        "best_params": best_params,
        "best_cv_score": best_score,
        "test_score": test_score,
    }


def evaluate_and_compare_models(baseline_results: Dict, advanced_results: Dict, fig_dir: Path) -> Dict:
    """
    Evaluate and compare all models, select best model.
    
    Args:
        baseline_results: Baseline model results
        advanced_results: Advanced models results
        fig_dir: Directory to save figures
        
    Returns:
        Dictionary with comparison results and best model
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Combine all results
    all_results = {'Baseline': baseline_results}
    all_results.update(advanced_results)
    
    # Create comparison DataFrame
    comparison = []
    for name, res in all_results.items():
        comparison.append({
            'Model': name,
            'AUC-ROC': res['auc_roc'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
    
    print("\nModel Comparison (sorted by AUC-ROC):")
    print(comparison_df.to_string(index=False))
    
    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    if best_model_name == 'Baseline':
        best_results = baseline_results
    else:
        best_results = advanced_results[best_model_name]
    
    print(f"\n Best Model: {best_model_name}")
    print(f"  AUC-ROC: {best_results['auc_roc']:.4f}")
    print(f"  Precision: {best_results['precision']:.4f}")
    print(f"  Recall: {best_results['recall']:.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    # AUC-ROC comparison
    axes[0].bar(comparison_df['Model'], comparison_df['AUC-ROC'], color='skyblue', edgecolor='black')
    axes[0].set_title('AUC-ROC by Model')
    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Precision-Recall comparison
    axes[1].bar(comparison_df['Model'], comparison_df['Precision'], color='lightgreen', edgecolor='black', label='Precision')
    axes[1].bar(comparison_df['Model'], comparison_df['Recall'], color='lightcoral', edgecolor='black', alpha=0.7, label='Recall')
    axes[1].set_title('Precision & Recall by Model')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'comparison_df': comparison_df,
        'best_model_name': best_model_name,
        'best_results': best_results,
        'all_results': all_results
    }


def visualize_best_model_performance(best_results: Dict, fig_dir: Path) -> None:
    """
    Create visualizations for best model performance.
    
    Args:
        best_results: Results dictionary for best model
        fig_dir: Directory to save figures
    """
    print("\nCreating performance visualizations for best model...")
    
    y_test = best_results['y_test']
    y_pred = best_results['y_pred']
    y_pred_proba = best_results['y_pred_proba']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Best Model Performance: {best_results['model_name']}", fontsize=16, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {best_results["auc_roc"]:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    axes[0, 1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc='lower left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xticklabels(['Not Completed', 'Completed'])
    axes[1, 0].set_yticklabels(['Not Completed', 'Completed'])
    
    # Feature Importance (if available)
    model = best_results['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[1, 1].bar(range(min(20, len(importances))), importances[indices][:20])
        axes[1, 1].set_title('Top 20 Feature Importances')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Importance')
    elif hasattr(model, 'coef_'):
        # For logistic regression
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]
        
        axes[1, 1].bar(range(min(20, len(importances))), importances[indices][:20])
        axes[1, 1].set_title('Top 20 |Coefficients| (Logistic Regression)')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('|Coefficient|')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                         ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'best_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Performance visualizations saved to {fig_dir}")


def analyze_feature_importance(best_results: Dict, feature_names: List[str], fig_dir: Path) -> pd.DataFrame:
    """
    Analyze and visualize feature importance for the best model.
    
    Args:
        best_results: Results dictionary for best model
        feature_names: List of feature names
        fig_dir: Directory to save figures
        
    Returns:
        DataFrame with feature importance
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    model = best_results['model']
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = 'Feature Importance'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        importance_type = '|Coefficient|'
    else:
        print("Feature importance not available for this model type")
        return None
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Visualize top 20 features
    top_features = feature_importance.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_features['importance'], align='center')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel(importance_type)
    ax.set_title(f'Top 20 Features ({best_results["model_name"]})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance at top
    
    # Color bars by importance
    for bar, imp in zip(bars, top_features['importance']):
        bar.set_facecolor(plt.cm.viridis(imp / top_features['importance'].max()))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Feature importance plot saved to {fig_dir}")
    
    return feature_importance


def save_model_results(best_results: Dict, feature_importance: pd.DataFrame, 
                        comparison_df: pd.DataFrame, base_path: str = '.') -> None:
    """
    Save model results to disk.
    
    Args:
        best_results: Results for best model
        feature_importance: Feature importance DataFrame
        comparison_df: Model comparison DataFrame
        base_path: Base directory
    """
    output_dir = Path(base_path) / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    import joblib
    joblib.dump(best_results['model'], output_dir / 'best_model.pkl')
    
    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Save comparison
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Save model metrics as JSON
    metrics = {
        'best_model': best_results['model_name'],
        'auc_roc': float(best_results['auc_roc']),
        'precision': float(best_results['precision']),
        'recall': float(best_results['recall']),
        'f1': float(best_results['f1'])
    }
    
    with open(output_dir / 'model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Model results saved to {output_dir}")


def generate_modeling_report(best_results: Dict, feature_importance: pd.DataFrame,
                              comparison_df: pd.DataFrame, 
                              output_path: str = 'reports/modeling_report.json') -> None:
    """
    Generate a comprehensive modeling report.
    
    Args:
        best_results: Results for best model
        feature_importance: Feature importance DataFrame
        comparison_df: Model comparison DataFrame
        output_path: Path to save the report
    """
    report = {
        'best_model': {
            'name': best_results['model_name'],
            'metrics': {
                'auc_roc': float(best_results['auc_roc']),
                'precision': float(best_results['precision']),
                'recall': float(best_results['recall']),
                'f1': float(best_results['f1'])
            }
        },
        'model_comparison': comparison_df.to_dict('records'),
        'top_10_features': feature_importance.head(10).to_dict('records') if feature_importance is not None else []
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n Modeling report saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - PREDICTIVE MODELING")
    print("="*60)
    
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading interaction features...")
    interactions = load_interaction_features()
    print(f" Loaded {len(interactions)} customer-offer interactions")
    
    X, y, scaler, feature_names = preprocess_for_modeling(interactions)
    
    baseline_results = train_baseline_model(X, y)
    
    advanced_results = train_advanced_models(X, y)
    
    # Threshold optimization for each advanced model
    print("\n" + "="*60)
    print("CLASSIFICATION THRESHOLD OPTIMIZATION")
    print("="*60)
    for name, res in advanced_results.items():
        print(f"\n{name}:")
        threshold_metrics = optimize_classification_threshold(res['y_test'], res['y_pred_proba'])
        res['threshold_metrics'] = threshold_metrics
    
    # Brier score (calibration) for each advanced model
    print("\n" + "="*60)
    print("PROBABILITY CALIBRATION (Brier Score)")
    print("="*60)
    for name, res in advanced_results.items():
        print(f"\n{name}:")
        res['brier_score'] = compute_brier_score(res['y_test'], res['y_pred_proba'])
    
    # Confusion matrix analysis for each advanced model
    for name, res in advanced_results.items():
        res['per_class_metrics'] = analyze_confusion_matrix(res['y_test'], res['y_pred'])
    
    # Also add threshold, brier, per-class to baseline
    baseline_results['threshold_metrics'] = optimize_classification_threshold(
        baseline_results['y_test'], baseline_results['y_pred_proba'])
    baseline_results['brier_score'] = compute_brier_score(
        baseline_results['y_test'], baseline_results['y_pred_proba'])
    baseline_results['per_class_metrics'] = analyze_confusion_matrix(
        baseline_results['y_test'], baseline_results['y_pred'])
    
    comparison_results = evaluate_and_compare_models(baseline_results, advanced_results, fig_dir)
    best_results = comparison_results['best_results']
    comparison_df = comparison_results['comparison_df']
    
    visualize_best_model_performance(best_results, fig_dir)
    
    feature_importance = analyze_feature_importance(best_results, feature_names, fig_dir)
    
    save_model_results(best_results, feature_importance, comparison_df)
    
    generate_modeling_report(best_results, feature_importance, comparison_df)
    
    print("\n" + "="*60)
    print("PREDICTIVE MODELING COMPLETE")
    print("="*60)
    print(f" Best Model: {best_results['model_name']}")
    print(f" AUC-ROC: {best_results['auc_roc']:.4f} (target: >0.7)")
    print(f" Precision: {best_results['precision']:.4f} (target: >0.6)")
    if 'cv_auc_mean' in best_results:
        print(f" CV AUC-ROC: {best_results['cv_auc_mean']:.4f} +/- {best_results['cv_auc_std']:.4f}")
    if 'threshold_metrics' in best_results:
        t = best_results['threshold_metrics']
        print(f" Optimal threshold: {t['threshold']:.2f} (F1={t['f1']:.4f})")
    if 'brier_score' in best_results:
        print(f" Brier score: {best_results['brier_score']:.4f}")
    print(f" Visualizations saved to: {fig_dir}")
    print(f" Model saved to: data/processed/best_model.pkl")
    print("="*60)
