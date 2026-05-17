"""Tests for the clustering module (src/models/clustering.py)."""

import pandas as pd
import numpy as np
from src.models.clustering import (
    preprocess_for_clustering,
    find_optimal_clusters,
)


class TestPreprocessForClustering:
    def test_returns_scaled_data(self):
        features = pd.DataFrame({
            "id": ["a", "b", "c", "d", "e"],
            "age_imputed": [25, 30, 35, 40, 45],
            "income_imputed": [50000, 60000, 70000, 80000, 90000],
            "tenure_months": [1, 12, 24, 36, 48],
            "gender_M": [1, 0, 1, 0, 1],
            "gender_F": [0, 1, 0, 1, 0],
            "trans_count": [0, 5, 10, 15, 20],
            "offers_received": [1, 3, 5, 7, 9],
            "offers_completed": [0, 1, 3, 5, 7],
            "completion_rate": [0.0, 0.33, 0.6, 0.71, 0.78],
        })
        X, scaler, feature_names = preprocess_for_clustering(features)
        assert X.shape[0] == 5
        assert abs(X.mean().mean()) < 0.5  # approximately centered


class TestFindOptimalClusters:
    def test_returns_optimization_results(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 4))
        results = find_optimal_clusters(X, max_k=6)
        assert "optimal_k_silhouette" in results
        assert "optimal_k_elbow" in results
        assert len(results["silhouette_scores"]) == 5  # k = 2..6
