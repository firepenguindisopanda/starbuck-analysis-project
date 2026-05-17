# Starbucks Customer Segmentation & Offer Recommendation

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org)
[![Built with uv](https://img.shields.io/badge/Built%20with-uv-blueviolet)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data science project that optimizes Starbucks offer targeting using customer segmentation and machine learning. Built for my data science portfolio to demonstrate skills in EDA, feature engineering, clustering, predictive modeling, and causal inference.

---

##  Business Problem

Starbucks sends promotional offers to mobile app users, but customer response varies significantly. This project aims to:
- Identify distinct customer segments based on demographics and behavior
- Predict offer completion probability using machine learning
- Build a recommendation system to optimize offer targeting
- Measure causal impact of offers on transaction spend

**Goal:** Increase offer completion rates by >10% through personalized targeting.

---

##  Key Results

### Customer Segments (4 Groups Identified)

| Segment | Size | Best Offer | Completion Rate |
|---------|------|-------------|-----------------|
| Cluster 0: Unknown Demographics | 12.8% | Informational | 11.4% |
| Cluster 1: Discount Responders | 24.9% | Discount | 69.7% |
| Cluster 2: BOGO Responders | 28.5% | BOGO | 70.9% |
| Cluster 3: Low Engagement | 33.9% | Informational | 14.8% |

### Predictive Model Performance

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **AUC-ROC** | >0.70 | **0.909** |  Exceeds |
| **Precision** | >0.60 | **0.633** |  Meets |
| **Model** | - | **XGBoost** | - |

### Recommendation System Lift

- **Random Targeting:** 43.5% completion rate
- **Rule-Based Targeting:** 47.0% completion rate
- **Lift:** +7.9% (close to +10% target)

---

##  Tech Stack

- **Language:** Python 3.13+
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn, plotly
- **Package Management:** uv (fast Python package manager)
- **Reproducibility:** Random seed 42, version-pinned dependencies

---

##  Quick Start

### Prerequisites

- Python 3.13+
- uv (fast Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/starbucks-project.git
cd starbucks-project

# Install dependencies (uv will create venv and install automatically)
uv sync
```

### Run the Full Pipeline

```bash
# Option A: Run everything at once
uv run python main.py

# Option B: Run individual phases
uv run python src/data/load_data.py              # Phase 1: Data Ingestion
uv run python src/data/eda.py                    # Phase 2: EDA
uv run python src/data/feature_engineering.py    # Phase 3: Feature Engineering
uv run python src/models/clustering.py           # Phase 4: Clustering
uv run python src/models/predictive_modeling.py  # Phase 5: Predictive Modeling
uv run python src/models/recommendation.py       # Phase 6: Recommendation
uv run python src/reporting/generate_report.py   # Phase 7: Reports
uv run python src/reporting/export_powerbi.py    # Phase 8: Power BI Export
```

### Run Tests

```bash
uv run pytest tests/ -v
```

### View Results

- **Executive Summary:** `reports/executive_summary.md`
- **Technical Appendix:** `reports/technical_appendix.md`
- **Visualizations:** `reports/figures/` (includes interactive HTML plots)

---

##  Key Insights

1. **Customer Segmentation Works:** 4 distinct segments with clear offer preferences
2. **XGBoost Outperforms:** 0.909 AUC-ROC, exceeding industry standards
3. **Behavioral Features Matter Most:** Historical completion is the strongest predictor
4. **Causal Impact is Modest:** +2.0% spend lift from any offer (ATE)
5. **Rule-Based System is Effective:** +7.9% lift with simple, interpretable rules

---

##  Detailed Analysis

For full details, see:
- **Business Overview:** [Executive Summary](reports/executive_summary.md)
- **Technical Deep-Dive:** [Technical Appendix](reports/technical_appendix.md)
- **Interactive Visualizations:** [reports/figures/](reports/figures/)

---

##  Skills Demonstrated

| Skill Area | Techniques |
|------------|-------------|
| **Data Analysis** | pandas, data cleaning, missing data imputation |
| **EDA** | Distribution analysis, funnel metrics, correlation analysis |
| **Visualization** | matplotlib, seaborn, plotly (interactive) |
| **Feature Engineering** | One-hot encoding, behavioral features, interaction terms |
| **Unsupervised Learning** | K-Means clustering, PCA, silhouette analysis |
| **Supervised Learning** | Logistic Regression, Random Forest, XGBoost |
| **Model Evaluation** | AUC-ROC, Precision-Recall, cross-validation |
| **Causal Inference** | Average Treatment Effect (ATE) calculation |
| **Business Strategy** | Rule-based recommendation system, lift calculation |
| **Software Engineering** | Modular code, type hints, docstrings, reproducibility |

---

##  Limitations

- **Simulated Data:** Results may not generalize to real Starbucks data
- **Missing Demographics:** 12.8% of customers have missing gender/income
- **Short Test Period:** 29 days; longer studies needed for seasonality
- **Lift Target Missed:** +7.9% vs. +10% target (due to large low-engagement segment)

---

##  Future Work

- A/B test the recommendation system in production
- Incorporate deep learning for offer recommendation
- Add SHAP explainability for model transparency
- Explore advanced causal methods (propensity score matching)
- Integrate with Starbucks' real-time offer system

---

##  Contact

**Project by:** Nicholas Smith  
**Email:** nicholas122008@hotmail.com  
**LinkedIn:** [LinkedIn](https://www.linkedin.com/in/nicholas-smith-933125148/)  
**GitHub:** [Github](https://github.com/firepenguindisopanda/starbuck-analysis-project)

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Dataset:** Udacity Capstone Challenge (simulated Starbucks Rewards App data)

*Last updated: May 2026*
