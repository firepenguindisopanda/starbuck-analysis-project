# Starbucks Customer Segmentation & Offer Recommendation

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org)
[![Built with uv](https://img.shields.io/badge/Built%20with-uv-blueviolet)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data science project that optimizes Starbucks offer targeting using customer segmentation, 
predictive modeling, and causal inference. Built for my data science portfolio to demonstrate skills in 
EDA, feature engineering, clustering, predictive modeling, statistical testing, and causal analysis.

---

## Business Problem

Starbucks sends promotional offers to mobile app users, but customer response varies significantly. This project aims to:
- Identify distinct customer segments based on demographics and behavior
- Predict offer completion probability using machine learning
- Build a recommendation system to optimize offer targeting
- Measure causal impact of offers on transaction spend with statistical rigor

**Goal:** Increase offer completion rates by >10% through personalized targeting.

---

## Key Results

### Customer Segments (4 Groups Identified)

| Segment | Size | Best Offer | Completion Rate | 30-Day Spend | Annual CLV |
|---------|------|-------------|-----------------|--------------|-------------|
| Unengaged Unknowns | 12.8% | Informational | 11.4% | $18.53 | $222 |
| Discount Seekers | 24.9% | Discount | 69.7% | $152.50 | $1,830 |
| BOGO Advocates | 28.5% | BOGO | 70.9% | $180.80 | $2,170 |
| Passive Browsers | 33.9% | Informational | 14.8% | $37.45 | $449 |

### Predictive Model Performance

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **AUC-ROC** | >0.70 | **0.909** |  Exceeds |
| **Precision** | >0.60 | **0.633** |  Meets |
| **5-Fold CV AUC** | - | **0.908 +/- 0.003** |  Stable |
| **Brier Score** | - | **0.133** |  Well-calibrated |

### Causal Impact & Statistical Rigor

| Comparison | ATE | 95% CI | p-value | Cohen's d | Effect |
|-----------|-----|--------|---------|-----------|--------|
| Any Offer vs. None | +$0.25 | [$0.17, $0.32] | <0.001 | 0.018 | Negligible |
| BOGO vs. No BOGO | +$0.24 | [$0.15, $0.32] | <0.001 | 0.015 | Negligible |
| Discount vs. No Discount | -$0.25 | [-$0.34, -$0.16] | <0.001 | -0.019 | Negligible |

> All statistically significant effects have **negligible** practical effect sizes (Cohen's d < 0.02). 
> Business value lies in offer **completion rates** (up to 70%), not transaction amount lift.

### Recommendation System Lift

- Random Targeting: 43.5% completion rate
- **Rule-Based Targeting: 47.0%** completion rate
- **Lift: +7.9%** (close to +10% target)

### Business KPIs

| Metric | Value |
|--------|-------|
| Total Addressable Market | 17,000 customers |
| High-Value Segments | 53.4% of customers generate 85.6% of revenue |
| Estimated Annual Incremental Revenue | $51,000 |
| Discount Seekers Offer ROI | 15.7x |
| BOGO Advocates Offer ROI | 7.4x |

---

## Tech Stack

- **Language:** Python 3.13+
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Statistical Testing:** scipy (Welch's t-test, Cohen's d, bootstrap CIs)
- **Causal Inference:** Propensity score matching, ATE estimation
- **Visualization:** matplotlib, seaborn, plotly
- **Package Management:** uv (fast Python package manager)
- **Reproducibility:** Random seed 42, version-pinned dependencies

---

## Methodology

1. **Data Ingestion & Validation:** Loaded 3 JSON datasets, validated schemas, handled missing data (12.8%)
2. **EDA:** Analyzed demographics, offer performance, transaction behavior, funnel metrics
3. **Feature Engineering:** 43+ features (demographic, behavioral, RFM, time-decay, channel)
4. **Customer Segmentation:** K-Means (k=4), validated with silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic, and ARI stability analysis
5. **Predictive Modeling:** XGBoost (AUC-ROC 0.909), 5-fold CV, threshold optimization, Brier score calibration
6. **Causal Inference:** ATE with bootstrap CIs, Welch's t-test, Cohen's d, propensity score matching, heterogeneous treatment effects
7. **Recommendation:** Rule-based system (+7.9% lift), A/B test simulation framework

---

## Quick Start

### Prerequisites

- Python 3.13+
- uv (fast Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/firepenguindisopanda/starbuck-analysis-project.git
cd starbuck-analysis-project

# Install dependencies (uv will create venv and install automatically)
uv sync
```

### Run the Full Pipeline

```bash
# Phase 1: Data Ingestion & Validation
uv run python src/data/load_data.py

# Phase 2: Exploratory Data Analysis
uv run python src/data/eda.py

# Phase 3: Feature Engineering
uv run python src/data/feature_engineering.py

# Phase 4: Customer Segmentation (Clustering)
uv run python src/models/clustering.py

# Phase 5: Predictive Modeling
uv run python src/models/predictive_modeling.py

# Phase 6: Causal Inference & Recommendation
uv run python src/models/recommendation.py

# Phase 7: Generate Reports
uv run python src/reporting/generate_report.py
```

### View Results

- **Executive Summary:** `reports/executive_summary.md` (business-focused with KPIs)
- **Technical Appendix:** `reports/technical_appendix.md` (methodology with validation)
- **Visualizations:** `reports/figures/` (includes interactive HTML plots)

---

## Key Insights

1. **Customer Segmentation Works:** 4 distinct segments with clear offer preferences, validated by ARI stability (0.85)
2. **XGBoost Outperforms:** 0.909 AUC-ROC with stable 5-fold CV (0.908 +/- 0.003)
3. **Behavioral Features Matter Most:** Historical completion is the strongest predictor (35.5% importance)
4. **Causal Impact is Modest:** +2.0% spend lift from any offer (ATE), but negligible effect sizes (Cohen's d < 0.02)
5. **Rule-Based System is Effective:** +7.9% lift with simple, interpretable rules
6. **Statistical Rigor Matters:** All ATE results have negligible effect sizes despite p < 0.001, 
   demonstrating the importance of considering practical significance alongside statistical significance

---

## Detailed Analysis

For full details, see:
- **Business Overview:** [Executive Summary](reports/executive_summary.md)
- **Technical Deep-Dive:** [Technical Appendix](reports/technical_appendix.md)
- **Interactive Visualizations:** [reports/figures/](reports/figures/)

---

## Skills Demonstrated

| Skill Area | Techniques |
|------------|-------------|
| **Data Analysis** | pandas, data cleaning, missing data imputation, groupby aggregations |
| **EDA** | Distribution analysis, funnel metrics, correlation analysis |
| **Visualization** | matplotlib, seaborn, plotly (interactive) |
| **Feature Engineering** | One-hot encoding, behavioral features, interaction terms, RFM, time-decay |
| **Unsupervised Learning** | K-Means clustering, PCA, silhouette/davies-bouldin/gap/ARI validation |
| **Supervised Learning** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Model Validation** | 5-fold CV, Brier score, threshold optimization, per-class metrics |
| **Statistical Testing** | Welch's t-test, Cohen's d, bootstrap CIs, Mann-Whitney U, Chi-squared |
| **Causal Inference** | ATE, propensity score matching, heterogeneous treatment effects, A/B simulation |
| **Business Strategy** | ROI analysis, CLV proxy, recommendation systems, lift calculation |
| **Software Engineering** | Modular code, type hints, docstrings, reproducibility (seed=42) |

---

## Limitations & Future Work

### Limitations

- **Simulated Data:** Results may not generalize to real Starbucks data
- **Missing Demographics:** 12.8% of customers have missing gender/income
- **Short Test Period:** 29 days; longer studies needed for seasonality
- **Lift Target:** +7.9% vs. +10% target (due to large low-engagement segment)
- **Low Silhouette Score:** k=4 yields 0.147 (moderate separation); k=3 yields 0.170
- **Selection Bias:** ATE estimates may overstate causal impact; PSM partially addresses this
- **Negligible Effect Sizes:** All Cohen's d < 0.02; business value is in completion rates, not spend lift

### Future Work

- A/B test the recommendation system in production
- Incorporate deep learning for offer recommendation
- Add SHAP explainability for model transparency (already in pipeline)
- Explore advanced causal methods (instrumental variables, difference-in-differences)
- Integrate with Starbucks' real-time offer system
- Add seasonal and lifecycle effects with longitudinal data
- Implement multi-armed bandit for online offer optimization

---

## Contact

**Project by:** Nicholas Smith  
**Email:** nicholas122008@hotmail.com  
**LinkedIn:** [linkedin.com/in/nicholas-smith-933125148](https://www.linkedin.com/in/nicholas-smith-933125148/)  
**GitHub:** [github.com/firepenguindisopanda/starbuck-analysis-project](https://github.com/firepenguindisopanda/starbuck-analysis-project)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Dataset:** Udacity Capstone Challenge (simulated Starbucks Rewards App data)

*Last updated: May 2026*
