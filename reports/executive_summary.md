# Starbucks Customer Segmentation - Executive Summary

**Project Date:** May 2026  
**Dataset:** Simulated Starbucks Rewards App Data (Udacity Capstone)  
**Data Size:** 17,000 customers, 10 offers, 306,534 events

---

## Business Problem

Starbucks sends promotional offers to mobile app users, but not all customers respond the same way. 
The goal is to **optimize offer targeting** by identifying customer segments and predicting offer completion, 
ultimately increasing campaign ROI and customer satisfaction.

**Key Business Questions:**
1. How do customer offer response rates vary across demographics?
2. Which offer types drive the highest engagement for different customer groups?
3. Can we identify distinct customer segments for targeted campaigns?
4. What is the incremental revenue lift from personalized vs. generic campaigns?

---

## Key Findings

### 1. Customer Segments Identified (4 Distinct Groups)

We identified **4 customer segments** using K-Means clustering:

| Segment | Size | Characteristics | Best Offer Type | Completion Rate |
|----------|------|------------------|------------------|-----------------|
| **Cluster 0** | 12.8% | Unknown demographics (missing data) | Informational | 11.4% |
| **Cluster 1** | 24.9% | High discount responders, $68K income | Discount | 69.7% |
| **Cluster 2** | 28.5% | High BOGO responders, $72K income | BOGO | 70.9% |
| **Cluster 3** | 33.9% | Low engagement, younger, $57K income | Informational | 14.8% |

### 2. Offer Performance Insights

- **BOGO offers:** 61.8% completion rate, strongest for Cluster 2
- **Discount offers:** 68.8% completion rate, strongest for Cluster 1
- **Informational offers:** No direct completion (awareness only), 71% view rate

**Funnel Metrics:**
- 75.7% of offers are viewed after receipt
- 44.0% of offers are completed
- Average time-to-view: 24.9 hours

### 3. Predictive Model Performance

We built an **XGBoost classifier** to predict offer completion probability:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| AUC-ROC | >0.70 | **0.909** |  Exceeds target |
| Precision | >0.60 | **0.633** |  Meets target |
| Recall | - | 0.536 | - |
| F1-Score | - | 0.581 | - |

**Top Predictive Features:**
1. Offers completed (historical)
2. Offer reward amount
3. Offer type (BOGO)
4. BOGO completion history
5. Discount completion history

### 4. Causal Impact & Revenue Lift

**Average Treatment Effect (ATE) of Offers on Transaction Spend:**
- Any offer vs. no offer: **+2.0% lift** (+$0.25 per transaction)
- BOGO offers: +1.9% lift
- Discount offers: -1.9% lift (customers may spend less on discounted items)
- Informational offers: 0% direct impact (awareness building)

### 5. Recommendation System Performance

We built a **rule-based recommendation system** using customer segments:

| Targeting Method | Completion Rate | Lift vs. Random |
|-----------------|-----------------|-------------------|
| Random Targeting (baseline) | 43.5% | - |
| **Rule-Based Targeting** | **47.0%** | **+7.9%** |

**Recommendation Rules:**
- Cluster 0 & 3 → Send informational offers (low engagement)
- Cluster 1 → Send discount offers (69.7% completion rate)
- Cluster 2 → Send BOGO offers (70.9% completion rate)

---

## Business Impact & ROI

If Starbucks implements this recommendation system:
- **+7.9% increase** in offer completion rates
- **More efficient ad spend** by avoiding low-response segments
- **Improved customer experience** by sending relevant offers
- **Scalable framework** for future offer optimization

---

## Methodology (High-Level)

1. **Data Ingestion & Validation:** Loaded 3 JSON datasets, validated schemas, handled missing data (12.8% missing gender/income)
2. **Exploratory Data Analysis:** Analyzed demographics, offer performance, transaction behavior
3. **Feature Engineering:** Created 43 customer features (demographic + behavioral)
4. **Customer Segmentation:** Applied K-Means clustering (k=4)
5. **Predictive Modeling:** Trained XGBoost classifier (AUC-ROC: 0.909)
6. **Causal Inference:** Calculated ATE using treatment vs. control groups
7. **Recommendation System:** Built rule-based system with +7.9% lift

---

## Limitations & Future Work

- **Simulated Data:** Results may not generalize to real Starbucks data
- **Informational Offers:** No completion events; measured via view rates and subsequent transactions
- **Time Constraints:** 29-day test period; longer studies needed for seasonal effects
- **Advanced Causal Methods:** Future work could use propensity score matching or instrumental variables

**Potential Improvements:**
- A/B test the recommendation system in production
- Incorporate more granular offer features (channel interactions)
- Explore deep learning models for offer recommendation
- Add SHAP explainability for model transparency

---

## Technical Skills Demonstrated

| Skill Area | Techniques Used |
|------------|-----------------|
| **Data Analysis (Pandas)** | Data cleaning, missing data imputation, groupby aggregations |
| **Exploratory Data Analysis** | Distribution analysis, funnel metrics, correlation analysis |
| **Data Visualization** | Matplotlib, Seaborn, Plotly (interactive) |
| **Feature Engineering** | One-hot encoding, behavioral features, interaction terms |
| **Unsupervised Learning** | K-Means clustering, PCA, silhouette analysis |
| **Supervised Learning** | Logistic Regression, Random Forest, XGBoost |
| **Model Evaluation** | AUC-ROC, Precision-Recall, cross-validation |
| **Causal Inference** | Average Treatment Effect (ATE) calculation |
| **Business Strategy** | Rule-based recommendation system, lift calculation |
| **Software Engineering** | Modular code, type hints, docstrings, reproducibility (seed=42) |

---

## Repository Structure

```
starbucks-project/
├-- data/processed/           # Engineered features and model artifacts
├-- reports/
│   ├-- figures/              # Publication-quality visualizations
│   ├-- *.json                # Detailed reports from each phase
│   └-- executive_summary.md  # This file
├-- src/
│   ├-- data/                 # Data loading and EDA modules
│   ├-- models/               # Clustering, predictive modeling, recommendation
│   └-- reporting/            # Report generation
├-- notebooks/                # Jupyter notebooks (optional)
├-- pyproject.toml           # Project dependencies (uv)
└-- README.md                 # Project overview and instructions
```

---

**For detailed technical methodology, see [Technical Appendix](technical_appendix.md).**

**Project completed by:** Nicholas Smith  
**Contact:** nicholas122008@hotmail.com  
**GitHub:** [github.com/yourusername/starbucks-project]
