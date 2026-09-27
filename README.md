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

### Predictive Model Performance (send-time model)

The question: when a BOGO or discount offer is sent, will the customer complete it
within its validity window? One row per offer actually sent (50,806 after dropping
10,236 sends whose window runs past the end of the data), features known at the
moment of sending, customers kept apart in every split
(`src/models/offer_response.py`, report in `reports/offer_response_report.json`).

| Metric | Value |
|--------|-------|
| **AUC, held-out customers** | **0.865** (95% CI 0.857 to 0.873, customer bootstrap) |
| **5-fold CV AUC, grouped by customer** | 0.865 +/- 0.005 |
| **AUC forward in time** (train on waves 0 to 336 h, test on later waves) | 0.878 |
| **Precision / recall / F1 at 0.5** | 0.797 / 0.822 / 0.809 |
| **Brier score** | 0.148, with a calibration curve close to the diagonal |
| Baselines: offer terms only / offer terms and demographics | 0.605 / 0.825 |

**Why this replaces the earlier 0.909 and 0.994 figures.** The earlier model
(removed; it is in the git history) crossed every customer with all ten offers, most never
received, scored "ever completed" with no window, used whether the offer was viewed
(after the send) and month-long completion counts (including the offer being scored),
and split rows at random. The leakage ladder takes those out one at a time:

| Step | AUC |
|------|-----|
| Original pipeline | 0.994 |
| One row per real send, month-long counts and viewed flag kept | 0.934 |
| Viewed flag removed | 0.931 |
| Month-long counts replaced by history before the send | 0.866 |
| Customers kept apart across folds | **0.865** |

`tests/test_offer_response.py` checks the window, repeat sends and censoring, and
fails if any feature changes when every event after a send is deleted.

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
5. **Predictive Modeling:** send-time offer response model, 0.865 AUC on held-out customers, customer-grouped CV, calibration, and a leakage ladder against the earlier framing
6. **Causal Inference:** ATE with bootstrap CIs, Welch's t-test, Cohen's d, propensity score matching, heterogeneous treatment effects
7. **Recommendation:** Rule-based system (+7.9% lift), A/B test simulation framework

---

## Data

The data is the simulated Starbucks Rewards app dataset that Starbucks made
available to Udacity's Data Scientist Nanodegree capstone. It is **not included in
this repository**: it is not mine to redistribute. To reproduce the results:

1. Download the three files from either source:
   - Kaggle mirror: [Starbucks app customer rewards program data](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data)
     (free Kaggle account; the page notes the data was provided to Udacity scholars)
   - Udacity: the capstone of the [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
2. Put them in `data/raw/`:

   ```
   data/raw/portfolio.json    10 offers
   data/raw/profile.json      17,000 customers
   data/raw/transcript.json   306,534 events
   ```

3. Optionally check you have the same files (SHA-256):

   | File | SHA-256 |
   |------|---------|
   | `portfolio.json` | `928a02da21961a9dd464a3000ef84af147efbe0101a7547425c6abe795ab9309` |
   | `profile.json` | `a23145d0b912c3d6a4c0f0d47d3e643ea3575a1014ebd3ad5af220c3cd181c76` |
   | `transcript.json` | `a975793c58e206168af7624497c8879c45d84968249947ddc4f2c38ef86a7006` |

`data/raw/` is git-ignored. Without the data, the unit tests that use synthetic
events still run and the ones that need the real files are skipped.

---

## Quick Start

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/firepenguindisopanda/starbuck-analysis-project.git
cd starbuck-analysis-project
uv sync                      # creates the virtual environment
# put the three data files in data/raw/ (see Data above)
uv run python main.py        # runs every phase in order, about 4 minutes
uv run pytest                # 47 tests
```

Each phase can also run on its own:

| Phase | Script | Output |
|-------|--------|--------|
| 1. Data validation | `src/data/load_data.py` | `reports/data_quality_report.json` |
| 2. Exploratory analysis | `src/data/eda.py` | `reports/eda_summary.json`, figures |
| 3. Feature engineering | `src/data/feature_engineering.py` | `data/processed/` |
| 4. Customer segmentation | `src/models/clustering.py` | `reports/clustering_report.json` |
| 5. Causal inference and recommendation | `src/models/recommendation.py` | `reports/causal_report.json` |
| 6. Send-time offer response model | `src/models/offer_response.py` | `reports/offer_response_report.json` |

Figures go to `reports/figures/` and intermediate tables to `data/processed/`; both
are regenerated by the pipeline and not committed.

---

## Repository Structure

```
starbuck-analysis-project/
|-- main.py                     runs the pipeline
|-- data/
|   |-- raw/                    the three Udacity files (not committed)
|   `-- processed/              pipeline output (not committed)
|-- notebooks/
|   `-- 01_eda_and_storytelling.ipynb
|-- reports/                    JSON results of each phase
|   `-- portfolio/              the same results in the shape my portfolio site reads
|-- src/
|   |-- data/                   loading and validation, EDA, feature engineering
|   `-- models/                 clustering, causal inference and recommendation, offer response model
`-- tests/                      unit tests (pytest)
```

---

## Key Insights

1. **Customer Segmentation Works:** 4 distinct segments with clear offer preferences, moderately stable (mean ARI 0.76 across resamples)
2. **An honest model beats an impressive one:** rebuilt at send time, 0.865 AUC on unseen customers (0.878 forward in time), where the leaky framing claimed 0.994
3. **Past spending matters most:** average spend before the send, tenure and income lead the permutation importance; offer terms alone reach 0.605
4. **Causal Impact is Modest:** +2.0% spend lift from any offer (ATE), but negligible effect sizes (Cohen's d < 0.02)
5. **Rule-Based System is Effective:** +7.9% lift with simple, interpretable rules
6. **Statistical Rigor Matters:** All ATE results have negligible effect sizes despite p < 0.001, 
   demonstrating the importance of considering practical significance alongside statistical significance

---

## Skills Demonstrated

| Skill Area | Techniques |
|------------|-------------|
| **Data Analysis** | pandas, data cleaning, missing data imputation, groupby aggregations |
| **EDA** | Distribution analysis, funnel metrics, correlation analysis |
| **Visualization** | matplotlib, seaborn, plotly (interactive) |
| **Feature Engineering** | One-hot encoding, behavioral features, interaction terms, RFM, time-decay |
| **Unsupervised Learning** | K-Means clustering, PCA, silhouette/davies-bouldin/gap/ARI validation |
| **Supervised Learning** | Logistic regression, histogram gradient boosting, XGBoost |
| **Model Validation** | Leakage audit, customer-grouped CV, held-out customers, forward-in-time check, calibration, permutation importance |
| **Statistical Testing** | Welch's t-test, Cohen's d, bootstrap CIs, Mann-Whitney U, Chi-squared |
| **Causal Inference** | ATE, propensity score matching, heterogeneous treatment effects, A/B simulation |
| **Business Strategy** | ROI analysis, CLV proxy, recommendation systems, lift calculation |
| **Software Engineering** | Modular pipeline, unit tests including a leak guard, type hints, reproducibility (seed=42) |

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
- Use the send-time model's probabilities to choose which offer to send, and compare that with the rule-based system
- Incorporate deep learning for offer recommendation
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

**Dataset:** simulated Starbucks Rewards app data from the Udacity Data Scientist Nanodegree capstone (not redistributed here; see [Data](#data))

*Last updated: September 2026*
