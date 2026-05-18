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
- Cluster 0 & 3 -> Send informational offers (low engagement)
- Cluster 1 -> Send discount offers (69.7% completion rate)
- Cluster 2 -> Send BOGO offers (70.9% completion rate)

---

## Business KPIs

### Total Addressable Market & Revenue

| Metric | Value |
|--------|-------|
| **Total Addressable Market** | 17,000 customers |
| **Total 30-Day Revenue** | $1,775,409 |
| **Avg. Revenue/Customer (30-Day)** | $104.44 |
| **Estimated Annual Revenue** | $21,304,907 |

### Revenue per Customer by Segment

| Segment | 30-Day Spend | Annual Spend (Est.) | Offers Completed (30-Day) | Offer Type | Reward Cost/Customer | **Offer ROI** |
|---------|-------------|---------------------|--------------------------|------------|---------------------|-------------|
| Unengaged Unknowns (C0) | $18.53 | $222 | 0.5 | Informational | $0.00 | N/A (no direct cost) |
| Discount Seekers (C1) | $152.50 | $1,830 | 3.2 | Discount | $9.74 | **15.7x** |
| BOGO Advocates (C2) | $180.80 | $2,170 | 3.3 | BOGO | $24.51 | **7.4x** |
| Passive Browsers (C3) | $37.45 | $449 | 0.6 | Informational | $0.00 | N/A (no direct cost) |

> **Offer ROI** = 30-Day Revenue / Reward Cost. "N/A" segments receive informational offers ($0 reward cost), 
> so traditional ROI is undefined; value manifests as brand awareness and long-term engagement.

### Customer Lifetime Value (CLV) Proxy

Using 30-day spend as a proxy (extrapolated annually with a conservative discount factor):

| Segment | CLV Proxy (12-Month) | Share of Revenue |
|---------|---------------------|-----------------|
| Unengaged Unknowns | $222 | 2.3% |
| Discount Seekers | $1,830 | 36.3% |
| BOGO Advocates | $2,170 | 49.3% |
| Passive Browsers | $449 | 12.2% |

> **53.3%** of customers (Discount Seekers + BOGO Advocates) generate 
> **85.6%** of total revenue.

### Estimated Annual Impact

| Scenario | Method | 30-Day Incremental Revenue | Annual Incremental Revenue |
|----------|--------|---------------------------|---------------------------|
| Current (Random) | Baseline | - | - |
| **Optimized (Rule-Based)** | Segment targeting | $4250 | $51,000 |
| **Best Case (Full Personalization)** | ML model + segments | $6375 | $76,500 |

> Estimates extrapolated from 30-day data with ATE of +$0.25/transaction. 
> "Best Case" assumes 50% improvement over simple rule-based targeting with ML model deployment.

---

## Statistical Confidence

### Hypothesis Test Results

Finding statistical significance does not imply practical significance. All effect sizes 
are **negligible** (Cohen's d < 0.02), meaning offers produce measurable but tiny differences 
in transaction amounts.

| Comparison | Test | p-value | Cohen's d | Effect Size | 95% CI (ATE) | Significant? |
|-----------|------|---------|-----------|-------------|---------------|-------------|
| Any Offer vs. None | Welch t-test | <0.001 | 0.018 | Negligible | [$0.17, $0.32] | Yes |
| BOGO vs. No BOGO | Welch t-test | <0.001 | 0.015 | Negligible | [$0.15, $0.32] | Yes |
| Discount vs. No Discount | Welch t-test | <0.001 | -0.019 | Negligible | [-$0.34, -$0.16] | Yes |
| Informational vs. None | Welch t-test | 0.976 | 0.0001 | Negligible | [-$0.08, $0.08] | No |

> **Interpretation:** All statistically significant results (p < 0.05) have negligible effect sizes. 
> This is common with large samples (N > 100K transactions): even tiny differences become statistically significant. 
> The real business impact lies in offer **completion rates** (up to 70%), not transaction amount lift.

### Confidence Intervals (Bootstrap, 1,000 iterations)

- ATE for any offer: **+$0.25** per transaction, 95% CI: [$0.17, $0.32]
- ATE for BOGO offers: **+$0.24** per transaction, 95% CI: [$0.15, $0.32]
- ATE for discount offers: **-$0.25** per transaction, 95% CI: [-$0.34, -$0.16]
- ATE for informational offers: **$0.00** per transaction, 95% CI: [-$0.08, $0.08]

### Model Prediction Confidence

| Metric | Value | Interpretation |
|--------|-------|---------------|
| AUC-ROC (5-fold CV) | 0.908 +/- 0.003 | Stable across folds; no overfitting |
| Brier Score | 0.133 | Well-calibrated probabilities |
| Optimal Threshold | 0.42 | Maximizes F1 (from default 0.50) |
| F1 at Optimal Threshold | 0.612 | +5.3% improvement over default threshold |

---

## Segment Deep Dive

### Segment 0: "Unengaged Unknowns" (12.8% - 2,170 customers)

**Profile:** Demographically opaque (100% missing gender/income), low engagement across all offer types.

| Metric | Value |
|--------|-------|
| Average 30-Day Spend | $18.53 |
| Completion Rate | 11.4% |
| BOGO Completion Rate | 9.3% |
| Discount Completion Rate | 15.8% |
| View Rate | 80.6% |
| Estimated CLV (12-month) | $222 |

**Actionable Recommendations:**
- **Primary Strategy:** Send informational offers only (lowest cost, awareness focus)
- **Re-engagement:** Deploy push notification campaigns to boost view-to-completion rates
- **Data Collection:** Offer profile-completion incentives to reduce missing demographics
- **Expected ROI:** Minimal direct ROI; value is long-term engagement and data collection
- **Risk:** High churn potential; requires A/B testing of re-engagement strategies

### Segment 1: "Discount Seekers" (24.9% - 4,228 customers)

**Profile:** Price-sensitive, high discount completion (89.2%), moderate income ($68K), active transactors.

| Metric | Value |
|--------|-------|
| Average 30-Day Spend | $152.50 |
| Completion Rate | 69.7% |
| Discount Completion Rate | 89.2% |
| BOGO Completion Rate | 56.0% |
| View Rate | 78.1% |
| Estimated CLV (12-month) | $1,830 |

**Actionable Recommendations:**
- **Primary Strategy:** Prioritize discount offers (89.2% completion rate)
- **Upsell:** Gradually introduce BOGO offers to shift spending upward (56% BOGO completion)
- **Offer Frequency:** 4-5 offers per 30-day cycle (current cadence is optimal)
- **Expected ROI:** **15.7x** (highest reward-to-spend ratio among discount-targeted segments)
- **Risk:** Over-discounting may erode perceived value; monitor for offer fatigue

### Segment 2: "BOGO Advocates" (28.5% - 4,837 customers)

**Profile:** Highest spenders ($181/mo), strong BOGO completion (87.1%), higher income ($72K), majority female (54%).

| Metric | Value |
|--------|-------|
| Average 30-Day Spend | $180.80 |
| Completion Rate | 70.9% |
| BOGO Completion Rate | 87.1% |
| Discount Completion Rate | 72.5% |
| View Rate | 82.4% |
| Estimated CLV (12-month) | $2,170 |

**Actionable Recommendations:**
- **Primary Strategy:** Send BOGO offers (87.1% completion - highest in dataset)
- **Cross-Sell:** Introduce premium product offers; this segment has highest income
- **Loyalty Programs:** Enroll in VIP/rewards tiers; they are your brand advocates
- **Expected ROI:** **7.4x** (strong return, though higher reward cost per offer)
- **Risk:** Ensure BOGO offers feel exclusive; avoid devaluing through overuse

### Segment 3: "Passive Browsers" (33.9% - 5,765 customers)

**Profile:** Younger (51.3), lower income ($57K), 71% male, lowest engagement (14.8% completion), lowest spend ($37/mo).

| Metric | Value |
|--------|-------|
| Average 30-Day Spend | $37.45 |
| Completion Rate | 14.8% |
| Discount Completion Rate | 19.5% |
| BOGO Completion Rate | 15.6% |
| View Rate | 67.3% |
| Estimated CLV (12-month) | $449 |

**Actionable Recommendations:**
- **Primary Strategy:** Minimal offer frequency (informational only); avoid offer fatigue
- **Channel:** Focus on mobile push notifications (67% view rate has room for improvement)
- **Simplification:** Use single-step completion offers (not multi-step); reduce friction
- **Expected ROI:** Low direct ROI; focus on retention and gradual engagement uplift
- **Risk:** Spam risk; excessive offers will drive disengagement or unsubscribes

---

## Risk & Limitations

### Data & Methodology Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Simulated Data** | Results may not generalize to real Starbucks customer behavior | Findings should be validated with production A/B tests before scaling |
| **Selection Bias in ATE** | Offer assignment may not be random; confounders (income, engagement) influence treatment | Propensity score matching (PSM) partially adjusts but cannot eliminate unobserved confounding |
| **Low Silhouette Score** | k=4 silhouette = 0.147 (moderate separation); k=3 yields 0.170 | Business interpretability justified k=4; stability analysis (ARI = 0.85) confirms reliability |
| **Negligible Effect Sizes** | All Cohen's d < 0.02 for ATE comparisons | Statistical significance is driven by large N; business impact should focus on completion rates, not transaction lift |
| **Informational Offer Attribution** | No completion event for informational offers; impact measured indirectly | Analysis treats informational as awareness-building; future work should track downstream conversion |
| **30-Day Window** | Single 29-day test period; no seasonality, lifecycle, or long-term effects | Annual extrapolations are pro-rated from 30-day data and should be treated as rough estimates |

### Business Assumptions

1. **CLV Proxy:** Annual spend is extrapolated from 30-day data without discount rate or churn adjustment
2. **Offer ROI:** Calculated as revenue / reward cost; does not include operational costs (marketing, distribution)
3. **Reward Cost Estimates:** BOGO avg. $7.50/offer, Discount avg. $3.00/offer based on portfolio data
4. **Recommendation Lift:** +7.9% improvement from historical simulation; requires live A/B test validation

### Statistical Caveats

- **p-Values with Large N:** With >100K transactions, even trivial differences reach p < 0.001. 
  Always consider effect sizes alongside statistical significance.
- **Multiple Testing:** Several simultaneous comparisons increase false positive risk; 
  no Bonferroni correction applied for exploratory analyses.
- **Bootstrap CIs:** Based on 1,000 resamples; may underestimate uncertainty for heavy-tailed distributions.

---

## Business Impact & ROI

If Starbucks implements this recommendation system:
- **+7.9% increase** in offer completion rates
- **85.6% of revenue** concentrated in 2 of 4 segments
- Targeted spend reduces waste on unresponsive segments (46.7% of customers)
- **Estimated annual incremental revenue:** $51,000 (conservative, from ATE)
- **Scalable framework** for future offer optimization and real-time personalization

---

## Methodology (High-Level)

1. **Data Ingestion & Validation:** Loaded 3 JSON datasets, validated schemas, handled missing data (12.8% missing gender/income)
2. **Exploratory Data Analysis:** Analyzed demographics, offer performance, transaction behavior
3. **Feature Engineering:** Created 43+ customer features (demographic + behavioral + interaction)
4. **Customer Segmentation:** K-Means clustering (k=4), validated with silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic, and stability analysis (ARI)
5. **Predictive Modeling:** XGBoost classifier with 5-fold CV (AUC-ROC: 0.909), threshold optimization, calibration (Brier score)
6. **Causal Inference:** ATE with bootstrap CIs, Welch's t-tests, Cohen's d, propensity score matching, heterogeneous treatment effects
7. **Recommendation System:** Rule-based with +7.9% lift, validated with A/B test simulation framework

---

## Technical Skills Demonstrated

| Skill Area | Techniques Used |
|------------|-----------------|
| **Data Analysis (Pandas)** | Data cleaning, missing data imputation, groupby aggregations |
| **Exploratory Data Analysis** | Distribution analysis, funnel metrics, correlation analysis |
| **Data Visualization** | Matplotlib, Seaborn, Plotly (interactive) |
| **Feature Engineering** | One-hot encoding, behavioral features, interaction terms, RFM metrics |
| **Unsupervised Learning** | K-Means clustering, PCA, silhouette analysis, gap statistic, stability analysis (ARI) |
| **Supervised Learning** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Model Evaluation** | AUC-ROC, Precision-Recall, 5-fold CV, Brier score, threshold optimization |
| **Statistical Testing** | Welch's t-test, Cohen's d, bootstrap CIs, Mann-Whitney U, Chi-squared |
| **Causal Inference** | ATE with bootstrap CIs, propensity score matching, heterogeneous treatment effects |
| **Business Strategy** | Rule-based recommendation, lift calculation, A/B test simulation, ROI analysis |
| **Software Engineering** | Modular code, type hints, docstrings, reproducibility (seed=42) |

---


**For detailed technical methodology, see [Technical Appendix](technical_appendix.md).**

**Project completed by:** [Your Name]  
**Contact:** [Your Email/LinkedIn]  
**GitHub:** [Repository Link]
