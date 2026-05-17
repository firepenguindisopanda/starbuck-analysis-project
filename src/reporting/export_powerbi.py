"""
Power BI Export Module for Starbucks Customer Segmentation Project.

This module exports all relevant data to CSV files optimized for Power BI import.
Power BI can directly consume CSV files and create interactive dashboards.

Why Power BI?
- Demonstrates business intelligence skills (not just Python)
- Creates executive-ready dashboards
- Shows ability to work with enterprise BI tools
- Complements Python/ML work with visual analytics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def export_customer_data(base_path: str = '.') -> pd.DataFrame:
    """
    Export customer master data with cluster assignments.
    
    Returns:
        Customer DataFrame ready for Power BI
    """
    print("\n" + "="*60)
    print("EXPORTING CUSTOMER DATA FOR POWER BI")
    print("="*60)
    
    # Load customer features
    customers = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_features.csv')
    
    # Load cluster assignments
    clusters = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'customer_clusters.csv')
    
    # Merge
    customers_export = customers.merge(clusters, on='id', how='left')
    
    # Add cluster labels for readability
    cluster_labels = {
        0: 'Unknown Demographics',
        1: 'Discount Responders',
        2: 'BOGO Responders',
        3: 'Low Engagement'
    }
    customers_export['cluster_label'] = customers_export['cluster'].map(cluster_labels)
    
    # Select and rename columns for Power BI
    export_cols = {
        'id': 'customer_id',
        'age_imputed': 'age',
        'income_imputed': 'income',
        'tenure_months': 'tenure_months',
        'gender_M': 'is_male',
        'gender_F': 'is_female',
        'gender_O': 'is_other_gender',
        'gender_Unknown': 'is_unknown_gender',
        'trans_count': 'transaction_count',
        'trans_total': 'total_spend',
        'trans_avg': 'avg_transaction',
        'offers_received': 'offers_received',
        'offers_viewed': 'offers_viewed',
        'offers_completed': 'offers_completed',
        'view_rate': 'view_rate',
        'completion_rate': 'completion_rate',
        'cluster': 'cluster_id',
        'cluster_label': 'cluster_label'
    }
    
    # Only keep columns that exist
    existing_cols = {k: v for k, v in export_cols.items() if k in customers_export.columns}
    customers_export = customers_export[list(existing_cols.keys())].rename(columns=existing_cols)
    
    # Add calculated columns for Power BI
    customers_export['customer_segment'] = customers_export['cluster_label']
    customers_export['engagement_level'] = pd.cut(
        customers_export['completion_rate'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    print(f" Exported {len(customers_export)} customers with {len(customers_export.columns)} columns")
    
    return customers_export


def export_offer_data(base_path: str = '.') -> pd.DataFrame:
    """
    Export offer master data.
    
    Returns:
        Offer DataFrame ready for Power BI
    """
    print("\nExporting offer data...")
    
    # Load portfolio
    portfolio = pd.read_json(Path(base_path) / 'portfolio.json', lines=True)
    
    # Explode channels into separate rows for Power BI (creates one row per offer-channel)
    portfolio_expanded = portfolio.explode('channels')
    
    # Rename columns
    portfolio_expanded = portfolio_expanded.rename(columns={
        'id': 'offer_id',
        'offer_type': 'offer_type',
        'difficulty': 'difficulty',
        'reward': 'reward',
        'duration': 'duration_days',
        'channels': 'channel'
    })
    
    # Add derived columns
    portfolio_expanded['offer_type_upper'] = portfolio_expanded['offer_type'].str.upper()
    portfolio_expanded['reward_per_day'] = portfolio_expanded['reward'] / portfolio_expanded['duration_days']
    portfolio_expanded['difficulty_per_day'] = portfolio_expanded['difficulty'] / portfolio_expanded['duration_days']
    
    print(f" Exported {len(portfolio_expanded)} offer-channel combinations")
    
    return portfolio_expanded


def export_transaction_data(base_path: str = '.') -> pd.DataFrame:
    """
    Export transaction data for Power BI.
    
    Returns:
        Transaction DataFrame ready for Power BI
    """
    print("\nExporting transaction data...")
    
    # Load transcript
    transcript = pd.read_json(Path(base_path) / 'transcript.json', lines=True)
    
    # Filter to transactions only
    transactions = transcript[transcript['event'] == 'transaction'].copy()
    
    # Extract amount
    transactions['amount'] = transactions['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transactions = transactions[transactions['amount'].notna()].copy()
    
    # Rename columns
    transactions = transactions.rename(columns={
        'person': 'customer_id',
        'time': 'time_hours',
        'amount': 'transaction_amount'
    })
    
    # Add derived columns
    transactions['transaction_date'] = pd.to_datetime('2018-07-26') - pd.to_timedelta(transactions['time_hours'], unit='h')
    transactions['day_of_week'] = transactions['transaction_date'].dt.day_name()
    transactions['hour_of_day'] = transactions['transaction_date'].dt.hour
    
    # Select columns
    export = transactions[['customer_id', 'transaction_amount', 'time_hours', 
                           'transaction_date', 'day_of_week', 'hour_of_day']].copy()
    
    print(f" Exported {len(export)} transactions")
    
    return export


def export_offer_events(base_path: str = '.') -> pd.DataFrame:
    """
    Export offer events (received, viewed, completed) for Power BI.
    
    Returns:
        Offer events DataFrame ready for Power BI
    """
    print("\nExporting offer events...")
    
    # Load transcript
    transcript = pd.read_json(Path(base_path) / 'transcript.json', lines=True)
    
    # Filter to offer events only
    offer_events = transcript[transcript['event'].isin(['offer received', 'offer viewed', 'offer completed'])].copy()
    
    # Extract offer_id
    offer_events['offer_id'] = offer_events['value'].apply(
        lambda x: x.get('offer id') or x.get('offer_id') if isinstance(x, dict) else None
    )
    offer_events = offer_events[offer_events['offer_id'].notna()].copy()
    
    # Rename columns
    offer_events = offer_events.rename(columns={
        'person': 'customer_id',
        'event': 'event_type',
        'time': 'time_hours'
    })
    
    # Add derived columns
    offer_events['event_date'] = pd.to_datetime('2018-07-26') - pd.to_timedelta(offer_events['time_hours'], unit='h')
    offer_events['day_of_week'] = offer_events['event_date'].dt.day_name()
    
    # Select columns
    export = offer_events[['customer_id', 'offer_id', 'event_type', 'time_hours', 
                           'event_date', 'day_of_week']].copy()
    
    print(f" Exported {len(export)} offer events")
    
    return export


def export_cluster_profiles(base_path: str = '.') -> pd.DataFrame:
    """
    Export cluster profiles for Power BI.
    
    Returns:
        Cluster profiles DataFrame
    """
    print("\nExporting cluster profiles...")
    
    # Load cluster profiles
    profiles = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'cluster_profiles.csv')
    
    # Add cluster labels
    cluster_labels = {
        0: 'Unknown Demographics',
        1: 'Discount Responders',
        2: 'BOGO Responders',
        3: 'Low Engagement'
    }
    profiles['cluster_label'] = profiles['cluster_id'].map(cluster_labels)
    
    # Reshape for Power BI (long format)
    metrics = ['completion_rate_mean', 'view_rate_mean', 'trans_avg_mean', 
               'income_imputed_mean', 'age_imputed_mean']
    
    profiles_long = []
    for _, row in profiles.iterrows():
        for metric in metrics:
            if metric in row.index:
                profiles_long.append({
                    'cluster_id': int(row['cluster_id']),
                    'cluster_label': row['cluster_label'],
                    'metric': metric.replace('_mean', '').replace('_', ' ').title(),
                    'value': row[metric]
                })
    
    profiles_export = pd.DataFrame(profiles_long)
    
    print(f" Exported {len(profiles_export)} cluster profile metrics")
    
    return profiles_export


def export_model_performance(base_path: str = '.') -> pd.DataFrame:
    """
    Export model performance metrics for Power BI.
    
    Returns:
        Model performance DataFrame
    """
    print("\nExporting model performance metrics...")
    
    # Load model comparison
    comparison = pd.read_csv(Path(base_path) / 'data' / 'processed' / 'model_comparison.csv')
    
    # Reshape for Power BI (long format)
    metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    
    performance_long = []
    for _, row in comparison.iterrows():
        for metric in metrics:
            if metric in row.index:
                performance_long.append({
                    'model_name': row['Model'],
                    'metric': metric,
                    'value': row[metric]
                })
    
    performance_export = pd.DataFrame(performance_long)
    
    print(f" Exported {len(performance_export)} model performance metrics")
    
    return performance_export


def export_ate_results(base_path: str = '.') -> pd.DataFrame:
    """
    Export Average Treatment Effect results for Power BI.
    
    Returns:
        ATE results DataFrame
    """
    print("\nExporting ATE results...")
    
    # Load causal report
    with open(Path(base_path) / 'reports' / 'causal_report.json', 'r') as f:
        causal_data = json.load(f)
    
    ate_results = causal_data.get('ate_results', {})
    
    # Convert to DataFrame
    ate_list = []
    for offer_type, values in ate_results.items():
        ate_list.append({
            'offer_type': offer_type.replace('_', ' ').title(),
            'control_mean': values.get('control_mean', 0),
            'treatment_mean': values.get('treatment_mean', 0),
            'ate_dollars': values.get('ate', 0),
            'ate_percent': values.get('ate_percent', 0)
        })
    
    ate_export = pd.DataFrame(ate_list)
    
    print(f" Exported {len(ate_export)} ATE results")
    
    return ate_export


def export_recommendation_results(base_path: str = '.') -> pd.DataFrame:
    """
    Export recommendation system results for Power BI.
    
    Returns:
        Recommendation results DataFrame
    """
    print("\nExporting recommendation results...")
    
    # Load recommendation results
    with open(Path(base_path) / 'data' / 'processed' / 'recommendation_results.json', 'r') as f:
        rec_data = json.load(f)
    
    # Create summary
    summary = {
        'method': ['Random Targeting', 'Rule-Based Targeting'],
        'completion_rate': [
            rec_data.get('random_completion_rate', 0),
            rec_data.get('rule_based_completion_rate', 0)
        ],
        'lift_percent': [0, rec_data.get('lift_percent', 0)]
    }
    
    rec_export = pd.DataFrame(summary)
    
    print(f" Exported recommendation system results")
    
    return rec_export


def export_calendar_table() -> pd.DataFrame:
    """
    Export a calendar/date dimension table for Power BI time intelligence.

    Covers the full date range present in the transaction and offer event data
    (approximately 2018-06-27 to 2018-07-26), padded slightly.

    Returns:
        Calendar DataFrame with date attributes
    """
    print("\nExporting calendar table...")

    start_date = pd.Timestamp('2018-06-01')
    end_date = pd.Timestamp('2018-07-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    calendar = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'quarter': dates.quarter,
        'month': dates.month,
        'month_name': dates.strftime('%B'),
        'day': dates.day,
        'day_of_week': dates.dayofweek,
        'day_name': dates.strftime('%A'),
        'week_of_year': dates.isocalendar().week.astype(int),
        'is_weekend': dates.dayofweek >= 5,
        'is_weekday': dates.dayofweek < 5
    })

    print(f" Exported {len(calendar)} calendar dates ({start_date.date()} to {end_date.date()})")

    return calendar


def save_all_exports(base_path: str = '.') -> None:
    """
    Save all exported DataFrames to CSV files for Power BI.
    
    Args:
        base_path: Base directory
    """
    print("\n" + "="*60)
    print("SAVING ALL EXPORTS FOR POWER BI")
    print("="*60)
    
    export_dir = Path(base_path) / 'powerbi_data'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export all datasets
    datasets = {
        'customers': export_customer_data(base_path),
        'offers': export_offer_data(base_path),
        'transactions': export_transaction_data(base_path),
        'offer_events': export_offer_events(base_path),
        'cluster_profiles': export_cluster_profiles(base_path),
        'model_performance': export_model_performance(base_path),
        'ate_results': export_ate_results(base_path),
        'recommendation_results': export_recommendation_results(base_path),
        'dim_date': export_calendar_table()
    }
    
    # Save to CSV
    for name, df in datasets.items():
        output_path = export_dir / f'{name}.csv'
        df.to_csv(output_path, index=False)
        print(f"   Saved: {output_path.name}")
    
    # Create a summary file
    summary = pd.DataFrame({
        'dataset': list(datasets.keys()),
        'rows': [len(df) for df in datasets.values()],
        'columns': [len(df.columns) for df in datasets.values()]
    })
    summary.to_csv(export_dir / 'export_summary.csv', index=False)
    
    print(f"\n All datasets saved to: {export_dir}")
    print(f" Total datasets: {len(datasets)}")
    print(f" Total rows: {summary['rows'].sum():,}")
    
    return export_dir, datasets


def create_powerbi_instructions(export_dir: Path, datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Create instructions for importing data into Power BI.

    Args:
        export_dir: Directory where CSV files are saved
        datasets: Dict of name -> DataFrame for dynamic row/col counts
    """
    d = datasets
    instructions = f"""# Power BI Dashboard Setup Instructions

## Overview
This folder contains CSV files exported from the Starbucks Customer Segmentation project,
optimized for import into Microsoft Power BI.

## Files Included

### Fact / Transaction Tables
1. **customers.csv** - Customer master data with cluster assignments
   - {len(d['customers']):,} rows × {len(d['customers'].columns)} columns
   - Key fields: customer_id, age, income, cluster_label, completion_rate, engagement_level

2. **offers.csv** - Offer master data with channels (exploded per channel)
   - {len(d['offers']):,} rows × {len(d['offers'].columns)} columns
   - Key fields: offer_id, offer_type, difficulty, reward, duration_days, channel

3. **transactions.csv** - Transaction data (138K purchase events)
   - {len(d['transactions']):,} rows × {len(d['transactions'].columns)} columns
   - Key fields: customer_id, transaction_amount, transaction_date, day_of_week

4. **offer_events.csv** - Offer lifecycle events (received → viewed → completed)
   - {len(d['offer_events']):,} rows × {len(d['offer_events'].columns)} columns
   - Key fields: customer_id, offer_id, event_type, event_date

### Dimension / Summary Tables
5. **dim_date.csv** - Calendar date dimension for time-intelligence DAX
   - {len(d['dim_date']):,} rows × {len(d['dim_date'].columns)} columns
   - Key fields: date, year, quarter, month, day_of_week, is_weekend

6. **cluster_profiles.csv** - Cluster profile metrics (long format)
   - {len(d['cluster_profiles']):,} rows × {len(d['cluster_profiles'].columns)} columns
   - Key fields: cluster_id, cluster_label, metric, value

7. **model_performance.csv** - Model comparison metrics (long format)
   - {len(d['model_performance']):,} rows × {len(d['model_performance'].columns)} columns
   - Key fields: model_name, metric, value

8. **ate_results.csv** - Causal inference (Average Treatment Effect)
   - {len(d['ate_results']):,} rows × {len(d['ate_results'].columns)} columns
   - Key fields: offer_type, control_mean, treatment_mean, ate_dollars, ate_percent

9. **recommendation_results.csv** - Recommendation system performance
   - {len(d['recommendation_results']):,} rows × {len(d['recommendation_results'].columns)} columns
   - Key fields: method, completion_rate, lift_percent

## Power BI Import Steps

### Option A: Get Data → Text/CSV (one at a time)
- Click "Get Data" → "Text/CSV"
- Navigate to `{export_dir}`, select a CSV, click "Load"

### Option B: Get Data → Folder (all at once, recommended)
- Click "Get Data" → "Folder"
- Navigate to `{export_dir}`, click "OK"
- In Power Query: Click "Combine & Load" → "Combine & Load To..."
- This auto-merges all CSVs into separate query tables

### 3. Power Query Data Type Fixes
| Table | Column | Set Type To |
|---|---|---|
| customers | age, income | Decimal |
| customers | is_male, is_female | Whole Number |
| transactions | transaction_date | Date |
| transactions | transaction_amount | Decimal |
| offer_events | event_date | Date |
| dim_date | date | Date |
| cluster_profiles | value | Decimal |
| model_performance | value | Decimal |

### 4. Star Schema Relationships
Create these in Model View (drag field between tables):

| From (Many) | To (One) | Cardinality | Cross-filter |
|---|---|---|---|
| customers[customer_id] | transactions[customer_id] | *:1 | Single |
| customers[customer_id] | offer_events[customer_id] | *:1 | Single |
| offers[offer_id] | offer_events[offer_id] | *:1 | Single |
| customers[cluster_id] | cluster_profiles[cluster_id] | *:1 | Single |
| dim_date[date] | transactions[transaction_date] | *:1 | Single |
| dim_date[date] | offer_events[event_date] | *:1 | Single |

**Tip**: Mark `dim_date` as the Date Table: right-click → "Mark as Date Table" → Date column.

## Visualizations - 4 Dashboard Pages

### Page 1: Executive Summary (KPIs + Segments)

| Visual | Fields | Notes |
|---|---|---|
| **KPI Card** | Total Customers = COUNT(customers[customer_id]) | Large number format |
| **KPI Card** | Avg Completion Rate | % format, 1 decimal |
| **KPI Card** | Avg Transaction | Currency $, 2 decimals |
| **KPI Card** | Rec System Lift % | Show green if > 0 |
| **Donut Chart** | Cluster Label (Count) | Starbucks green/gold colors |
| **Clustered Bar Chart** | Axis: cluster_label, Values: completion_rate, view_rate, avg_transaction | Compare segments side-by-side |
| **Gauge** | XGBoost AUC-ROC vs target line of 0.70 | Green zone 0.8-1.0 |
| **Decomposition Tree** | Analyze: Avg Completion Rate, By: cluster_label → engagement_level → gender | Right-click to drill |

**Slicers** (top ribbon): Cluster label, Engagement level, Offer type

---

### Page 2: Offer Funnel & Performance

| Visual | Fields | Notes |
|---|---|---|
| **Funnel Chart** | Values: offer_events[event_type] (count) | Stages: received → viewed → completed |
| **Funnel chart** (per segment) | Copy funnel, add cluster_label filter in Visual-Level Filters | Shows funnel differs by cluster |
| **Scatter Plot** | X-axis: offers[difficulty], Y-axis: offers[reward], Size: COUNT(offer_events[completed]), Legend: offer_type | Identify sweet-spot offers |
| **Heatmap (Matrix)** | Rows: offer_type, Columns: channel, Values: AVG(customers[completion_rate]) | Shows which channel×type combo works best |
| **100% Stacked Bar** | Axis: offer_type, Values: event_type count (%), Legend: event_type | Funnel conversion rate per offer type |
| **Ribbon Chart** | Axis: event_date, Values: COUNT(offer_events), Legend: event_type | Event volume trends over time |

---

### Page 3: Customer Segments Deep Dive

| Visual | Fields | Notes |
|---|---|---|
| **Scatter Plot** | X: customers[income], Y: customers[age], Legend: cluster_label | PCA-style view of clusters |
| **Small Multiples (Line Chart)** | Axis: transaction_date, Values: SUM(transaction_amount), Small multiples: cluster_label | Spend trends per cluster |
| **Clustered Bar Chart** | Axis: cluster_label, Values: AVG(income), AVG(age), AVG(tenure_months), AVG(transaction_count) | Demographics comparison |
| **Stacked Column** | Axis: cluster_label, Values: COUNT(customers), Legend: engagement_level | Engagement distribution per cluster |
| **Key Influencers** | Analyze: customers[cluster_label], Explain by: income, age, completion_rate, view_rate, transaction_count | ML-driven "what drives segment membership" |

---

### Page 4: Model & Business Impact

| Visual | Fields | Notes |
|---|---|---|
| **Radar Chart** | (Custom shape) Compare 4 models on AUC-ROC, Precision, Recall, F1 | Use "Shape" or Python visual |
| **Bar Chart** | Axis: ate_results[offer_type], Values: ate_results[ate_dollars] | Add data label with ± |
| **Clustered Bar** | Axis: offer_type, Values: control_mean, treatment_mean | Before/after comparison |
| **Table** | Fields: model_name, metric, value | Matrix style with metric as columns |
| **Gauge cluster** | One per model: AUC-ROC value vs 0.70 target | Color = green if > 0.8 |
| **Card** | Rule-Based Targeting lift % (7.9%) | Large, with "+" prefix |
| **Card** | ATE Any Offer +$0.25 | Currency format |

---

## DAX Measures

Copy these into the model (Home → New Measure):

```dax
-- ============================================
-- CUSTOMER METRICS
-- ============================================

-- Total customer count (distinct)
Total Customers = DISTINCTCOUNT(customers[customer_id])

-- Average completion rate (weighted)
Avg Completion Rate = AVERAGE(customers[completion_rate])

-- Average view rate
Avg View Rate = AVERAGE(customers[view_rate])

-- Average transaction value
Avg Transaction = AVERAGE(customers[avg_transaction])

-- High-engagement customers
High Engagement Customers =
    CALCULATE(
        DISTINCTCOUNT(customers[customer_id]),
        customers[completion_rate] >= 0.5
    )

-- ============================================
-- OFFER FUNNEL
-- ============================================

-- Total offers received
Offers Received = COUNTROWS(FILTER(offer_events, offer_events[event_type] = "offer received"))

-- Offers viewed
Offers Viewed = COUNTROWS(FILTER(offer_events, offer_events[event_type] = "offer viewed"))

-- Offers completed
Offers Completed = COUNTROWS(FILTER(offer_events, offer_events[event_type] = "offer completed"))

-- Funnel conversion: received → viewed
Received to Viewed % =
    DIVIDE([Offers Viewed], [Offers Received])

-- Funnel conversion: received → completed
Received to Completed % =
    DIVIDE([Offers Completed], [Offers Received])

-- ============================================
-- TIME INTELLIGENCE (requires dim_date marked as date table)
-- ============================================

-- Running total spend
Running Total Spend =
    CALCULATE(
        SUM(transactions[transaction_amount]),
        DATESYTD(dim_date[date])
    )

-- Spend vs previous period
Spend vs Previous Period =
    CALCULATE(
        SUM(transactions[transaction_amount]),
        SAMEPERIODLASTYEAR(dim_date[date])
    )

-- Transaction count MTD
Transactions MTD =
    TOTALMTD(
        COUNTROWS(transactions),
        dim_date[date]
    )

-- ============================================
-- MODEL PERFORMANCE
-- ============================================

-- Best model AUC-ROC
Best AUC-ROC = MAXX(
    FILTER(model_performance, model_performance[metric] = "AUC-ROC"),
    model_performance[value]
)

-- Best model name
Best Model = MINX(
    TOPN(1,
        FILTER(model_performance, model_performance[metric] = "AUC-ROC"),
        model_performance[value], DESC
    ),
    model_performance[model_name]
)

-- ============================================
-- BUSINESS IMPACT
-- ============================================

-- Lift from rule-based targeting (percentage)
Lift % = SUM(recommendation_results[lift_percent])

-- Lift from rule-based targeting (points)
Lift Points =
    MAXX(recommendation_results, recommendation_results[completion_rate]) -
    MINX(recommendation_results, recommendation_results[completion_rate])

-- ============================================
-- DYNAMIC MEASURES (for parameter-style switching)
-- ============================================

-- Create a disconnected table for measure selection (in Power BI: Home > Enter Data):
--   Table name: param_metrics, Column: Metric (string), Rows: "Completion Rate", "View Rate", "Avg Spend", "Transaction Count"

Selected KPI =
    SWITCH(
        SELECTEDVALUE(param_metrics[Metric]),
        "Completion Rate", [Avg Completion Rate],
        "View Rate", [Avg View Rate],
        "Avg Spend", [Avg Transaction],
        "Transaction Count", COUNTROWS(transactions),
        [Avg Completion Rate]
    )

-- Dynamic title responds to slicers
Report Title =
    "Starbucks Segments" &
        IF(
            HASONEVALUE(customers[cluster_label]),
            " - " & VALUES(customers[cluster_label]),
            ""
        )
```

## Starbucks Theme Setup

1. **View tab** → **Themes** → **Customize current theme**
2. Set colors:
   - **Data colors**: `#00704A` (green primary), `#6B8E23` (olive), `#FFC72C` (gold), `#1E3932` (dark green), `#D4E9E2` (light mint)
   - **Background**: White `#FFFFFF`
   - **Cards & tiles**: `#F5F5F5` (light gray)
   - **Font**: Segoe UI, 12pt base
3. Save as `starbucks-theme.json` for reuse

## Layout & Interactivity Tips

- **Slicer panel**: Add a consistent left-rail with slicers for cluster_label, engagement_level, offer_type. Sync across all 4 pages via "View → Sync Slicers"
- **Page navigation**: Insert shape buttons (rounded rectangle) at the top of each page, assign Page Navigation action
- **Drill-through**: Right-click page tab → "Drill-through" → add cluster_label as drill-through field. Any page can now right-click → drill to details
- **Tooltip pages**: Create a mini page (200x300) with cluster profile metrics. Assign it as the tooltip for cluster visuals
- **Mobile layout**: View → Mobile Layout → arrange cards vertically for phone viewing

## Next Steps

1. Run `python src/reporting/export_powerbi.py` to regenerate CSVs with latest data
2. Open Power BI Desktop → Get Data → Folder → select `{export_dir}`
3. Set data types per the table above
4. Create relationships per the star schema above
5. Mark dim_date as date table
6. Paste DAX measures from this guide
7. Build the 4 dashboard pages following the visual tables above
8. Apply the Starbucks theme
9. Publish to Power BI Service → Share on portfolio!

---

**For questions or issues, refer to:**
- Power BI Documentation: https://docs.microsoft.com/power-bi/
- Project Repository: [your-repo-url]
- Technical Appendix: reports/technical_appendix.md
"""

    with open(export_dir / 'POWERBI_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    
    print(f"\n Power BI instructions saved to: {export_dir / 'POWERBI_INSTRUCTIONS.md'}")


if __name__ == "__main__":
    print("="*60)
    print("STARBUCKS CUSTOMER SEGMENTATION - POWER BI EXPORT")
    print("="*60)
    
    # Export all datasets
    export_dir, datasets = save_all_exports()
    
    # Create instructions with dynamic data
    create_powerbi_instructions(export_dir, datasets)
    
    print("\n" + "="*60)
    print("POWER BI EXPORT COMPLETE")
    print("="*60)
    print(f" All CSV files saved to: {export_dir}")
    print(f" Read instructions: {export_dir / 'POWERBI_INSTRUCTIONS.md'}")
    print(f" Ready to import into Power BI Desktop!")
    print("="*60)
    print("\n Next steps:")
    print("  1. Open Power BI Desktop")
    print("  2. Import CSV files from 'powerbi_data/' folder")
    print("  3. Create relationships between tables")
    print("  4. Build interactive dashboards")
    print("  5. Share on your portfolio with screenshots!")
