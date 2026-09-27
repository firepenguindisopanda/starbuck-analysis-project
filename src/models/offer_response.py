"""
Send-time offer response model.

Answers the question the targeting decision actually faces: when an offer is
sent, how likely is this customer to complete it within the offer's validity
window? Every feature is something known at the moment of sending.

This replaces an earlier framing (removed; see the git history), which scored
0.99 AUC for reasons that do not survive contact with a real send:
  - it crossed every customer with all ten offers, so most rows were offers
    never received (target trivially 0), informational offers included;
  - its target was "ever completed", with no window;
  - its features included whether the offer was viewed (after the send) and
    completion counts over the whole month (including the offer being scored);
  - it split rows at random, so a customer's other offers sat in the test set.

The leakage ladder below measures how much each of those contributed.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
END_OF_DATA = 714  # hours; the last event in the transcript
# Test AUC of the earlier framing (every customer x every offer, random split),
# as reported by the pipeline at commit 85f7726 before it was removed.
ORIGINAL_PIPELINE_AUC = 0.9936
CHANNELS = ['email', 'mobile', 'social', 'web']

OFFER_COLS = ['difficulty', 'reward', 'duration', 'is_bogo', 'n_channels'] + [f'ch_{c}' for c in CHANNELS]
DEMO_COLS = ['age', 'age_missing', 'income', 'income_missing', 'tenure_days',
             'gender_F', 'gender_M', 'gender_O']
HISTORY_COLS = ['prior_sends', 'prior_completions', 'prior_completion_rate',
                'prior_views', 'prior_view_rate', 'prior_sends_same_offer',
                'prior_transactions', 'prior_spend', 'prior_avg_spend',
                'hours_since_last_transaction']
HONEST_COLS = OFFER_COLS + DEMO_COLS + HISTORY_COLS
LEAKY_EXTRA_COLS = ['month_completions', 'month_completion_rate', 'viewed']


# ---------------------------------------------------------------------------
# Building the send-level dataset
# ---------------------------------------------------------------------------

def build_send_table(portfolio: pd.DataFrame, transcript: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    One row per BOGO or discount offer received, with its validity window and
    the target: completed within that window.

    A completion is credited to the latest receipt of the same offer by the same
    person whose window contains it, so an offer received twice is scored as two
    separate sends. Sends whose window runs past the end of the data are dropped
    as censored: whether they would have completed is unknown.
    """
    offers = portfolio.rename(columns={'id': 'offer_id'})
    offers = offers[offers['offer_type'].isin(['bogo', 'discount'])]

    sends = transcript.loc[transcript['event'] == 'offer received', ['person', 'offer_id', 'time']]
    sends = sends.merge(offers[['offer_id', 'duration']], on='offer_id', how='inner')
    sends = sends.rename(columns={'time': 'sent_at'}).reset_index(drop=True)
    sends['send_id'] = np.arange(len(sends))
    sends['window_end'] = sends['sent_at'] + 24 * sends['duration']

    completions = transcript.loc[transcript['event'] == 'offer completed', ['person', 'offer_id', 'time']]
    matched = sends[['send_id', 'person', 'offer_id', 'sent_at', 'window_end']].merge(
        completions, on=['person', 'offer_id'], how='inner')
    matched = matched[(matched['time'] >= matched['sent_at']) & (matched['time'] <= matched['window_end'])]
    # Credit each completion to the latest eligible send.
    credited = matched.sort_values('sent_at').groupby(['person', 'offer_id', 'time']).tail(1)
    sends['completed'] = sends['send_id'].isin(credited['send_id']).astype(int)

    viewed = transcript.loc[transcript['event'] == 'offer viewed', ['person', 'offer_id', 'time']]
    seen = sends[['send_id', 'person', 'offer_id', 'sent_at', 'window_end']].merge(
        viewed, on=['person', 'offer_id'], how='inner')
    seen = seen[(seen['time'] >= seen['sent_at']) & (seen['time'] <= seen['window_end'])]
    sends['viewed'] = sends['send_id'].isin(seen['send_id']).astype(int)

    censored = sends['window_end'] > END_OF_DATA
    counts = {
        'sends_bogo_discount': int(len(sends)),
        'sends_censored_dropped': int(censored.sum()),
        'sends_kept': int((~censored).sum()),
    }
    return sends[~censored].drop(columns='duration').reset_index(drop=True), counts


def _prior_count(sends: pd.DataFrame, events: pd.DataFrame, name: str,
                 amount_col: str = None, by: List[str] = None) -> pd.DataFrame:
    """
    For each send, count the events (and sum `amount_col`) strictly before
    `sent_at`, per person (or per `by` keys). Strictly before: an event at the
    same hour as the send is not known when deciding to send.
    """
    by = by or ['person']
    ev = events.sort_values('time').copy()
    ev[f'{name}'] = ev.groupby(by).cumcount() + 1
    cols = by + ['time', name]
    if amount_col:
        ev[f'{name}_amount'] = ev.groupby(by)[amount_col].cumsum()
        cols.append(f'{name}_amount')
    ev['last_time'] = ev['time']
    cols.append('last_time')
    left = sends.sort_values('sent_at')
    out = pd.merge_asof(left, ev[cols].rename(columns={'time': 'ev_time'}),
                        left_on='sent_at', right_on='ev_time', by=by,
                        allow_exact_matches=False, direction='backward')
    keep = ['send_id', name] + ([f'{name}_amount'] if amount_col else []) + ['last_time']
    out = out[keep].rename(columns={'last_time': f'{name}_last_time'})
    out[name] = out[name].fillna(0)
    if amount_col:
        out[f'{name}_amount'] = out[f'{name}_amount'].fillna(0)
    return out


def add_send_time_features(sends: pd.DataFrame, portfolio: pd.DataFrame,
                           profile: pd.DataFrame, transcript: pd.DataFrame) -> pd.DataFrame:
    """Offer terms, demographics, and history strictly before the send."""
    df = sends.copy()

    offers = portfolio.rename(columns={'id': 'offer_id'})[['offer_id', 'offer_type', 'difficulty', 'reward', 'duration', 'channels']].copy()
    offers['is_bogo'] = (offers['offer_type'] == 'bogo').astype(int)
    for c in CHANNELS:
        offers[f'ch_{c}'] = offers['channels'].apply(lambda chs, c=c: int(c in chs))
    offers['n_channels'] = offers['channels'].apply(len)
    df = df.merge(offers.drop(columns=['channels']), on='offer_id', how='left')

    prof = profile.rename(columns={'id': 'person'})[['person', 'age', 'gender', 'income', 'became_member_on']].copy()
    prof['age_missing'] = (prof['age'] == 118).astype(int)
    prof['age'] = prof['age'].where(prof['age'] != 118)
    prof['income_missing'] = prof['income'].isna().astype(int)
    joined = pd.to_datetime(prof['became_member_on'].astype(str), format='%Y%m%d', errors='coerce')
    # Tenure at the start of the experiment: known before any send.
    prof['tenure_days'] = (joined.max() - joined).dt.days
    for g in ['F', 'M', 'O']:
        prof[f'gender_{g}'] = (prof['gender'] == g).astype(int)
    df = df.merge(prof.drop(columns=['gender', 'became_member_on']), on='person', how='left')

    ev = transcript[['person', 'event', 'time', 'offer_id', 'transaction_amount']]
    all_sends = ev[ev['event'] == 'offer received']
    parts = [
        _prior_count(df, all_sends, 'prior_sends'),
        _prior_count(df, ev[ev['event'] == 'offer completed'], 'prior_completions'),
        _prior_count(df, ev[ev['event'] == 'offer viewed'], 'prior_views'),
        _prior_count(df, ev[ev['event'] == 'transaction'], 'prior_transactions', amount_col='transaction_amount'),
        _prior_count(df, all_sends, 'prior_sends_same_offer', by=['person', 'offer_id']),
    ]
    for p in parts:
        df = df.merge(p, on='send_id', how='left')

    df['prior_completion_rate'] = df['prior_completions'] / df['prior_sends'].replace(0, np.nan)
    df['prior_view_rate'] = df['prior_views'] / df['prior_sends'].replace(0, np.nan)
    df['prior_spend'] = df['prior_transactions_amount']
    df['prior_avg_spend'] = df['prior_spend'] / df['prior_transactions'].replace(0, np.nan)
    df['hours_since_last_transaction'] = df['sent_at'] - df['prior_transactions_last_time']
    drop = [c for c in df.columns if c.endswith('_last_time') or c == 'prior_transactions_amount']
    return df.drop(columns=drop).sort_values('send_id').reset_index(drop=True)


def add_leaky_features(df: pd.DataFrame, transcript: pd.DataFrame) -> pd.DataFrame:
    """
    The kind of features the original pipeline used, for the ladder only:
    completions over the whole month (the send being scored included) and
    whether the offer was viewed (which happens after the send).
    """
    out = df.copy()
    month = transcript[transcript['event'] == 'offer completed'].groupby('person').size().rename('month_completions')
    received = transcript[transcript['event'] == 'offer received'].groupby('person').size().rename('month_sends')
    out = out.merge(month, left_on='person', right_index=True, how='left')
    out = out.merge(received, left_on='person', right_index=True, how='left')
    out['month_completions'] = out['month_completions'].fillna(0)
    out['month_completion_rate'] = out['month_completions'] / out['month_sends']
    return out.drop(columns='month_sends')


def build_dataset(base_path: str = '.') -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load the raw JSON and return the send-level modelling table."""
    raw = Path(base_path) / 'data' / 'raw'
    portfolio = pd.read_json(raw / 'portfolio.json', lines=True)
    profile = pd.read_json(raw / 'profile.json', lines=True)
    transcript = pd.read_json(raw / 'transcript.json', lines=True)
    transcript['offer_id'] = transcript['value'].apply(lambda v: v.get('offer id') or v.get('offer_id'))
    transcript['transaction_amount'] = transcript['value'].apply(lambda v: v.get('amount'))

    sends, counts = build_send_table(portfolio, transcript)
    df = add_send_time_features(sends, portfolio, profile, transcript)
    df = add_leaky_features(df, transcript)
    return df, counts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)


def _models() -> Dict[str, object]:
    return {
        'Logistic regression': make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                             StandardScaler(), LogisticRegression(max_iter=2000)),
        'Histogram gradient boosting': _hgb(),
        'XGBoost': xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric='logloss', random_state=SEED, verbosity=0),
    }


def leakage_ladder(df: pd.DataFrame) -> List[Dict]:
    """
    Take the leaks away one at a time, same model and data, 5-fold CV each.
    Rows of the ladder are cumulative.
    """
    y = df['completed'].values
    groups = df['person'].values
    honest_no_history = OFFER_COLS + DEMO_COLS
    rungs = [
        ('Month-long completion counts, viewed flag, random row split',
         honest_no_history + LEAKY_EXTRA_COLS, 'rows'),
        ('Viewed flag removed',
         honest_no_history + ['month_completions', 'month_completion_rate'], 'rows'),
        ('Month-long counts replaced by history before the send',
         HONEST_COLS, 'rows'),
        ('Customers kept apart across folds (the honest setup)',
         HONEST_COLS, 'customers'),
    ]
    out = []
    for label, cols, split in rungs:
        if split == 'rows':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            scores = cross_val_score(_hgb(), df[cols], y, cv=cv, scoring='roc_auc')
        else:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            scores = cross_val_score(_hgb(), df[cols], y, cv=cv, groups=groups, scoring='roc_auc')
        out.append({'step': label, 'split': split, 'n_features': len(cols),
                    'cv_auc_mean': round(float(scores.mean()), 4),
                    'cv_auc_std': round(float(scores.std()), 4)})
        print(f"  {out[-1]['cv_auc_mean']:.4f} +/- {out[-1]['cv_auc_std']:.4f}  {label}")
    return out


def _grouped_bootstrap_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int = 1000) -> Tuple[float, float]:
    """95% interval for AUC, resampling customers rather than rows."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_by_group = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array)
    aucs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate(idx_by_group.loc[pick].values)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_honest(df: pd.DataFrame) -> Dict:
    """Grouped CV, a held-out set of customers, and a later-sends time check."""
    y = df['completed'].values
    groups = df['person'].values
    X = df[HONEST_COLS]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    g_tr, g_te = groups[train_idx], groups[test_idx]

    results = {'models': {}, 'baselines': {}}

    base_rate = float(y_tr.mean())
    results['baselines']['Base rate (predict the average)'] = {
        'holdout_auc': 0.5, 'holdout_brier': round(float(brier_score_loss(y_te, np.full(len(y_te), base_rate))), 4)}
    offer_only = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr[OFFER_COLS], y_tr)
    p_offer = offer_only.predict_proba(X_te[OFFER_COLS])[:, 1]
    results['baselines']['Offer terms only (logistic regression)'] = {
        'holdout_auc': round(float(roc_auc_score(y_te, p_offer)), 4),
        'holdout_brier': round(float(brier_score_loss(y_te, p_offer)), 4)}
    no_history = OFFER_COLS + DEMO_COLS
    nh = _hgb().fit(X_tr[no_history], y_tr)
    p_nh = nh.predict_proba(X_te[no_history])[:, 1]
    results['baselines']['Offer terms and demographics, no history (gradient boosting)'] = {
        'holdout_auc': round(float(roc_auc_score(y_te, p_nh)), 4),
        'holdout_brier': round(float(brier_score_loss(y_te, p_nh)), 4)}

    best_name, best_auc, best_pred, best_model = None, -1, None, None
    for name, model in _models().items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, groups=g_tr, scoring='roc_auc')
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p)
        lo, hi = _grouped_bootstrap_auc(y_te, p, g_te)
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        results['models'][name] = {
            'cv_auc_mean': round(float(cv_scores.mean()), 4),
            'cv_auc_std': round(float(cv_scores.std()), 4),
            'holdout_auc': round(float(auc), 4),
            'holdout_auc_ci95': [round(lo, 4), round(hi, 4)],
            'holdout_pr_auc': round(float(average_precision_score(y_te, p)), 4),
            'holdout_brier': round(float(brier_score_loss(y_te, p)), 4),
            'at_threshold_0_5': {
                'precision': round(float(precision_score(y_te, pred)), 4),
                'recall': round(float(recall_score(y_te, pred)), 4),
                'f1': round(float(f1_score(y_te, pred)), 4),
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            },
        }
        print(f"  {name}: CV {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}, holdout {auc:.4f} [{lo:.4f}, {hi:.4f}]")
        if auc > best_auc:
            best_name, best_auc, best_pred, best_model = name, auc, p, model

    frac_pos, mean_pred = calibration_curve(y_te, best_pred, n_bins=10, strategy='quantile')
    results['best_model'] = best_name
    results['calibration'] = {'mean_predicted': [round(float(v), 4) for v in mean_pred],
                              'fraction_completed': [round(float(v), 4) for v in frac_pos]}

    imp = permutation_importance(best_model, X_te, y_te, scoring='roc_auc', n_repeats=5, random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    results['permutation_importance'] = [
        {'feature': HONEST_COLS[i], 'auc_drop': round(float(imp.importances_mean[i]), 4)} for i in order]

    # Time check: learn from the first three send waves, score the later ones.
    early = df['sent_at'] <= 336
    late = ~early
    tm = _hgb().fit(X[early], y[early])
    results['time_check'] = {
        'train_sends': int(early.sum()), 'test_sends': int(late.sum()),
        'train_waves_hours': sorted(int(t) for t in df.loc[early, 'sent_at'].unique()),
        'test_waves_hours': sorted(int(t) for t in df.loc[late, 'sent_at'].unique()),
        'test_auc': round(float(roc_auc_score(y[late], tm.predict_proba(X[late])[:, 1])), 4),
    }
    print(f"  time check (train waves <= 336 h, test later): {results['time_check']['test_auc']:.4f}")

    results['holdout'] = {'customers': int(len(np.unique(g_te))), 'sends': int(len(test_idx)),
                          'train_sends': int(len(train_idx)),
                          'completion_rate': round(float(y_te.mean()), 4)}
    results['_best_pred'] = best_pred
    results['_y_te'] = y_te
    return results


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

def plot_ladder(ladder: List[Dict], original_auc: float, path: Path) -> None:
    labels = ['Original pipeline\n(every customer x every offer)'] + [r['step'].replace(', ', ',\n', 1) for r in ladder]
    values = [original_auc] + [r['cv_auc_mean'] for r in ladder]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ['#b8b8b8'] * (len(values) - 1) + ['#1f5f8b']
    ax.barh(range(len(values))[::-1], values, color=colors)
    for i, v in enumerate(values):
        ax.text(v + 0.005, len(values) - 1 - i, f'{v:.3f}', va='center', fontsize=10)
    ax.set_yticks(range(len(values))[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.5, 1.05)
    ax.axvline(0.5, color='black', lw=0.8)
    ax.set_xlabel('ROC AUC')
    ax.set_title('Taking the leaks out, one at a time')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(cal: Dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=1, label='Perfect calibration')
    ax.plot(cal['mean_predicted'], cal['fraction_completed'], marker='o', color='#1f5f8b', label='Model')
    ax.set_xlabel('Predicted probability of completion')
    ax.set_ylabel('Share actually completed')
    ax.set_title('Calibration on held-out customers')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_for_portfolio(report: Dict, y_te: np.ndarray, p_te: np.ndarray, out_dir: Path) -> None:
    """The three JSON files the portfolio's interactive analysis block reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    best_name = report['best_model']
    best = report['models'][best_name]
    base = report['models']['Logistic regression']
    card = lambda m: {'auc_roc': m['holdout_auc'], **{k: m['at_threshold_0_5'][k] for k in ('precision', 'recall', 'f1')}}
    pos = report['holdout']['completion_rate']
    metrics = {
        'best': card(best),
        'baseline': card(base),
        'improvement': {
            'auc_roc_over_baseline_pct': round(100 * (best['holdout_auc'] - base['holdout_auc']) / base['holdout_auc'], 1),
            'recall_gain_pct': round(100 * (best['at_threshold_0_5']['recall'] - base['at_threshold_0_5']['recall']), 1),
        },
        'training_samples': report['holdout']['train_sends'],
        'test_samples': report['holdout']['sends'],
        'best_model': best_name,
        'baseline_model': 'Logistic regression',
        'imbalance_ratio': f'{pos / (1 - pos):.1f}:1',
        'positive_pct': round(100 * pos, 1),
        'models_trained': len(report['models']),
        'cross_validation': {'cv_folds': 5, 'cv_auc_mean': best['cv_auc_mean'], 'cv_auc_std': best['cv_auc_std'],
                             'interpretation': 'Customers kept apart across folds'},
        'brier_score': best['holdout_brier'],
    }
    (out_dir / 'model_metrics.json').write_text(json.dumps(metrics, indent=2))

    importance = {
        'features': [{'name': r['feature'], 'importance': r['auc_drop']} for r in report['permutation_importance'][:10]],
        'total_features': len(HONEST_COLS),
        'model_type': f"{best_name} (permutation, AUC lost when shuffled)",
        'dataset': 'Starbucks offer completion at send time',
    }
    (out_dir / 'feature_importance.json').write_text(json.dumps(importance, indent=2))

    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.where(y_te == c)[0], 25, replace=False) for c in (0, 1)])
    rng.shuffle(idx)
    cm = best['at_threshold_0_5']['confusion_matrix']
    samples = {
        'samples': [{'actual': int(y_te[i]), 'predicted': int(p_te[i] >= 0.5),
                     'predicted_probability': round(float(p_te[i]), 3),
                     'error': int(int(p_te[i] >= 0.5) != y_te[i])} for i in idx],
        'sample_size': int(len(idx)),
        'sampling_method': 'stratified',
        'model': best_name,
        'dataset': 'held-out customers',
        'positive_label': 'Completed offers',
        'negative_label': 'not completed',
        'confusion_matrix': {**cm,
                             'sensitivity': round(cm['tp'] / (cm['tp'] + cm['fn']), 4),
                             'specificity': round(cm['tn'] / (cm['tn'] + cm['fp']), 4),
                             'precision': round(cm['tp'] / (cm['tp'] + cm['fp']), 4)},
    }
    (out_dir / 'predictions_sample.json').write_text(json.dumps(samples, indent=2))


def run(base_path: str = '.') -> Dict:
    base = Path(base_path)
    fig_dir = base / 'reports' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 60)
    print('SEND-TIME OFFER RESPONSE MODEL')
    print('=' * 60)
    df, counts = build_dataset(base_path)
    print(f"Sends: {counts['sends_kept']:,} kept, {counts['sends_censored_dropped']:,} censored "
          f"(window past hour {END_OF_DATA}), completion rate {df['completed'].mean():.3f}")

    original_auc = ORIGINAL_PIPELINE_AUC

    print('\nLeakage ladder (histogram gradient boosting, 5-fold CV):')
    ladder = leakage_ladder(df)
    print('\nHonest models:')
    honest = evaluate_honest(df)

    plot_ladder(ladder, original_auc, fig_dir / 'offer_response_ladder.png')
    plot_calibration(honest['calibration'], fig_dir / 'offer_response_calibration.png')

    report = {
        'question': 'When a BOGO or discount offer is sent, will this customer complete it within its validity window?',
        'unit': 'one row per offer received (BOGO and discount; informational offers have no completion event)',
        'features': 'offer terms, demographics, and the customer\'s history strictly before the send',
        'counts': counts,
        'completion_rate': round(float(df['completed'].mean()), 4),
        'customers': int(df['person'].nunique()),
        'original_pipeline_test_auc': round(original_auc, 4),
        'leakage_ladder': ladder,
        **{k: v for k, v in honest.items() if not k.startswith('_')},
    }
    (base / 'reports' / 'offer_response_report.json').write_text(json.dumps(report, indent=2))
    export_for_portfolio(report, honest['_y_te'], honest['_best_pred'], base / 'reports' / 'portfolio')
    print(f"\nWrote reports/offer_response_report.json, reports/portfolio/*.json and two figures in {fig_dir}")
    return report


if __name__ == '__main__':
    run('.')
