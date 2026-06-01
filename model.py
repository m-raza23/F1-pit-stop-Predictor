# ================================================================
# F1 PIT STOP PREDICTION — PRODUCTION EXPORT PIPELINE
# ================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

import os

print("⏳ Loading data...")
project_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(project_dir, 'trainf1.csv'))
test_df = pd.read_csv(os.path.join(project_dir, 'testf1.csv'))

NEG = (df['PitNextLap'] == 0).sum()
POS = (df['PitNextLap'] == 1).sum()
RATIO = round(NEG / POS)

# ── 1. Basic Encoders ───────────────────────────────────────────
compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
compound_life_map = {0: 20, 1: 30, 2: 40, 3: 25, 4: 35}

df['Compound'] = df['Compound'].map(compound_map).fillna(-1).astype(int)
test_df['Compound'] = test_df['Compound'].map(compound_map).fillna(-1).astype(int)

# Use dictionary maps instead of LabelEncoder for easier real-time lookup later
# We save these mappings explicitly so our local dashboard can decode incoming text strings
driver_list = sorted(pd.concat([df['Driver'], test_df['Driver']]).astype(str).unique())
race_list = sorted(pd.concat([df['Race'], test_df['Race']]).astype(str).unique())

driver_map = {name: i for i, name in enumerate(driver_list)}
race_map = {name: i for i, name in enumerate(race_list)}

df['Driver'] = df['Driver'].astype(str).map(driver_map)
df['Race'] = df['Race'].astype(str).map(race_map)
test_df['Driver'] = test_df['Driver'].astype(str).map(driver_map)
test_df['Race'] = test_df['Race'].astype(str).map(race_map)

# ── 2. Feature Engineering Function ─────────────────────────────
def engineer_features(data):
    data = data.copy()
    grp = data.groupby(['Driver', 'Race'])

    data['tyre_life_pct'] = data['TyreLife'] / data['Compound'].map(compound_life_map).replace(-1, 30).fillna(30)
    data['tyre_life_remaining'] = 1 - data['tyre_life_pct']
    data['tyre_critical'] = (data['tyre_life_pct'] > 0.80).astype(int)
    data['tyre_danger_zone'] = (data['tyre_life_pct'] > 0.90).astype(int)
    data['expected_life'] = data['Compound'].map(compound_life_map).fillna(30)
    data['laps_until_overdue'] = data['expected_life'] - data['TyreLife']

    data['rolling_deg_2'] = grp['LapTime_Delta'].transform(lambda x: x.rolling(2, min_periods=1).mean())
    data['rolling_deg_3'] = grp['LapTime_Delta'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_deg_5'] = grp['LapTime_Delta'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    data['rolling_deg_std'] = grp['LapTime_Delta'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    data['deg_acceleration'] = grp['LapTime_Delta'].transform(lambda x: x.diff()).fillna(0)
    data['deg_trend'] = grp['LapTime_Delta'].transform(lambda x: x.rolling(5, min_periods=2).apply(
        lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True)).fillna(0)
    data['cumulative_deg_rate'] = data['Cumulative_Degradation'] / (data['TyreLife'] + 1)

    data['in_first_window'] = ((data['RaceProgress'] >= 0.25) & (data['RaceProgress'] <= 0.40)).astype(int)
    data['in_second_window'] = ((data['RaceProgress'] >= 0.55) & (data['RaceProgress'] <= 0.70)).astype(int)
    data['too_late_to_pit'] = (data['RaceProgress'] > 0.90).astype(int)
    data['in_any_pit_window'] = ((data['in_first_window'] == 1) | (data['in_second_window'] == 1)).astype(int)
    data['is_first_stint'] = (data['Stint'] == 1).astype(int)
    data['is_second_stint'] = (data['Stint'] == 2).astype(int)
    data['race_progress_sq'] = data['RaceProgress'] ** 2
    data['lap_in_race_pct'] = grp['LapNumber'].transform(lambda x: x / x.max())

    data['is_frontrunner'] = (data['Position'] <= 5).astype(int)
    data['is_backmarker'] = (data['Position'] >= 16).astype(int)
    data['in_midfield'] = ((data['Position'] >= 6) & (data['Position'] <= 15)).astype(int)
    data['losing_positions'] = (data['Position_Change'] < -1).astype(int)
    data['gaining_positions'] = (data['Position_Change'] > 1).astype(int)
    data['rolling_pos_change'] = grp['Position_Change'].transform(lambda x: x.rolling(3, min_periods=1).mean()).fillna(0)
    data['laptime_normalized'] = grp['LapTime (s)'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    data['tyre_x_window'] = data['tyre_life_pct'] * data['in_any_pit_window']
    data['tyre_x_deg'] = data['tyre_life_pct'] * data['rolling_deg_3']
    data['critical_x_window'] = data['tyre_critical'] * data['in_any_pit_window']
    data['stint_x_tyre'] = data['Stint'] * data['tyre_life_pct']
    data['progress_x_tyre'] = data['RaceProgress'] * data['tyre_life_pct']

    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(0)
    return data

print("⚙️ Running feature engineering...")
df = engineer_features(df)
test_df = engineer_features(test_df)

# ── 3. Target Encoding Fit ──────────────────────────────────────
TARGET_ENCODE_COLS = ['Driver', 'Race', 'Compound']
te_full = TargetEncoder(cols=TARGET_ENCODE_COLS, smoothing=10)
te_full.fit(df[TARGET_ENCODE_COLS], df['PitNextLap'])

te_train = te_full.transform(df[TARGET_ENCODE_COLS])
for col in TARGET_ENCODE_COLS:
    df[f'{col}_te'] = te_train[col].values

features = [
    'TyreLife', 'tyre_life_pct', 'tyre_life_remaining', 'tyre_critical', 'tyre_danger_zone', 'expected_life', 'laps_until_overdue',
    'LapTime_Delta', 'Cumulative_Degradation', 'rolling_deg_2', 'rolling_deg_3', 'rolling_deg_5', 'rolling_deg_std', 'deg_acceleration', 'deg_trend', 'cumulative_deg_rate',
    'RaceProgress', 'race_progress_sq', 'lap_in_race_pct', 'in_first_window', 'in_second_window', 'too_late_to_pit', 'in_any_pit_window',
    'Stint', 'is_first_stint', 'is_second_stint', 'Position', 'Position_Change', 'is_frontrunner', 'is_backmarker', 'in_midfield', 'losing_positions', 'gaining_positions', 'rolling_pos_change',
    'laptime_normalized', 'tyre_x_window', 'tyre_x_deg', 'critical_x_window', 'stint_x_tyre', 'progress_x_tyre',
    'Driver', 'Race', 'Compound', 'Driver_te', 'Race_te', 'Compound_te'
]

X = df[features]
y = df['PitNextLap']

# ── 4. Train Final Level-1 Machine Learning Models ──────────────
print("🔥 Training production-level models (XGBoost, LightGBM, CatBoost)...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_store = {m: np.zeros(len(X)) for m in ['xgb', 'lgb', 'cat']}
trained_models = {'xgb': [], 'lgb': [], 'cat': []}

for fold, (ti, vi) in enumerate(skf.split(X, y)):
    X_tr, X_vl = X.iloc[ti], X.iloc[vi]
    y_tr, y_vl = y.iloc[ti], y.iloc[vi]
    
    # XGBoost
    m_xgb = xgb.XGBClassifier(scale_pos_weight=RATIO, eval_metric='auc', random_state=SEED, n_jobs=-1)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    oof_store['xgb'][vi] = m_xgb.predict_proba(X_vl)[:, 1]
    trained_models['xgb'].append(m_xgb)
    
    # LightGBM
    m_lgb = lgb.LGBMClassifier(scale_pos_weight=RATIO, random_state=SEED, verbose=-1, n_jobs=-1)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_store['lgb'][vi] = m_lgb.predict_proba(X_vl)[:, 1]
    trained_models['lgb'].append(m_lgb)
    
    # CatBoost
    m_cat = CatBoostClassifier(scale_pos_weight=RATIO, eval_metric='AUC', random_seed=SEED, verbose=0)
    m_cat.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=50)
    oof_store['cat'][vi] = m_cat.predict_proba(X_vl)[:, 1]
    trained_models['cat'].append(m_cat)

print(f"-> XGB OOF AUC: {roc_auc_score(y, oof_store['xgb']):.4f}")
print(f"-> LGB OOF AUC: {roc_auc_score(y, oof_store['lgb']):.4f}")
print(f"-> CAT OOF AUC: {roc_auc_score(y, oof_store['cat']):.4f}")

# ── 5. Train Stacking Meta-Model ───────────────────────────────
print("🧠 Training Stacking Meta-Model...")
meta_train = np.column_stack([oof_store['xgb'], oof_store['lgb'], oof_store['cat']])

# Pairwise interactions for meta-features
pairs = [meta_train[:, 0] * meta_train[:, 1], meta_train[:, 0] * meta_train[:, 2], meta_train[:, 1] * meta_train[:, 2]]
meta_train = np.column_stack([meta_train] + pairs)

scaler = RobustScaler()
meta_train_scaled = scaler.fit_transform(meta_train)

meta_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED)
meta_model.fit(meta_train_scaled, y)

# Hardcoded optimal ensemble blend weights from Kaggle run
weights = {'meta': 0.401, 'lgb': 0.350, 'cat': 0.140, 'xgb': 0.109}

# ── 6. Save Everything for Local Production Deployment ──────────
print("💾 Exporting components to disk...")
artifacts = {
    'models': trained_models,        # List of 5 folds for each algorithm
    'meta_model': meta_model,        # Meta-classifier
    'scaler': scaler,                # Scaling transformation
    'target_encoder': te_full,       # Encodes names to values leakage-free
    'driver_map': driver_map,        # String-to-Int maps
    'race_map': race_map,
    'compound_map': compound_map,
    'features_list': features,
    'blend_weights': weights
}

joblib.dump(artifacts, 'f1_production_assets.pkl', compress=3)
print("🚀 ALL PRODUCTION ASSETS SAVED SUCCESSFULLY TO 'f1_production_assets.pkl'!")