"""
train_fx_pro.py

FX PRO ML Model - Specialized Random Forest for Forex Trading

Trained ONLY on FX trades from the 812-trade backup dataset.
Learns patterns specific to forex markets (sessions, spreads, etc.)

Usage: python train_fx_pro.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime

print("=" * 60)
print("FX PRO ML - Random Forest Training")
print("=" * 60)

# ============================================================
# 1. LOAD AND FILTER FX TRADES
# ============================================================

df = pd.read_csv('training_dataset_backup_812.csv')
print(f"[INFO] Loaded {len(df)} total trades from backup")

# Non-FX keywords to exclude
crypto_keys = ['BTC','ETH','XRP','ADA','SOL','DOGE','LTC','LINK','UNI','AAVE',
               'DOT','BNB','XLM','AVAX','MATIC','SHIB','ATOM','ALGO','FTM','SAND',
               'MANA','AXS','CRV','COMP','SUSHI','YFI','SNX','MKR','ENJ','BAT',
               'ZEC','DASH','XMR','EOS','TRX','XTZ','LRC','BCH','NEAR','APE',
               'FIL','THETA','VET','ICP','HBAR','EGLD','FLOW','CHZ','GALA','IMX']

metals = ['XAUUSD','XAGUSD','XAUEUR','XAUGBP','XAUAUD','XPTUSD']

indices_keys = ['US30','US500','US2000','USTECH','GER40','UK100','DAX','NAS100',
                'FT100','VIX','DJ30','S&P','AUS200','CN50','EU50','FRA40','HK50',
                'JPN225','NK225','CAC40','SPI200','EUSTX','HSI','IT40','SWI20',
                'SPA35','NETH25','SGFREE','CHINA50','USDINDEX']

energies = ['BRENT','WTI','USOIL','UKOIL','NATGAS']
softs = ['COCOA','COFFEE','COPPER','SOYBEAN']

def is_forex(symbol):
    s = symbol.upper()
    # Exclude crypto
    if any(k in s for k in crypto_keys) or '-USD' in s:
        return False
    # Exclude metals
    if any(m in s for m in metals):
        return False
    # Exclude indices
    if any(i in s for i in indices_keys):
        return False
    # Exclude energies
    if any(e in s for e in energies):
        return False
    # Exclude softs
    if any(c in s for c in softs):
        return False
    # Must contain USD, EUR, GBP, JPY, etc. (FX pairs)
    fx_currencies = ['USD','EUR','GBP','JPY','AUD','NZD','CAD','CHF','SEK','NOK',
                     'PLN','HUF','CZK','ZAR','MXN','SGD','HKD','CNH','TRY','THB',
                     'INR','BRL','KRW','TWD']
    return any(c in s for c in fx_currencies)

df_fx = df[df['symbol'].apply(is_forex)].copy()
print(f"[INFO] Filtered to {len(df_fx)} FX trades")

# Show unique symbols
print(f"[INFO] Unique FX pairs: {df_fx['symbol'].nunique()}")

# ============================================================
# 2. FEATURE SELECTION (FX-optimized)
# ============================================================

# Features that matter for FX
feature_cols = [
    'score',           # Signal strength
    'rsi',             # RSI (mean-reversion matters in FX)
    'atr',             # Volatility
    'adx',             # Trend strength
    'macd_hist',       # MACD histogram
    'trend_bias',      # Trend direction
    'volatility_regime',# Volatility state
    'spread_ratio',    # Spread cost (critical for FX!)
    'session_state',   # Trading session (London/NY/Tokyo)
    'ema_fast',        # Fast EMA
    'ema_slow',        # Slow EMA
]

# Check which features exist
available_features = [f for f in feature_cols if f in df_fx.columns]
print(f"[INFO] Using {len(available_features)} features: {available_features}")

X = df_fx[available_features].copy()
y = df_fx['label'].copy()

# Handle missing values
X = X.fillna(0)

# Remove any infinite values
X = X.replace([np.inf, -np.inf], 0)

print(f"[INFO] Features shape: {X.shape}")
print(f"[INFO] Labels: {y.value_counts().to_dict()}")

wins = (y == 1).sum()
losses = (y == 0).sum()
print(f"[INFO] Win Rate: {wins}/{len(y)} = {wins/len(y)*100:.1f}%")

# ============================================================
# 3. TRAIN ENSEMBLE MODEL
# ============================================================

print("\n[TRAINING] Building FX Pro Ensemble...")

# Three RF models with different configurations
rf1 = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf2 = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=123,
    n_jobs=-1
)

rf3 = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=12,
    min_samples_leaf=6,
    class_weight='balanced',
    random_state=456,
    n_jobs=-1
)

# Ensemble voting
ensemble = VotingClassifier(
    estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)],
    voting='soft'
)

# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

print("\n[CV] Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')

print(f"[CV] Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
print(f"[CV] Per-fold: {[f'{s*100:.1f}%' for s in cv_scores]}")

# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================

print("\n[TRAINING] Training final model on all data...")
ensemble.fit(X, y)

# Feature importance (from first RF in ensemble)
importances = ensemble.estimators_[0].feature_importances_
feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)

print("\n[FEATURES] Top features for FX:")
for feat, imp in feat_imp[:10]:
    print(f"  {feat:20} {imp:.4f}")

# ============================================================
# 6. FINAL EVALUATION
# ============================================================

y_pred = ensemble.predict(X)
y_proba = ensemble.predict_proba(X)[:, 1]

print("\n[EVAL] Classification Report:")
print(classification_report(y, y_pred, target_names=['Loss', 'Win']))

print("[EVAL] Confusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(f"  TN={cm[0,0]:4}  FP={cm[0,1]:4}")
print(f"  FN={cm[1,0]:4}  TP={cm[1,1]:4}")

try:
    auc = roc_auc_score(y, y_proba)
    print(f"\n[EVAL] ROC AUC: {auc:.3f}")
except:
    auc = 0

# ============================================================
# 7. SAVE MODEL
# ============================================================

model_path = 'model_fx_pro.joblib'
features_path = 'model_fx_pro_features.txt'
report_path = 'model_fx_pro_report.txt'

joblib.dump(ensemble, model_path)
print(f"\n[SAVED] Model: {model_path}")

with open(features_path, 'w') as f:
    for feat in available_features:
        f.write(f"{feat}\n")
print(f"[SAVED] Features: {features_path}")

# Save report
with open(report_path, 'w') as f:
    f.write("FX PRO ML Model Report\n")
    f.write("=" * 50 + "\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Training samples: {len(df_fx)}\n")
    f.write(f"Unique pairs: {df_fx['symbol'].nunique()}\n")
    f.write(f"Features: {len(available_features)}\n")
    f.write(f"Win Rate in data: {wins/len(y)*100:.1f}%\n")
    f.write(f"CV Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)\n")
    f.write(f"ROC AUC: {auc:.3f}\n")
    f.write("\nTop Features:\n")
    for feat, imp in feat_imp[:10]:
        f.write(f"  {feat}: {imp:.4f}\n")
    f.write("\n" + classification_report(y, y_pred, target_names=['Loss', 'Win']))
print(f"[SAVED] Report: {report_path}")

print("\n" + "=" * 60)
print("FX PRO ML MODEL READY!")
print("=" * 60)
print(f"  Model: {model_path}")
print(f"  Trades: {len(df_fx)}")
print(f"  CV Accuracy: {cv_scores.mean()*100:.1f}%")
print("=" * 60)
