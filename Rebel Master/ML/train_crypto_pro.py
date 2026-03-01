"""
train_crypto_pro.py

CRYPTO PRO ML Model - Specialized Random Forest for Crypto Trading

Trained ONLY on crypto trades from the 812-trade backup dataset.
Learns patterns specific to crypto markets (24/7, high volatility, etc.)

Usage: python train_crypto_pro.py
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
print("CRYPTO PRO ML - Random Forest Training")
print("=" * 60)

# ============================================================
# 1. LOAD AND FILTER CRYPTO TRADES
# ============================================================

df = pd.read_csv('training_dataset_backup_812.csv')
print(f"[INFO] Loaded {len(df)} total trades from backup")

# Crypto keywords
crypto_keys = ['BTC','ETH','XRP','ADA','SOL','DOGE','LTC','LINK','UNI','AAVE',
               'DOT','BNB','XLM','AVAX','MATIC','SHIB','ATOM','ALGO','FTM','SAND',
               'MANA','AXS','CRV','COMP','SUSHI','YFI','SNX','MKR','ENJ','BAT',
               'ZEC','DASH','XMR','EOS','TRX','XTZ','LRC','BCH','NEAR','APE',
               'FIL','THETA','VET','ICP','HBAR','EGLD','FLOW','CHZ','GALA','IMX']

def is_crypto(symbol):
    s = symbol.upper()
    return any(k in s for k in crypto_keys) or '-USD' in s

df_crypto = df[df['symbol'].apply(is_crypto)].copy()
print(f"[INFO] Filtered to {len(df_crypto)} crypto trades")

# ============================================================
# 2. FEATURE SELECTION (Crypto-optimized)
# ============================================================

# Features that matter for crypto
feature_cols = [
    'score',           # Signal strength
    'rsi',             # Momentum (crypto is momentum-driven)
    'atr',             # Volatility (high for crypto)
    'adx',             # Trend strength
    'macd_hist',       # MACD histogram
    'trend_bias',      # Trend direction
    'volatility_regime',# Volatility state
    'spread_ratio',    # Spread cost
    'ema_fast',        # Fast EMA
    'ema_slow',        # Slow EMA
]

# Check which features exist
available_features = [f for f in feature_cols if f in df_crypto.columns]
print(f"[INFO] Using {len(available_features)} features: {available_features}")

X = df_crypto[available_features].copy()
y = df_crypto['label'].copy()

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

print("\n[TRAINING] Building Crypto Pro Ensemble...")

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

print("\n[FEATURES] Top features for crypto:")
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

model_path = 'model_crypto_pro.joblib'
features_path = 'model_crypto_pro_features.txt'
report_path = 'model_crypto_pro_report.txt'

joblib.dump(ensemble, model_path)
print(f"\n[SAVED] Model: {model_path}")

with open(features_path, 'w') as f:
    for feat in available_features:
        f.write(f"{feat}\n")
print(f"[SAVED] Features: {features_path}")

# Save report
with open(report_path, 'w') as f:
    f.write("CRYPTO PRO ML Model Report\n")
    f.write("=" * 50 + "\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Training samples: {len(df_crypto)}\n")
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
print("CRYPTO PRO ML MODEL READY!")
print("=" * 60)
print(f"  Model: {model_path}")
print(f"  Trades: {len(df_crypto)}")
print(f"  CV Accuracy: {cv_scores.mean()*100:.1f}%")
print("=" * 60)
