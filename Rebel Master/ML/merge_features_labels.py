"""
STEP C — Merge Trade Features + Labels into Unified ML Training Dataset

Pipeline:
1. Load trade_features.csv (features captured at trade ENTRY time, with ticket)
2. Apply ML feature engineering (add 27 new features)
3. Merge with labels.csv (trade outcomes, with ticket)
4. Save training_dataset.csv

Output: training_dataset.csv
- Each row = one trade with features + engineered features + outcome
- Ready for XGBoost, RandomForest, LightGBM, LSTM, etc.
"""

import pandas as pd
import os

# Import feature engineering
from ml_feature_engineering import add_ml_features

BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"

# Use trade_features.csv (logged at trade entry with ticket)
FEATURES_FILE = os.path.join(BASE_PATH, "trade_features.csv")
LABELS_FILE = os.path.join(BASE_PATH, "labels.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "training_dataset.csv")

# Drop break-even trades to avoid noisy labels
DROP_BREAK_EVEN = True
BREAK_EVEN_EPS = 1e-8
BREAK_EVEN_RR_THRESHOLD = 0.05

print("=== ML PIPELINE: FEATURES + ENGINEERING + LABELS ===")

# ------------------------------------------------
# STEP 1: LOAD FEATURES
# ------------------------------------------------
if not os.path.exists(FEATURES_FILE):
    print(f"[ERROR] Features file not found: {FEATURES_FILE}")
    print("[INFO] Run REBEL bot with auto_trade enabled to generate trade_features.csv")
    print("[INFO] Features are logged when trades are opened.")
    exit(1)

df_features = pd.read_csv(FEATURES_FILE)
print(f"[INFO] Loaded {len(df_features)} trade feature rows from trade_features.csv")

# ------------------------------------------------
# STEP 2: APPLY ML FEATURE ENGINEERING
# ------------------------------------------------
print("[INFO] Applying ML feature engineering...")
df_ml = add_ml_features(df_features)
print(f"[OK] Added {len(df_ml.columns) - len(df_features.columns)} engineered features")
print(f"[INFO] Total features: {len(df_ml.columns)}")

# ------------------------------------------------
# STEP 3: LOAD LABELS
# ------------------------------------------------
if not os.path.exists(LABELS_FILE):
    print(f"[ERROR] Labels file not found: {LABELS_FILE}")
    exit(1)

# Handle inconsistent labels.csv format - skip bad lines
try:
    labels = pd.read_csv(LABELS_FILE, on_bad_lines='skip')
    print(f"[INFO] Loaded {len(labels)} label rows from labels.csv")
except Exception as e:
    print(f"[ERROR] Failed to load labels: {e}")
    exit(1)

# ------------------------------------------------
# STEP 4: OPTIONAL LABEL CLEANUP
# ------------------------------------------------
if DROP_BREAK_EVEN:
    before = len(labels)
    removed = 0
    if "pnl" in labels.columns:
        labels = labels[labels["pnl"].abs() > BREAK_EVEN_EPS]
        removed = before - len(labels)
    if "rr" in labels.columns:
        before_rr = len(labels)
        labels = labels[labels["rr"].abs() > BREAK_EVEN_RR_THRESHOLD]
        removed += before_rr - len(labels)
    if removed:
        print(f"[INFO] Break-even labels removed: {removed} (from {before} -> {len(labels)})")

# ------------------------------------------------
# STEP 5: PREP TICKETS FOR MERGE
# ------------------------------------------------
if "ticket" not in df_ml.columns:
    print("[ERROR] trade_features.csv missing 'ticket' column!")
    exit(1)

if "ticket" not in labels.columns:
    print("[ERROR] labels.csv missing 'ticket' column!")
    print("[INFO] Your labels file may be using an older format.")
    print("[INFO] Labels need to include ticket ID for proper merging.")
    exit(1)

df_ml["ticket"] = pd.to_numeric(df_ml["ticket"], errors="coerce")
labels["ticket"] = pd.to_numeric(labels["ticket"], errors="coerce")

# Drop rows with invalid tickets
df_ml = df_ml.dropna(subset=["ticket"])
labels = labels.dropna(subset=["ticket"])

print(f"[INFO] Valid feature rows with ticket: {len(df_ml)}")
print(f"[INFO] Valid label rows with ticket: {len(labels)}")

# ------------------------------------------------
# STEP 6: MERGE ON TICKET
# ------------------------------------------------
merged = pd.merge(
    df_ml,
    labels,
    on="ticket",
    how="inner",
    suffixes=("_feat", "_label")
)

print(f"[OK] Merged dataset has {len(merged)} rows")

if len(merged) == 0:
    print("[WARNING] No rows matched!")
    print("[DEBUG] Feature tickets sample:", df_ml["ticket"].head(5).tolist())
    print("[DEBUG] Label tickets sample:", labels["ticket"].head(5).tolist())
    print("[INFO] Make sure labels are generated for trades that have features logged.")
else:
    # Show merge stats
    print(f"[INFO] Features without labels: {len(df_ml) - len(merged)}")
    print(f"[INFO] Labels without features: {len(labels) - len(merged)}")

# ------------------------------------------------
# STEP 7: CLEAN UP DUPLICATE COLUMNS
# ------------------------------------------------
# Prefer feature columns over label columns for duplicates
for col in list(merged.columns):
    if col.endswith("_feat"):
        base_col = col.replace("_feat", "")
        label_col = base_col + "_label"
        if label_col in merged.columns:
            # Keep the feature version, drop label version
            merged = merged.drop(columns=[label_col])
        merged = merged.rename(columns={col: base_col})
    elif col.endswith("_label"):
        base_col = col.replace("_label", "")
        if base_col not in merged.columns:
            merged = merged.rename(columns={col: base_col})

# ------------------------------------------------
# STEP 8: SAVE OUTPUT
# ------------------------------------------------
merged.to_csv(OUTPUT_FILE, index=False)
print(f"=== TRAINING DATASET SAVED -> {OUTPUT_FILE} ===")
print(f"[INFO] Total columns: {len(merged.columns)}")
print(f"[INFO] Total rows: {len(merged)}")

# Label summary (after any filtering)
if "label" in merged.columns:
    wins = int((merged["label"] == 1).sum())
    losses = int((merged["label"] == 0).sum())
    total = wins + losses
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    print(f"[INFO] Labels: wins={wins} losses={losses} win_rate={win_rate:.1f}%")

# Show feature summary
print("\n[INFO] Feature categories:")
print(f"  - Base features: 22")
print(f"  - Engineered features: 27")
print(f"  - Label columns: ~10-15")
print(f"  - Total: {len(merged.columns)}")
