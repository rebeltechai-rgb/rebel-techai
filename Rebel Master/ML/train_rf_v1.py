"""
train_rf_v1.py

Random Forest Model v1 — REBEL Trading Bot ML

Trains a RandomForest classifier on trading data to predict win/loss outcomes.

Usage:
    python train_rf_v1.py

Input:
    - training_dataset.csv (merged features + labels)

Output:
    - model_rf_v1.joblib (trained model)
    - model_rf_v1_report.txt (performance report)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump as joblib_dump

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"
DATASET_FILE = os.path.join(BASE_PATH, "training_dataset.csv")
MODEL_OUTPUT = os.path.join(BASE_PATH, "model_rf_v1.joblib")
REPORT_OUTPUT = os.path.join(BASE_PATH, "model_rf_v1_report.txt")

# Random Forest hyperparameters (v1 - baseline)
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",  # Handle imbalanced classes
}

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to drop (non-features)
# These are either identifiers or OUTCOME data (not available at trade entry)
DROP_COLUMNS = [
    # Identifiers
    "timestamp",
    "ticket",
    "symbol",
    "reason",
    "comment",
    "close_timestamp",
    
    # OUTCOME COLUMNS (data leakage - these don't exist at entry time!)
    "outcome_class",
    "reward_ratio",
    "vol_norm_reward",
    "volatility_normalized_reward",
    "pnl",
    "rr",
    "norm_pnl",
    "mfe",              # Max Favorable Excursion (only known after trade)
    "mae",              # Max Adverse Excursion (only known after trade)
    "time_in_trade",    # Only known after trade closes
    "exit_price",       # Only known after trade closes
    "profit_points",    # Only known after trade closes
    "r_multiple",       # Only known after trade closes
    "max_heat",         # Only known during/after trade
    "max_favorable",    # Only known during/after trade
    "duration_seconds", # Only known after trade closes
]

# Target column (what we're predicting)
TARGET_COLUMN = "label"  # 1 = win, 0 = loss


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate the training dataset."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset not found: {filepath}")
        print("[INFO] Run merge_features_labels.py first to create training_dataset.csv")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features (X) and target (y) for training.
    
    Returns:
        (X, y, feature_names)
    """
    df = df.copy()
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        # Try alternative label columns
        if "correctness" in df.columns:
            df[TARGET_COLUMN] = df["correctness"].astype(int)
            print(f"[INFO] Using 'correctness' as target (1=win, 0=loss)")
        elif "outcome_class" in df.columns:
            df[TARGET_COLUMN] = (df["outcome_class"] == "win").astype(int)
            print(f"[INFO] Converted 'outcome_class' to binary target")
        else:
            print(f"[ERROR] No valid target column found!")
            print(f"[INFO] Available columns: {list(df.columns)}")
            sys.exit(1)
    
    # Extract target
    y = df[TARGET_COLUMN].values
    
    # Drop non-feature columns
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Also drop the target column from features
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col in df.columns:
            # Simple label encoding for now
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"[INFO] Encoded categorical column: {col}")
    
    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Get feature names
    feature_names = df.columns.tolist()
    
    # Convert to numpy array
    X = df.values
    
    print(f"[INFO] Prepared {X.shape[0]} samples with {X.shape[1]} features")
    print(f"[INFO] Target distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_names


def train_model(X_train, y_train):
    """Train the Random Forest model."""
    print("\n[INFO] Training Random Forest v1...")
    print(f"[INFO] Parameters: {RF_PARAMS}")
    
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    
    print("[OK] Model training complete!")
    return model


def evaluate_model(model, X_test, y_test, feature_names) -> str:
    """Evaluate the model and return a report string."""
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=["Loss", "Win"])
    
    # Feature importance
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build report
    report = []
    report.append("=" * 60)
    report.append("  RANDOM FOREST v1 - TRAINING REPORT")
    report.append("=" * 60)
    report.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Dataset: {DATASET_FILE}")
    report.append("")
    report.append("  PERFORMANCE METRICS")
    report.append("-" * 60)
    report.append(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    report.append(f"  Precision: {precision:.4f}")
    report.append(f"  Recall:    {recall:.4f}")
    report.append(f"  F1 Score:  {f1:.4f}")
    report.append(f"  ROC AUC:   {roc_auc:.4f}")
    report.append("")
    report.append("  CONFUSION MATRIX")
    report.append("-" * 60)
    report.append(f"                 Predicted")
    report.append(f"                 Loss    Win")
    report.append(f"  Actual Loss    {cm[0][0]:5d}  {cm[0][1]:5d}")
    report.append(f"  Actual Win     {cm[1][0]:5d}  {cm[1][1]:5d}")
    report.append("")
    report.append("  CLASSIFICATION REPORT")
    report.append("-" * 60)
    report.append(class_report)
    report.append("")
    report.append("  TOP 20 FEATURE IMPORTANCES")
    report.append("-" * 60)
    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        bar = "#" * int(imp * 50)
        report.append(f"  {i:2d}. {feat:30s} {imp:.4f} {bar}")
    report.append("")
    report.append("=" * 60)
    report.append(f"  Model saved to: {MODEL_OUTPUT}")
    report.append("=" * 60)
    
    return "\n".join(report)


def cross_validate(model, X, y, cv=5):
    """Perform cross-validation."""
    print(f"\n[INFO] Running {cv}-fold cross-validation...")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"[CV] Accuracy scores: {scores}")
    print(f"[CV] Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  RANDOM FOREST v1 — TRAINING SCRIPT")
    print("=" * 60 + "\n")
    
    # 1. Load dataset
    df = load_dataset(DATASET_FILE)
    
    # 2. Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Check if we have enough data
    if len(X) < 50:
        print(f"\n[WARNING] Only {len(X)} samples — model may not generalize well")
        print("[INFO] Recommend collecting at least 200+ trades before training")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Train set: {len(X_train)} samples")
    print(f"[INFO] Test set: {len(X_test)} samples")
    
    # 4. Train model
    model = train_model(X_train, y_train)
    
    # 5. Evaluate
    report = evaluate_model(model, X_test, y_test, feature_names)
    print(report)
    
    # 6. Cross-validation (on full dataset)
    cross_validate(RandomForestClassifier(**RF_PARAMS), X, y)
    
    # 7. Save model
    joblib_dump(model, MODEL_OUTPUT)
    print(f"\n[OK] Model saved to: {MODEL_OUTPUT}")
    
    # 8. Save report
    with open(REPORT_OUTPUT, "w") as f:
        f.write(report)
    print(f"[OK] Report saved to: {REPORT_OUTPUT}")
    
    # 9. Save feature names for inference
    feature_file = os.path.join(BASE_PATH, "model_rf_v1_features.txt")
    with open(feature_file, "w") as f:
        f.write("\n".join(feature_names))
    print(f"[OK] Feature list saved to: {feature_file}")
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Random Forest v1 Ready!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

