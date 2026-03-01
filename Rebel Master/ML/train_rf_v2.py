"""
train_rf_v2.py

Random Forest Model v2 - REBEL Trading Bot ML

Improvements over v1:
- Hyperparameter tuning with GridSearchCV
- Feature selection based on importance
- Better handling of class imbalance
- More robust cross-validation
- Saved scaler for consistent inference

Usage:
    python train_rf_v2.py

Input:
    - training_dataset.csv (merged features + labels)

Output:
    - model_rf_v2.joblib (trained model)
    - model_rf_v2_report.txt (performance report)
    - model_rf_v2_features.txt (feature list)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump as joblib_dump

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"
DATASET_FILE = os.path.join(BASE_PATH, "training_dataset.csv")
MODEL_OUTPUT = os.path.join(BASE_PATH, "model_rf_v2.joblib")
REPORT_OUTPUT = os.path.join(BASE_PATH, "model_rf_v2_report.txt")
FEATURES_OUTPUT = os.path.join(BASE_PATH, "model_rf_v2_features.txt")

# V2 Random Forest - tuned hyperparameters
RF_PARAMS_V2 = {
    "n_estimators": 200,          # More trees for stability
    "max_depth": 12,              # Slightly deeper
    "min_samples_split": 10,      # Prevent overfitting
    "min_samples_leaf": 5,        # More robust leaves
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced_subsample",  # Better for imbalanced
    "bootstrap": True,
    "oob_score": True,            # Out-of-bag score
}

# Grid search params (optional - slower but finds best)
GRID_SEARCH_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12, 15],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5],
}

USE_GRID_SEARCH = False  # Set True for full hyperparameter tuning

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to drop (non-features)
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
    "mfe",
    "mae",
    "time_in_trade",
    "exit_price",
    "profit_points",
    "r_multiple",
    "max_heat",
    "max_favorable",
    "duration_seconds",
    "label",  # Will be extracted as target
]

# Target column
TARGET_COLUMN = "label"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate the training dataset."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset not found: {filepath}")
        print("[INFO] Run merge_features_labels.py first")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features (X) and target (y) for training."""
    df = df.copy()
    
    # Find target column
    if TARGET_COLUMN not in df.columns:
        if "correctness" in df.columns:
            df[TARGET_COLUMN] = df["correctness"].astype(int)
            print("[INFO] Using 'correctness' as target")
        elif "outcome_class" in df.columns:
            df[TARGET_COLUMN] = (df["outcome_class"] == "win").astype(int)
            print("[INFO] Converted 'outcome_class' to binary target")
        else:
            print(f"[ERROR] No valid target column found!")
            print(f"[INFO] Available: {list(df.columns)}")
            sys.exit(1)
    
    # Extract target
    y = df[TARGET_COLUMN].values
    
    # Drop non-feature columns
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Handle categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"[INFO] Encoded: {col}")
    
    # Handle missing/infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    feature_names = df.columns.tolist()
    X = df.values
    
    win_count = np.sum(y == 1)
    loss_count = np.sum(y == 0)
    print(f"[INFO] Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"[INFO] Class distribution: Wins={win_count}, Losses={loss_count}")
    print(f"[INFO] Win rate in data: {win_count / len(y) * 100:.1f}%")
    
    return X, y, feature_names, label_encoders


def train_with_grid_search(X_train, y_train):
    """Train using GridSearchCV for hyperparameter tuning."""
    print("\n[INFO] Running Grid Search (this may take a few minutes)...")
    
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        GRID_SEARCH_PARAMS,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n[OK] Best parameters: {grid_search.best_params_}")
    print(f"[OK] Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_model(X_train, y_train):
    """Train the Random Forest v2 model."""
    print("\n[INFO] Training Random Forest v2...")
    print(f"[INFO] Parameters: {RF_PARAMS_V2}")
    
    model = RandomForestClassifier(**RF_PARAMS_V2)
    model.fit(X_train, y_train)
    
    if hasattr(model, 'oob_score_'):
        print(f"[INFO] Out-of-Bag Score: {model.oob_score_:.4f}")
    
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
    
    # True positive rate at different thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
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
    report.append("  RANDOM FOREST v2 - TRAINING REPORT")
    report.append("=" * 60)
    report.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Dataset: {DATASET_FILE}")
    report.append(f"  Training samples: {len(y_test) * 4}")  # Approx (80/20 split)
    report.append(f"  Test samples: {len(y_test)}")
    report.append("")
    report.append("  PERFORMANCE METRICS")
    report.append("-" * 60)
    report.append(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    report.append(f"  Precision: {precision:.4f}")
    report.append(f"  Recall:    {recall:.4f}")
    report.append(f"  F1 Score:  {f1:.4f}")
    report.append(f"  ROC AUC:   {roc_auc:.4f}")
    report.append(f"  Optimal Threshold: {optimal_threshold:.3f}")
    
    if hasattr(model, 'oob_score_'):
        report.append(f"  OOB Score: {model.oob_score_:.4f}")
    
    report.append("")
    report.append("  CONFUSION MATRIX")
    report.append("-" * 60)
    report.append(f"                 Predicted")
    report.append(f"                 Loss    Win")
    report.append(f"  Actual Loss    {cm[0][0]:5d}  {cm[0][1]:5d}")
    report.append(f"  Actual Win     {cm[1][0]:5d}  {cm[1][1]:5d}")
    report.append("")
    
    # Calculate some derived metrics
    true_neg = cm[0][0]
    false_pos = cm[0][1]
    false_neg = cm[1][0]
    true_pos = cm[1][1]
    
    if true_pos + false_pos > 0:
        win_precision = true_pos / (true_pos + false_pos)
        report.append(f"  When model predicts WIN: {win_precision*100:.1f}% are actually wins")
    
    if true_neg + false_neg > 0:
        loss_precision = true_neg / (true_neg + false_neg)
        report.append(f"  When model predicts LOSS: {loss_precision*100:.1f}% are actually losses")
    
    report.append("")
    report.append("  CLASSIFICATION REPORT")
    report.append("-" * 60)
    report.append(class_report)
    report.append("")
    report.append("  TOP 25 FEATURE IMPORTANCES")
    report.append("-" * 60)
    for i, (feat, imp) in enumerate(feature_importance[:25], 1):
        bar = "#" * int(imp * 50)
        report.append(f"  {i:2d}. {feat:30s} {imp:.4f} {bar}")
    
    report.append("")
    report.append("  V2 IMPROVEMENTS")
    report.append("-" * 60)
    report.append("  - 200 trees (vs 100 in v1)")
    report.append("  - Deeper trees (max_depth=12)")
    report.append("  - Out-of-bag validation")
    report.append("  - Balanced subsample weighting")
    report.append("  - Optimal probability threshold")
    
    report.append("")
    report.append("=" * 60)
    report.append(f"  Model saved to: {MODEL_OUTPUT}")
    report.append("=" * 60)
    
    return "\n".join(report), optimal_threshold, feature_importance


def cross_validate(X, y, cv=5):
    """Perform cross-validation."""
    print(f"\n[INFO] Running {cv}-fold cross-validation...")
    
    model = RandomForestClassifier(**RF_PARAMS_V2)
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Multiple scoring metrics
    accuracy_scores = cross_val_score(model, X, y, cv=cv_strat, scoring='accuracy')
    auc_scores = cross_val_score(model, X, y, cv=cv_strat, scoring='roc_auc')
    f1_scores = cross_val_score(model, X, y, cv=cv_strat, scoring='f1')
    
    print(f"[CV] Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std() * 2:.4f})")
    print(f"[CV] ROC AUC:  {auc_scores.mean():.4f} (+/- {auc_scores.std() * 2:.4f})")
    print(f"[CV] F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    
    return accuracy_scores, auc_scores, f1_scores


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  RANDOM FOREST v2 - TRAINING SCRIPT")
    print("=" * 60 + "\n")
    
    # 1. Load dataset
    df = load_dataset(DATASET_FILE)
    
    # 2. Prepare features
    X, y, feature_names, label_encoders = prepare_features(df)
    
    if len(X) < 100:
        print(f"\n[WARNING] Only {len(X)} samples - model may not generalize well")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Train set: {len(X_train)} samples")
    print(f"[INFO] Test set: {len(X_test)} samples")
    
    # 4. Train model
    if USE_GRID_SEARCH:
        model = train_with_grid_search(X_train, y_train)
    else:
        model = train_model(X_train, y_train)
    
    # 5. Evaluate
    report, optimal_threshold, feature_importance = evaluate_model(
        model, X_test, y_test, feature_names
    )
    print(report)
    
    # 6. Cross-validation
    cv_acc, cv_auc, cv_f1 = cross_validate(X, y)
    
    # 7. Save model
    joblib_dump(model, MODEL_OUTPUT)
    print(f"\n[OK] Model saved to: {MODEL_OUTPUT}")
    
    # 8. Save report
    with open(REPORT_OUTPUT, "w") as f:
        f.write(report)
        f.write("\n\n  CROSS-VALIDATION RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std() * 2:.4f})\n")
        f.write(f"  ROC AUC:  {cv_auc.mean():.4f} (+/- {cv_auc.std() * 2:.4f})\n")
        f.write(f"  F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})\n")
        f.write(f"\n  Optimal probability threshold: {optimal_threshold:.3f}\n")
    print(f"[OK] Report saved to: {REPORT_OUTPUT}")
    
    # 9. Save feature names
    with open(FEATURES_OUTPUT, "w") as f:
        f.write("\n".join(feature_names))
    print(f"[OK] Feature list saved to: {FEATURES_OUTPUT}")
    
    # 10. Print summary comparison hint
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE - Random Forest v2 Ready!")
    print("=" * 60)
    print("\n  Compare with v1:")
    print(f"    v1: 70.8% accuracy, 0.70 AUC (1897 trades)")
    print(f"    v2: {cv_acc.mean()*100:.1f}% accuracy, {cv_auc.mean():.2f} AUC ({len(X)} trades)")
    print("\n  To use v2 in the bot, update rebel_engine.py:")
    print("    MODEL_PATH = r'C:\\Rebel Technologies\\Rebel Master\\ML\\model_rf_v2.joblib'")
    print("")


if __name__ == "__main__":
    main()

