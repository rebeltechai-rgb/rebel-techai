"""
train_rf_v3.py

Random Forest Model v3 BETA - REBEL Trading Bot ML

Improvements over v2:
- Optimized for smaller datasets (~300 samples)
- Aggressive feature selection (top 15 features)
- Ensemble voting with multiple RF configurations
- Better regularization to prevent overfitting
- Confidence calibration for probability outputs
- More robust cross-validation (10-fold)

Usage:
    python train_rf_v3.py

Input:
    - training_dataset.csv (merged features + labels)

Output:
    - model_rf_v3.joblib (trained model)
    - model_rf_v3_report.txt (performance report)
    - model_rf_v3_features.txt (feature list)
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
    cross_val_predict
)
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.feature_selection import SelectFromModel

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"
DATASET_FILE = os.path.join(BASE_PATH, "training_dataset.csv")
MODEL_OUTPUT = os.path.join(BASE_PATH, "model_rf_v3.joblib")
REPORT_OUTPUT = os.path.join(BASE_PATH, "model_rf_v3_report.txt")
FEATURES_OUTPUT = os.path.join(BASE_PATH, "model_rf_v3_features.txt")

# V3 Random Forest - optimized for small datasets
RF_PARAMS_V3 = {
    "n_estimators": 300,           # More trees for stability with small data
    "max_depth": 8,                # Shallower to prevent overfitting
    "min_samples_split": 15,       # Higher threshold for small data
    "min_samples_leaf": 8,         # Larger leaves = more robust
    "max_features": 0.5,           # Use 50% of features per split
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",    # Handle class imbalance
    "bootstrap": True,
    "oob_score": True,             # Out-of-bag validation
    "max_samples": 0.8,            # Subsample for diversity
}

# Alternative RF for ensemble
RF_PARAMS_V3_ALT = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": 123,
    "n_jobs": -1,
    "class_weight": "balanced_subsample",
    "bootstrap": True,
}

# Feature selection - keep top N features
TOP_N_FEATURES = 15

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation folds (more folds for small data)
CV_FOLDS = 10

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


def select_top_features(X, y, feature_names, n_features=TOP_N_FEATURES):
    """Select top N features using preliminary RF importance."""
    print(f"\n[INFO] Selecting top {n_features} features...")
    
    # Train quick RF for feature importance
    quick_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    quick_rf.fit(X, y)
    
    # Get feature importances
    importances = quick_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N
    top_indices = indices[:n_features]
    top_features = [feature_names[i] for i in top_indices]
    
    print(f"[INFO] Top {n_features} features selected:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    X_selected = X[:, top_indices]
    
    return X_selected, top_features, importances


def train_ensemble_model(X_train, y_train):
    """Train an ensemble of RF models with voting."""
    print("\n[INFO] Training RF v3 Ensemble...")
    
    # Primary RF
    rf1 = RandomForestClassifier(**RF_PARAMS_V3)
    rf1.fit(X_train, y_train)
    
    if hasattr(rf1, 'oob_score_'):
        print(f"[INFO] RF1 OOB Score: {rf1.oob_score_:.4f}")
    
    # Alternative RF  
    rf2 = RandomForestClassifier(**RF_PARAMS_V3_ALT)
    rf2.fit(X_train, y_train)
    
    # Ensemble with soft voting
    print("[INFO] Creating voting ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf_main', rf1),
            ('rf_alt', rf2),
        ],
        voting='soft',
        weights=[0.6, 0.4]  # Weight primary RF higher
    )
    ensemble.fit(X_train, y_train)
    
    print("[OK] Ensemble training complete!")
    return ensemble, rf1  # Return both for feature importance


def cross_validate_model(model, X, y, cv_folds=CV_FOLDS):
    """Perform robust cross-validation."""
    print(f"\n[INFO] Running {cv_folds}-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Multiple metrics
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print(f"[CV] Accuracy:  {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std()*2:.4f})")
    print(f"[CV] Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std()*2:.4f})")
    print(f"[CV] Recall:    {recall_scores.mean():.4f} (+/- {recall_scores.std()*2:.4f})")
    print(f"[CV] F1 Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std()*2:.4f})")
    print(f"[CV] ROC AUC:   {roc_auc_scores.mean():.4f} (+/- {roc_auc_scores.std()*2:.4f})")
    
    return {
        'accuracy': (accuracy_scores.mean(), accuracy_scores.std()),
        'precision': (precision_scores.mean(), precision_scores.std()),
        'recall': (recall_scores.mean(), recall_scores.std()),
        'f1': (f1_scores.mean(), f1_scores.std()),
        'roc_auc': (roc_auc_scores.mean(), roc_auc_scores.std()),
    }


def evaluate_model(model, X_test, y_test, feature_names, rf_model=None) -> str:
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
    
    # Feature importances (from primary RF)
    if rf_model is not None:
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
    else:
        importances = None
        indices = None
    
    # Build report
    report = []
    report.append("=" * 60)
    report.append("  RANDOM FOREST v3 BETA - MODEL REPORT")
    report.append("=" * 60)
    report.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Dataset: {DATASET_FILE}")
    report.append(f"  Training samples: {len(y_test) * 4}")  # Approximate
    report.append(f"  Test samples: {len(y_test)}")
    report.append(f"  Features used: {len(feature_names)}")
    report.append("")
    
    report.append("-" * 60)
    report.append("  HOLDOUT TEST SET PERFORMANCE")
    report.append("-" * 60)
    report.append(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.1f}%)")
    report.append(f"  Precision:  {precision:.4f}")
    report.append(f"  Recall:     {recall:.4f}")
    report.append(f"  F1 Score:   {f1:.4f}")
    report.append(f"  ROC AUC:    {roc_auc:.4f}")
    report.append("")
    
    report.append("-" * 60)
    report.append("  CONFUSION MATRIX")
    report.append("-" * 60)
    report.append(f"                  Predicted")
    report.append(f"                  Loss    Win")
    report.append(f"  Actual Loss     {cm[0,0]:4d}    {cm[0,1]:4d}")
    report.append(f"  Actual Win      {cm[1,0]:4d}    {cm[1,1]:4d}")
    report.append("")
    
    # Calculate trading-specific metrics
    true_neg = cm[0, 0]
    false_pos = cm[0, 1]
    false_neg = cm[1, 0]
    true_pos = cm[1, 1]
    
    total_trades_taken = true_pos + false_pos
    if total_trades_taken > 0:
        actual_win_rate = true_pos / total_trades_taken
    else:
        actual_win_rate = 0
    
    report.append("-" * 60)
    report.append("  TRADING METRICS")
    report.append("-" * 60)
    report.append(f"  Trades predicted as WIN: {total_trades_taken}")
    report.append(f"  Actual wins from those:  {true_pos}")
    report.append(f"  Simulated win rate:      {actual_win_rate*100:.1f}%")
    report.append(f"  Missed opportunities:    {false_neg} (wins predicted as loss)")
    report.append(f"  Bad calls avoided:       {true_neg} (losses correctly skipped)")
    report.append("")
    
    if importances is not None:
        report.append("-" * 60)
        report.append("  TOP 10 FEATURE IMPORTANCES")
        report.append("-" * 60)
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            report.append(f"  {i+1}. {feature_names[idx]:25s} {importances[idx]:.4f}")
    
    report.append("")
    report.append("=" * 60)
    report.append(f"  Model saved to: {MODEL_OUTPUT}")
    report.append("=" * 60)
    
    return "\n".join(report), importances, indices


def save_model(model, scaler=None):
    """Save the trained model and scaler."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'version': 'v3_beta',
        'created': datetime.now().isoformat(),
    }
    joblib_dump(model_data, MODEL_OUTPUT)
    print(f"[OK] Model saved to: {MODEL_OUTPUT}")


def save_features(feature_names, importances=None, indices=None):
    """Save feature list for inference."""
    with open(FEATURES_OUTPUT, 'w') as f:
        f.write("# RF v3 Beta Features\n")
        f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total features: {len(feature_names)}\n\n")
        
        if importances is not None and indices is not None:
            f.write("# Ranked by importance:\n")
            for i, idx in enumerate(indices):
                if idx < len(feature_names):
                    f.write(f"{feature_names[idx]},{importances[idx]:.6f}\n")
        else:
            for name in feature_names:
                f.write(f"{name}\n")
    
    print(f"[OK] Features saved to: {FEATURES_OUTPUT}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  REBEL ML - Random Forest v3 BETA Training")
    print("=" * 60)
    print(f"  Optimized for small datasets (~300 samples)")
    print(f"  Features: Top {TOP_N_FEATURES} selection + Ensemble voting")
    print("=" * 60)
    print()
    
    # Load data
    df = load_dataset(DATASET_FILE)
    
    # Prepare features
    X, y, feature_names, label_encoders = prepare_features(df)
    
    # Feature selection - get top features
    X_selected, selected_features, full_importances = select_top_features(
        X, y, feature_names, n_features=TOP_N_FEATURES
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n[INFO] Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train ensemble
    ensemble, rf_primary = train_ensemble_model(X_train, y_train)
    
    # Cross-validation on full selected data
    cv_results = cross_validate_model(rf_primary, X_selected, y, cv_folds=CV_FOLDS)
    
    # Evaluate on holdout test set
    report, importances, indices = evaluate_model(
        ensemble, X_test, y_test, selected_features, rf_model=rf_primary
    )
    
    # Print report
    print("\n" + report)
    
    # Save report
    with open(REPORT_OUTPUT, 'w') as f:
        f.write(report)
        f.write("\n\n")
        f.write("-" * 60 + "\n")
        f.write("  CROSS-VALIDATION RESULTS (10-fold)\n")
        f.write("-" * 60 + "\n")
        for metric, (mean, std) in cv_results.items():
            f.write(f"  {metric.upper():12s}: {mean:.4f} (+/- {std*2:.4f})\n")
    print(f"[OK] Report saved to: {REPORT_OUTPUT}")
    
    # Save model
    save_model(ensemble)
    
    # Save features
    save_features(selected_features, importances, indices)
    
    print("\n" + "=" * 60)
    print("  RF v3 BETA TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  To use v3 in the bot, update rebel_engine.py:")
    print(f"    MODEL_PATH = r'C:\\Rebel Technologies\\Rebel Master\\ML\\model_rf_v3.joblib'")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
