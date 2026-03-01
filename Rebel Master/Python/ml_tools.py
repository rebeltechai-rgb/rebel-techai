"""
ML Tools - Dataset Preview & Validation Utilities
Provides quick inspection and integrity checking for ML datasets.
"""

import csv
import math
import os


# =============================================================================
# DATASET PREVIEW FUNCTIONS
# =============================================================================

def preview_features(path="C:/Rebel Technologies/Rebel Master/ML/features.csv", n=20):
    """
    Preview first N rows of the features dataset.
    
    Args:
        path: Path to features.csv
        n: Number of rows to preview (default: 20)
        
    Returns:
        List of dictionaries (one per row)
    
    Usage:
        from ml_tools import preview_features
        print(preview_features())
    """
    if not os.path.exists(path):
        print(f"[ML_TOOLS] File not found: {path}")
        return []
    
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    
    print(f"[ML_TOOLS] Loaded {len(rows)} feature rows from {path}")
    return rows


def preview_labels(path="C:/Rebel Technologies/Rebel Master/ML/labels.csv", n=20):
    """
    Preview first N rows of the labels dataset.
    
    Args:
        path: Path to labels.csv
        n: Number of rows to preview (default: 20)
        
    Returns:
        List of dictionaries (one per row)
    
    Usage:
        from ml_tools import preview_labels
        print(preview_labels())
    """
    if not os.path.exists(path):
        print(f"[ML_TOOLS] File not found: {path}")
        return []
    
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    
    print(f"[ML_TOOLS] Loaded {len(rows)} label rows from {path}")
    return rows


# =============================================================================
# DATA INTEGRITY CHECKER
# =============================================================================

def validate_dataset(path="C:/Rebel Technologies/Rebel Master/ML/features.csv"):
    """
    Validate dataset for missing values, NaNs, and corrupt rows.
    
    Args:
        path: Path to CSV file to validate
        
    Returns:
        List of tuples: (row_index, column_name, issue_type)
        
    Usage:
        from ml_tools import validate_dataset
        errors = validate_dataset()
        print(errors)
        
    Output example:
        [(123, 'atr', 'missing'), (128, 'spread_ratio', 'NaN'), ...]
    """
    if not os.path.exists(path):
        print(f"[ML_TOOLS] File not found: {path}")
        return [(-1, "file", "not_found")]
    
    issues = []
    row_count = 0
    
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row_count += 1
            for k, v in row.items():
                # Check for missing/empty values
                if v == "" or v is None:
                    issues.append((i, k, "missing"))
                else:
                    # Check for NaN values in numeric fields
                    try:
                        fval = float(v)
                        if math.isnan(fval):
                            issues.append((i, k, "NaN"))
                        elif math.isinf(fval):
                            issues.append((i, k, "Inf"))
                    except ValueError:
                        # Not a numeric field, that's okay
                        pass
    
    print(f"[ML_TOOLS] Validated {row_count} rows, found {len(issues)} issues")
    return issues


def dataset_stats(path="C:/Rebel Technologies/Rebel Master/ML/features.csv"):
    """
    Get basic statistics about the dataset.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Dictionary with dataset statistics
    """
    if not os.path.exists(path):
        print(f"[ML_TOOLS] File not found: {path}")
        return {}
    
    stats = {
        "total_rows": 0,
        "columns": [],
        "symbols": set(),
        "missing_by_column": {},
    }
    
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        stats["columns"] = reader.fieldnames or []
        
        # Initialize missing counts
        for col in stats["columns"]:
            stats["missing_by_column"][col] = 0
        
        for row in reader:
            stats["total_rows"] += 1
            
            # Track unique symbols
            if "symbol" in row:
                stats["symbols"].add(row["symbol"])
            
            # Count missing values per column
            for col in stats["columns"]:
                if row.get(col, "") == "" or row.get(col) is None:
                    stats["missing_by_column"][col] += 1
    
    stats["symbols"] = list(stats["symbols"])
    stats["unique_symbols"] = len(stats["symbols"])
    
    print(f"[ML_TOOLS] Dataset: {stats['total_rows']} rows, {len(stats['columns'])} columns, {stats['unique_symbols']} symbols")
    return stats


def compare_feature_label_counts(
    features_path="C:/Rebel Technologies/Rebel Master/ML/features.csv",
    labels_path="C:/Rebel Technologies/Rebel Master/ML/labels.csv"
):
    """
    Compare row counts between features and labels to check alignment.
    
    Returns:
        Dictionary with counts and alignment status
    """
    feature_count = 0
    label_count = 0
    
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            feature_count = sum(1 for _ in f) - 1  # Subtract header
    
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            label_count = sum(1 for _ in f) - 1  # Subtract header
    
    result = {
        "features": feature_count,
        "labels": label_count,
        "aligned": feature_count == label_count,
        "difference": abs(feature_count - label_count)
    }
    
    print(f"[ML_TOOLS] Features: {feature_count}, Labels: {label_count}, Aligned: {result['aligned']}")
    return result


# =============================================================================
# QUICK USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n=== ML Tools Demo ===\n")
    
    # Preview features
    print("--- Features Preview ---")
    features = preview_features(n=5)
    for f in features[:3]:
        print(f)
    
    # Preview labels
    print("\n--- Labels Preview ---")
    labels = preview_labels(n=5)
    for l in labels[:3]:
        print(l)
    
    # Validate
    print("\n--- Validation ---")
    issues = validate_dataset()
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  Row {issue[0]}, Column '{issue[1]}': {issue[2]}")
    else:
        print("No issues found!")
    
    # Stats
    print("\n--- Dataset Stats ---")
    stats = dataset_stats()
    print(f"Columns: {stats.get('columns', [])}")

