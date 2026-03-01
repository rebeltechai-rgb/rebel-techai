"""
REBEL TRADER — Build RF Training Dataset
Merges rf_features.csv (entry-time indicators) with trades.jsonl (outcomes)
to produce a labelled dataset ready for Random Forest training.

Usage:
    python tools/build_rf_dataset.py

Output:
    logs/rf_training_dataset.csv
"""

import csv
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "logs", "rf_features.csv")
TRADES_PATH = os.path.join(BASE_DIR, "logs", "trades.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "logs", "rf_training_dataset.csv")


def load_outcomes(trades_path: str) -> dict:
    """Load closed-trade outcomes keyed by ticket number."""
    outcomes = {}
    if not os.path.exists(trades_path):
        return outcomes
    with open(trades_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("direction") != "CLOSE":
                continue
            ticket = rec.get("ticket")
            if ticket is None:
                continue
            outcomes[int(ticket)] = {
                "profit": rec.get("profit", 0),
                "outcome": rec.get("outcome", ""),
                "exit_price": rec.get("exit_price", 0),
                "close_time": rec.get("close_time", ""),
            }
    return outcomes


def build_dataset():
    if not os.path.exists(FEATURES_PATH):
        print(f"[ERROR] Feature file not found: {FEATURES_PATH}")
        print("  Run Rebel Trader to collect features first.")
        sys.exit(1)

    outcomes = load_outcomes(TRADES_PATH)
    print(f"[INFO] Loaded {len(outcomes)} closed-trade outcomes from trades.jsonl")

    matched = 0
    unmatched = 0
    rows = []

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_cols = reader.fieldnames or []
        for row in reader:
            # Backward compat: default governance fields for older rows
            if not row.get("decision_source"):
                row["decision_source"] = "RULES"
            if not row.get("rules_version_id"):
                row["rules_version_id"] = "UNKNOWN_PRE_VERSION"

            ticket = row.get("ticket")
            try:
                ticket_int = int(ticket)
            except (TypeError, ValueError):
                unmatched += 1
                continue
            if ticket_int in outcomes:
                out = outcomes[ticket_int]
                row["profit"] = out["profit"]
                row["outcome"] = out["outcome"]
                row["label"] = 1 if out["outcome"] == "WIN" else 0
                row["exit_price"] = out["exit_price"]
                row["close_time"] = out["close_time"]
                matched += 1
            else:
                unmatched += 1
                continue
            rows.append(row)

    if not rows:
        print("[WARN] No matched rows — trades may still be open. Run again later.")
        return

    out_cols = feature_cols + ["profit", "outcome", "label", "exit_price", "close_time"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(rows)

    wins = sum(1 for r in rows if r["label"] == 1)
    losses = len(rows) - wins
    wr = (wins / len(rows) * 100) if rows else 0
    print(f"\n{'='*50}")
    print(f"RF TRAINING DATASET BUILT")
    print(f"{'='*50}")
    print(f"  Matched rows:   {matched}")
    print(f"  Still open:     {unmatched}")
    print(f"  Wins / Losses:  {wins} / {losses}  ({wr:.1f}%)")
    print(f"  Output:         {OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    build_dataset()
