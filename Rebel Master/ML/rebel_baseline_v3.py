"""
rebel_baseline_v3.py
- Looser baseline with rule pass-rate stats
- Uses reward_risk_ratio for PF if available
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"

# Very light filters to gauge raw edge
BASE_RULES = {
    "spread_tight": lambda df: df.get("spread_ratio", 999) < 0.050,
    "rr_decent": lambda df: df.get("reward_risk_ratio", 0) >= 1.2,
    "rsi_ok": lambda df: (df.get("rsi", 50) < 80) & (df.get("rsi", 50) > 20),
}


def safe_col(df, name):
    if name not in df.columns:
        print(f"[WARN] Missing column: {name}")
        return pd.Series(False, index=df.index)
    return df[name]


def apply_rules(df, rules):
    mask = pd.Series(True, index=df.index)
    for name, func in rules.items():
        try:
            condition = func(df)
            mask &= condition
            print(f"Rule {name} pass: {condition.sum()} / {len(df)}")
        except Exception as e:
            print(f"Error in rule '{name}': {e}")
            mask &= False
    print(f"ALL rules pass count: {mask.sum()} / {len(df)}")
    return mask


def simulate_trades(df):
    df = df.copy().sort_values("timestamp") if "timestamp" in df.columns else df
    mask = apply_rules(df, BASE_RULES)
    trades = df[mask].copy()
    trades["actual"] = safe_col(trades, "label")
    return trades


def calculate_metrics(trades):
    if len(trades) == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    wins = (trades["actual"] == 1).sum()
    losses = (trades["actual"] == 0).sum()
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0

    # PF using reward_risk_ratio if available
    if "reward_risk_ratio" in trades.columns:
        rr = trades["reward_risk_ratio"].fillna(0)
        profit = rr[trades["actual"] == 1].sum()
        loss = rr[trades["actual"] == 0].sum()
        profit_factor = profit / max(loss, 1e-6)
    else:
        profit_factor = wins / max(losses, 1e-6)

    # Simple equity curve
    equity = np.cumsum(np.where(trades["actual"] == 1, 1, -1))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-6)
    max_dd = dd.min() if len(dd) > 0 else 0.0

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\n=== Spread & RR Quick View ===")
if "spread_ratio" in df.columns:
    sr = df["spread_ratio"]
    print(sr.describe())
    print(f"% < 0.008:  {(sr < 0.008).mean()*100:.1f}%")
    print(f"% < 0.012:  {(sr < 0.012).mean()*100:.1f}%")
    print(f"% < 0.015:  {(sr < 0.015).mean()*100:.1f}%")
    print(f"% < 0.020:  {(sr < 0.020).mean()*100:.1f}%")
    print(f"Median:     {sr.median():.5f}")
else:
    print("[WARN] spread_ratio column missing")

if "reward_risk_ratio" in df.columns:
    rr = df["reward_risk_ratio"]
    print(rr.describe())
    print(f"% >=1.0:   {(rr >= 1.0).mean()*100:.1f}%")
    print(f"% >=1.5:   {(rr >= 1.5).mean()*100:.1f}%")
else:
    print("[WARN] reward_risk_ratio column missing")

print("\nFull dataset simulation...")
all_trades = simulate_trades(df)
metrics_all = calculate_metrics(all_trades)
print("\nFull dataset metrics:")
for k, v in metrics_all.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nWalk-forward validation...")
tscv = TimeSeriesSplit(n_splits=5)
oos_results = []

for fold, (_, test_idx) in enumerate(tscv.split(df), 1):
    test_df = df.iloc[test_idx]
    oos_trades = simulate_trades(test_df)
    metrics = calculate_metrics(oos_trades)
    print(f"\nFold {fold}: {len(test_df)} rows → {metrics['trades']} trades")
    print(f"  win_rate: {metrics['win_rate']:.3f}")
    print(f"  profit_factor: {metrics['profit_factor']:.2f}")
    print(f"  max_dd: {metrics['max_drawdown']:.3f}")
    oos_results.append(metrics)

if oos_results:
    avg_wr = np.mean([r["win_rate"] for r in oos_results])
    avg_pf = np.mean([r["profit_factor"] for r in oos_results])
    print("\nAverage OOS:")
    print(f"  win_rate: {avg_wr:.3f}")
    print(f"  profit_factor: {avg_pf:.2f}")
