"""
rebel_baseline_v1.py
- Simple deterministic baseline strategy (no ML yet)
- Goal: Test if your core signals have any persistent edge BEFORE adding RF/XGBoost back
- Uses your existing features (from trade_features.csv style)
- Walk-forward simulation on historical/merged data
- Outputs basic stats: win rate, PF, max DD, trade count

Run: python rebel_baseline_v1.py
(assumes training_dataset.csv exists from your merge)

Next steps after this:
1. Run on your current rows → see OOS performance
2. If PF > 1.2 in walk-forward → green light for ML v2
3. If <1.1 → need new rules/features/regime filters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# ────────────────────────────────────────────────
# CONFIG - Tweak these based on your intuition / data
# ────────────────────────────────────────────────

DATA_PATH = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"

# Core entry rules (AND conditions - all must be true)
BUY_RULES = {
    "ema_trend": lambda df: df["ema_fast"] > df["ema_slow"],
    "rsi_oversold": lambda df: df["rsi"] < 45,
    "adx_trending": lambda df: df["adx"] > 20,
    "rr_min": lambda df: df["reward_risk_ratio"] >= 1.5,
    "spread_ok": lambda df: df["spread_ratio"] < 0.0035,
}

SELL_RULES = {
    "ema_trend": lambda df: df["ema_fast"] < df["ema_slow"],
    "rsi_overbought": lambda df: df["rsi"] > 55,
    "adx_trending": lambda df: df["adx"] > 20,
    "rr_min": lambda df: df["reward_risk_ratio"] >= 1.5,
    "spread_ok": lambda df: df["spread_ratio"] < 0.0035,
}

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────


def apply_rules(df, rules, direction="buy"):
    """Returns boolean mask where ALL rules are True."""
    mask = pd.Series(True, index=df.index)
    for _, func in rules.items():
        condition = func(df)
        mask &= condition
    print(f"{direction.upper()} rule passed count: {mask.sum()} / {len(df)}")
    return mask


def simulate_trades(df):
    """Simple sim: assume entry on rule signal, outcome = label."""
    df = df.copy().sort_values("timestamp")  # ensure chronological

    buy_signals = apply_rules(df, BUY_RULES, "buy")
    sell_signals = apply_rules(df, SELL_RULES, "sell")

    # Combine: 1 = buy win, 0 = buy loss, -1 = sell win, etc.
    # For simplicity here: treat buy/sell symmetrically (label=1 = win)
    signals = pd.Series(0, index=df.index)
    signals[buy_signals] = 1
    signals[sell_signals] = 1  # assuming label=1 means directionally correct win

    trades = df[signals == 1].copy()
    trades["predicted"] = signals[signals == 1]
    trades["actual"] = trades["label"]

    return trades


def calculate_metrics(trades):
    if len(trades) == 0:
        return {"trades": 0, "win_rate": 0, "pf": 0, "max_dd": 0}

    wins = (trades["actual"] == 1).sum()
    losses = (trades["actual"] == 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Very basic profit factor (assuming 1:1 RR for baseline)
    # In real: use your actual R from reward_risk_ratio column
    profit_factor = wins / max(losses, 1e-6)

    # Simple equity curve for drawdown
    equity = np.cumsum(np.where(trades["actual"] == 1, 1, -1))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-6)
    max_dd = dd.min()

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# Assume timestamp exists and is datetime
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nRunning full-dataset simulation (for quick check)...")
all_trades = simulate_trades(df)
metrics_all = calculate_metrics(all_trades)
print("\nFull dataset metrics:")
for k, v in metrics_all.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

# ────────────────────────────────────────────────
# Walk-forward validation (real test)
# ────────────────────────────────────────────────
print("\nRunning walk-forward (TimeSeriesSplit)...")

tscv = TimeSeriesSplit(n_splits=5)
oos_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
    test_df = df.iloc[test_idx]

    # Simulate on OOS only
    oos_trades = simulate_trades(test_df)
    metrics = calculate_metrics(oos_trades)

    print(f"\nFold {fold}: {len(test_df)} rows")
    print(f"  OOS trades: {metrics['trades']}")
    print(f"  OOS win rate: {metrics['win_rate']:.3f}")
    print(f"  OOS PF: {metrics['profit_factor']:.2f}")
    print(f"  OOS max DD: {metrics['max_drawdown']:.3f}")

    oos_results.append(metrics)

# Average OOS
if oos_results:
    avg_win = np.mean([r["win_rate"] for r in oos_results])
    avg_pf = np.mean([r["profit_factor"] for r in oos_results])
    print(f"\nAverage OOS across folds:")
    print(f"  Win rate: {avg_win:.3f}")
    print(f"  Profit Factor: {avg_pf:.2f}")

print("\nDone. If OOS PF > 1.2–1.3 consistently → good sign to add ML filter on top.")
print("If <1.1 → regime filters or new features needed before ML.")
