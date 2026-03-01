"""
rebel_baseline_v2.py - Fixed crash + looser starting rules
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"

# ─── Much looser starting rules (adjust these based on your 57% period) ───
BUY_RULES = {
    "ema_trend":     lambda df: df["ema_fast"] > df["ema_slow"],               # keep
    "rsi_not_overbought": lambda df: df["rsi"] < 70,                           # was too strict <45
    "spread_ok":     lambda df: df["spread_ratio"] < 0.005,                    # was 0.0035 → allow more
    # "adx_trending": lambda df: df["adx"] > 20,                               # comment out if too restrictive
    # "rr_min":       lambda df: df["reward_risk_ratio"] >= 1.5,               # comment out to test volume
}

SELL_RULES = {
    "ema_trend":     lambda df: df["ema_fast"] < df["ema_slow"],
    "rsi_not_oversold": lambda df: df["rsi"] > 30,
    "spread_ok":     lambda df: df["spread_ratio"] < 0.005,
    # "adx_trending": lambda df: df["adx"] > 20,
    # "rr_min":       lambda df: df["reward_risk_ratio"] >= 1.5,
}


def apply_rules(df, rules, direction="buy"):
    mask = pd.Series(True, index=df.index)
    for name, func in rules.items():
        try:
            condition = func(df)
            mask &= condition
        except Exception as e:
            print(f"Error in rule '{name}' ({direction}): {e}")
            mask &= False
    print(f"{direction.upper()} rule passed count: {mask.sum()} / {len(df)}")
    return mask


def simulate_trades(df):
    df = df.copy().sort_values("timestamp") if "timestamp" in df.columns else df
    buy_signals = apply_rules(df, BUY_RULES, "buy")
    sell_signals = apply_rules(df, SELL_RULES, "sell")
    signals = pd.Series(0, index=df.index)
    signals[buy_signals] = 1
    signals[sell_signals] = 1   # treating both directions as "win if label=1"
    trades = df[signals == 1].copy()
    trades["signal"] = signals[signals == 1]
    trades["actual"] = trades["label"]
    return trades


def calculate_metrics(trades):
    if len(trades) == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    wins = (trades["actual"] == 1).sum()
    losses = (trades["actual"] == 0).sum()
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0

    # Basic PF (wins / losses) — later replace with real R:multiple
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


# ─── MAIN ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

print("\nFull dataset simulation...")
all_trades = simulate_trades(df)
metrics_all = calculate_metrics(all_trades)
print("\nFull dataset metrics:")
for k, v in metrics_all.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nWalk-forward validation...")
tscv = TimeSeriesSplit(n_splits=5)
oos_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
    test_df = df.iloc[test_idx]
    oos_trades = simulate_trades(test_df)
    metrics = calculate_metrics(oos_trades)
    print(f"\nFold {fold}: {len(test_df)} rows → {metrics['trades']} trades")
    print(f"  win_rate:    {metrics['win_rate']:.3f}")
    print(f"  profit_factor: {metrics['profit_factor']:.2f}")
    print(f"  max_dd:      {metrics['max_drawdown']:.3f}")
    oos_results.append(metrics)

if oos_results:
    avg_wr = np.mean([r["win_rate"] for r in oos_results])
    avg_pf = np.mean([r["profit_factor"] for r in oos_results])
    print("\nAverage OOS:")
    print(f"  win_rate:      {avg_wr:.3f}")
    print(f"  profit_factor: {avg_pf:.2f}")
