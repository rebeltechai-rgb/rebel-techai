"""
rebel_baseline_v4.py
- Split buy/sell properly
- Add light EMA trend bias
- Keep spread <0.05 as main gate (strong performer)
- rr >=1.2, rsi 20–80
- Full walk-forward + per-direction metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"

# Symbols to exclude due to extreme spreads in dataset
SYMBOL_BLACKLIST = {
    "KSM-USD", "LRC-USD", "SUSHI-USD", "COMP-USD", "SOYBEAN.FS",
    "USDPLN", "XTZ-USD", "CRV-USD", "DOTUSD", "UNI-USD",
}

# ─── Rules ───────────────────────────────────────────────────────────────
BUY_RULES = {
    "spread_tight": lambda df: df.get("spread_ratio", 999) < 0.040,
    "rr_decent": lambda df: df.get("reward_risk_ratio", 0) >= 1.2,
    "rsi_ok": lambda df: (df.get("rsi", 50) < 80) & (df.get("rsi", 50) > 20),
    "ema_trend": lambda df: df.get("ema_fast", 0) > df.get("ema_slow", 0),
    "symbol_ok": lambda df: ~df.get("symbol", "").astype(str).str.upper().isin(SYMBOL_BLACKLIST),
}

SELL_RULES = {
    "spread_tight": lambda df: df.get("spread_ratio", 999) < 0.040,
    "rr_decent": lambda df: df.get("reward_risk_ratio", 0) >= 1.2,
    "rsi_ok": lambda df: (df.get("rsi", 50) < 80) & (df.get("rsi", 50) > 20),
    "ema_trend": lambda df: df.get("ema_fast", 0) < df.get("ema_slow", 0),
    "symbol_ok": lambda df: ~df.get("symbol", "").astype(str).str.upper().isin(SYMBOL_BLACKLIST),
}


def apply_rules_cumulative(df, rules, direction="buy"):
    print(f"\n{direction.upper()} Rule pass rates (cumulative):")
    mask = pd.Series(True, index=df.index)
    for name, func in rules.items():
        try:
            condition = func(df)
            mask &= condition
            print(f"  After {name:<18}: {mask.sum():4d} / {len(df):4d}  ({mask.mean():5.1%})")
        except Exception as e:
            print(f"  Error in {direction} rule '{name}': {e}")
            mask &= False
    return mask


def simulate_trades(df):
    df = df.copy().sort_values("timestamp") if "timestamp" in df.columns else df

    buy_mask = apply_rules_cumulative(df, BUY_RULES, "buy")
    sell_mask = apply_rules_cumulative(df, SELL_RULES, "sell")

    # Combine: buy=1, sell=-1
    signals = pd.Series(0, index=df.index)
    signals[buy_mask] = 1
    signals[sell_mask] = -1

    trades = df[signals != 0].copy()
    trades["direction"] = signals[signals != 0]  # 1=buy, -1=sell
    trades["actual"] = trades["label"]

    # R profit: +RR if win, -1 if loss (regardless of buy/sell)
    rr = trades.get("reward_risk_ratio", pd.Series(1.0, index=trades.index)).clip(lower=0.1)
    trades["r_profit"] = np.where(trades["actual"] == 1, rr, -1.0)
    trades["cum_r"] = trades["r_profit"].cumsum()
    return trades


def calculate_metrics(trades):
    if len(trades) == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "max_dd_r": 0.0,
            "buy_trades": 0,
            "sell_trades": 0,
        }

    wins = (trades["actual"] == 1).sum()
    total = len(trades)
    win_rate = wins / total if total > 0 else 0.0

    gross_profit = trades[trades["actual"] == 1]["reward_risk_ratio"].sum()
    gross_loss = trades[trades["actual"] == 0]["reward_risk_ratio"].abs().sum()
    profit_factor = gross_profit / max(gross_loss, 1e-6)

    equity_r = trades["cum_r"]
    peak = equity_r.cummax()
    dd_r = (equity_r - peak) / (peak.abs() + 1e-6)
    max_dd_r = dd_r.min()

    net_r = equity_r.iloc[-1] if len(equity_r) > 0 else 0.0

    buy_count = (trades["direction"] == 1).sum()
    sell_count = (trades["direction"] == -1).sum()

    return {
        "trades": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": net_r,
        "max_dd_r": max_dd_r,
        "buy_trades": buy_count,
        "sell_trades": sell_count,
    }


print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\n=== Quick Symbol View ===")
if "symbol" in df.columns:
    print("Top symbols:")
    print(df["symbol"].value_counts().head(10))
    if "spread_ratio" in df.columns:
        print("\nAvg spread by symbol:")
        print(df.groupby("symbol")["spread_ratio"].mean().sort_values(ascending=False).head(10))

print("\nFull dataset simulation...")
trades_full = simulate_trades(df)
metrics_full = calculate_metrics(trades_full)
print("\nFull metrics:")
for k, v in metrics_full.items():
    print(f"  {k:<14}: {v:.3f}" if isinstance(v, float) else f"  {k:<14}: {v}")

print("\nWalk-forward (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)
oos_results = []

for fold, (_, test_idx) in enumerate(tscv.split(df), 1):
    test_df = df.iloc[test_idx].copy()
    print(f"\nFold {fold} ({len(test_df)} rows)")
    trades_oos = simulate_trades(test_df)
    metrics = calculate_metrics(trades_oos)
    print("  OOS metrics:")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v:.3f}" if isinstance(v, float) else f"    {k:<12}: {v}")
    oos_results.append(metrics)

if oos_results:
    avg_wr = np.mean([r["win_rate"] for r in oos_results])
    avg_pf = np.mean([r["profit_factor"] for r in oos_results])
    avg_net_r = np.mean([r["net_r"] for r in oos_results])
    print("\nAverage OOS:")
    print(f"  win_rate:      {avg_wr:.3f}")
    print(f"  profit_factor: {avg_pf:.2f}")
    print(f"  net R:         {avg_net_r:.2f}")

print("\nDone. Look for:")
print("- Total trades >80–150")
print("- Win rate >60%?")
print("- PF >1.5?")
print("- Balanced buy/sell counts?")
print("If strong: this baseline can go into production filter.")
