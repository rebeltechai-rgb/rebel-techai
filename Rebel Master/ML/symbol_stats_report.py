"""
symbol_stats_report.py
- Build per-symbol stats from trade_limiter_state.json + trades.csv
"""

import os
import json
import argparse
from datetime import datetime

import pandas as pd


LOG_DIR = r"C:\Rebel Technologies\Rebel Master\logs"
LIMITER_STATE_PATH = os.path.join(LOG_DIR, "trade_limiter_state.json")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "trades.csv")
DEFAULT_OUT = os.path.join(LOG_DIR, "symbol_stats_report.csv")


def load_limiter_stats():
    if not os.path.exists(LIMITER_STATE_PATH):
        return pd.DataFrame()
    with open(LIMITER_STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    stats = state.get("symbol_stats", {}) or {}
    rows = []
    for symbol, data in stats.items():
        wins = int(data.get("wins", 0) or 0)
        losses = int(data.get("losses", 0) or 0)
        total = wins + losses
        win_rate = wins / total if total > 0 else 0.0
        rows.append({
            "symbol": symbol.upper(),
            "wins": wins,
            "losses": losses,
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "consecutive_losses": int(data.get("consecutive_losses", 0) or 0),
            "last_loss_time": data.get("last_loss_time"),
            "cooldown_until": data.get("cooldown_until"),
            "last_session": data.get("last_session"),
            "last_session_day": data.get("last_session_day"),
            "last_loss_session": data.get("last_loss_session"),
            "last_loss_session_day": data.get("last_loss_session_day"),
        })
    return pd.DataFrame(rows)


def load_trade_stats():
    if not os.path.exists(TRADES_CSV_PATH):
        return pd.DataFrame()
    usecols = ["timestamp", "symbol", "direction", "score", "rsi", "adx", "atr", "result", "comment"]
    df = pd.read_csv(TRADES_CSV_PATH, usecols=[c for c in usecols if c in pd.read_csv(TRADES_CSV_PATH, nrows=1).columns])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper()

    agg = {
        "timestamp": "max",
        "direction": "count",
    }
    if "score" in df.columns:
        agg["score"] = "mean"
    if "rsi" in df.columns:
        agg["rsi"] = "mean"
    if "adx" in df.columns:
        agg["adx"] = "mean"
    if "atr" in df.columns:
        agg["atr"] = "mean"
    trade_stats = df.groupby("symbol").agg(agg).reset_index()
    trade_stats = trade_stats.rename(columns={
        "timestamp": "last_trade_time",
        "direction": "trade_count",
        "score": "avg_score",
        "rsi": "avg_rsi",
        "adx": "avg_adx",
        "atr": "avg_atr",
    })
    return trade_stats


def build_report():
    limiter_df = load_limiter_stats()
    trades_df = load_trade_stats()

    if limiter_df.empty and trades_df.empty:
        return pd.DataFrame()

    if limiter_df.empty:
        report = trades_df
    elif trades_df.empty:
        report = limiter_df
    else:
        report = pd.merge(trades_df, limiter_df, on="symbol", how="outer")

    # Fill numeric columns
    for col in ["trade_count", "wins", "losses", "total_trades", "consecutive_losses"]:
        if col in report.columns:
            report[col] = report[col].fillna(0).astype(int)

    for col in ["avg_score", "avg_rsi", "avg_adx", "avg_atr", "win_rate"]:
        if col in report.columns:
            report[col] = report[col].astype(float).round(4)

    # Normalize win_rate if missing but wins/losses exist
    if "win_rate" in report.columns and "wins" in report.columns and "losses" in report.columns:
        mask = report["win_rate"].isna()
        if mask.any():
            totals = report.loc[mask, "wins"] + report.loc[mask, "losses"]
            report.loc[mask, "win_rate"] = (report.loc[mask, "wins"] / totals.replace(0, pd.NA)).fillna(0.0)

    report = report.sort_values(by=["total_trades", "trade_count"], ascending=False, na_position="last")
    return report


def print_summary(report: pd.DataFrame, min_trades: int):
    if report.empty:
        print("No data available.")
        return

    print("\n=== TOP WORST WIN RATE (min trades) ===")
    if "win_rate" in report.columns:
        subset = report[report["total_trades"] >= min_trades]
        print(subset.sort_values(by="win_rate", ascending=True).head(15).to_string(index=False))

    print("\n=== MOST LOSSES ===")
    if "losses" in report.columns:
        print(report.sort_values(by="losses", ascending=False).head(15).to_string(index=False))

    print("\n=== HIGHEST CONSECUTIVE LOSSES ===")
    if "consecutive_losses" in report.columns:
        print(report.sort_values(by="consecutive_losses", ascending=False).head(15).to_string(index=False))

    print("\n=== MOST RECENT TRADES ===")
    if "last_trade_time" in report.columns:
        print(report.sort_values(by="last_trade_time", ascending=False).head(15).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Per-symbol stats report")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades for win-rate ranking")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    report = build_report()
    print_summary(report, args.min_trades)

    if not report.empty:
        out_path = args.out
        report.to_csv(out_path, index=False)
        print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
