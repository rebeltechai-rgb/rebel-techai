"""
daily_checkin_report.py
- Daily/24h check-in summary for Rebel Master
"""

import os
import re
import json
from datetime import datetime, timedelta

import pandas as pd
import yaml

ROOT = r"C:\Rebel Technologies\Rebel Master"
LOG_DIR = os.path.join(ROOT, "logs")
CONFIG_PATH = os.path.join(ROOT, "Config", "master_config.yaml")
LIMITER_PATH = os.path.join(LOG_DIR, "trade_limiter_state.json")
TRADES_PATH = os.path.join(LOG_DIR, "trades.csv")
REJECTIONS_PATH = os.path.join(LOG_DIR, "filter_rejections.txt")
OUT_PATH = os.path.join(ROOT, "DAILY_CHECKIN.md")


def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def summarize_trades(since: datetime):
    if not os.path.exists(TRADES_PATH):
        return {}
    df = pd.read_csv(TRADES_PATH)
    if "timestamp" not in df.columns:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"] >= since]
    if df.empty:
        return {"count": 0}

    summary = {
        "count": len(df),
        "symbols": df["symbol"].value_counts().head(10).to_dict(),
    }
    for col in ("score", "rsi", "adx"):
        if col in df.columns:
            summary[f"avg_{col}"] = float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
    return summary


def summarize_rejections(since: datetime):
    if not os.path.exists(REJECTIONS_PATH):
        return {}
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| ([^|]+) \| (REJECTED|SOFT_ALLOW) \| Gate=([^|]+) \|")
    counts = {}
    with open(REJECTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            if ts < since:
                continue
            gate = match.group(4)
            counts[gate] = counts.get(gate, 0) + 1
    return counts


def build_report():
    now = datetime.now()
    since = now - timedelta(hours=24)
    cfg = load_yaml(CONFIG_PATH)
    limiter = load_json(LIMITER_PATH)

    lines = []
    lines.append("# REBEL MASTER DAILY CHECK-IN")
    lines.append(f"Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Window: last 24h since {since.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Core config
    strategy = cfg.get("strategy", {})
    trading = cfg.get("trading", {})
    risk_engine = cfg.get("risk_engine", {})
    baseline = cfg.get("baseline_locked", {})
    lines.append("## Core Settings")
    lines.append(f"- Auto trading: {trading.get('auto_trade', False)}")
    lines.append(f"- Strategy mode: {strategy.get('mode', 'unknown')}")
    lines.append(f"- Min score: {strategy.get('min_score', {}).get(strategy.get('mode', ''), 'n/a')}")
    lines.append(f"- Risk per trade: {risk_engine.get('percent_risk_per_trade', 'n/a')}%")
    lines.append(f"- Baseline lock: {baseline.get('enabled', False)}")
    lines.append("")

    # Trade limiter snapshot
    lines.append("## Trade Limiter (Today)")
    if limiter:
        lines.append(f"- Trades today total: {limiter.get('trades_today_total', 'n/a')}")
        lines.append(f"- Wins/Losses: {limiter.get('total_wins', 'n/a')} / {limiter.get('total_losses', 'n/a')}")
        per_class = limiter.get("trades_today_by_class", {}) or {}
        if per_class:
            class_parts = [f"{k}:{v}" for k, v in per_class.items()]
            lines.append(f"- By class: {', '.join(class_parts)}")
    else:
        lines.append("- No limiter state found.")
    lines.append("")

    # Trades summary
    lines.append("## Trades (Last 24h)")
    trade_summary = summarize_trades(since)
    if not trade_summary:
        lines.append("- No trades.csv data found.")
    else:
        lines.append(f"- Trade rows: {trade_summary.get('count', 0)}")
        for col in ("avg_score", "avg_rsi", "avg_adx"):
            if col in trade_summary:
                lines.append(f"- {col}: {trade_summary[col]:.2f}")
        if trade_summary.get("symbols"):
            lines.append("- Top symbols:")
            for sym, count in trade_summary["symbols"].items():
                lines.append(f"  - {sym}: {count}")
    lines.append("")

    # Rejections summary
    lines.append("## Filter Rejections (Last 24h)")
    rej = summarize_rejections(since)
    if not rej:
        lines.append("- No rejection log data.")
    else:
        for gate, count in sorted(rej.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {gate}: {count}")

    return "\n".join(lines)


def main():
    report = build_report()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
