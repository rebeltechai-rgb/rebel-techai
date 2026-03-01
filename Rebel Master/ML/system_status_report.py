"""
system_status_report.py
- Build a concise system status snapshot for Rebel Master
"""

import os
import json
from datetime import datetime

import yaml


ROOT = r"C:\Rebel Technologies\Rebel Master"
CONFIG_PATH = os.path.join(ROOT, "Config", "master_config.yaml")
BASELINE_PATH = os.path.join(ROOT, "ML", "baseline_locked.yaml")
LIMITER_PATH = os.path.join(ROOT, "logs", "trade_limiter_state.json")
OUT_PATH = os.path.join(ROOT, "STATUS.md")


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


def build_status():
    cfg = load_yaml(CONFIG_PATH)
    baseline = load_yaml(BASELINE_PATH)
    limiter = load_json(LIMITER_PATH)

    lines = []
    lines.append("# REBEL MASTER STATUS")
    lines.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Core mode
    strategy = cfg.get("strategy", {})
    trading = cfg.get("trading", {})
    risk = cfg.get("risk", {})
    risk_engine = cfg.get("risk_engine", {})
    ml = cfg.get("ml_filter", {})
    staging = (ml.get("staging") or {})

    lines.append("## Core Settings")
    lines.append(f"- Strategy mode: {strategy.get('mode', 'unknown')}")
    lines.append(f"- Auto trading: {trading.get('auto_trade', False)}")
    lines.append(f"- SL/TP ATR: {trading.get('sl_atr_multiplier')} / {trading.get('tp_atr_multiplier')}")
    lines.append(f"- Risk per trade (risk engine): {risk_engine.get('percent_risk_per_trade', 'n/a')}%")
    lines.append(f"- Max open trades: {risk.get('max_open_trades', 'n/a')}")
    lines.append(f"- Max daily drawdown: {risk.get('max_daily_drawdown', 'n/a')}")
    lines.append("")

    lines.append("## ML Filter")
    lines.append(f"- Enabled: {ml.get('enabled', False)}")
    lines.append(f"- Model: {ml.get('model', 'n/a')}")
    lines.append(f"- Min win prob: {ml.get('min_win_prob', 'n/a')}")
    lines.append(f"- Staging enabled: {staging.get('enabled', False)}")
    lines.append("")

    lines.append("## Baseline Locked")
    if baseline:
        rules = baseline.get("rules", {})
        lines.append(f"- Name: {baseline.get('name', 'n/a')}")
        lines.append(f"- Spread max: {rules.get('spread_tight', 'n/a')}")
        lines.append(f"- RR min: {rules.get('rr_decent', 'n/a')}")
        lines.append(f"- RSI range: {rules.get('rsi_ok', 'n/a')}")
        lines.append(f"- EMA bias: {rules.get('ema_trend_buy', 'n/a')} / {rules.get('ema_trend_sell', 'n/a')}")
    else:
        lines.append("- Baseline locked file not found.")
    lines.append("")

    lines.append("## Trade Limiter (Today)")
    if limiter:
        lines.append(f"- Trades today total: {limiter.get('trades_today_total', 'n/a')}")
        lines.append(f"- Wins / Losses: {limiter.get('total_wins', 'n/a')} / {limiter.get('total_losses', 'n/a')}")
        per_class = limiter.get("trades_today_by_class", {}) or {}
        if per_class:
            class_parts = [f"{k}:{v}" for k, v in per_class.items()]
            lines.append(f"- By class: {', '.join(class_parts)}")
    else:
        lines.append("- Trade limiter state not found.")

    lines.append("")
    return "\n".join(lines)


def main():
    status = build_status()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(status)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
