import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.jsonl")

trades = []
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("outcome")]

def get_session(ts_str):
    if not ts_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        h = dt.hour
        if 7 <= h < 13:
            return "LONDON"
        elif 13 <= h < 21:
            return "NEW_YORK"
        elif 0 <= h < 7 or h >= 21:
            return "TOKYO"
        else:
            return "OTHER"
    except:
        return "unknown"

ny_trades = [t for t in closed if get_session(t.get("close_time")) == "NEW_YORK"]

print("=" * 70)
print("  NEW YORK SESSION - FULL BREAKDOWN")
print(f"  Total NY trades: {len(ny_trades)}")
print("=" * 70)

# By group
print(f"\n  {'GROUP':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WR%':>6} {'P/L':>10} {'AVG_W':>8} {'AVG_L':>8}")
print("-" * 70)

groups = {}
for t in ny_trades:
    g = t.get("group", "unknown")
    if g not in groups:
        groups[g] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0, "win_pnl": 0.0, "loss_pnl": 0.0}
    groups[g]["trades"] += 1
    p = float(t.get("profit", 0))
    groups[g]["pnl"] += p
    if t["outcome"] == "WIN":
        groups[g]["wins"] += 1
        groups[g]["win_pnl"] += p
    else:
        groups[g]["losses"] += 1
        groups[g]["loss_pnl"] += p

for g in sorted(groups, key=lambda x: groups[x]["pnl"]):
    d = groups[g]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    aw = d["win_pnl"] / d["wins"] if d["wins"] else 0
    al = d["loss_pnl"] / d["losses"] if d["losses"] else 0
    print(f"  {g:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['pnl']:>+10.2f} {aw:>+8.2f} {al:>+8.2f}")

# By symbol (worst first)
print(f"\n  {'SYMBOL':<14} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WR%':>6} {'P/L':>10} {'AVG_W':>8} {'AVG_L':>8}")
print("-" * 70)

syms = {}
for t in ny_trades:
    s = t["symbol"]
    if s not in syms:
        syms[s] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0, "win_pnl": 0.0, "loss_pnl": 0.0}
    syms[s]["trades"] += 1
    p = float(t.get("profit", 0))
    syms[s]["pnl"] += p
    if t["outcome"] == "WIN":
        syms[s]["wins"] += 1
        syms[s]["win_pnl"] += p
    else:
        syms[s]["losses"] += 1
        syms[s]["loss_pnl"] += p

for s in sorted(syms, key=lambda x: syms[x]["pnl"]):
    d = syms[s]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    aw = d["win_pnl"] / d["wins"] if d["wins"] else 0
    al = d["loss_pnl"] / d["losses"] if d["losses"] else 0
    print(f"  {s:<14} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['pnl']:>+10.2f} {aw:>+8.2f} {al:>+8.2f}")
